//! Removing, disabling and re-keying a provider.
//!
//! Guards: G5 (clearing a key is in-use guarded, a rotation never is), G6
//! (removal is in-use guarded, deletes the key first and puts it back if the
//! record cannot be removed, and **never clears the default**), G7 (disabling is
//! in-use guarded and never scrubs a reference; a route to a disabled provider
//! fails closed), G21 (entry zero is read-only), G13 (a credential write drops
//! the scope's cached catalogs and the provider's health).

use crate::descriptor::RecordOrigin;
use crate::error::{HubError, InputField, InvalidInput, NotFound, Operation, UsedBy};
use crate::hub::{Confirm, Hub, Mutation, MutationStatus};
use crate::ids::{ScopeKey, Slug};
use crate::ports::HubEvent;
use crate::secret::Secret;
use crate::taxonomy::ProviderGroup;

impl Hub {
    /// Everything that references `slug`: the hub's own references, read from
    /// `config`, plus the host's.
    fn merge_used_by(config: &crate::config::HubConfig, slug: &Slug, host: &UsedBy) -> UsedBy {
        let mut used = Self::config_used_by(config, slug);
        for agent in &host.agents {
            if !used.agents.contains(agent) {
                used.agents.push(agent.clone());
            }
        }
        for workload in &host.workloads {
            if !used.workloads.contains(workload) {
                used.workloads.push(workload.clone());
            }
        }
        used.other.extend(host.other.iter().cloned());
        used.default_choice |= host.default_choice;
        used
    }

    /// Removes a provider and its key.
    ///
    /// Refused while anything references it (the default, an agent pin, a
    /// workload route, or something the host reports) unless the caller
    /// confirms; a confirmed removal leaves every reference in place, where it
    /// fails closed on the turn path. **The default is never cleared.** The key
    /// is deleted first and put back if the record cannot be removed.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`]; [`HubError::Unsupported`] for the managed
    /// provider and the read-only entry-zero record; [`HubError::InUse`];
    /// [`HubError::Conflict`] when the provider was removed and added again while
    /// this waited (it decided about another provider); and the stores' errors.
    pub async fn remove(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        confirm: Confirm,
    ) -> Result<Mutation, HubError> {
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        if self.group_of(&record) == ProviderGroup::Managed
            || record.origin == RecordOrigin::EntryZero
        {
            return Err(HubError::Unsupported {
                op: Operation::Remove,
                kind: record.kind.clone(),
            });
        }
        let host = self.host_used_by(scope, slug).await?;
        let used = Self::merge_used_by(&config, slug, &host);
        if !used.is_empty() && !confirm.in_use {
            return Err(HubError::InUse(used));
        }
        let view = self.view(scope, &record, &config).await;

        // The key delete and the record's removal are one unit against every
        // other key operation on this provider (a concurrent add of the slug
        // must not find the slot half-cleared, nor a key write land in it).
        let _guard = self.slot_lock(scope, slug).await;
        // What was read before the lock may have been replaced while this waited
        // for it (a removal and a new add of the slug): this call decided about
        // one provider and must not delete another. The hub-owned references are
        // re-checked by the transaction below.
        let now = self.read_config(scope).await?;
        match now.provider(slug) {
            None => return Err(HubError::NotFound(NotFound::Provider(slug.clone()))),
            Some(current) if current.id != record.id => return Err(HubError::Conflict),
            Some(_) => {}
        }
        let previous = self.read_slot(scope, slug).await?;
        if previous.is_some()
            && let Err(error) = self.delete_slot(scope, slug).await
        {
            // A store can delete and then report a failure: the key goes back,
            // so "nothing happened" is true.
            self.restore_or_announce(scope, slug, previous.clone())
                .await;
            return Err(error);
        }
        let committed = self
            .transact(scope, |config| {
                if config.provider(slug).is_none() {
                    return Err(HubError::NotFound(NotFound::Provider(slug.clone())));
                }
                let used = Self::merge_used_by(config, slug, &host);
                if !used.is_empty() && !confirm.in_use {
                    return Err(HubError::InUse(used));
                }
                config.providers.retain(|p| &p.slug != slug);
                Ok(used)
            })
            .await;
        let used = match committed {
            Ok(committed) => committed.value,
            Err(error) => {
                // A record somebody else removed while this ran took its key with
                // it: putting the key back would leave a slot no record owns.
                if matches!(error, HubError::NotFound(_)) {
                    return Err(error);
                }
                // A store that committed the removal and then reported a failure
                // is looked at, not assumed away: the record is read back.
                let still_there = self
                    .read_config(scope)
                    .await
                    .ok()
                    .map(|c| c.provider(slug).is_some_and(|p| p.id == record.id));
                // What the transaction itself refused did not commit: a record
                // that is gone then was removed by somebody else.
                let refused = matches!(
                    error,
                    HubError::Conflict | HubError::InUse(_) | HubError::Invalid(_)
                );
                match (still_there, previous) {
                    (Some(false), _) if refused => {
                        return Err(HubError::NotFound(NotFound::Provider(slug.clone())));
                    }
                    // A store error is ambiguous: the record is gone, so the
                    // removal is taken to be this call's (at worst another writer's
                    // removal is announced twice). Carry on as one.
                    (Some(false), _) => Self::merge_used_by(&now, slug, &host),
                    // Still there: the key goes back beside it.
                    (Some(true), Some(previous)) => {
                        self.restore_or_announce(scope, slug, Some(previous)).await;
                        return Err(error);
                    }
                    // Unreadable: cannot tell, so the slot is left as it is and
                    // said so.
                    (None, Some(_)) => {
                        self.announce_key_state(scope, slug).await;
                        return Err(error);
                    }
                    _ => return Err(error),
                }
            }
        };
        self.inner.cache.evict_scope(scope);
        self.forget_health(scope, slug).await;
        self.inner.events.emit(HubEvent::ProviderRemoved {
            scope: scope.clone(),
            slug: slug.clone(),
        });
        Ok(Mutation {
            status: MutationStatus::Saved,
            note: format!("{} was removed.", record.label),
            probe: None,
            used_by: (!used.is_empty()).then_some(used),
            record: Some(view),
        })
    }

    /// Enables or disables a provider. Disabling is in-use guarded and never
    /// touches a reference: a route to a disabled provider fails closed.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`]; [`HubError::Unsupported`] for the read-only
    /// entry-zero record; [`HubError::InUse`] when disabling a referenced
    /// provider without confirmation; [`HubError::Conflict`] when the provider was
    /// removed and added again while this ran; and the stores' errors.
    pub async fn set_enabled(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        on: bool,
        confirm: Confirm,
    ) -> Result<Mutation, HubError> {
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        if record.origin == RecordOrigin::EntryZero {
            return Err(HubError::Unsupported {
                op: Operation::SetEnabled,
                kind: record.kind.clone(),
            });
        }
        if on
            && let Some(descriptor) = crate::catalogue::descriptor(record.kind.as_str())
            && !descriptor.endpoint_editable
            && let Some(preset) = descriptor.default_endpoint
            && !crate::policy::same_origin(&record.base_url, preset)
        {
            // An import that found a cloud row on another origin left it
            // disabled: switching it on would send its key there.
            return Err(HubError::Invalid(InvalidInput::Malformed {
                field: InputField::Endpoint,
                reason: "this cloud provider's stored endpoint is not its preset's; remove it and add it again",
            }));
        }
        let host = if on {
            UsedBy::default()
        } else {
            self.host_used_by(scope, slug).await?
        };
        // Ordered with an endpoint move of this provider, which parks the record
        // disabled and restores its flag at the end: a switch made in between
        // would otherwise be overwritten by that restore.
        let _guard = self.slot_lock(scope, slug).await;
        let committed = self
            .transact(scope, |config| {
                if !on {
                    let used = Self::merge_used_by(config, slug, &host);
                    if !used.is_empty() && !confirm.in_use {
                        return Err(HubError::InUse(used));
                    }
                }
                let current = config
                    .provider_mut(slug)
                    .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?;
                // The guards above were checked against `record`; a provider
                // that took its slug since is not what they were about.
                if current.id != record.id {
                    return Err(HubError::Conflict);
                }
                current.enabled = on;
                Ok(())
            })
            .await?;
        if committed.changed {
            self.inner.events.emit(HubEvent::EnabledChanged {
                scope: scope.clone(),
                slug: slug.clone(),
                enabled: on,
            });
        }
        let config = self.read_config(scope).await?;
        let after = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        let view = self.view(scope, &after, &config).await;
        let used = Self::merge_used_by(&config, slug, &host);
        Ok(Mutation {
            status: if committed.changed {
                MutationStatus::Saved
            } else {
                MutationStatus::Unchanged
            },
            note: match (committed.changed, on) {
                (false, _) => "Nothing changed.".to_string(),
                (true, true) => format!("{} was enabled.", after.label),
                (true, false) => format!("{} was disabled.", after.label),
            },
            probe: None,
            used_by: (!on && !used.is_empty()).then_some(used),
            record: Some(view),
        })
    }

    /// Stores a key for a provider, replacing any earlier one. Never guarded:
    /// rotating a key is what an operator does to fix a problem. Nothing is sent
    /// to the provider; call [`Hub::test`] to check it.
    ///
    /// The provider's health and the scope's cached catalogs are dropped.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`]; [`HubError::Invalid`] for an empty key;
    /// [`HubError::Conflict`] when the provider was removed and added again while
    /// this waited; and [`HubError::StoreUnreadable`].
    pub async fn set_key(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        key: Secret,
    ) -> Result<Mutation, HubError> {
        let key = Secret::new(key.expose().trim());
        if key.is_empty() {
            return Err(HubError::Invalid(InvalidInput::Empty(InputField::Key)));
        }
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        if self.group_of(&record) == ProviderGroup::Cli {
            return Err(HubError::Unsupported {
                op: Operation::SetKey,
                kind: record.kind.clone(),
            });
        }
        // Under the provider's lock the record cannot be removed (or re-added)
        // between the check and the write: the check is repeated once the lock is
        // held, and the write happens inside it.
        let _guard = self.slot_lock(scope, slug).await;
        let now = self.read_config(scope).await?;
        match now.provider(slug) {
            None => return Err(HubError::NotFound(NotFound::Provider(slug.clone()))),
            // The group was checked against another provider: not this key's to
            // store.
            Some(current) if current.id != record.id => return Err(HubError::Conflict),
            Some(_) => {}
        }
        let previous = self.read_slot(scope, slug).await?;
        if let Err(error) = self.write_slot(scope, slug, key).await {
            // A store can take the write and then report a failure: what the
            // caller is told did not happen must not have happened.
            self.restore_or_announce(scope, slug, previous).await;
            return Err(error);
        }
        // Kept for a store shared with another hub, whose removal the lock does
        // not order: the key just written would be owned by nothing.
        let after = match self.read_config(scope).await {
            Ok(after) => after,
            Err(error) => {
                // The key is stored; only the confirmation could not be read.
                self.announce_key_state(scope, slug).await;
                return Err(error);
            }
        };
        if after.provider(slug).is_none() {
            self.delete_orphan_slot(scope, slug).await;
            return Err(HubError::NotFound(NotFound::Provider(slug.clone())));
        }
        self.after_key_change(scope, slug, true).await;
        let view = self.view(scope, &record, &after).await;
        Ok(Mutation {
            status: MutationStatus::Saved,
            note: format!("The key for {} was saved.", record.label),
            probe: None,
            used_by: None,
            record: Some(view),
        })
    }

    /// Deletes a provider's stored key. In-use guarded (the provider would stop
    /// working for whatever references it); the managed provider's key can be
    /// cleared, after which the rest of its chain answers.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`], [`HubError::InUse`] and the stores' errors.
    pub async fn clear_key(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        confirm: Confirm,
    ) -> Result<Mutation, HubError> {
        let config = self.read_config(scope).await?;
        if config.provider(slug).is_none() {
            return Err(HubError::NotFound(NotFound::Provider(slug.clone())));
        }
        let host = self.host_used_by(scope, slug).await?;
        let used = Self::merge_used_by(&config, slug, &host);
        if !used.is_empty() && !confirm.in_use {
            return Err(HubError::InUse(used));
        }
        let _guard = self.slot_lock(scope, slug).await;
        // Everything read before the lock may have changed while this waited for
        // it (a removal and a new add of the slug, a reference that appeared):
        // decide again on the provider as it is now.
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        let used = Self::merge_used_by(&config, slug, &host);
        if !used.is_empty() && !confirm.in_use {
            return Err(HubError::InUse(used));
        }
        let previous = self.read_slot(scope, slug).await?;
        let had_key = previous.is_some();
        if had_key {
            if let Err(error) = self.delete_slot(scope, slug).await {
                self.restore_or_announce(scope, slug, previous).await;
                return Err(error);
            }
            self.after_key_change(scope, slug, false).await;
        }
        let view = self.view(scope, &record, &config).await;
        Ok(Mutation {
            status: if had_key {
                MutationStatus::Saved
            } else {
                MutationStatus::Unchanged
            },
            note: if had_key {
                format!("The key for {} was removed.", record.label)
            } else {
                "There was no key to remove.".to_string()
            },
            probe: None,
            used_by: (had_key && !used.is_empty()).then_some(used),
            record: Some(view),
        })
    }
}
