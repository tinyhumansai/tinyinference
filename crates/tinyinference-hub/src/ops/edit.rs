//! Editing a saved provider.
//!
//! Guards: G2 (a cloud preset's endpoint is not typed), G3 (a credential does
//! not follow an endpoint to another origin: changing the origin with a stored
//! key needs the key entered again), G14 (model ids), G21 (entry zero is
//! read-only). A key rotation is never guarded (G5). The kind never changes.

use super::edit_move::MovePlan;
use crate::descriptor::{ProviderRecord, RecordOrigin};
use crate::error::{HubError, InputField, InvalidInput, NotFound, Operation};
use crate::hub::{Hub, Mutation, MutationStatus, ProviderPatch};
use crate::ids::{ScopeKey, Slug, check_provider_name};
use crate::policy::same_origin;
use crate::ports::HubEvent;
use crate::secret::Secret;
use crate::taxonomy::ProviderGroup;

impl Hub {
    /// Changes a saved provider's label, endpoint, model or key.
    ///
    /// The slug never changes (it addresses the key). A typed endpoint is
    /// ignored for a catalogue preset. Changing an endpoint to another origin
    /// while a stored key exists and no new key is supplied is refused, so a key
    /// cannot be sent somewhere its owner never chose. A new key drops the
    /// provider's health and the scope's cached catalogs.
    ///
    /// Moving the endpoint to another origin **with** a new key is done in a
    /// fixed order that never lets a credential meet an origin it was not entered
    /// for: the record is committed at the new origin and disabled, the key is
    /// written, and the record is switched back to what it was. The provider
    /// reads as disabled for those few calls (see `edit_move.rs`).
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`]; [`HubError::Unsupported`] for the read-only
    /// entry-zero record or a label/endpoint change on the managed provider;
    /// [`HubError::Invalid`] and [`HubError::Policy`] for a patch that fails
    /// validation; [`HubError::Conflict`] when another writer moved the
    /// endpoint's origin, or removed and re-added the provider, after this edit
    /// validated its patch; and the stores' errors. **A failure while moving the
    /// endpoint with a new key can leave the change partly applied, always in a
    /// state in which no credential the hub stored can meet the wrong origin:**
    /// if only switching the provider back on failed, the edit **succeeds with a
    /// warning** ([`MutationStatus::SavedWithWarning`]; call [`Hub::set_enabled`]);
    /// if the move could not be undone, the caller gets the failure and the
    /// provider is left disabled at the new endpoint holding at most the key
    /// entered for it (if it has none, enter one with [`Hub::set_key`] before
    /// testing, listing or enabling it: a credential a host source supplies would
    /// otherwise answer there). Both are logged and announced (`ProviderEdited`,
    /// `KeyChanged`). Like every operation that
    /// changes two stores this is **not cancellation-safe**: run it to completion
    /// (do not put it under a timeout or a `select!` that can drop it).
    pub async fn edit(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        patch: ProviderPatch,
    ) -> Result<Mutation, HubError> {
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        if record.origin == RecordOrigin::EntryZero {
            return Err(HubError::Unsupported {
                op: Operation::Edit,
                kind: record.kind.clone(),
            });
        }
        let driver = self.driver(&record.kind)?;
        let descriptor = driver.descriptor();
        if descriptor.group == ProviderGroup::Managed
            && (patch.label.is_some() || patch.base_url.is_some())
        {
            return Err(HubError::Unsupported {
                op: Operation::Edit,
                kind: record.kind.clone(),
            });
        }

        let label = match patch.label.as_deref().map(str::trim) {
            Some(label) => {
                check_provider_name(label).map_err(|e| e.into_hub_error(label))?;
                Some(label.to_string())
            }
            None => None,
        };
        // Guard G2: a cloud preset's endpoint is data, so a typed one is ignored.
        let base_url = match patch.base_url.as_deref() {
            Some(typed) if descriptor.endpoint_editable => {
                Some(self.plan_endpoint(descriptor, Some(typed))?)
            }
            _ => None,
        };
        let model = match &patch.model {
            Some(model) => Some(self.check_model(model)?),
            None => None,
        };
        let key = match &patch.key {
            Some(key) if key.expose().trim().is_empty() => {
                return Err(HubError::Invalid(InvalidInput::Empty(InputField::Key)));
            }
            Some(key) => Some(Secret::new(key.expose().trim())),
            None => None,
        };

        // Everything below that touches the key slot, or moves the endpoint,
        // runs under the provider's lock: another key change or edit of the
        // same provider waits, and what this edit read is what it changes.
        let slot_guard = if key.is_some() || base_url.is_some() {
            Some(self.slot_lock(scope, slug).await)
        } else {
            None
        };
        // The record as it is now that nobody else can be changing this
        // provider's key or endpoint.
        let record = match &slot_guard {
            Some(_) => {
                let fresh = self
                    .read_config(scope)
                    .await?
                    .provider(slug)
                    .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
                    .clone();
                // The patch was validated against one provider (its kind, its
                // endpoint rules); if the slug now names another, it is not this
                // edit's to change.
                if fresh.id != record.id {
                    return Err(HubError::Conflict);
                }
                fresh
            }
            None => record,
        };
        let origin_changes = base_url
            .as_deref()
            .is_some_and(|new| !same_origin(&record.base_url, new));
        // Only an origin move without a new key reads the chain: a label rename or
        // a key rotation must not fail because a source is unreadable.
        if origin_changes && key.is_none() && self.credential(scope, &record).await?.key.is_some() {
            return Err(HubError::Invalid(InvalidInput::Malformed {
                field: InputField::Endpoint,
                reason: "changing the endpoint to another origin needs the key entered again",
            }));
        }

        // A key entered together with an origin move is for the NEW origin, so
        // no credential may ever meet an origin it was not entered for. That takes
        // more than an order of two writes (the chain has other sources than the
        // slot), so the move runs as its own sequence: `edit_move.rs`.
        let move_with_key = origin_changes && key.is_some();
        let previous = match &key {
            Some(_) => Some(self.read_slot(scope, slug).await?),
            None => None,
        };
        if let (Some(key), Some(target), true) = (&key, base_url.as_deref(), move_with_key) {
            // (The slot's old key is deleted by the move itself, first.)
            let plan = MovePlan {
                label: label.as_deref(),
                model: model.as_ref(),
                was_label: &record.label,
                was_model: record.model.as_ref(),
                target,
                validated_base: &record.base_url,
                key,
                was_enabled: record.enabled,
                previous: previous.clone().flatten(),
            };
            let outcome = self.move_origin_with_key(scope, slug, plan).await?;
            drop(slot_guard);
            let finished = self
                .finish_edit(
                    scope,
                    slug,
                    &record,
                    base_url.as_deref(),
                    true,
                    outcome.changed,
                )
                .await;
            if !outcome.left_disabled {
                return finished;
            }
            // The edit went through and only the flag could not be restored: it
            // is a warning on a successful edit, and stays one if reading the
            // result back fails too (the same outage that stopped the flag).
            let label = match &finished {
                Ok(m) => m
                    .record
                    .as_ref()
                    .map_or(record.label.clone(), |v| v.record.label.clone()),
                Err(_) => record.label.clone(),
            };
            let mut mutation = finished.unwrap_or(Mutation {
                status: MutationStatus::SavedWithWarning,
                note: String::new(),
                probe: None,
                used_by: None,
                record: None,
            });
            mutation.status = MutationStatus::SavedWithWarning;
            mutation.note = format!(
                "{label} was moved and its key saved, but it could not be switched back on; enable it once the store recovers."
            );
            return Ok(mutation);
        }
        if let Some(key) = &key
            && let Err(error) = self.write_slot(scope, slug, key.clone()).await
        {
            // A store can commit the write and then report a failure: what the
            // caller is told did not happen must not have happened.
            if let Some(previous) = previous {
                self.restore_or_announce(scope, slug, previous).await;
            }
            return Err(error);
        }
        let validated_base = record.base_url.clone();
        let validated_id = record.id.clone();
        let committed = self
            .transact(scope, |config| {
                let record = config
                    .provider_mut(slug)
                    .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?;
                // The patch was validated against one provider; a label or model
                // edit takes no lock, so the slug may have been re-added since.
                if record.id != validated_id {
                    return Err(HubError::Conflict);
                }
                if let Some(label) = &label {
                    record.label.clone_from(label);
                }
                if let Some(url) = &base_url {
                    // G3 again, against the record as it is now: an edit through
                    // another hub over this store may have moved the origin since
                    // the check above, and this edit was validated against the
                    // old one.
                    if !same_origin(&record.base_url, &validated_base) {
                        return Err(HubError::Conflict);
                    }
                    record.base_url.clone_from(url);
                }
                if let Some(model) = &model {
                    record.model = Some(model.clone());
                }
                Ok(())
            })
            .await;
        let committed = match committed {
            Ok(committed) => committed,
            Err(error) => {
                // A record somebody removed while this ran took its key with it:
                // restoring the old one would resurrect it, and the new one just
                // written belongs to nothing. Anything else puts the old key back.
                if matches!(error, HubError::NotFound(_)) {
                    if key.is_some() {
                        self.delete_orphan_slot(scope, slug).await;
                    }
                } else if let Some(previous) = previous {
                    // The operation's own error is the reason; a failed restore
                    // must not replace it.
                    self.restore_or_announce(scope, slug, previous).await;
                }
                return Err(error);
            }
        };
        let changed = committed.changed;
        drop(slot_guard);
        self.finish_edit(
            scope,
            slug,
            &record,
            base_url.as_deref(),
            key.is_some(),
            changed,
        )
        .await
    }

    /// What every successful edit does last: hooks for a new key or endpoint, the
    /// event, and the view.
    async fn finish_edit(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        record: &ProviderRecord,
        base_url: Option<&str>,
        key_set: bool,
        committed_changed: bool,
    ) -> Result<Mutation, HubError> {
        if key_set {
            // What is in the slot now, not what this edit wrote: an operation on
            // the same provider may have run since the lock was released, and
            // its event must not be contradicted by a stale `present: true`.
            self.announce_key_state(scope, slug).await;
        }
        if let Some(new) = base_url
            && new != record.base_url
            && !key_set
        {
            // A different endpoint says nothing about what the old one said. (A
            // key change, announced above, has already dropped both.)
            self.inner.cache.evict_scope(scope);
            self.forget_health(scope, slug).await;
        }
        let changed = committed_changed || key_set;
        if changed {
            self.inner.events.emit(HubEvent::ProviderEdited {
                scope: scope.clone(),
                slug: slug.clone(),
            });
        }
        let config = self.read_config(scope).await?;
        let after = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        let view = self.view(scope, &after, &config).await;
        Ok(Mutation {
            status: if changed {
                MutationStatus::Saved
            } else {
                MutationStatus::Unchanged
            },
            note: if changed {
                format!("{} was updated.", after.label)
            } else {
                "Nothing changed.".to_string()
            },
            probe: None,
            used_by: None,
            record: Some(view),
        })
    }
}
