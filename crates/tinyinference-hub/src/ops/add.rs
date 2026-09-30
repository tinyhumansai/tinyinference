//! Adding a provider: `add` (no probe) and `connect` (add, probe, roll back).
//!
//! Guards carried over from OpenCompany: G2 (a cloud preset's endpoint is not
//! typed), G4 (one row per kind when the host asks), G9 (the first provider
//! becomes the default), G11 and G12 (only a rejected credential rolls the add
//! back, and the previous key is restored), G14 and G15 (model and name
//! validation), G16 and G17 (endpoint policy).

use crate::catalogue::is_reserved_slug;
use crate::config::{DefaultChoice, HubConfig, ProviderDraft};
use crate::descriptor::ProviderRecord;
use crate::endpoint::normalize_local_endpoint;
use crate::error::{HubError, InputField, InvalidInput, Operation, PolicyViolation, describe};
use crate::hub::{ConnectOptions, Hub, Mutation, MutationStatus};
use crate::ids::{ModelId, ScopeKey, Slug, check_provider_name, check_slug, slugify};
use crate::policy::check_endpoint;
use crate::ports::HubEvent;
use crate::secret::Secret;
use crate::taxonomy::{ProviderGroup, TestDepth};

/// A validated draft: everything decided before anything is written.
pub(crate) struct AddPlan {
    pub(crate) kind: crate::ids::KindId,
    pub(crate) group: ProviderGroup,
    pub(crate) slug: Slug,
    pub(crate) label: String,
    pub(crate) base_url: String,
    pub(crate) model: Option<ModelId>,
    pub(crate) key: Option<Secret>,
    /// Whether a key is *required* to use the kind: a keyless add is saved
    /// unchecked only when one is.
    pub(crate) key_required: bool,
}

impl Hub {
    /// Validates a draft without touching any store. The guards that need the
    /// current configuration (a taken slug) run later, inside the change.
    pub(crate) fn plan_add(
        &self,
        op: Operation,
        draft: &ProviderDraft,
    ) -> Result<AddPlan, HubError> {
        let driver = self.driver(&draft.kind)?;
        let descriptor = driver.descriptor();
        let group = descriptor.group;
        if matches!(
            group,
            ProviderGroup::Cli | ProviderGroup::Managed | ProviderGroup::OAuthBacked
        ) {
            return Err(HubError::Unsupported {
                op,
                kind: descriptor.kind.clone(),
            });
        }

        let named = draft
            .label
            .as_deref()
            .map(str::trim)
            .filter(|label| !label.is_empty());
        if group == ProviderGroup::Custom && named.is_none() {
            return Err(HubError::Invalid(InvalidInput::Empty(
                InputField::ProviderName,
            )));
        }
        let label = named.map_or_else(|| descriptor.label.to_string(), str::to_string);
        check_provider_name(&label).map_err(|e| e.into_hub_error(&label))?;

        let kind_slug = descriptor.kind.as_str();
        let is_kind_row = group != ProviderGroup::Custom
            && named.is_none_or(|l| slugify(l) == kind_slug || l == descriptor.label);
        let slug_text = if is_kind_row {
            kind_slug.to_string()
        } else {
            slugify(&label)
        };
        check_slug(std::iter::empty::<&str>(), &slug_text, |s| {
            !(is_kind_row && s == kind_slug) && is_reserved_slug(s)
        })
        .map_err(|e| e.into_hub_error(&slug_text))?;
        let slug = Slug::parse(&slug_text).map_err(HubError::Invalid)?;

        let base_url = self.plan_endpoint(descriptor, draft.base_url.as_deref())?;

        let model = match &draft.model {
            Some(model) => Some(self.check_model(model)?),
            None => None,
        };
        let key = match &draft.key {
            Some(key) if key.expose().trim().is_empty() => {
                return Err(HubError::Invalid(InvalidInput::Empty(InputField::Key)));
            }
            Some(key) => Some(Secret::new(key.expose().trim())),
            None => None,
        };
        Ok(AddPlan {
            kind: descriptor.kind.clone(),
            group,
            slug,
            label,
            base_url,
            model,
            key,
            key_required: descriptor.needs_key && descriptor.auth.needs_credential(),
        })
    }

    /// The endpoint a record is saved with: the preset for a catalogue cloud
    /// kind (a typed URL is ignored, G2), the normalised URL for a local
    /// runtime, the typed URL for everything else; typed ones go through the
    /// endpoint policy.
    pub(crate) fn plan_endpoint(
        &self,
        descriptor: &crate::descriptor::ProviderDescriptor,
        typed: Option<&str>,
    ) -> Result<String, HubError> {
        let typed = typed.map(str::trim).filter(|t| !t.is_empty());
        if !descriptor.endpoint_editable {
            return descriptor
                .default_endpoint
                .map(str::to_string)
                .ok_or(HubError::Invalid(InvalidInput::Empty(InputField::Endpoint)));
        }
        let raw = typed
            .or(descriptor.default_endpoint)
            .ok_or(HubError::Invalid(InvalidInput::Empty(InputField::Endpoint)))?;
        let url = if descriptor.group == ProviderGroup::Local {
            normalize_local_endpoint(raw).ok_or(HubError::Invalid(InvalidInput::Malformed {
                field: InputField::Endpoint,
                reason: "that is not an http or https endpoint with a host",
            }))?
        } else {
            raw.to_string()
        };
        check_endpoint(&url, &self.inner.policy)
            .map_err(|refusal| HubError::Policy(PolicyViolation::from(refusal)))?;
        Ok(url)
    }

    /// Model-id validation (guard G14) with the host's reserved words.
    pub(crate) fn check_model(&self, model: &ModelId) -> Result<ModelId, HubError> {
        let reserved: Vec<&str> = self
            .inner
            .hub_policy
            .reserved_model_words
            .iter()
            .map(String::as_str)
            .collect();
        ModelId::parse_with_reserved(model.as_str(), &reserved).map_err(HubError::Invalid)
    }

    /// Inserts the planned record, enforcing the guards that read the
    /// configuration. Returns the record and, when the add changed the default, what it was
    /// before (so an undo can restore exactly that).
    fn insert_planned(
        &self,
        config: &mut HubConfig,
        plan: &AddPlan,
        id: &str,
        make_default: bool,
    ) -> Result<(ProviderRecord, Option<DefaultChoice>), HubError> {
        if config.contains(&plan.slug) {
            return Err(HubError::AlreadyExists {
                slug: plan.slug.clone(),
            });
        }
        if self.inner.hub_policy.one_row_per_kind
            && plan.group != ProviderGroup::Custom
            && let Some(existing) = config.providers.iter().find(|p| p.kind == plan.kind)
        {
            return Err(HubError::AlreadyExists {
                slug: existing.slug.clone(),
            });
        }
        let mut record = ProviderRecord::new(
            id,
            plan.slug.clone(),
            plan.label.clone(),
            plan.kind.clone(),
            plan.base_url.clone(),
        );
        record.model.clone_from(&plan.model);
        config.providers.push(record.clone());
        let operator_rows = config
            .providers
            .iter()
            .filter(|p| self.group_of(p) != ProviderGroup::Managed)
            .count();
        let mut default_was: Option<DefaultChoice> = None;
        if make_default {
            let model = plan
                .model
                .clone()
                .ok_or(HubError::Invalid(InvalidInput::Empty(InputField::ModelId)))?;
            default_was = Some(std::mem::replace(
                &mut config.default,
                DefaultChoice::Full {
                    provider: plan.slug.clone(),
                    model,
                },
            ));
        } else if config.default == DefaultChoice::Unset
            && operator_rows == 1
            && let Some(model) = plan.model.clone()
        {
            // Guard G9: the first provider ever becomes the default. Because
            // this runs inside the change, "still unset and exactly one row" is
            // re-checked against the version that is saved.
            config.default = DefaultChoice::Full {
                provider: plan.slug.clone(),
                model,
            };
            default_was = Some(DefaultChoice::Unset);
        }
        Ok((record, default_was))
    }

    /// Adds a provider **without** checking it. Nothing is sent anywhere.
    ///
    /// The record is saved first and the key written after it; a key that cannot
    /// be written takes the record back out, and a provider that was removed in
    /// between is a `NotFound` that leaves no key behind.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`] for an unknown kind, [`HubError::Unsupported`] for
    /// a CLI, OAuth or managed kind, [`HubError::Invalid`] and
    /// [`HubError::Policy`] for a draft that fails validation,
    /// [`HubError::AlreadyExists`] for a taken slug,
    /// [`HubError::StoreUnreadable`] or [`HubError::Conflict`] from the stores,
    /// [`HubError::NotFound`] when another writer removed the provider before its
    /// key could be stored, and [`HubError::Conflict`] when another writer moved
    /// its endpoint first (the key, entered for the endpoint it was added at, is
    /// not stored; the record stays as they left it).
    pub async fn add(&self, scope: &ScopeKey, draft: ProviderDraft) -> Result<Mutation, HubError> {
        let plan = self.plan_add(Operation::Add, &draft)?;
        let added = self.save_new(scope, &plan, false).await?;
        let config = self.read_config(scope).await.unwrap_or_default();
        let view = self.view(scope, &added.record, &config).await;
        Ok(Mutation {
            status: MutationStatus::Saved,
            note: format!("{} was added.", plan.label),
            probe: None,
            used_by: None,
            record: Some(view),
        })
    }

    /// Saves the record, **then** writes the key.
    ///
    /// The record goes first on purpose: a writer that loses (the slug is taken,
    /// the compare-and-swap keeps failing) has touched nothing else, so it can
    /// never overwrite, or delete, the key of the provider that won. The window
    /// where the row exists without its key fails closed: a turn to it is
    /// `NoKey`. If the key cannot be written the record is taken out again.
    async fn save_new(
        &self,
        scope: &ScopeKey,
        plan: &AddPlan,
        make_default: bool,
    ) -> Result<Added, HubError> {
        // Read the slot before anything changes: an unreadable store stops the
        // add here. (What a rollback puts back is read again, under the lock,
        // once the record is committed.)
        if plan.key.is_some() {
            self.read_slot(scope, &plan.slug).await?;
        }
        let id = self.new_record_id(&plan.slug);
        let mut inserted = None;
        self.transact(scope, |config| {
            inserted = Some(self.insert_planned(config, plan, &id, make_default)?);
            Ok(())
        })
        .await?;
        let (record, default_was) = inserted.ok_or(HubError::Conflict)?;
        let mut added = Added {
            record,
            default_was,
            key_was: None,
        };
        // Announced as soon as the record is committed, so every later failure
        // (an undo, another writer's edit or removal) reads as a change to a
        // provider the host already knows about.
        self.inner.events.emit(HubEvent::ProviderAdded {
            scope: scope.clone(),
            slug: plan.slug.clone(),
        });
        if let Some(key) = &plan.key {
            // The record is committed; the key goes in under the provider's lock,
            // after confirming the record is still this add's. Without the lock, a
            // removal and a second add of the same slug can run between this add's
            // record and its key: this add would then write its key over the second
            // add's and, finding its record gone, delete the slot (finding 4.6).
            let guard = self.slot_lock(scope, &plan.slug).await;
            let ownership = match self.read_config(scope).await {
                Ok(config) => Ownership::of(&config, &added.record),
                Err(error) => {
                    // The add is half done and cannot be confirmed: undo it,
                    // and report why.
                    drop(guard);
                    self.undo_add(scope, &added).await;
                    return Err(error);
                }
            };
            match ownership {
                Ownership::Ours => {}
                // Somebody removed the provider meanwhile: the slot is not ours
                // to write or to delete.
                Ownership::Gone => {
                    return Err(HubError::NotFound(crate::error::NotFound::Provider(
                        plan.slug.clone(),
                    )));
                }
                // Somebody moved it to another endpoint meanwhile: this key was
                // entered for the one it was added at, so it is not stored. The
                // record is theirs now and stays as they left it.
                Ownership::Moved => return Err(HubError::Conflict),
            }
            // What to put back if this add is undone: read under the lock, so it
            // is the slot as this add found it, not as it was before a removal.
            let previous = match self.read_slot(scope, &plan.slug).await {
                Ok(previous) => previous,
                Err(error) => {
                    drop(guard);
                    self.undo_add(scope, &added).await;
                    return Err(error);
                }
            };
            if let Err(error) = self.write_slot(scope, &plan.slug, key.clone()).await {
                // A store can commit and then time out: put back what was there.
                added.key_was = Some(previous);
                drop(guard);
                self.undo_add(scope, &added).await;
                return Err(error);
            }
            added.key_was = Some(previous);
            // Once more after the write, for a removal made through another hub
            // over the same store, which the lock does not order: the key just
            // written would belong to nothing, and to whoever adds that slug next.
            // (Under the lock, so it deletes only what this add wrote.)
            let still = match self.read_config(scope).await {
                Ok(config) => Ownership::of(&config, &added.record),
                Err(error) => {
                    drop(guard);
                    self.undo_add(scope, &added).await;
                    return Err(error);
                }
            };
            if still != Ownership::Ours {
                self.delete_orphan_slot(scope, &plan.slug).await;
                // A key came and went.
                self.announce_key_state(scope, &plan.slug).await;
                return Err(match still {
                    Ownership::Moved => HubError::Conflict,
                    _ => HubError::NotFound(crate::error::NotFound::Provider(plan.slug.clone())),
                });
            }
            drop(guard);
            // What is in the slot now (an operation on this provider may have
            // run since the lock was released), not what this add wrote.
            self.announce_key_state(scope, &plan.slug).await;
        }
        Ok(added)
    }

    /// Adds a provider **and checks it**, rolling the add back when the
    /// provider rejects the credential.
    ///
    /// The check runs at `options.depth` (default: read the catalog) and is
    /// recorded as health. Only a rejected credential undoes the add (a local
    /// runtime also when nothing answers, or it times out), and never with
    /// `options.add_anyway`; an endpoint the policy refuses is undone whatever
    /// `add_anyway` says. When the add is undone the previous key is put back
    /// and the error carries the reason. A row saved without a key for a kind
    /// that needs one is saved unchecked, with a warning.
    ///
    /// # Errors
    ///
    /// Everything [`Hub::add`] returns, plus the provider's failure
    /// ([`HubError::Provider`], or [`HubError::Policy`] for a refused endpoint)
    /// when the add was undone. [`HubError::NotFound`] or [`HubError::Conflict`]
    /// when another writer removed or edited the provider between the add and the
    /// check: what is there is theirs, so it is neither undone nor checked.
    pub async fn connect(
        &self,
        scope: &ScopeKey,
        draft: ProviderDraft,
        options: ConnectOptions,
    ) -> Result<Mutation, HubError> {
        let plan = self.plan_add(Operation::Connect, &draft)?;
        let driver = self.driver(&plan.kind)?;
        if !driver.descriptor().supports_depth(options.depth) {
            return Err(HubError::Unsupported {
                op: Operation::Test(options.depth),
                kind: plan.kind.clone(),
            });
        }
        if options.depth == TestDepth::Completion && plan.model.is_none() {
            return Err(HubError::Invalid(InvalidInput::Empty(InputField::ModelId)));
        }
        if options.make_default && plan.model.is_none() {
            return Err(HubError::Invalid(InvalidInput::Empty(InputField::ModelId)));
        }
        let added = self.save_new(scope, &plan, options.make_default).await?;
        let record = added.record.clone();

        // Is there anything to check with? A keyed kind saved without a key is
        // saved unchecked.
        let credential = match self.credential_checked(scope, &record).await {
            Ok(credential) => credential,
            // Somebody else removed or edited the provider between this connect's
            // commit and its check: what is there now is theirs, so it is not
            // undone (that would delete their edit), and the check is not made.
            Err(error @ (HubError::Conflict | HubError::NotFound(_))) => return Err(error),
            Err(error) => {
                self.undo_add(scope, &added).await;
                return Err(error);
            }
        };
        let probes = !plan.key_required || credential.key.is_some();
        if !probes {
            let config = self.read_config(scope).await.unwrap_or_default();
            let view = self.view(scope, &record, &config).await;
            return Ok(Mutation {
                status: MutationStatus::SavedWithWarning,
                note: format!("{} was added without a key; add one to use it.", plan.label),
                probe: None,
                used_by: None,
                record: Some(view),
            });
        }

        let report = match self
            .probe_record(
                scope,
                &record,
                options.depth,
                plan.model.as_ref(),
                &credential,
            )
            .await
        {
            Ok(report) => report,
            Err(error) => {
                self.undo_add(scope, &added).await;
                return Err(error);
            }
        };
        if let Some(failure) = &report.failure {
            let refused = report.refusal.is_some();
            if refused || (failure.rolls_back(plan.group) && !options.add_anyway) {
                self.undo_add(scope, &added).await;
                let error = report
                    .clone()
                    .into_result()
                    .err()
                    .unwrap_or(HubError::Provider(failure.clone()));
                return Err(error);
            }
        }
        let config = self.read_config(scope).await.unwrap_or_default();
        let view = self.view(scope, &record, &config).await;
        Ok(match &report.failure {
            None => Mutation {
                status: MutationStatus::Saved,
                note: format!("{} is connected.", plan.label),
                probe: Some(report),
                used_by: None,
                record: Some(view),
            },
            Some(failure) => Mutation {
                status: MutationStatus::SavedWithWarning,
                note: describe(failure.reason, &plan.label),
                probe: Some(report),
                used_by: None,
                record: Some(view),
            },
        })
    }

    /// Undoes an add exactly: the record goes (only the record this add made,
    /// found by id), the default goes back to what it was (if this add changed it
    /// and nobody has changed it since), the key slot goes back to what it held
    /// (if this add wrote a key and its record was still there), and what was
    /// learned about the new key is forgotten. Compensating events are emitted,
    /// so a host that mirrors state from events sees the add and its undoing.
    ///
    /// Best effort by construction: each step runs whatever the others did, and a
    /// step that fails is logged. The caller is reporting *why the add failed*;
    /// a failure of the cleanup must not replace that reason.
    async fn undo_add(&self, scope: &ScopeKey, added: &Added) {
        let (slug, id) = (added.record.slug.clone(), added.record.id.clone());
        // Held to the end: nothing else may write this provider's key between
        // the restore and the record's removal, or a later add of the slug could
        // have its key overwritten by this restore.
        let _guard = self.slot_lock(scope, &slug).await;
        // Put the key slot back while this add's record still owns the slug: no
        // other add of that slug can be writing it, so this cannot clobber a
        // winner's key. A record already gone means the slot is not ours.
        // One retry: a transient failure here would otherwise leave the key slot
        // to be restored only after the record is gone.
        let first = match self.read_config(scope).await {
            Ok(config) => Ok(config),
            Err(_) => self.read_config(scope).await,
        };
        // "Ours" is the record this add committed, at the endpoint it was added
        // at. One somebody edited while the check was in flight (it takes no lock
        // while the network is being asked) is theirs now: neither it nor its key
        // is undone.
        let ours = |p: &ProviderRecord| p.id == id && p.base_url == added.record.base_url;
        let (existed, pre_read_failed) = match first {
            Ok(config) => (config.providers.iter().any(ours), false),
            Err(_) => (false, true),
        };
        if existed && let Some(previous) = added.key_was.clone() {
            let present = previous.is_some();
            match self.restore_slot(scope, &slug, previous).await {
                Ok(()) => self.inner.events.emit(HubEvent::KeyChanged {
                    scope: scope.clone(),
                    slug: slug.clone(),
                    present,
                }),
                Err(error) => {
                    tracing::warn!(%slug, reason = %error.reason(), "could not restore the key an undone add replaced");
                    self.announce_key_state(scope, &slug).await;
                }
            }
        }
        let mut existed = existed;
        let restore_after = pre_read_failed;
        let removed = self
            .transact(scope, |config| {
                existed = config.providers.iter().any(ours);
                if !existed {
                    return Ok(());
                }
                config.providers.retain(|p| !ours(p));
                if let Some(was) = &added.default_was
                    && matches!(&config.default, DefaultChoice::Full { provider, .. } if *provider == slug)
                {
                    config.default = was.clone();
                }
                Ok(())
            })
            .await;
        // A store that committed the removal and then reported a failure is looked
        // at: the record is read back, and a record that is gone was removed.
        let removed_ok = match &removed {
            Ok(_) => true,
            Err(error) => {
                tracing::warn!(%slug, reason = %error.reason(), "could not take an undone add's record out");
                match self.read_config(scope).await {
                    Ok(config) => !config.providers.iter().any(ours),
                    Err(_) => false,
                }
            }
        };
        if restore_after
            && existed
            && let Some(previous) = added.key_was.clone()
        {
            self.restore_or_announce(scope, &slug, previous).await;
        }
        self.forget_health(scope, &slug).await;
        self.inner.cache.evict_scope(scope);
        if existed && removed_ok {
            self.inner.events.emit(HubEvent::ProviderRemoved {
                scope: scope.clone(),
                slug,
            });
        }
    }
}

/// Whether an add's record is still the one it committed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Ownership {
    /// The record is there, at the endpoint it was added at.
    Ours,
    /// The record is there at another endpoint (somebody edited it).
    Moved,
    /// The record is gone.
    Gone,
}

impl Ownership {
    fn of(config: &HubConfig, added: &ProviderRecord) -> Self {
        match config.providers.iter().find(|p| p.id == added.id) {
            None => Self::Gone,
            // A change of path or query is not a move: the key was entered for
            // the origin, and it is still there (the same rule G3 applies).
            Some(now) if !crate::policy::same_origin(&now.base_url, &added.base_url) => Self::Moved,
            Some(_) => Self::Ours,
        }
    }
}

/// What an add did, so it can be undone exactly.
pub(crate) struct Added {
    pub(crate) record: ProviderRecord,
    /// The default before the add, when the add changed it.
    pub(crate) default_was: Option<DefaultChoice>,
    /// The key slot's value before the add wrote a key (`None`: it wrote none).
    pub(crate) key_was: Option<Option<Secret>>,
}
