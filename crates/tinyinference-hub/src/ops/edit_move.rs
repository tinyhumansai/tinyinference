//! Moving a provider to another origin while entering its key: the one edit
//! whose two stores (the record and the key slot) must change in an order that
//! never pairs a credential with an origin it was not entered for.
//!
//! The order, under the provider's lock:
//!
//! 1. delete the old key from the slot (a failure here aborts with nothing
//!    changed);
//! 2. commit the record at the **new** origin and **disabled**, together with
//!    the rest of the patch;
//! 3. write the new key;
//! 4. commit the record's original `enabled` flag back.
//!
//! While the record is disabled nothing is sent for it by a route or a kept
//! model (`stale_route`), whatever any chain source (the slot, an environment
//! variable, a keychain) would answer: that is what closes the window between
//! steps 2 and 4 for the sources the hub does not own. Step 1 closes the other
//! half: the old key is out of the slot before the record moves, so no failure
//! afterwards, including a store that stops answering mid-way, can leave the old
//! key beside the new origin (a state a disabled record does not protect: a
//! `test` of a disabled provider still sends its credential). Whatever fails,
//! the slot holds nothing or the new key while the record is at the new origin,
//! and nothing or the old key while it is at the old one.
//!
//! A store that commits and then reports a failure is looked at, not assumed
//! away: after an error from the record commit the record is read back before
//! the old key is restored, and if it cannot be read the slot is left empty.
//!
//! An undo runs in the order that keeps that true at every instant: empty the
//! slot, move the record back and re-enable it, then restore the old key. A step
//! that cannot run stops the undo there; the record stays **disabled at the new
//! origin** (unusable). The caller's error is the reason the move failed; the
//! stuck state is logged and announced (`ProviderEdited`, and `KeyChanged` with
//! what the slot holds). If only the final "switch back on" failed the edit
//! **succeeds with a warning** (`MutationStatus::SavedWithWarning`): the endpoint
//! moved and the key is in place, and `Hub::set_enabled` finishes it. If the undo
//! failed the caller gets the failure and the provider is left disabled at the
//! new endpoint holding at most the key entered for it (no key of the old
//! endpoint): enter a key (`set_key`) **before** testing, listing or switching it
//! on if it has none, because a host credential in the chain would otherwise
//! answer at the new endpoint (a disabled record is not protected: `test` and
//! `list_models` do not check the flag). The old key is lost only when the stores fail twice in a row.
//!
//! Not cancellation-safe: the move is several awaits over two stores. A future
//! dropped between them (a request timeout, a `select!`) leaves the state a
//! failure would, minus the undo and the announcement. Run it to completion.

use crate::error::{HubError, NotFound};
use crate::hub::Hub;
use crate::ids::{ScopeKey, Slug};
use crate::secret::Secret;

/// How a completed move ended.
pub(super) struct MoveOutcome {
    /// Whether the record's commit changed the document.
    pub(super) changed: bool,
    /// The endpoint moved and the key is in place, but the record could not be
    /// switched back on: reported as a warning on a successful edit.
    pub(super) left_disabled: bool,
}

/// Everything an origin move needs, gathered by `Hub::edit` under the lock.
pub(super) struct MovePlan<'a> {
    pub(super) label: Option<&'a str>,
    pub(super) model: Option<&'a crate::ids::ModelId>,
    /// The label and model the record had, put back by an undo (the patch's own
    /// label and model are committed with the move, so an undone edit must not
    /// keep them).
    pub(super) was_label: &'a str,
    pub(super) was_model: Option<&'a crate::ids::ModelId>,
    /// The endpoint being moved to.
    pub(super) target: &'a str,
    /// The endpoint the move was validated against (G3 is re-checked against it
    /// inside the transaction).
    pub(super) validated_base: &'a str,
    pub(super) key: &'a Secret,
    /// The record's `enabled` flag before the move, restored at the end.
    pub(super) was_enabled: bool,
    /// What the slot held before (it is deleted first, and put back if the move
    /// is undone).
    pub(super) previous: Option<Secret>,
}

impl Hub {
    /// Performs the move described by `plan` and returns whether the first
    /// commit changed the document.
    ///
    /// # Errors
    ///
    /// Whatever the first commit returns (nothing was changed), or the failure
    /// that made the move be undone.
    pub(super) async fn move_origin_with_key(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        plan: MovePlan<'_>,
    ) -> Result<MoveOutcome, HubError> {
        // 1. The old key leaves the slot before anything else moves. A store can
        // delete and then report a failure: the old key goes back (the record has
        // not moved), so "nothing changed" is true of what the caller is told.
        if plan.previous.is_some()
            && let Err(error) = self.delete_slot(scope, slug).await
        {
            // The key is gone though the record never moved: back it goes, or it
            // is announced.
            self.restore_or_announce(scope, slug, plan.previous.clone())
                .await;
            return Err(error);
        }
        // 2. The record moves, disabled.
        let committed = self
            .transact(scope, |config| {
                let record = config
                    .provider_mut(slug)
                    .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?;
                // G3 against the record as it is now: an edit through another hub
                // over this store may have moved the origin since this move was
                // validated.
                if !crate::policy::same_origin(&record.base_url, plan.validated_base) {
                    return Err(HubError::Conflict);
                }
                if let Some(label) = plan.label {
                    record.label = label.to_string();
                }
                if let Some(model) = plan.model {
                    record.model = Some(model.clone());
                }
                record.base_url = plan.target.to_string();
                record.enabled = false;
                Ok(())
            })
            .await;
        let changed = match committed {
            Ok(committed) => committed.changed,
            Err(error) => {
                // What the closure or the compare-and-swap refused did not
                // commit. Any other error is a store that may have committed and
                // then failed to say so: look, rather than guess.
                let moved = match &error {
                    HubError::NotFound(_) | HubError::Conflict | HubError::Invalid(_) => {
                        Some(false)
                    }
                    _ => self.record_is_at(scope, slug, plan.target).await,
                };
                match moved {
                    // It did commit: carry on from there.
                    Some(true) => true,
                    // Nothing moved: the old key goes back, unless the provider is
                    // gone (its key went with it) or the record is not verifiably
                    // still at the endpoint that key was entered for (another
                    // hub moved it elsewhere: the old key beside a third origin
                    // is exactly what must not happen).
                    Some(false) => {
                        if matches!(error, HubError::NotFound(_)) {
                            return Err(error);
                        }
                        if self.record_is_at(scope, slug, plan.validated_base).await == Some(true) {
                            self.restore_or_announce(scope, slug, plan.previous.clone())
                                .await;
                        } else {
                            self.announce_key_state(scope, slug).await;
                        }
                        return Err(error);
                    }
                    // Cannot tell where the record is: leave the slot empty. An
                    // old key restored beside a record that may be at the new
                    // origin is the one pairing this must never produce.
                    None => {
                        tracing::warn!(%slug, reason = %error.reason(), "could not tell whether the endpoint moved; the key slot is left empty");
                        // The record may be at the new origin: announce it as a
                        // change, whatever the slot held.
                        self.announce_move(scope, slug).await;
                        return Err(error);
                    }
                }
            }
        };

        // 3. The new key, for the new origin.
        if let Err(error) = self.write_slot(scope, slug, plan.key.clone()).await {
            self.undo_move(scope, slug, &plan).await;
            return Err(error);
        }
        if plan.was_enabled
            && let Err(error) = self
                .transact(scope, |config| {
                    if let Some(record) = config.provider_mut(slug)
                        && record.base_url == plan.target
                    {
                        record.enabled = true;
                    }
                    Ok(())
                })
                .await
        {
            // A store that committed the flag and then reported a failure is
            // looked at, not assumed away.
            if self.record_is_enabled_at(scope, slug, plan.target).await == Some(true) {
                return Ok(MoveOutcome {
                    changed,
                    left_disabled: false,
                });
            }
            // The move and the key are in place; only switching it back on
            // failed. Unusable rather than half-usable: warn, on a successful
            // edit (the caller's `finish_edit` announces the change, once).
            // Switching it on is then `set_enabled`.
            tracing::warn!(%slug, reason = %error.reason(), "the provider was moved and its key saved but it could not be switched back on");
            return Ok(MoveOutcome {
                changed,
                left_disabled: true,
            });
        }
        Ok(MoveOutcome {
            changed,
            left_disabled: false,
        })
    }

    /// What a move that changed the record and could not finish tells the rest of
    /// the system: the endpoint and key really changed, so what was learned,
    /// cached or announced about the old ones is stale (health, catalogs, and the
    /// `KeyChanged` and `ProviderEdited` events a host mirrors state from).
    async fn announce_move(&self, scope: &ScopeKey, slug: &Slug) {
        self.announce_key_state(scope, slug).await;
        self.inner
            .events
            .emit(crate::ports::HubEvent::ProviderEdited {
                scope: scope.clone(),
                slug: slug.clone(),
            });
    }

    /// Whether the record is at `target` **and enabled** now: `None` when the
    /// store cannot say.
    async fn record_is_enabled_at(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        target: &str,
    ) -> Option<bool> {
        let config = self.read_config(scope).await.ok()?;
        Some(
            config
                .provider(slug)
                .is_some_and(|r| r.base_url == target && r.enabled),
        )
    }

    /// Whether the record is at `target` now: `None` when the store cannot say.
    async fn record_is_at(&self, scope: &ScopeKey, slug: &Slug, target: &str) -> Option<bool> {
        let config = self.read_config(scope).await.ok()?;
        Some(config.provider(slug).is_some_and(|r| r.base_url == target))
    }

    /// Undoes a move whose key could not be written, in the order that is safe
    /// at every instant (see the module docs). Best effort; each step's failure
    /// is logged and never replaces the reason the move failed.
    async fn undo_move(&self, scope: &ScopeKey, slug: &Slug, plan: &MovePlan<'_>) {
        // The slot may hold the new key (a write that committed and then timed
        // out), which would meet the old origin once the record is switched back
        // on, so it goes first. If it cannot be emptied the record stays at the
        // new origin, where whatever is in the slot is the new key or nothing.
        if let Err(error) = self.delete_slot(scope, slug).await {
            tracing::warn!(%slug, reason = %error.reason(), "could not empty the key slot; the provider stays disabled at the new endpoint");
            self.announce_move(scope, slug).await;
            return;
        }
        let back = self
            .transact(scope, |config| {
                if let Some(record) = config.provider_mut(slug)
                    && record.base_url == plan.target
                {
                    record.base_url = plan.validated_base.to_string();
                    record.enabled = plan.was_enabled;
                    // Only what this move set: a label or model another writer
                    // changed since (those edits take no lock) is theirs.
                    if plan.label == Some(record.label.as_str()) {
                        record.label = plan.was_label.to_string();
                    }
                    if plan.model.is_some() && plan.model == record.model.as_ref() {
                        record.model = plan.was_model.cloned();
                    }
                }
                Ok(())
            })
            .await;
        // A store that committed the move back and then reported a failure is
        // looked at, not assumed away (as after the move's own commit).
        if let Err(error) = back
            && self.record_is_at(scope, slug, plan.validated_base).await != Some(true)
        {
            tracing::warn!(%slug, reason = %error.reason(), "could not move the endpoint back; the provider stays disabled at the new endpoint with no key");
            self.announce_move(scope, slug).await;
            return;
        }
        // The old key goes back only beside a record verifiably at the endpoint
        // it was entered for (the transaction above is a no-op when another hub
        // has moved the record elsewhere meanwhile).
        if self.record_is_at(scope, slug, plan.validated_base).await == Some(true) {
            self.restore_or_announce(scope, slug, plan.previous.clone())
                .await;
        } else {
            self.announce_key_state(scope, slug).await;
        }
    }
}
