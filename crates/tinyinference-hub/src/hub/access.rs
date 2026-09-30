//! Shared plumbing for the operations: drivers, credentials, views, hooks.

use std::sync::atomic::Ordering;
use std::sync::{Arc, PoisonError};

use crate::catalogue;
use crate::config::HubConfig;
use crate::credential::{CredentialChain, CredentialOrigin};
use crate::descriptor::{ProviderDescriptor, ProviderRecord};
use crate::error::{HubError, NotFound, ProviderFailure, UsedBy};
use crate::health::ProviderHealth;
use crate::ids::{KindId, ScopeKey, Slug};
use crate::kinds::{DriverContext, KindDriver};
use crate::ports::HubEvent;
use crate::secret::Secret;
use crate::taxonomy::{AuthStyle, ProviderGroup};

use super::Hub;
use super::types::{KeyState, ProviderView};

/// A resolved credential and the source that supplied it.
pub(crate) struct Credential {
    pub(crate) key: Option<Secret>,
    pub(crate) origin: Option<CredentialOrigin>,
    /// The provider's health epoch **as read before the chain ran**: anything
    /// measured with this credential is dropped if the credential changed since.
    pub(crate) epoch: u64,
}

impl Credential {
    /// The identity a catalog cache entry is tied to: the credential, so a list
    /// read with a replaced key is never served to its successor. A source that
    /// rotates its own token (the platform token, a browser login) has none: the
    /// account owns the list, and a fresh token every minute must not refetch it
    /// every minute.
    ///
    /// `credentialed` is whether the request presents the credential at all: a
    /// keyless read (an auth style of none, a local runtime) shares one slot
    /// whatever key happens to be configured, so it has no identity either.
    pub(crate) fn cache_identity(&self, credentialed: bool) -> Option<crate::secret::SecretId> {
        if !credentialed {
            return None;
        }
        match self.origin {
            Some(
                CredentialOrigin::InstanceIdentity
                | CredentialOrigin::SessionJwt
                | CredentialOrigin::OAuth,
            ) => None,
            _ => self.key.as_ref().map(Secret::id),
        }
    }
}

impl Hub {
    pub(crate) fn driver(&self, kind: &KindId) -> Result<Arc<dyn KindDriver>, HubError> {
        self.inner
            .registry
            .get(kind)
            .ok_or_else(|| HubError::NotFound(NotFound::Kind(kind.clone())))
    }

    pub(crate) fn chain_for(&self, kind: &KindId) -> &CredentialChain {
        self.inner
            .chains
            .get(kind)
            .unwrap_or(&self.inner.default_chain)
    }

    /// The header policy for a provider presented with `auth`: the built-in one
    /// plus the header a custom auth style carries the credential in, so a
    /// redirect to another origin strips it too.
    pub(crate) fn headers_for(&self, auth: &AuthStyle) -> crate::policy::HeaderPolicy {
        self.inner.headers.clone().with_auth(auth)
    }

    /// The driver context every request is made with.
    pub(crate) fn cx<'a>(&'a self, headers: &'a crate::policy::HeaderPolicy) -> DriverContext<'a> {
        let cx = DriverContext::new(
            &*self.inner.http,
            &self.inner.policy,
            &*self.inner.clock,
            headers,
        );
        match &self.inner.product {
            Some(product) => cx.with_product(product),
            None => cx,
        }
    }

    /// The auth style a record is presented with.
    pub(crate) fn auth_of(record: &ProviderRecord, descriptor: &ProviderDescriptor) -> AuthStyle {
        record
            .auth_override
            .clone()
            .unwrap_or_else(|| descriptor.auth.clone())
    }

    /// Runs the record's credential chain.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when a source cannot be read. Never treated
    /// as "no key".
    pub(crate) async fn credential(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
    ) -> Result<Credential, HubError> {
        let epoch = self.inner.health.epoch(scope, &record.slug);
        let resolved = self
            .chain_for(&record.kind)
            .resolve(scope, &record.slug)
            .await?;
        Ok(match resolved {
            Some((key, origin)) => Credential {
                key: Some(key),
                origin: Some(origin),
                epoch,
            },
            None => Credential {
                key: None,
                origin: None,
                epoch,
            },
        })
    }

    /// [`Hub::credential`] for a caller that is about to **send** the credential
    /// to `record`'s endpoint: resolved under the provider's lock, after
    /// re-reading the record.
    ///
    /// An endpoint move holds the same lock across every step, so a credential
    /// resolved here is one the record's endpoint, as of now, was entered for: a
    /// probe or a listing that read the record just before a move cannot pair it
    /// with the key the move writes, nor read mid-move state. Only a kind whose
    /// endpoint can be edited can move, so the others (the catalogue's clouds, the
    /// managed provider, whose endpoint is the host's) skip the lock and the
    /// extra read. The lock is held across one credential-chain read, so a source
    /// must not call back into the hub from `resolve`.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`] when the provider went while the caller was
    /// waiting; [`HubError::Conflict`] when its endpoint changed since the caller
    /// read the record (retry against the new one). Otherwise as
    /// [`Hub::credential`].
    pub(crate) async fn credential_checked(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
    ) -> Result<Credential, HubError> {
        // Only a provider that can move **and** sends a credential has anything
        // to pair: a local runtime with no auth style never presents one.
        let movable = self.inner.registry.get(&record.kind).is_some_and(|driver| {
            let descriptor = driver.descriptor();
            descriptor.endpoint_editable && Self::auth_of(record, descriptor).needs_credential()
        });
        if !movable || self.group_of(record) == ProviderGroup::Managed {
            return self.credential(scope, record).await;
        }
        let _guard = self.slot_lock(scope, &record.slug).await;
        let config = self.read_config(scope).await?;
        let now = config
            .provider(&record.slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(record.slug.clone())))?;
        if now.base_url != record.base_url {
            return Err(HubError::Conflict);
        }
        self.credential(scope, record).await
    }

    /// [`Hub::credential_checked`] that answers a benign concurrent endpoint move
    /// by looking at the record again (up to three times) instead of failing:
    /// returns the record the credential was resolved for, which the caller
    /// must use, with the credential.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`] when the provider went; [`HubError::Conflict`] when
    /// it was replaced by another provider under the same slug, or kept moving.
    pub(crate) async fn checked_credential(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
    ) -> Result<(ProviderRecord, Credential), HubError> {
        let mut current = record.clone();
        for _ in 0..3 {
            match self.credential_checked(scope, &current).await {
                Ok(credential) => return Ok((current, credential)),
                Err(HubError::Conflict) => {
                    let fresh = self
                        .read_config(scope)
                        .await?
                        .provider(&record.slug)
                        .ok_or_else(|| HubError::NotFound(NotFound::Provider(record.slug.clone())))?
                        .clone();
                    if fresh.id != record.id {
                        return Err(HubError::Conflict);
                    }
                    current = fresh;
                }
                Err(other) => return Err(other),
            }
        }
        Err(HubError::Conflict)
    }

    pub(crate) fn group_of(&self, record: &ProviderRecord) -> ProviderGroup {
        self.inner.registry.get(&record.kind).map_or_else(
            || catalogue::group_of(record.kind.as_str()),
            |d| d.descriptor().group,
        )
    }

    pub(crate) async fn key_state(&self, scope: &ScopeKey, record: &ProviderRecord) -> KeyState {
        match self.credential(scope, record).await {
            Ok(Credential {
                origin: Some(origin),
                ..
            }) => KeyState::Configured(origin),
            Ok(_) => KeyState::Missing,
            Err(_) => KeyState::Unreadable,
        }
    }

    /// A provider as an operator sees it. Never fails: an unreadable credential
    /// store is shown as such.
    pub(crate) async fn view(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
        config: &HubConfig,
    ) -> ProviderView {
        let is_default = match &config.default {
            crate::config::DefaultChoice::Unset => false,
            crate::config::DefaultChoice::ProviderOnly { provider }
            | crate::config::DefaultChoice::Full { provider, .. } => *provider == record.slug,
        };
        ProviderView {
            group: self.group_of(record),
            kind_label: self.inner.registry.get(&record.kind).map_or_else(
                || record.kind.to_string(),
                |d| d.descriptor().label.to_string(),
            ),
            key: self.key_state(scope, record).await,
            is_default,
            record: record.clone(),
        }
    }

    /// What a provider's health reads as to an operator (see
    /// [`ProviderStatus::health`](super::ProviderStatus)).
    pub(crate) async fn effective_health(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
        key: &KeyState,
    ) -> Result<(ProviderHealth, crate::health::HealthSnapshot), HubError> {
        let mut snapshot = self.inner.health.snapshot(scope, &record.slug).await?;
        // A stored "signed out" is a fact about a moment with no credential. The
        // host signs a user in through its own token source, which the hub is
        // not told about (no epoch moves), so a mark written just before that
        // sign-in would otherwise outlive it. A credential is answering now:
        // whatever the mark said no longer holds, and the next probe or turn
        // decides (finding 4.8).
        if snapshot.health == ProviderHealth::SignedOut && matches!(key, KeyState::Configured(_)) {
            snapshot.health = ProviderHealth::Unknown;
        }
        let group = self.group_of(record);
        let offline_excluded = !self.inner.policy.allow_public
            && !matches!(group, ProviderGroup::Local | ProviderGroup::Cli);
        let health = if !record.enabled || offline_excluded {
            ProviderHealth::Disabled
        } else if group == ProviderGroup::Managed && matches!(key, KeyState::Missing) {
            ProviderHealth::SignedOut
        } else {
            snapshot.health
        };
        Ok((health, snapshot))
    }

    /// Forgets a provider's health **after** the change that made it stale has
    /// been committed.
    ///
    /// Best effort by design: the operation already succeeded, and health is
    /// derived data. Reporting a failed operation because the health store had an
    /// outage would leave the caller believing a committed change did not happen.
    /// A health store that is down is reported by the reads that need it
    /// (`health`, `status`).
    pub(crate) async fn forget_health(&self, scope: &ScopeKey, slug: &Slug) {
        self.inner
            .retests
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .remove(&(scope.clone(), slug.clone()));
        if let Err(error) = self.inner.health.forget(scope, slug).await {
            tracing::warn!(%slug, reason = %error.reason(), "could not forget a provider's health");
        }
    }

    /// The hooks every credential change runs: the provider's health is
    /// forgotten (what was learned about the old credential says nothing about
    /// the new one) and the scope's cached catalogs are dropped.
    pub(crate) async fn after_key_change(&self, scope: &ScopeKey, slug: &Slug, present: bool) {
        // Health first (it bumps the epoch), the cache second: a list that read the
        // old key and fills the cache after this sees the bumped epoch and evicts
        // itself; one that filled it before is evicted here.
        self.forget_health(scope, slug).await;
        self.inner.cache.evict_scope(scope);
        self.inner.events.emit(HubEvent::KeyChanged {
            scope: scope.clone(),
            slug: slug.clone(),
            present,
        });
    }

    /// A rejected credential: tell the source that supplied it so a host that
    /// caches (a rotating token) refreshes instead of replaying it, naming the
    /// credential that was rejected so a source that has already rotated keeps its
    /// fresh one.
    pub(crate) fn note_rejection(
        &self,
        scope: &ScopeKey,
        kind: &KindId,
        credential: &Credential,
        failure: &ProviderFailure,
    ) {
        if let Some(origin) = credential.origin.as_ref()
            && let Some(key) = credential.key.as_ref()
            && failure.is_rejection()
        {
            self.chain_for(kind)
                .invalidate_origin_rejected(scope, origin, key.id());
        }
    }

    /// A rejection reported by the host for a turn whose credential the hub does
    /// not know (`record_outcome`): the credential answering now may not be the
    /// one that was rejected, so the source is told without naming one and
    /// refreshes as it always did.
    pub(crate) fn note_rejection_unattributed(
        &self,
        scope: &ScopeKey,
        kind: &KindId,
        credential: &Credential,
        failure: &ProviderFailure,
    ) {
        if let Some(origin) = credential.origin.as_ref()
            && failure.is_rejection()
        {
            self.chain_for(kind).invalidate_origin(scope, origin);
        }
    }

    /// What references a provider: the default, the hub's own pins and routes,
    /// and whatever the host's [`UsageQuery`](crate::ports::UsageQuery) adds.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the host cannot say: a guard that
    /// cannot see must not assume nothing is in use.
    pub(crate) async fn host_used_by(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
    ) -> Result<UsedBy, HubError> {
        match &self.inner.usage {
            Some(usage) => usage
                .used_by(scope, slug)
                .await
                .map_err(|e| e.into_hub(crate::error::PortName::Config)),
            None => Ok(UsedBy::default()),
        }
    }

    /// The references inside the hub's own configuration.
    pub(crate) fn config_used_by(config: &HubConfig, slug: &Slug) -> UsedBy {
        use crate::config::DefaultChoice;
        use crate::route::RouteTarget;
        UsedBy {
            default_choice: match &config.default {
                DefaultChoice::Unset => false,
                DefaultChoice::ProviderOnly { provider } | DefaultChoice::Full { provider, .. } => {
                    provider == slug
                }
            },
            agents: config
                .agent_pins
                .iter()
                .filter(|(_, choice)| &choice.provider == slug)
                .map(|(agent, _)| agent.clone())
                .collect(),
            workloads: config
                .workload_routes
                .iter()
                .filter(|(_, route)| matches!(&route.target, RouteTarget::Provider(s) if s == slug))
                .map(|(workload, _)| workload.clone())
                .collect(),
            other: Vec::new(),
        }
    }

    /// A deterministic opaque record id.
    pub(crate) fn new_record_id(&self, slug: &Slug) -> String {
        use std::hash::{Hash, Hasher};
        let n = self.inner.ids.fetch_add(1, Ordering::SeqCst);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (slug.as_str(), self.inner.clock.wall_ms(), n).hash(&mut hasher);
        let a = hasher.finish();
        (a, n).hash(&mut hasher);
        format!("prv_{a:016x}{:016x}", hasher.finish())
    }
}

impl Hub {
    /// The current value of a provider's key slot: `Ok(None)` is "no key",
    /// `Err` is "could not find out" and stops the caller before it writes.
    pub(crate) async fn read_slot(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
    ) -> Result<Option<Secret>, HubError> {
        self.inner
            .credentials
            .get(scope, &slug.key_slot())
            .await
            .map_err(|e| e.into_hub(crate::error::PortName::Credentials))
    }

    pub(crate) async fn write_slot(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        key: Secret,
    ) -> Result<(), HubError> {
        self.inner
            .credentials
            .set(scope, &slug.key_slot(), key)
            .await
            .map_err(|e| e.into_hub(crate::error::PortName::Credentials))
    }

    pub(crate) async fn delete_slot(&self, scope: &ScopeKey, slug: &Slug) -> Result<(), HubError> {
        self.inner
            .credentials
            .delete(scope, &slug.key_slot())
            .await
            .map_err(|e| e.into_hub(crate::error::PortName::Credentials))
    }

    /// Announces that a provider's key **may have changed** where the operation
    /// cannot say how it ended (a store that failed part-way): reads the slot as
    /// it is now, so the event says what is there rather than what was hoped.
    ///
    /// When the slot cannot be read (a store that just failed part-way is often
    /// still down) no claim about the key is made: health and cached catalogs are
    /// dropped, and no `KeyChanged` event says something that may be false.
    pub(crate) async fn announce_key_state(&self, scope: &ScopeKey, slug: &Slug) {
        match self.read_slot(scope, slug).await {
            Ok(key) => self.after_key_change(scope, slug, key.is_some()).await,
            Err(error) => {
                tracing::warn!(%slug, reason = %error.reason(), "could not read the key slot to announce it");
                self.forget_health(scope, slug).await;
                self.inner.cache.evict_scope(scope);
            }
        }
    }

    /// Puts a slot back to `previous` after an operation that is failing; if that
    /// cannot be done the key really is not what the caller is told (nothing
    /// changed), so it is logged and announced. One place, so the announcement is
    /// part of restoring rather than something each caller remembers.
    pub(crate) async fn restore_or_announce(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        previous: Option<Secret>,
    ) {
        if let Err(error) = self.restore_slot(scope, slug, previous).await {
            tracing::warn!(%slug, reason = %error.reason(), "could not restore the previous key");
            self.announce_key_state(scope, slug).await;
        }
    }

    /// Deletes a slot that belongs to nothing (its provider went, or moved, while
    /// a key was being stored). A failure cannot be reported to the caller, whose
    /// operation is already failing for another reason, but it is logged: the key
    /// left behind would answer for the next provider of that slug.
    pub(crate) async fn delete_orphan_slot(&self, scope: &ScopeKey, slug: &Slug) {
        if let Err(error) = self.delete_slot(scope, slug).await {
            tracing::warn!(%slug, reason = %error.reason(), "could not delete a key that belongs to no provider");
        }
    }

    /// Puts a slot back to what it held before an operation touched it.
    pub(crate) async fn restore_slot(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        previous: Option<Secret>,
    ) -> Result<(), HubError> {
        match previous {
            Some(key) => self.write_slot(scope, slug, key).await,
            None => self.delete_slot(scope, slug).await,
        }
    }
}
