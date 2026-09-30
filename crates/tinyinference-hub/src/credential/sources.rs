//! The built-in [`CredentialSource`]s.

use std::fmt::{self, Debug};
use std::sync::Arc;

use async_trait::async_trait;

use crate::error::PortName;
use crate::ids::{ScopeKey, Slug};
use crate::ports::{CredentialStore, EnvSource, PortError, TokenSource};
use crate::secret::{Secret, SecretId};

use super::{CredentialOrigin, CredentialSource};

/// Where a [`StoreSource`] looks in the store.
#[derive(Clone, Debug)]
enum SlotRule {
    /// `provider/<slug>/key`: one key per provider.
    PerProvider,
    /// One slot whatever the provider (the company account key).
    Fixed(String),
}

/// Reads a key from a [`CredentialStore`].
#[derive(Clone)]
pub struct StoreSource {
    store: Arc<dyn CredentialStore>,
    slot: SlotRule,
    origin: CredentialOrigin,
}

impl StoreSource {
    /// The provider's own pasted key, at [`Slug::key_slot`].
    pub fn provider_key(store: Arc<dyn CredentialStore>) -> Self {
        Self {
            store,
            slot: SlotRule::PerProvider,
            origin: CredentialOrigin::ProviderKey,
        }
    }

    /// One fixed slot for every provider, reporting `origin` (the company
    /// account key is `AccountKey` at a slot of the host's choosing).
    pub fn fixed_slot(
        store: Arc<dyn CredentialStore>,
        slot: impl Into<String>,
        origin: CredentialOrigin,
    ) -> Self {
        Self {
            store,
            slot: SlotRule::Fixed(slot.into()),
            origin,
        }
    }
}

impl Debug for StoreSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoreSource")
            .field("slot", &self.slot)
            .field("origin", &self.origin)
            .finish()
    }
}

#[async_trait]
impl CredentialSource for StoreSource {
    fn origin(&self) -> CredentialOrigin {
        self.origin.clone()
    }

    async fn resolve(&self, scope: &ScopeKey, slug: &Slug) -> Result<Option<Secret>, PortError> {
        match &self.slot {
            SlotRule::PerProvider => self.store.get(scope, &slug.key_slot()).await,
            SlotRule::Fixed(slot) => self.store.get(scope, slot).await,
        }
    }
}

/// Reads a key from an environment variable through the [`EnvSource`] port.
#[derive(Clone)]
pub struct EnvVarSource {
    env: Arc<dyn EnvSource>,
    name: String,
}

impl EnvVarSource {
    /// A source for the variable `name`.
    pub fn new(env: Arc<dyn EnvSource>, name: impl Into<String>) -> Self {
        Self {
            env,
            name: name.into(),
        }
    }
}

impl Debug for EnvVarSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EnvVarSource")
            .field("name", &self.name)
            .finish()
    }
}

#[async_trait]
impl CredentialSource for EnvVarSource {
    fn origin(&self) -> CredentialOrigin {
        CredentialOrigin::Env(self.name.clone())
    }

    async fn resolve(&self, _scope: &ScopeKey, _slug: &Slug) -> Result<Option<Secret>, PortError> {
        // A trailing newline or a blank value is not a key: detection treats
        // them as unset, and so must the chain.
        Ok(self
            .env
            .var(&self.name)
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .map(Secret::new))
    }
}

/// A key fixed at construction: a one-shot CLI given `--api-key`, a test.
#[derive(Clone)]
pub struct StaticSource {
    secret: Secret,
}

impl StaticSource {
    /// A source that always answers `secret`.
    pub fn new(secret: Secret) -> Self {
        Self { secret }
    }
}

impl Debug for StaticSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StaticSource")
            .field("secret", &self.secret)
            .finish()
    }
}

#[async_trait]
impl CredentialSource for StaticSource {
    fn origin(&self) -> CredentialOrigin {
        CredentialOrigin::Static
    }

    async fn resolve(&self, _scope: &ScopeKey, _slug: &Slug) -> Result<Option<Secret>, PortError> {
        Ok(Some(self.secret.clone()))
    }
}

/// Adapts a rotating [`TokenSource`] (the managed platform token, an OAuth
/// access token) into the chain. It asks on every call, and passes
/// [`CredentialSource::invalidate`] through.
#[derive(Clone)]
pub struct TokenSourceAdapter {
    source: Arc<dyn TokenSource>,
    origin: CredentialOrigin,
}

impl TokenSourceAdapter {
    /// Wraps `source`, reporting `origin` (`InstanceIdentity`, `SessionJwt`,
    /// `OAuth`).
    pub fn new(source: Arc<dyn TokenSource>, origin: CredentialOrigin) -> Self {
        Self { source, origin }
    }
}

impl Debug for TokenSourceAdapter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenSourceAdapter")
            .field("origin", &self.origin)
            .finish()
    }
}

#[async_trait]
impl CredentialSource for TokenSourceAdapter {
    fn origin(&self) -> CredentialOrigin {
        self.origin.clone()
    }

    fn port(&self) -> PortName {
        PortName::Token
    }

    async fn resolve(&self, scope: &ScopeKey, _slug: &Slug) -> Result<Option<Secret>, PortError> {
        self.source.token(scope).await
    }

    fn invalidate(&self, scope: &ScopeKey) {
        self.source.invalidate(scope);
    }

    fn invalidate_rejected(&self, scope: &ScopeKey, rejected: SecretId) {
        self.source.invalidate_rejected(scope, rejected);
    }
}

/// A predicate over `(scope, slug)` deciding whether a legacy slot applies.
type Gate = dyn Fn(&ScopeKey, &Slug) -> bool + Send + Sync;

/// OpenCompany's legacy flat key (`inference/key`), which predates per-provider
/// slots and is read only for the row it always belonged to: entry zero, the
/// managed provider on the default harness. The gate says whether it applies to
/// this `(scope, slug)`; outside the gate the source answers "not here".
///
/// Never written by the hub. Removing the writer is OpenCompany's to do.
#[derive(Clone)]
pub struct LegacyFlatSlot {
    store: Arc<dyn CredentialStore>,
    slot: String,
    gate: Arc<Gate>,
}

impl LegacyFlatSlot {
    /// Reads `slot`, but only where `gate(scope, slug)` is true.
    pub fn new(
        store: Arc<dyn CredentialStore>,
        slot: impl Into<String>,
        gate: impl Fn(&ScopeKey, &Slug) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            store,
            slot: slot.into(),
            gate: Arc::new(gate),
        }
    }
}

impl Debug for LegacyFlatSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LegacyFlatSlot")
            .field("slot", &self.slot)
            .finish()
    }
}

#[async_trait]
impl CredentialSource for LegacyFlatSlot {
    fn origin(&self) -> CredentialOrigin {
        CredentialOrigin::ProviderKey
    }

    async fn resolve(&self, scope: &ScopeKey, slug: &Slug) -> Result<Option<Secret>, PortError> {
        if (self.gate)(scope, slug) {
            self.store.get(scope, &self.slot).await
        } else {
            Ok(None)
        }
    }
}
