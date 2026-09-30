//! [`Hub`]: the facade a host embeds, and [`HubBuilder`], which wires the ports.
//!
//! # State machine (the one spec every operation follows)
//!
//! **Configuration** is one document per scope, changed only by
//! *load, validate the guards, save(expect = the version loaded)*, retried up
//! to three times on a lost compare-and-swap with the guards **re-run** against
//! what was just loaded (a guard decision is only valid for the version it
//! read). References the hub does not store (a credential slot, a host's own
//! pins) are read once before the loop; they are not protected by the
//! configuration's version and the operations say where that matters.
//!
//! **Credentials** live in the [`CredentialStore`]
//! under [`Slug::key_slot`](crate::Slug::key_slot), never on a record, and are
//! resolved through a per-kind [`CredentialChain`] on every request. An
//! operation that changes a key and a record orders the two so that a writer
//! that *loses* has touched nothing it does not own:
//!
//! * **add**: read the previous slot value, save the record, *then* write the
//!   key; a lost race (the slug is taken, the compare-and-swap keeps failing)
//!   never reaches the slot, and if the key cannot be written the record is
//!   taken out again. Undoing a connect puts back exactly what it changed: the
//!   record, the default it may have replaced, the previous key;
//! * **edit** and **set_key**: write the key, then the record (or, for
//!   `set_key`, re-check the record); a provider that was removed meanwhile is a
//!   `NotFound`, the key just written is deleted again and the old one is not
//!   restored;
//! * **remove**: delete the key, remove the record, and put the key back if the
//!   record cannot be removed (unless it was already gone).
//!
//! What no ordering can fix is a `set_key` racing a `remove` in the instant
//! between two stores; the re-check narrows that window to two adjacent calls.
//!
//! **A key change** (set, clear, rotate, edit with a key, remove) drops the
//! provider's health snapshot and every catalog cached for the scope, because a
//! new credential can change what the endpoint answers without changing
//! anything the hub can see. A rejected credential (`401`, or reason `auth`)
//! seen on any request invalidates the source that supplied it.
//!
//! **A result measured against an old credential is dropped.** Every probe and
//! every turn through [`Hub::chat_model`] reads the provider's health *epoch*
//! **before it resolves the credential**; forgetting health (a key change, a removal) bumps it, and a
//! result that arrives carrying an older epoch is not recorded. A failing check
//! made with the previous key therefore cannot mark a working new key `Down`.
//!
//! **Health** is folded from probes and from real turns
//! ([`Hub::record_outcome`], which the client also calls). A provider that is
//! `Down` on a rejected key or an exhausted account has no turns to clear it, so
//! a host schedules [`Hub::retest_down`]; the hub itself never spawns anything.
//!
//! **The default choice** is never silently changed by another operation: it is
//! set by [`Hub::set_default`], by the first provider ever added (guard G9), and
//! cleared only by [`Hub::clear_default`]. Removing a provider leaves it (and
//! every pin and route that names it) in place, where it fails closed on the
//! turn path.
//!
//! **The managed provider** is always listed and sorted first when the host
//! configures one ([`ManagedConfig`]); it cannot be removed, can be disabled,
//! and takes a key like any other provider (D5).

mod access;
mod builder;
#[cfg(test)]
pub(crate) mod fixtures;
mod slot_lock;
mod tx;
mod types;

#[cfg(test)]
#[path = "test.rs"]
mod tests;

pub(crate) use access::Credential;
pub use builder::{HubBuilder, ManagedConfig};
pub use types::{
    Confirm, ConnectOptions, HubPolicy, HubStatus, KeyState, Mutation, MutationStatus,
    ProviderPatch, ProviderStatus, ProviderView, Retested,
};

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::catalog::{CatalogCache, ModelMetadataSource, ModelOverride};
use crate::client::ModelFactory;
use crate::credential::CredentialChain;
use crate::health::HealthTracker;
use crate::ids::KindId;
use crate::kinds::DriverRegistry;
use crate::policy::{EndpointPolicy, HeaderPolicy};
use crate::ports::{
    Clock, ConfigStore, CredentialStore, Detector, EnvSource, EventSink, Http, UsageQuery,
};

/// How many times a configuration change is attempted before it reports
/// [`HubError::Conflict`](crate::HubError::Conflict).
pub(crate) const MAX_CAS_ATTEMPTS: usize = 3;

pub(crate) struct Inner {
    pub(crate) credentials: Arc<dyn CredentialStore>,
    pub(crate) config: Arc<dyn ConfigStore>,
    pub(crate) http: Arc<dyn Http>,
    pub(crate) clock: Arc<dyn Clock>,
    pub(crate) events: Arc<dyn EventSink>,
    pub(crate) env: Option<Arc<dyn EnvSource>>,
    pub(crate) policy: EndpointPolicy,
    pub(crate) headers: HeaderPolicy,
    pub(crate) product: Option<(String, String)>,
    pub(crate) registry: DriverRegistry,
    pub(crate) cache: CatalogCache,
    pub(crate) health: HealthTracker,
    pub(crate) chains: HashMap<KindId, CredentialChain>,
    pub(crate) default_chain: CredentialChain,
    pub(crate) managed_endpoint: Option<String>,
    pub(crate) hub_policy: HubPolicy,
    pub(crate) metadata: Option<Arc<dyn ModelMetadataSource>>,
    pub(crate) overrides: Vec<ModelOverride>,
    pub(crate) usage: Option<Arc<dyn UsageQuery>>,
    pub(crate) detector: Option<Arc<dyn Detector>>,
    #[cfg(feature = "cli")]
    pub(crate) spawner: Option<Arc<dyn crate::ports::ProcessSpawner>>,
    pub(crate) models: Arc<dyn ModelFactory>,
    pub(crate) ids: AtomicU64,
    /// When each provider was last re-tested (wall milliseconds), so a re-test
    /// that cannot clear what it looked at still waits before the next one.
    pub(crate) retests: std::sync::Mutex<HashMap<(crate::ids::ScopeKey, crate::ids::Slug), u64>>,
    /// One async lock per `(scope, slug)` that every operation touching a key
    /// slot holds across its record-and-key sequence (see `slot_lock`).
    pub(crate) slot_locks: std::sync::Mutex<slot_lock::SlotLocks>,
}

/// The hub: every operation a host needs to manage and use inference providers.
///
/// Cheap to clone (an `Arc` inside), `Send + Sync`, and runtime-agnostic: it
/// never spawns a task, so a host that wants periodic work (re-testing a
/// provider that is down on a rejected key) calls [`Hub::retest_down`] itself.
/// Build one with [`Hub::builder`].
///
/// ```
/// use tinyinference_hub::ports::SystemClock;
/// use tinyinference_hub::ports::memory::{MemoryConfig, MemoryCredentials};
/// use tinyinference_hub::ports::{Http, HttpError, HubRequest, HubResponse};
/// use tinyinference_hub::{EndpointPolicy, Hub, ScopeKey};
///
/// // A host with no network: every request fails to connect.
/// #[derive(Debug)]
/// struct NoNet;
///
/// #[async_trait::async_trait]
/// impl Http for NoNet {
///     async fn send(&self, _: HubRequest, _: &EndpointPolicy) -> Result<HubResponse, HttpError> {
///         Err(HttpError::ConnectFailed)
///     }
/// }
///
/// let hub = Hub::builder()
///     .credentials(MemoryCredentials::new())
///     .config(MemoryConfig::new())
///     .http(NoNet)
///     .clock(SystemClock)
///     .build()
///     .expect("every required port is set");
/// let status = futures::executor::block_on(hub.status(&ScopeKey::new("user:local"))).unwrap();
/// assert!(status.providers.is_empty());
/// ```
#[derive(Clone)]
pub struct Hub {
    pub(crate) inner: Arc<Inner>,
}

impl Hub {
    /// Starts building a hub.
    pub fn builder() -> HubBuilder {
        HubBuilder::new()
    }

    /// The endpoint policy every request goes through.
    pub fn policy(&self) -> &EndpointPolicy {
        &self.inner.policy
    }

    /// The driver registry (which kinds this hub knows).
    pub fn kinds(&self) -> &DriverRegistry {
        &self.inner.registry
    }
}

impl fmt::Debug for Hub {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hub")
            .field("policy", &self.inner.policy)
            .field("kinds", &self.inner.registry.len())
            .field("managed", &self.inner.managed_endpoint.is_some())
            .finish_non_exhaustive()
    }
}
