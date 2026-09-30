//! [`HubBuilder`] and [`ManagedConfig`].

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use crate::catalog::{CatalogCache, ModelMetadataSource, ModelOverride};
use crate::catalogue;
use crate::client::{LlmModelFactory, ModelFactory};
use crate::credential::{CredentialChain, CredentialSource, EnvVarSource, StoreSource};
use crate::detect::env_vars_for_kind;
use crate::error::{HubError, InputField, InvalidInput};
use crate::health::HealthTracker;
use crate::ids::KindId;
use crate::kinds::{DriverRegistry, KindDriver, ManagedDriver};
use crate::policy::{EndpointPolicy, HeaderPolicy};
use crate::ports::memory::{MemoryHealth, NoopEvents};
use crate::ports::{
    Clock, ConfigStore, CredentialStore, Detector, EnvSource, EventSink, HealthStore, Http,
    UsageQuery,
};
use crate::taxonomy::ProviderGroup;

use super::{Hub, HubPolicy, Inner};

/// How the host reaches its managed (TinyHumans) backend.
///
/// The hub hardcodes neither backend's URL (OpenCompany and OpenHuman reach
/// different ones): the host names the endpoint and, if the catalog is
/// OpenAI-shaped, the query it needs. The **first** source of the managed
/// credential chain is always the key pasted for the provider itself; the
/// sources added here follow it (a company account key, an instance identity, a
/// session token).
#[non_exhaustive]
pub struct ManagedConfig {
    /// The endpoint requests go to.
    pub base_url: String,
    /// `Some(query)` when the catalog is OpenAI-shaped (OpenHuman) rather than
    /// the paged envelope (OpenCompany); the query is appended to `/models`.
    pub openai_catalog_query: Option<String>,
    /// Sources tried after the pasted key.
    pub sources: Vec<Box<dyn CredentialSource>>,
    /// A product-identity header, sent only to first-party hosts (guard G26).
    pub product_header: Option<(String, String)>,
}

impl ManagedConfig {
    /// A managed backend at `base_url` with the paged catalog.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            openai_catalog_query: None,
            sources: Vec::new(),
            product_header: None,
        }
    }

    /// The catalog is OpenAI-shaped; `query` (without the `?`) is appended.
    #[must_use]
    pub fn openai_shaped(mut self, query: impl Into<String>) -> Self {
        self.openai_catalog_query = Some(query.into());
        self
    }

    /// Adds a credential source after the pasted key.
    #[must_use]
    pub fn source(mut self, source: impl CredentialSource + 'static) -> Self {
        self.sources.push(Box::new(source));
        self
    }

    /// Sets the product-identity header.
    #[must_use]
    pub fn product_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.product_header = Some((name.into(), value.into()));
        self
    }
}

impl fmt::Debug for ManagedConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ManagedConfig")
            .field(
                "base_url",
                &crate::endpoint::redact_endpoint(&self.base_url),
            )
            .field("openai_catalog_query", &self.openai_catalog_query)
            .field("sources", &self.sources.len())
            .finish()
    }
}

/// Builds a [`Hub`]. Four ports are required (credentials, configuration,
/// HTTP, clock); everything else has a default.
pub struct HubBuilder {
    credentials: Option<Arc<dyn CredentialStore>>,
    config: Option<Arc<dyn ConfigStore>>,
    http: Option<Arc<dyn Http>>,
    clock: Option<Arc<dyn Clock>>,
    health: Option<Arc<dyn HealthStore>>,
    events: Option<Arc<dyn EventSink>>,
    env: Option<Arc<dyn EnvSource>>,
    env_credentials: bool,
    policy: EndpointPolicy,
    hub_policy: HubPolicy,
    registry: DriverRegistry,
    extra_sources: Vec<(KindId, Box<dyn CredentialSource>)>,
    managed: Option<ManagedConfig>,
    metadata: Option<Arc<dyn ModelMetadataSource>>,
    overrides: Vec<ModelOverride>,
    usage: Option<Arc<dyn UsageQuery>>,
    detector: Option<Arc<dyn Detector>>,
    #[cfg(feature = "cli")]
    spawner: Option<Arc<dyn crate::ports::ProcessSpawner>>,
    models: Option<Arc<dyn ModelFactory>>,
}

fn missing(what: &'static str) -> HubError {
    HubError::Invalid(InvalidInput::Malformed {
        field: InputField::Config,
        reason: what,
    })
}

impl HubBuilder {
    pub(super) fn new() -> Self {
        Self {
            credentials: None,
            config: None,
            http: None,
            clock: None,
            health: None,
            events: None,
            env: None,
            env_credentials: false,
            policy: EndpointPolicy::hosted(),
            hub_policy: HubPolicy::default(),
            registry: DriverRegistry::with_builtin(),
            extra_sources: Vec::new(),
            managed: None,
            metadata: None,
            overrides: Vec::new(),
            usage: None,
            detector: None,
            #[cfg(feature = "cli")]
            spawner: None,
            models: None,
        }
    }

    /// The credential store (required).
    #[must_use]
    pub fn credentials(self, store: impl CredentialStore + 'static) -> Self {
        self.credentials_arc(Arc::new(store))
    }

    /// The credential store, already shared.
    #[must_use]
    pub fn credentials_arc(mut self, store: Arc<dyn CredentialStore>) -> Self {
        self.credentials = Some(store);
        self
    }

    /// The configuration store (required).
    #[must_use]
    pub fn config(self, store: impl ConfigStore + 'static) -> Self {
        self.config_arc(Arc::new(store))
    }

    /// The configuration store, already shared.
    #[must_use]
    pub fn config_arc(mut self, store: Arc<dyn ConfigStore>) -> Self {
        self.config = Some(store);
        self
    }

    /// The HTTP transport (required).
    #[must_use]
    pub fn http(self, http: impl Http + 'static) -> Self {
        self.http_arc(Arc::new(http))
    }

    /// The HTTP transport, already shared.
    #[must_use]
    pub fn http_arc(mut self, http: Arc<dyn Http>) -> Self {
        self.http = Some(http);
        self
    }

    /// The clock (required).
    #[must_use]
    pub fn clock(self, clock: impl Clock + 'static) -> Self {
        self.clock_arc(Arc::new(clock))
    }

    /// The clock, already shared.
    #[must_use]
    pub fn clock_arc(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Where health survives a restart (default: in memory).
    #[must_use]
    pub fn health_store(mut self, store: Arc<dyn HealthStore>) -> Self {
        self.health = Some(store);
        self
    }

    /// Where events go (default: dropped).
    #[must_use]
    pub fn events(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.events = Some(sink);
        self
    }

    /// The environment, for [`env_credentials`](Self::env_credentials) and
    /// detection. Never read otherwise.
    #[must_use]
    pub fn env(mut self, env: Arc<dyn EnvSource>) -> Self {
        self.env = Some(env);
        self
    }

    /// Lets each catalogue kind's well-known environment variable
    /// (`OPENAI_API_KEY` and its siblings) answer as a credential **after** the
    /// stored key, so a CLI or CI run needs no persisted key. Off by default;
    /// requires [`env`](Self::env). The value is read per request and never
    /// copied anywhere.
    #[must_use]
    pub fn env_credentials(mut self, on: bool) -> Self {
        self.env_credentials = on;
        self
    }

    /// The endpoint policy (default [`EndpointPolicy::hosted`], the most
    /// restrictive).
    #[must_use]
    pub fn policy(mut self, policy: EndpointPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// The hub's own behaviour knobs.
    #[must_use]
    pub fn hub_policy(mut self, policy: HubPolicy) -> Self {
        self.hub_policy = policy;
        self
    }

    /// Registers (or replaces) a kind driver. The built-in kinds are registered
    /// already.
    #[must_use]
    pub fn kind(mut self, driver: Arc<dyn KindDriver>) -> Self {
        self.registry.register(driver);
        self
    }

    /// Registers the built-in kinds. Already done by default; kept so a builder
    /// that began with [`no_builtin_kinds`](Self::no_builtin_kinds) can say so.
    #[must_use]
    pub fn with_builtin_kinds(mut self) -> Self {
        let mut fresh = DriverRegistry::with_builtin();
        for kind in self.registry.kinds() {
            if let Some(driver) = self.registry.get(&kind) {
                fresh.register(driver);
            }
        }
        self.registry = fresh;
        self
    }

    /// Starts from an empty registry (a host that offers only its own kinds).
    #[must_use]
    pub fn no_builtin_kinds(mut self) -> Self {
        self.registry = DriverRegistry::new();
        self
    }

    /// Adds a credential source to one kind's chain, after the stored key.
    #[must_use]
    pub fn credential_source(
        mut self,
        kind: impl AsRef<str>,
        source: impl CredentialSource + 'static,
    ) -> Self {
        self.extra_sources
            .push((KindId::new(kind), Box::new(source)));
        self
    }

    /// Configures the managed provider (see [`ManagedConfig`]).
    #[must_use]
    pub fn managed(mut self, managed: ManagedConfig) -> Self {
        self.managed = Some(managed);
        self
    }

    /// A registry of model facts (D10); nothing is bundled.
    #[must_use]
    pub fn metadata(mut self, source: Arc<dyn ModelMetadataSource>) -> Self {
        self.metadata = Some(source);
        self
    }

    /// Operator corrections applied to every model list.
    #[must_use]
    pub fn overrides(mut self, overrides: Vec<ModelOverride>) -> Self {
        self.overrides = overrides;
        self
    }

    /// References only the host knows (its own pins), for the in-use guard.
    #[must_use]
    pub fn usage_query(mut self, usage: Arc<dyn UsageQuery>) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Replaces first-run detection.
    #[must_use]
    pub fn detector(mut self, detector: Arc<dyn Detector>) -> Self {
        self.detector = Some(detector);
        self
    }

    /// The subprocess spawner that CLI-login readiness runs through.
    #[cfg(feature = "cli")]
    #[must_use]
    pub fn process_spawner(mut self, spawner: Arc<dyn crate::ports::ProcessSpawner>) -> Self {
        self.spawner = Some(spawner);
        self
    }

    /// How `chat_model` builds the model behind a resolved turn (default: the
    /// `tinyinference-llm` builders).
    #[must_use]
    pub fn model_factory(mut self, factory: Arc<dyn ModelFactory>) -> Self {
        self.models = Some(factory);
        self
    }

    /// Builds the hub.
    ///
    /// # Errors
    ///
    /// [`HubError::Invalid`] naming the port when a required one is not set.
    pub fn build(self) -> Result<Hub, HubError> {
        let credentials = self
            .credentials
            .ok_or_else(|| missing("the credential store port is not set"))?;
        let config = self
            .config
            .ok_or_else(|| missing("the configuration store port is not set"))?;
        let http = self
            .http
            .ok_or_else(|| missing("the http port is not set"))?;
        let clock = self
            .clock
            .ok_or_else(|| missing("the clock port is not set"))?;
        let events = self.events.unwrap_or_else(|| Arc::new(NoopEvents));
        let health_store = self.health.unwrap_or_else(|| Arc::new(MemoryHealth::new()));

        let mut registry = self.registry;
        let mut product = None;
        let mut managed_endpoint = None;
        let mut managed_sources = Vec::new();
        if let Some(managed) = self.managed {
            managed_endpoint = Some(managed.base_url.trim().to_string());
            product = managed.product_header;
            managed_sources = managed.sources;
            if let Some(query) = managed.openai_catalog_query
                && let Some(descriptor) = catalogue::descriptors_in(ProviderGroup::Managed).next()
            {
                registry.register(Arc::new(ManagedDriver::openai_shaped(
                    descriptor.clone(),
                    query,
                )));
            }
        }

        let store_chain =
            || CredentialChain::new().with(StoreSource::provider_key(credentials.clone()));
        let mut chains: HashMap<KindId, CredentialChain> = HashMap::new();
        for kind in registry.kinds() {
            chains.insert(kind, store_chain());
        }
        if self.env_credentials
            && let Some(env) = &self.env
        {
            for (kind, chain) in &mut chains {
                for var in env_vars_for_kind(kind.as_str()) {
                    let taken = std::mem::take(chain);
                    *chain = taken.with(EnvVarSource::new(env.clone(), var));
                }
            }
        }
        for (kind, source) in self.extra_sources {
            let chain = chains.entry(kind).or_insert_with(store_chain);
            let taken = std::mem::take(chain);
            *chain = taken.with_boxed(source);
        }
        if let Some(descriptor) = catalogue::descriptors_in(ProviderGroup::Managed).next() {
            let chain = chains
                .entry(descriptor.kind.clone())
                .or_insert_with(store_chain);
            for source in managed_sources {
                let taken = std::mem::take(chain);
                *chain = taken.with_boxed(source);
            }
        }

        let cache = CatalogCache::new(clock.clone());
        let health = HealthTracker::new(health_store, clock.clone(), events.clone());
        Ok(Hub {
            inner: Arc::new(Inner {
                credentials,
                config,
                http,
                clock,
                events,
                env: self.env,
                policy: self.policy,
                headers: HeaderPolicy::builtin(),
                product,
                registry,
                cache,
                health,
                chains,
                default_chain: CredentialChain::new(),
                managed_endpoint,
                hub_policy: self.hub_policy,
                metadata: self.metadata,
                overrides: self.overrides,
                usage: self.usage,
                detector: self.detector,
                #[cfg(feature = "cli")]
                spawner: self.spawner,
                models: self
                    .models
                    .unwrap_or_else(|| Arc::new(LlmModelFactory::new())),
                ids: AtomicU64::new(0),
                retests: std::sync::Mutex::new(HashMap::new()),
                slot_locks: std::sync::Mutex::new(Default::default()),
            }),
        })
    }
}

impl fmt::Debug for HubBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HubBuilder")
            .field("policy", &self.policy)
            .field("kinds", &self.registry.len())
            .finish_non_exhaustive()
    }
}
