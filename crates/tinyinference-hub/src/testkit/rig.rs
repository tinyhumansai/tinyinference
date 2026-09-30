//! [`MemoryPorts`]: every port a [`Hub`](crate::Hub) needs, in memory, wired to
//! one fake clock and one scripted transport.

use std::sync::Arc;

use crate::hub::{Hub, HubBuilder};
use crate::policy::EndpointPolicy;
use crate::ports::memory::{MapEnv, MemoryConfig, MemoryCredentials, MemoryEvents, MemoryHealth};

use super::{FakeClock, ScriptedHttp};

/// The in-memory ports of a simulated host. Every handle is public so a test
/// can inspect what the hub did (the credential slots, the stored document, the
/// events, the requests it sent) and inject faults.
#[derive(Clone, Debug)]
pub struct MemoryPorts {
    /// The one timeline every component shares.
    pub clock: FakeClock,
    /// The scripted transport. An unscripted request panics.
    pub http: Arc<ScriptedHttp>,
    /// The credential store.
    pub credentials: Arc<MemoryCredentials>,
    /// The configuration store (real compare-and-swap versions).
    pub config: Arc<MemoryConfig>,
    /// The health store.
    pub health: Arc<MemoryHealth>,
    /// Every event the hub emitted.
    pub events: Arc<MemoryEvents>,
    /// The environment the hub may read (empty by default).
    pub env: Arc<MapEnv>,
}

impl MemoryPorts {
    /// Fresh, empty ports.
    pub fn new() -> Self {
        let clock = FakeClock::new();
        Self {
            http: Arc::new(ScriptedHttp::new(clock.clone())),
            clock,
            credentials: Arc::new(MemoryCredentials::new()),
            config: Arc::new(MemoryConfig::new()),
            health: Arc::new(MemoryHealth::new()),
            events: Arc::new(MemoryEvents::new()),
            env: Arc::new(MapEnv::new()),
        }
    }

    /// Replaces the environment.
    #[must_use]
    pub fn with_env(mut self, env: MapEnv) -> Self {
        self.env = Arc::new(env);
        self
    }

    /// A builder with every required port set and the endpoint policy at
    /// [`EndpointPolicy::desktop`] (loopback allowed, so local kinds work).
    /// Tweak it and call `build`.
    pub fn builder(&self) -> HubBuilder {
        Hub::builder()
            .credentials_arc(self.credentials.clone())
            .config_arc(self.config.clone())
            .http_arc(self.http.clone())
            .clock(self.clock.clone())
            .health_store(self.health.clone())
            .events(self.events.clone())
            .env(self.env.clone())
            .policy(EndpointPolicy::desktop())
    }

    /// A hub over these ports with the default builder.
    ///
    /// # Panics
    ///
    /// Never: every required port is set.
    pub fn hub(&self) -> Hub {
        self.builder()
            .build()
            .expect("MemoryPorts::builder sets every required port")
    }
}

impl Default for MemoryPorts {
    fn default() -> Self {
        Self::new()
    }
}
