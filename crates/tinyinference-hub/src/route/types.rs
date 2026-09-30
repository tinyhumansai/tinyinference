//! The data types of routing: what a turn asks for and what it resolves to.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::credential::CredentialOrigin;
use crate::endpoint::redact_endpoint;
use crate::ids::{AgentKey, KindId, ModelId, Slug, WorkloadKey};
use crate::taxonomy::{AuthStyle, CliKind, LocalRuntime, Protocol, ProviderGroup};

/// A sampling temperature: finite by construction, which is what lets a route
/// (and so the whole configuration) be `Eq`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "f32", into = "f32")]
pub struct Temperature(f32);

impl Eq for Temperature {}

impl Temperature {
    /// Wraps a finite temperature; `None` for NaN or infinity.
    pub fn new(value: f32) -> Option<Self> {
        value.is_finite().then_some(Self(value))
    }

    /// The temperature.
    pub fn get(self) -> f32 {
        self.0
    }
}

impl TryFrom<f32> for Temperature {
    type Error = &'static str;

    fn try_from(value: f32) -> Result<Self, Self::Error> {
        Self::new(value).ok_or("a temperature must be a finite number")
    }
}

impl From<Temperature> for f32 {
    fn from(value: Temperature) -> Self {
        value.0
    }
}

/// What a route points at.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum RouteTarget {
    /// Whatever the scope's default choice says.
    Default,
    /// One configured provider.
    Provider(Slug),
    /// The managed provider.
    Managed,
    /// A local runtime: a specific one, or (`None`) the first enabled local
    /// record.
    Local(Option<LocalRuntime>),
    /// A CLI login. Not an HTTP provider: the host runs the binary.
    Cli(CliKind),
    /// A caller-supplied endpoint that is never persisted. Parseable so an
    /// importer can name it; the hub does not resolve it (see
    /// [`Hub::resolve_for_turn`](crate::Hub::resolve_for_turn)).
    Ephemeral,
}

/// A structured route: a target, an optional model and an optional temperature.
/// Replaces OpenCompany's route strings and OpenHuman's provider strings.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProviderRoute {
    /// What the route points at.
    pub target: RouteTarget,
    /// The model, verbatim as the provider lists it. The record's own model is
    /// used when a route names none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelId>,
    /// A temperature for the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Temperature>,
}

impl std::hash::Hash for Temperature {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl ProviderRoute {
    /// A route to `target` with no model and no temperature.
    pub fn new(target: RouteTarget) -> Self {
        Self {
            target,
            model: None,
            temperature: None,
        }
    }

    /// The default route.
    pub fn default_route() -> Self {
        Self::new(RouteTarget::Default)
    }

    /// A route to a provider.
    pub fn provider(slug: Slug) -> Self {
        Self::new(RouteTarget::Provider(slug))
    }

    /// Sets the model.
    #[must_use]
    pub fn with_model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the temperature.
    #[must_use]
    pub fn with_temperature(mut self, temperature: Temperature) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// The provider this route names, if it names one directly.
    pub fn provider_slug(&self) -> Option<&Slug> {
        match &self.target {
            RouteTarget::Provider(slug) => Some(slug),
            _ => None,
        }
    }
}

/// What a turn asks the hub to resolve.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TurnQuery {
    /// The agent taking the turn, for its pin.
    pub agent: Option<AgentKey>,
    /// The workload (a host tier or role), for its route.
    pub workload: Option<WorkloadKey>,
    /// A route the caller forces for this turn only (a per-task override).
    pub override_route: Option<ProviderRoute>,
}

impl TurnQuery {
    /// A query with nothing set: the default choice.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the agent.
    #[must_use]
    pub fn with_agent(mut self, agent: AgentKey) -> Self {
        self.agent = Some(agent);
        self
    }

    /// Sets the workload.
    #[must_use]
    pub fn with_workload(mut self, workload: WorkloadKey) -> Self {
        self.workload = Some(workload);
        self
    }

    /// Sets the per-turn override.
    #[must_use]
    pub fn with_override(mut self, route: ProviderRoute) -> Self {
        self.override_route = Some(route);
        self
    }
}

/// Which rule chose the provider.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResolvedVia {
    /// The caller's per-turn override.
    Override,
    /// The agent's pin.
    Pin,
    /// The workload's route.
    Workload,
    /// The scope's default choice.
    Default,
}

/// What a turn will call. Carries no credential: the client resolves the
/// credential chain on every request.
#[non_exhaustive]
#[derive(Clone, PartialEq)]
pub struct ResolvedTurn {
    /// The provider.
    pub slug: Slug,
    /// Its kind (exact, so telemetry never says `unknown` for a catalogue kind).
    pub kind: KindId,
    /// Its group.
    pub group: ProviderGroup,
    /// The endpoint, empty for a CLI login.
    pub base_url: String,
    /// The model, verbatim. `None` only for a CLI route that names none.
    pub model: Option<ModelId>,
    /// The chat wire protocol.
    pub protocol: Protocol,
    /// How the credential is presented.
    pub auth: AuthStyle,
    /// Which rule chose this provider.
    pub via: ResolvedVia,
    /// Which credential source answered when the route was resolved.
    pub origin: Option<CredentialOrigin>,
    /// The turn's temperature, if the route set one.
    pub temperature: Option<Temperature>,
    /// The CLI login, for a CLI route.
    pub cli: Option<CliKind>,
}

impl fmt::Debug for ResolvedTurn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResolvedTurn")
            .field("slug", &self.slug)
            .field("kind", &self.kind)
            .field("base_url", &redact_endpoint(&self.base_url))
            .field("model", &self.model)
            .field("protocol", &self.protocol)
            .field("via", &self.via)
            .field("origin", &self.origin)
            .finish_non_exhaustive()
    }
}
