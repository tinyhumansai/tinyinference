//! Model capabilities tagged with where each fact came from (D12).

use serde::{Deserialize, Serialize};

/// A tri-state capability. It never defaults to `Yes`: an unknown capability
/// must not be promoted to a claim.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tri {
    /// Supported.
    Yes,
    /// Not supported.
    No,
    /// Nobody has said.
    #[default]
    Unknown,
}

impl Tri {
    /// Whether the capability is known to be supported.
    pub fn is_yes(self) -> bool {
        matches!(self, Self::Yes)
    }
}

/// Where a capability value came from, in decreasing order of trust.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapSource {
    /// The provider's own model listing.
    ProviderApi,
    /// A probe of a local runtime.
    LocalProbe,
    /// A metadata registry (`ModelMetadataSource`).
    Registry,
    /// An operator override.
    UserOverride,
    /// Nothing supplied it; the value is a placeholder.
    #[default]
    Default,
}

/// A value plus the source that supplied it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sourced<T> {
    /// The value.
    pub value: T,
    /// Where the value came from.
    pub source: CapSource,
}

impl<T> Sourced<T> {
    /// A value from a named source.
    pub fn new(value: T, source: CapSource) -> Self {
        Self { value, source }
    }
}

impl<T: Default> Default for Sourced<T> {
    fn default() -> Self {
        Self {
            value: T::default(),
            source: CapSource::Default,
        }
    }
}

/// What a model can do, each fact with its source. Every field defaults to
/// unknown, never to "yes".
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Context window in tokens.
    pub context_window: Sourced<Option<u64>>,
    /// Maximum output tokens.
    pub max_output: Sourced<Option<u64>>,
    /// Tool calling.
    pub tools: Sourced<Tri>,
    /// Image input.
    pub vision: Sourced<Tri>,
    /// Reasoning / thinking output.
    pub reasoning: Sourced<Tri>,
    /// Whether a `temperature` parameter is accepted.
    pub temperature: Sourced<Tri>,
    /// Structured (JSON-schema) output.
    pub structured_output: Sourced<Tri>,
}
