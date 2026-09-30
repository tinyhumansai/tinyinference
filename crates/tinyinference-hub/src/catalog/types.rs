//! The data types of a model catalog.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::descriptor::{CapSource, Capabilities, Sourced};
use crate::error::{InvalidInput, ProviderFailure};
use crate::ids::ModelId;

/// Where a model entry came from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EntrySource {
    /// The provider's own listing.
    #[default]
    ProviderApi,
    /// Reserved for a registry that supplies a model the provider did not list.
    /// No registry ships in phase 1, so nothing produces it yet.
    Registry,
    /// The operator added it (an Azure deployment name, a fine-tune).
    User,
}

/// Where a model is in its life.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleStatus {
    /// Current.
    Active,
    /// Still served, scheduled to go.
    Deprecated,
    /// No longer served.
    Retired,
}

/// A model's deprecation state.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Lifecycle {
    /// The status.
    pub status: LifecycleStatus,
    /// The retirement date as the source wrote it (`2026-11-01`).
    pub retires_on: Option<String>,
    /// What to use instead.
    pub replacement: Option<ModelId>,
}

/// One model a provider offers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelEntry {
    /// The id exactly as the provider spells it.
    pub id: ModelId,
    /// A human name, when the listing supplies one.
    pub display_name: Option<String>,
    /// The organisation that owns the model, when advertised.
    pub owned_by: Option<String>,
    /// What the model can do, each fact tagged with its source.
    pub capabilities: Capabilities,
    /// Charged input price in USD per million tokens, when published.
    pub input_per_1m: Option<f64>,
    /// Charged output price in USD per million tokens, when published.
    pub output_per_1m: Option<f64>,
    /// Where the entry came from.
    pub origin: EntrySource,
    /// Whether the provider currently serves it.
    pub available: bool,
    /// Deprecation state, when a registry knows it.
    pub lifecycle: Option<Lifecycle>,
    /// The model this id currently points at, for a moving alias.
    pub alias_of: Option<ModelId>,
}

impl ModelEntry {
    /// An entry with an id and nothing else known: every capability unknown,
    /// available, from the provider's listing.
    pub fn new(id: ModelId) -> Self {
        Self {
            id,
            display_name: None,
            owned_by: None,
            capabilities: Capabilities::default(),
            input_per_1m: None,
            output_per_1m: None,
            origin: EntrySource::ProviderApi,
            available: true,
            lifecycle: None,
            alias_of: None,
        }
    }

    /// Sets the context window, tagged as coming from `source`.
    #[must_use]
    pub fn with_context_window(mut self, tokens: u64, source: CapSource) -> Self {
        self.capabilities.context_window = Sourced::new(Some(tokens), source);
        self
    }

    /// Sets the display name.
    #[must_use]
    pub fn with_display_name(mut self, name: impl Into<String>) -> Self {
        self.display_name = Some(name.into());
        self
    }

    /// Sets the prices (USD per million tokens).
    #[must_use]
    pub fn with_prices(mut self, input: Option<f64>, output: Option<f64>) -> Self {
        self.input_per_1m = input;
        self.output_per_1m = output;
        self
    }
}

impl TryFrom<tinyinference_llm::catalog::ModelInfo> for ModelEntry {
    type Error = InvalidInput;

    /// Lifts llm's listing entry into the hub's.
    ///
    /// # Errors
    ///
    /// [`InvalidInput`] when the listing's id is not a valid [`ModelId`] (it
    /// contains whitespace or control characters, or is over 256 characters).
    /// Such an entry cannot be addressed by a route, so callers drop it and
    /// count it rather than rewriting the provider's id.
    fn try_from(info: tinyinference_llm::catalog::ModelInfo) -> Result<Self, Self::Error> {
        let mut entry = Self::new(ModelId::parse(&info.id)?);
        entry.display_name = info.display_name;
        entry.owned_by = info.owned_by;
        if let Some(window) = info.context_window {
            entry.capabilities.context_window = Sourced::new(Some(window), CapSource::ProviderApi);
        }
        entry.input_per_1m = info.input_per_1m;
        entry.output_per_1m = info.output_per_1m;
        Ok(entry)
    }
}

/// How fresh the models in a [`ModelList`] are.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Freshness {
    /// Fetched by this call.
    Fresh,
    /// Served from the cache, within its time to live.
    Cached,
    /// The provider could not be reached and an older list is served instead.
    /// The failure is the typed warning the UI shows beside the list.
    Stale {
        /// Why the refresh failed.
        failure: ProviderFailure,
    },
}

/// A provider's models and how they were obtained.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ModelList {
    /// The models, in the order the provider listed them. Shared with the
    /// cache, so a cache hit is a pointer copy however long the list is.
    pub models: Arc<Vec<ModelEntry>>,
    /// How fresh the list is.
    pub freshness: Freshness,
    /// The listing had more pages than one read follows; the list is a prefix.
    pub truncated: bool,
}

impl ModelList {
    /// The list sorted by model id, for a picker. The provider's own order is
    /// whatever it felt like; setup's "leading model" wants that order, a
    /// picker wants this one.
    #[must_use]
    pub fn sorted(mut self) -> Self {
        Arc::make_mut(&mut self.models).sort_by(|a, b| a.id.cmp(&b.id));
        self
    }

    /// The ids only.
    pub fn ids(&self) -> Vec<&str> {
        self.models.iter().map(|m| m.id.as_str()).collect()
    }

    /// Whether the list came from the cache past its time to live.
    pub fn is_stale(&self) -> bool {
        matches!(self.freshness, Freshness::Stale { .. })
    }
}
