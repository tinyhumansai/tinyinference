//! Token usage accounting types.

use serde::{Deserialize, Serialize};

/// A provider-reported charge expressed in integer micro-units.
///
/// The inference boundary carries the provider's measured charge but deliberately
/// does not calculate prices. Hosts retain pricing policy and convert their
/// provider DTOs into this fixed-point representation at the adapter boundary.
/// The host selects the unit currency; a call sequence combined into one
/// [`Usage`] must therefore have a common currency.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChargedAmount {
    /// Charge in one-millionth host-selected currency units.
    pub micros: i64,
}

impl ChargedAmount {
    /// Creates an amount from integer micro-units.
    #[must_use]
    pub fn new(micros: i64) -> Self {
        Self { micros }
    }

    /// Creates a USD amount from integer micro-dollars.
    #[must_use]
    pub fn usd_micros(micros: i64) -> Self {
        Self::new(micros)
    }
}

/// Normalized token usage for a single model call.
///
/// Providers expose different breakdowns; fields default to zero so partial
/// data still produces a valid record. Detail fields (cache, reasoning) do not
/// need to sum to the totals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Prompt/input tokens.
    #[serde(default)]
    pub input_tokens: u64,
    /// Completion/output tokens.
    #[serde(default)]
    pub output_tokens: u64,
    /// Total tokens (input + output) as reported by the provider when known.
    #[serde(default)]
    pub total_tokens: u64,
    /// Input tokens served from a provider prompt/KV cache.
    #[serde(default)]
    pub cache_read_tokens: u64,
    /// Input tokens written into a provider prompt/KV cache.
    #[serde(default)]
    pub cache_creation_tokens: u64,
    /// Reasoning/thinking output tokens when the provider exposes them.
    #[serde(default)]
    pub reasoning_tokens: u64,
    /// Provider-reported charge for this call, when the provider exposes one.
    ///
    /// This is measured metadata, not a local price calculation. Hosts own
    /// pricing policy and may leave it absent for local or unmetered providers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub charged_amount: Option<ChargedAmount>,
    /// The model context window used for this call, when the provider reports
    /// it or the host resolved it authoritatively.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window_tokens: Option<u64>,
}

/// Aggregate usage across many calls, tracking both the call count and the
/// summed [`Usage`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct UsageTotals {
    /// Number of accumulated calls.
    #[serde(default)]
    pub calls: u64,
    /// Summed usage across all accumulated calls.
    #[serde(default)]
    pub usage: Usage,
}
