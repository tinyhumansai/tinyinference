//! The data types of provider health.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::ReasonCode;
use crate::taxonomy::TestDepth;

/// One provider's status, as a UI shows it.
///
/// `SignedOut` is a state of its own, not a failure: "sign in" and "no models"
/// and a red error are three different screens.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "state", content = "reason", rename_all = "snake_case")]
pub enum ProviderHealth {
    /// Nothing has been checked yet, or the record was just re-keyed.
    #[default]
    Unknown,
    /// Working.
    Ok,
    /// Partly working: some checks or turns fail, others succeed, or a
    /// transient failure has not yet repeated.
    Degraded(ReasonCode),
    /// Not working.
    Down(ReasonCode),
    /// The managed provider has no credential source that answers.
    SignedOut,
    /// The operator turned the provider off. Derived by the hub, never stored.
    Disabled,
}

impl ProviderHealth {
    /// Whether the provider can be expected to serve a turn.
    pub fn is_usable(self) -> bool {
        matches!(self, Self::Ok | Self::Degraded(_) | Self::Unknown)
    }
}

/// A failure, kept without its raw text.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureNote {
    /// What kind of failure.
    pub reason: ReasonCode,
    /// The HTTP status, when there was one.
    pub status: Option<u16>,
    /// When it happened, in milliseconds since the Unix epoch.
    pub at_ms: u64,
}

/// The latest result of one probe depth.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProbeSignal {
    /// Whether the depth passed.
    pub ok: bool,
    /// A later success in a lane that proves this failure is over cleared it
    /// (see the fold rules). Sticky: overwriting the *other* lane later does not
    /// bring the failure back.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub superseded: bool,
    /// Why it failed.
    pub reason: Option<ReasonCode>,
    /// When it was recorded (it finished), in milliseconds since the Unix epoch.
    pub at_ms: u64,
    /// When the probe **started**, when known. A pass supersedes only failures
    /// recorded before this instant: whatever failed while the probe was in
    /// flight is newer than the probe's evidence. Absent on snapshots written
    /// before it was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_ms: Option<u64>,
    /// How long it took.
    pub latency_ms: Option<u64>,
}

/// The latest result of a real turn.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnSignal {
    /// How long the turn took, when the host reported it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
    /// Whether the turn succeeded.
    pub ok: bool,
    /// A later success in a lane that proves this failure is over cleared it.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub superseded: bool,
    /// Why it failed.
    pub reason: Option<ReasonCode>,
    /// When, in milliseconds since the Unix epoch.
    pub at_ms: u64,
}

/// Everything remembered about one provider's health.
///
/// Every field has a default, so a snapshot written by an older or newer build
/// (a host persists these) still loads.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct HealthSnapshot {
    /// The folded status.
    pub health: ProviderHealth,
    /// When the status last changed, in milliseconds since the Unix epoch.
    pub changed_at_ms: u64,
    /// The last success of any kind.
    pub last_ok_ms: Option<u64>,
    /// The last failure of any kind.
    pub last_failure: Option<FailureNote>,
    /// Failed real turns in a row since the last success.
    pub consecutive_failures: u32,
    /// The latest result per probe depth.
    pub probes: BTreeMap<TestDepth, ProbeSignal>,
    /// The latest real turn.
    pub turn: Option<TurnSignal>,
}
