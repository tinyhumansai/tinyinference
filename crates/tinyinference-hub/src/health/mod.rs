//! Provider health: what probes and real turns have said about a provider
//! lately, folded into one status a UI can show.
//!
//! * `types`: [`ProviderHealth`], [`HealthSnapshot`] and the signals inside it.
//! * `fold`: the latching rules that turn signals into a status.
//! * `tracker`: [`HealthTracker`], which feeds the snapshot from probes and from
//!   real turns ([`Outcome`]) and tells the host when the status changes.
//!
//! Health is fed by probes **and** by real turns, so a provider that passes a
//! catalog read but fails every completion does not look green. The router
//! crate (a later phase) consumes these signals; this crate only exposes them.

mod fold;
mod tracker;
mod types;

pub use fold::FAILURES_TO_DOWN;
pub use tracker::{HealthTracker, Outcome};
pub use types::{FailureNote, HealthSnapshot, ProbeSignal, ProviderHealth, TurnSignal};

#[cfg(test)]
#[path = "test.rs"]
mod tests;
