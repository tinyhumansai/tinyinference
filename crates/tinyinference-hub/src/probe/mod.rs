//! Probing a provider at three depths (D12).
//!
//! * [`TestDepth::KeyOnly`](crate::TestDepth): a cheap key validation call
//!   (OpenRouter's `GET /key`). Proves the key without reading a catalog.
//! * `Catalog`: read the model listing. The cheapest check every kind has, but
//!   **not proof of a key** for the kinds whose listing is public (Hugging Face,
//!   Venice): [`ProbeReport::proves_key`] says so.
//! * `Completion`: a real one-token ping. Proves the key *and* the model.
//!
//! A kind declares which depths it supports; any other depth is a typed
//! [`Unsupported`](crate::HubError::Unsupported). A provider *failing* a check is
//! not an error of the probe but a fact in the [`ProbeReport`], because the add
//! flow decides from it whether to roll a key back and the health tracker
//! records it either way. Only things that stop a probe from running at all (an
//! unsupported depth, a missing key, a missing model, signed out) are `Err`.

mod run;
mod types;

pub use run::run_probe;
pub use types::{ProbeNote, ProbeReport};

#[cfg(test)]
#[path = "test.rs"]
mod tests;
