//! Descriptors (static templates for a kind) and records (configured
//! instances).
//!
//! A [`ProviderDescriptor`] is one catalogue row: where a kind lives, how it
//! authenticates, which test depths it supports, and its typed quirks. A
//! [`ProviderRecord`] is a configured instance and never holds a credential
//! (invariant 1).

mod capabilities;
mod record;
mod types;

pub use capabilities::{CapSource, Capabilities, Sourced, Tri};
pub use record::{ExtractedCredential, LegacyFields, ProviderRecord, RecordOrigin};
pub use types::{ProviderDescriptor, Quirk};

#[cfg(test)]
#[path = "test.rs"]
mod tests;
