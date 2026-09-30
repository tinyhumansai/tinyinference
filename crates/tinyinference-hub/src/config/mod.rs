//! The hub's persisted configuration and the drafts operations take.
//!
//! Only the data types live here in this milestone: the ports need
//! [`HubConfig`] to describe what a [`ConfigStore`](crate::ports::ConfigStore)
//! saves. The operations that mutate it (add, edit, remove, defaults, pins)
//! arrive with `Hub`.

mod draft;
mod types;

pub use draft::ProviderDraft;
pub use types::{CONFIG_SCHEMA_VERSION, DefaultChoice, HubConfig, ModelChoice};

#[cfg(test)]
#[path = "test.rs"]
mod tests;
