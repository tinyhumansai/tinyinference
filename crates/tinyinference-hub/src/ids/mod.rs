//! Validated identifiers: scopes, slugs, model ids, kind ids, and the opaque
//! host keys for agents and workloads.
//!
//! The types live in `types`; the validators ported from OpenCompany's
//! `store.rs` (slugify, name/slug/model-id checks) live in `validate`.

mod types;
mod validate;

pub use types::{AgentKey, KindId, ModelId, ScopeKey, Slug, WorkloadKey};
pub use validate::{
    MAX_MODEL_ID_CHARS, MAX_PROVIDER_NAME_CHARS, SlugError, check_model_id, check_provider_name,
    check_slug, slugify,
};

#[cfg(test)]
#[path = "test.rs"]
mod tests;
