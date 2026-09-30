//! Model catalogs: parsing what a provider lists, caching it safely, and
//! merging what other sources know about the models in it.
//!
//! * `types`: [`ModelEntry`], [`ModelList`] and the small enums around them.
//! * `parse`: the tolerant OpenAI-shaped parser, Ollama's `/api/tags` and LM
//!   Studio's `/api/v0/models`.
//! * `paged`: the TinyHumans paged envelope, ported whole from OpenCompany.
//! * `cache`: the per-endpoint cache with the invariants OpenCompany learned
//!   the hard way (never keyed on a credential; partitioned by scope when a
//!   credential was sent; a `401`/`403` is never remembered).
//! * `merge` and `registry`: filling gaps from a [`ModelMetadataSource`] and
//!   operator overrides, with each fact tagged by where it came from.
//!
//! Every parser here is **tolerant per entry**: one malformed row costs that
//! row, never the whole list, and no model id is filtered, preferred or
//! rejected by vendor or name; any id an endpoint returns is valid.

mod cache;
mod merge;
mod paged;
mod parse;
mod registry;
mod types;

pub use cache::{
    CATALOG_TTL, CatalogCache, CatalogKey, EMPTY_CATALOG_TTL, FAILURE_TTL, Fetched, MAX_SLOTS,
    STALE_RETENTION,
};
pub use merge::{ModelOverride, merge_metadata};
pub use paged::{Collector, MAX_PAGES, NextPage, PAGE_LIMIT, Page, page_path, parse_page};
pub use parse::{
    ParsedCatalog, parse_lmstudio_v0, parse_ollama_tags, parse_openai, parse_openai_value,
};
pub(crate) use parse::{too_large, unreadable};
pub use registry::{ModelMeta, ModelMetadataSource};
pub use types::{EntrySource, Freshness, Lifecycle, LifecycleStatus, ModelEntry, ModelList};

#[cfg(test)]
#[path = "cache_identity_test.rs"]
mod cache_identity_tests;
#[cfg(test)]
#[path = "cache_test.rs"]
mod cache_tests;
#[cfg(test)]
#[path = "paged_test.rs"]
mod paged_tests;
#[cfg(test)]
#[path = "test.rs"]
mod tests;
