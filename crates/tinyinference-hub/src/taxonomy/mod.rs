//! The provider taxonomy: groups, transports, protocols, auth styles, catalog
//! shapes, the unified [`LocalRuntime`], and the CLI kinds.
//!
//! Every enum is `#[non_exhaustive]` with snake_case serde spellings; legacy
//! spellings from OpenCompany and OpenHuman are accepted as serde aliases so
//! stored data keeps loading (`docs/spec` compat rule 3).

mod local_runtime;
mod types;

pub use local_runtime::{LocalRuntime, UnsupportedRuntime};
pub use types::{AuthStyle, CatalogShape, CliKind, Protocol, ProviderGroup, TestDepth, Transport};

#[cfg(test)]
#[path = "test.rs"]
mod tests;
