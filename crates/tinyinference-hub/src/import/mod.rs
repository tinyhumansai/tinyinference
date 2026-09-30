//! Pure readers that turn OpenCompany's and OpenHuman's stored shapes into a
//! [`HubConfig`] and a [`LossReport`].
//!
//! Nothing here touches a store or a network: a host reads its own records into
//! the plain input structs (which also `Deserialize` from the stored JSON) and
//! calls [`oc::import`] or [`oh::import`]. The report says, entry by entry, what
//! could not be carried over faithfully; a reader never guesses silently.

pub mod oc;
pub mod oh;
mod report;

pub use oh::StoredKey;
pub use report::{LossEntry, LossKind, LossReport};

use crate::catalog::ModelOverride;
use crate::config::HubConfig;
use crate::health::ProviderHealth;
use crate::ids::Slug;
use crate::secret::Secret;

/// A credential an input carried that the host should write to its credential
/// store; the configuration itself never holds it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ImportedCredential {
    /// The provider it belongs to.
    pub slug: Slug,
    /// The key; redacts itself in `Debug`.
    pub key: Secret,
}

/// What an import produced.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Imported {
    /// The configuration.
    pub config: HubConfig,
    /// What was lost, normalised, synthesised or refused.
    pub loss: LossReport,
    /// Keys to move into the credential store (never on the configuration).
    pub credentials: Vec<ImportedCredential>,
    /// Model facts the source carried, as operator overrides.
    pub overrides: Vec<ModelOverride>,
    /// Health states the source remembered (states only; timestamps are
    /// dropped and the report says so).
    pub health: std::collections::BTreeMap<Slug, ProviderHealth>,
}

impl Imported {
    pub(crate) fn new() -> Self {
        Self {
            config: HubConfig::new(),
            loss: LossReport::new(),
            credentials: Vec::new(),
            overrides: Vec::new(),
            health: std::collections::BTreeMap::new(),
        }
    }
}

#[cfg(test)]
#[path = "test.rs"]
mod tests;
