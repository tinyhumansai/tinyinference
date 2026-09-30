//! [`ModelMetadataSource`]: an optional registry of model facts (D10).
//!
//! Phase 1 ships the trait and the merge, not a bundled registry: any id a
//! provider lists is valid, and what a registry knows about it is an addition
//! tagged with its source, never a filter.

use std::fmt::Debug;

use crate::descriptor::Capabilities;
use crate::ids::{KindId, ModelId};

use super::types::Lifecycle;

/// What a registry knows about one model.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelMeta {
    /// A human name.
    pub display_name: Option<String>,
    /// Capabilities. The source tags a registry supplies should be
    /// [`CapSource::Registry`](crate::descriptor::CapSource::Registry); the
    /// merge overwrites any other tag so a registry cannot claim to be the
    /// provider.
    pub capabilities: Capabilities,
    /// Deprecation state.
    pub lifecycle: Option<Lifecycle>,
    /// The model an alias currently points at.
    pub alias_of: Option<ModelId>,
    /// Charged input price in USD per million tokens.
    pub input_per_1m: Option<f64>,
    /// Charged output price in USD per million tokens.
    pub output_per_1m: Option<f64>,
}

/// A source of model facts the provider's listing does not carry.
pub trait ModelMetadataSource: Send + Sync + Debug {
    /// What is known about `model` on a `kind`, if anything.
    fn lookup(&self, kind: &KindId, model: &str) -> Option<ModelMeta>;
}
