//! [`Detector`]: first-run discovery of what is already on this machine.

use std::fmt::Debug;

use async_trait::async_trait;

use crate::config::ProviderDraft;

/// What a detection pass may look at.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DetectOptions {
    /// Ports to skip because the host itself listens there (OpenCompany
    /// defaults to 8080, which is also llama.cpp's port).
    pub exclude_ports: Vec<u16>,
}

/// Finds providers without being told about them. Never persists anything: a
/// detection result is a list of drafts the operator confirms.
#[async_trait]
pub trait Detector: Send + Sync + Debug {
    /// The drafts found, in a stable order.
    async fn detect(&self, options: &DetectOptions) -> Vec<ProviderDraft>;
}
