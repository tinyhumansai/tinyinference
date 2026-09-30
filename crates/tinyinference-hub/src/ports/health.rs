//! [`HealthStore`]: where per-provider health survives a restart.

use std::fmt::Debug;

use async_trait::async_trait;

use crate::health::HealthSnapshot;
use crate::ids::{ScopeKey, Slug};

use super::PortError;

/// Persistence for [`HealthSnapshot`]s. The default is in memory
/// ([`memory::MemoryHealth`](super::memory::MemoryHealth)); a host that wants
/// health to survive a restart (OpenCompany keeps it under `inference/health`)
/// supplies its own.
#[async_trait]
pub trait HealthStore: Send + Sync + Debug {
    /// The snapshot for a provider, if one was recorded.
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the store cannot be read.
    async fn get(&self, scope: &ScopeKey, slug: &Slug)
    -> Result<Option<HealthSnapshot>, PortError>;

    /// Records a snapshot.
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the write did not happen.
    async fn put(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        snapshot: HealthSnapshot,
    ) -> Result<(), PortError>;

    /// Forgets a provider's health (it was removed or re-keyed).
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the delete did not happen.
    async fn forget(&self, scope: &ScopeKey, slug: &Slug) -> Result<(), PortError>;
}
