//! [`UsageQuery`]: references to a provider that only the host knows about.

use std::fmt::Debug;

use async_trait::async_trait;

use crate::error::UsedBy;
use crate::ids::{ScopeKey, Slug};

use super::PortError;

/// Tells the in-use guard about references the hub does not store.
///
/// The hub knows the default choice, its own agent pins and its workload
/// routes. A host that keeps pins elsewhere (OpenCompany keeps them on the
/// company record) implements this so `remove`, `disable` and `clear_key` still
/// refuse to orphan them without confirmation. Whatever it returns is merged
/// into the hub's own answer.
#[async_trait]
pub trait UsageQuery: Send + Sync + Debug {
    /// What the host has pointing at `slug` in `scope`.
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the host cannot say. The hub then refuses
    /// the guarded operation rather than assuming nothing is in use.
    async fn used_by(&self, scope: &ScopeKey, slug: &Slug) -> Result<UsedBy, PortError>;
}
