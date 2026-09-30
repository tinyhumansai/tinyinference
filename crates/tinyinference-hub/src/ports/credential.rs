//! [`CredentialStore`]: where keys live.

use std::fmt::Debug;

use async_trait::async_trait;

use crate::ids::ScopeKey;
use crate::secret::Secret;

use super::PortError;

/// A host's secret store (OS keychain, an encrypted KV, a vault).
///
/// `slot` is an opaque name; the hub uses [`Slug::key_slot`](crate::Slug::key_slot)
/// (`provider/<slug>/key`, OpenCompany's naming, so adopting the hub needs no
/// key migration).
///
/// **`Err` is not `None`.** `Ok(None)` says "there is no key here". `Err` says
/// "I could not find out". The hub stops a credential chain on `Err` and
/// reports [`HubError::StoreUnreadable`](crate::HubError::StoreUnreadable): a
/// caller that read an outage as "no key" would fall through to the next
/// source, which for a managed provider is the operator's own account.
#[async_trait]
pub trait CredentialStore: Send + Sync + Debug {
    /// The key in `slot`, if there is one.
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the store cannot be read.
    async fn get(&self, scope: &ScopeKey, slot: &str) -> Result<Option<Secret>, PortError>;

    /// Writes `value` to `slot`, replacing what was there.
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the write did not happen.
    async fn set(&self, scope: &ScopeKey, slot: &str, value: Secret) -> Result<(), PortError>;

    /// Removes `slot`. Deleting an absent slot succeeds. This replaces
    /// OpenCompany's "overwrite with an empty string", which left a slot behind.
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the delete did not happen.
    async fn delete(&self, scope: &ScopeKey, slot: &str) -> Result<(), PortError>;
}
