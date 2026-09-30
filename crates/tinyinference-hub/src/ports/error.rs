//! [`PortError`]: what a host store reports.

use crate::error::{HubError, PortName};
use crate::secret::LogOnly;

/// A host port failed.
///
/// Deliberately tiny: a host's own error text is [`LogOnly`], because a
/// database driver's message can echo a connection string.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PortError {
    /// A compare-and-swap lost: the stored version is not the one expected.
    #[error("the stored value changed concurrently")]
    Conflict,
    /// The store could not be read or written. Never means "absent".
    #[error("the store is unavailable")]
    Unavailable(LogOnly<String>),
}

impl PortError {
    /// An outage, with the store's own message kept log-only.
    pub fn unavailable(detail: impl Into<String>) -> Self {
        Self::Unavailable(LogOnly::new(detail.into()))
    }

    /// The hub error this becomes when `port` failed: a lost compare-and-swap
    /// is [`HubError::Conflict`], everything else is
    /// [`HubError::StoreUnreadable`] naming the port.
    pub fn into_hub(self, port: PortName) -> HubError {
        match self {
            Self::Conflict => HubError::Conflict,
            Self::Unavailable(detail) => HubError::StoreUnreadable { port, detail },
        }
    }
}
