//! [`ConfigStore`]: where the hub's configuration lives, with compare-and-swap.

use std::fmt::Debug;

use async_trait::async_trait;

use crate::config::HubConfig;
use crate::ids::ScopeKey;

use super::PortError;

/// An opaque, host-chosen version of a stored configuration.
///
/// A counter, a row revision or a content hash: the hub only compares versions
/// for equality and passes back the one it was given.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Version(u64);

impl Version {
    /// Wraps a host version.
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    /// The host's number.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// A host's configuration store.
///
/// Every mutation is load, change, `save(expect = the version loaded)`. Two
/// writers in one process are serialised by a mutex; two processes (or a
/// desktop app and a CLI) are not, so the store itself must refuse the loser.
#[async_trait]
pub trait ConfigStore: Send + Sync + Debug {
    /// The scope's configuration and its version, or `None` when nothing has
    /// been saved yet.
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the store cannot be read. A corrupt
    /// document is also an error: an empty configuration would be silently
    /// overwritten by the next save.
    async fn load(&self, scope: &ScopeKey) -> Result<Option<(HubConfig, Version)>, PortError>;

    /// Saves `config` if the stored version is `expect` (`None` meaning nothing
    /// is stored yet) and returns the new version.
    ///
    /// # Errors
    ///
    /// [`PortError::Conflict`] when the stored version is not `expect`;
    /// [`PortError::Unavailable`] when the store cannot be written.
    async fn save(
        &self,
        scope: &ScopeKey,
        config: &HubConfig,
        expect: Option<Version>,
    ) -> Result<Version, PortError>;
}
