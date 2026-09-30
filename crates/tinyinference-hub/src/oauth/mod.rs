//! OAuth-backed providers (feature `oauth`): **types only**.
//!
//! No flow is enabled and none can be added without the operator recording a
//! policy check per vendor (D12, open question Q12). Claude subscription OAuth
//! is excluded permanently: the hub never mints or replays those tokens; Claude
//! is a CLI login (see the `cli` feature). This module defines the shapes a
//! future flow implements and makes every entry point answer
//! [`HubError::Unsupported`], so a host can already write its UI against them.

use std::fmt::Debug;

use async_trait::async_trait;

use crate::error::{HubError, Operation};
use crate::hub::Hub;
use crate::ids::{KindId, ScopeKey};
use crate::secret::Secret;

/// The first half of an authorisation flow: where to send the user.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OAuthStart {
    /// The page the user opens.
    pub authorize_url: String,
    /// The opaque state the completion must present.
    pub state: String,
}

/// The second half: what a completed flow yields. The credential is written to
/// the credential store, never to the configuration.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct OAuthGrant {
    /// The credential; redacts itself in `Debug`.
    pub credential: Secret,
}

/// An authorisation flow a host may plug in for a kind. **No implementation is
/// shipped and none is enabled.**
#[async_trait]
pub trait LoginFlow: Send + Sync + Debug {
    /// Starts the flow.
    ///
    /// # Errors
    ///
    /// Whatever the flow reports.
    async fn start(&self, scope: &ScopeKey) -> Result<OAuthStart, HubError>;

    /// Finishes the flow with what the user pasted or the redirect delivered.
    ///
    /// # Errors
    ///
    /// Whatever the flow reports.
    async fn complete(&self, scope: &ScopeKey, input: &str) -> Result<OAuthGrant, HubError>;
}

impl Hub {
    /// Starts an OAuth login. Always [`HubError::Unsupported`] in this build.
    ///
    /// # Errors
    ///
    /// [`HubError::Unsupported`], always.
    pub async fn oauth_start(
        &self,
        _scope: &ScopeKey,
        kind: &KindId,
    ) -> Result<OAuthStart, HubError> {
        Err(HubError::Unsupported {
            op: Operation::OAuth,
            kind: kind.clone(),
        })
    }

    /// Finishes an OAuth login. Always [`HubError::Unsupported`] in this build.
    ///
    /// # Errors
    ///
    /// [`HubError::Unsupported`], always.
    pub async fn oauth_complete(
        &self,
        _scope: &ScopeKey,
        kind: &KindId,
        _input: &str,
    ) -> Result<(), HubError> {
        Err(HubError::Unsupported {
            op: Operation::OAuth,
            kind: kind.clone(),
        })
    }
}

#[cfg(test)]
#[path = "test.rs"]
mod tests;
