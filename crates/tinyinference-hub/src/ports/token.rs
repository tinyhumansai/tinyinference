//! [`TokenSource`]: a rotating token (the managed platform token, an OAuth
//! access token).

use std::fmt::Debug;

use async_trait::async_trait;

use crate::ids::ScopeKey;
use crate::secret::{Secret, SecretId};

use super::PortError;

/// A source of a token that changes over time.
///
/// The hub asks on **every** request and never caches the answer, so a rotated
/// token is used on the next call. On a `401` a caller invokes
/// [`TokenSource::invalidate`] so the host refreshes instead of replaying a
/// dead token (OpenCompany's `Credential::invalidate`).
#[async_trait]
pub trait TokenSource: Send + Sync + Debug {
    /// The current token, or `None` when the user is signed out.
    ///
    /// # Errors
    ///
    /// [`PortError::Unavailable`] when the source could not be consulted; that
    /// is not "signed out".
    async fn token(&self, scope: &ScopeKey) -> Result<Option<Secret>, PortError>;

    /// The token was rejected; drop whatever the host cached.
    fn invalidate(&self, scope: &ScopeKey) {
        let _ = scope;
    }

    /// The token identified by `rejected` (see [`Secret::id`]) was rejected.
    ///
    /// Unlike [`TokenSource::invalidate`] this says **which** token failed, so a
    /// source that has already rotated can ignore a stale rejection instead of
    /// discarding its fresh token: two requests that both used token A and both
    /// got a `401` must cause one refresh, not two. The default forwards to
    /// [`TokenSource::invalidate`], which is the old behaviour, so an existing
    /// implementation is unchanged; override this to compare `rejected` with
    /// `current.id()` of the token you would hand out now.
    fn invalidate_rejected(&self, scope: &ScopeKey, rejected: SecretId) {
        let _ = rejected;
        self.invalidate(scope);
    }
}
