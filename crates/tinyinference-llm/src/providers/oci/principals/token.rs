//! Shared session-token plumbing: JWT expiry parsing, a thread-safe renewing
//! cache, and a blocking bridge for the async HTTP client.

use std::future::Future;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine as _;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use serde_json::Value;

use crate::providers::oci::auth::OciSessionMaterial;
use crate::{Error, Result};

/// Tokens are renewed this long before they expire (oci-go-sdk `jwt.go`,
/// `bufferTimeBeforeTokenExpiration`).
pub(super) const REFRESH_BUFFER_SECS: i64 = 300;

/// The `iat`/`exp` claims of a session token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TokenTimes {
    pub(super) iat: Option<i64>,
    pub(super) exp: i64,
}

impl TokenTimes {
    /// Whether the token may still be used at `now` (unix seconds). With
    /// `soft`, the token is also retired at the midpoint of its lifetime
    /// (`rpstValidForRatio` 0.5 in the SDK's OKE flow).
    pub(super) fn valid_at(&self, now: i64, soft: bool) -> bool {
        let horizon = now + REFRESH_BUFFER_SECS;
        if self.exp <= horizon {
            return false;
        }
        if soft && let Some(iat) = self.iat {
            let soft_exp = iat + (self.exp - iat) / 2;
            return soft_exp > horizon;
        }
        true
    }
}

/// Reads the (unverified) claims needed for caching from a JWT.
pub(super) fn parse_token_times(token: &str) -> Result<TokenTimes> {
    let mut parts = token.split('.');
    let (Some(_), Some(payload), Some(_), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return Err(Error::Model(
            "OCI session token is not a three-part JWT".into(),
        ));
    };
    let bytes = URL_SAFE_NO_PAD
        .decode(payload.trim_end_matches('='))
        .map_err(|_| Error::Model("OCI session token payload is not base64url".into()))?;
    let claims: Value = serde_json::from_slice(&bytes)
        .map_err(|_| Error::Model("OCI session token payload is not JSON".into()))?;
    let exp = claims["exp"]
        .as_f64()
        .ok_or_else(|| Error::Model("OCI session token has no exp claim".into()))?;
    Ok(TokenTimes {
        iat: claims["iat"].as_f64().map(|v| v as i64),
        exp: exp as i64,
    })
}

pub(super) fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

struct Cached {
    material: OciSessionMaterial,
    times: TokenTimes,
}

/// A mutex-guarded token that renews itself when it nears expiry. The lock is
/// held during renewal so concurrent callers share one exchange.
#[derive(Default)]
pub(super) struct TokenCache {
    inner: Mutex<Option<Cached>>,
}

impl std::fmt::Debug for TokenCache {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TokenCache([redacted])")
    }
}

impl TokenCache {
    /// Returns the cached material, renewing through `renew` when absent or
    /// no longer valid at `now`.
    pub(super) fn get_or_renew(
        &self,
        now: i64,
        soft: bool,
        renew: impl FnOnce() -> Result<OciSessionMaterial>,
    ) -> Result<OciSessionMaterial> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| Error::Model("OCI token cache lock poisoned".into()))?;
        if let Some(cached) = guard.as_ref()
            && cached.times.valid_at(now, soft)
        {
            return Ok(cached.material.clone());
        }
        tracing::debug!(target: "tinyinference::oci", "renewing oci principal session token");
        let material = renew()?;
        let times = parse_token_times(&material.token)?;
        *guard = Some(Cached {
            material: material.clone(),
            times,
        });
        Ok(material)
    }
}

/// Runs an async HTTP exchange to completion from synchronous code.
///
/// The future runs on a scoped helper thread with its own current-thread
/// runtime, so this is safe to call from inside a Tokio worker (it blocks the
/// caller only during a renewal).
pub(super) fn block_on<F, T>(future: F) -> Result<T>
where
    F: Future<Output = Result<T>> + Send,
    T: Send,
{
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| Error::Model(format!("could not start runtime: {error}")))?
                    .block_on(future)
            })
            .join()
            .map_err(|_| Error::Model("OCI principal exchange thread panicked".into()))?
    })
}

#[cfg(test)]
#[path = "token_tests.rs"]
mod tests;
