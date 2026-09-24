//! Authenticated HTTP transport for OpenRouter's media APIs.
//!
//! One transport serves two deployments that speak the same wire format:
//!
//! - **Direct** — `https://openrouter.ai/api/v1` with the caller's OpenRouter
//!   API key ([`MediaAuth::ApiKey`]).
//! - **Proxied** — a host backend that forwards OpenRouter's request bodies
//!   verbatim and returns OpenRouter's response body, optionally wrapped in a
//!   `{"success": true, "data": …}` envelope that is unwrapped transparently
//!   ([`unwrap_envelope`]) (for example TinyHumans'
//!   `/agent-integrations/openrouter`), authenticated with a host-owned bearer
//!   ([`MediaAuth::Bearer`]) so the credential lifecycle stays in the host.
//!
//! Retry policy is billing-aware. `GET` calls (model listings, job polls,
//! content downloads) retry on 429/5xx/transport failures. A `POST` that may
//! have started a paid generation is retried **only** on HTTP 429, where the
//! provider rejected the request before doing any work; a 5xx or a dropped
//! connection after a submit is surfaced as-is, because resubmitting could
//! bill a second generation.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::Serialize;
use serde::de::DeserializeOwned;
use tinyinference_core::retry_after::{MAX_RETRIES, backoff_ms_for_attempt};
use tinyinference_core::sanitize::sanitize_api_error;

use crate::{Error, Result};

/// OpenRouter's public API base URL.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// Environment variable read by [`MediaAuth::from_env`].
pub const OPENROUTER_API_KEY_ENV: &str = "OPENROUTER_API_KEY";
/// Default cap on a downloaded or decoded media body (512 MiB).
pub const DEFAULT_MAX_MEDIA_BYTES: usize = 512 * 1024 * 1024;
/// Cap on an error body read into a message.
const MAX_ERROR_BODY_BYTES: usize = 16 * 1024;

/// Resolves the current bearer token for each request.
pub type BearerResolver = Arc<dyn Fn() -> Result<String> + Send + Sync>;

/// How requests authenticate.
#[derive(Clone)]
pub enum MediaAuth {
    /// A static API key sent as `Authorization: Bearer <key>`.
    ApiKey(String),
    /// A host-owned resolver called before every request.
    Bearer(BearerResolver),
}

impl MediaAuth {
    /// Reads an OpenRouter API key from [`OPENROUTER_API_KEY_ENV`].
    ///
    /// # Errors
    ///
    /// [`Error::Auth`] when the variable is unset or blank.
    pub fn from_env() -> Result<Self> {
        match std::env::var(OPENROUTER_API_KEY_ENV) {
            Ok(key) if !key.trim().is_empty() => Ok(Self::ApiKey(key.trim().to_owned())),
            _ => Err(Error::Auth(format!("{OPENROUTER_API_KEY_ENV} is not set"))),
        }
    }

    fn token(&self) -> Result<String> {
        let token = match self {
            Self::ApiKey(key) => key.clone(),
            Self::Bearer(resolve) => resolve()?,
        };
        let token = token.trim().to_owned();
        if token.is_empty() {
            return Err(Error::Auth(
                "no credential available for media generation".into(),
            ));
        }
        Ok(token)
    }
}

impl std::fmt::Debug for MediaAuth {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ApiKey(_) => formatter.write_str("MediaAuth::ApiKey(<redacted>)"),
            Self::Bearer(_) => formatter.write_str("MediaAuth::Bearer(<resolver>)"),
        }
    }
}

/// Whether a request may start billable work, which decides its retry policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Billing {
    /// Safe to repeat: listings, polls, downloads.
    Idempotent,
    /// May start a paid generation: only a 429 is retried.
    Billable,
}

/// HTTP client bound to one OpenRouter-compatible base URL and credential.
#[derive(Clone)]
pub struct MediaTransport {
    client: reqwest::Client,
    base_url: String,
    auth: MediaAuth,
    headers: HeaderMap,
    max_retries: u32,
    max_media_bytes: usize,
}

impl MediaTransport {
    /// Creates a transport for OpenRouter's public API.
    #[must_use]
    pub fn new(auth: MediaAuth) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: OPENROUTER_BASE_URL.to_owned(),
            auth,
            headers: HeaderMap::new(),
            max_retries: MAX_RETRIES,
            max_media_bytes: DEFAULT_MAX_MEDIA_BYTES,
        }
    }

    /// Points the transport at another OpenRouter-compatible base URL, such as
    /// a host backend that proxies OpenRouter's media routes.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = base_url.as_ref().trim().trim_end_matches('/').to_owned();
        self
    }

    /// Uses a caller-supplied HTTP client (timeouts, proxies, TLS roots).
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Adds a header sent on every request (for example `HTTP-Referer`,
    /// `X-Title`, or a host's client-identification header).
    ///
    /// Invalid header names or values are ignored with a warning rather than
    /// failing construction.
    #[must_use]
    pub fn with_header(mut self, name: &str, value: &str) -> Self {
        match (
            HeaderName::from_bytes(name.as_bytes()),
            HeaderValue::from_str(value),
        ) {
            (Ok(name), Ok(value)) => {
                self.headers.insert(name, value);
            }
            _ => tracing::warn!(
                header = name,
                "[tinyinference-image] ignoring invalid header"
            ),
        }
        self
    }

    /// Sets how many times a retryable failure is retried (default 3).
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Caps the size of any single media body (default 512 MiB).
    #[must_use]
    pub fn with_max_media_bytes(mut self, max_media_bytes: usize) -> Self {
        self.max_media_bytes = max_media_bytes;
        self
    }

    /// The configured base URL, without a trailing slash.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The configured media size cap in bytes.
    #[must_use]
    pub fn max_media_bytes(&self) -> usize {
        self.max_media_bytes
    }

    /// Sends a JSON `POST` that may start a paid generation.
    ///
    /// # Errors
    ///
    /// [`Error::Auth`], [`Error::Http`], [`Error::Transport`] or
    /// [`Error::Decode`]. Only HTTP 429 is retried.
    pub async fn post_json<B, T>(&self, path: &str, body: &B) -> Result<T>
    where
        B: Serialize + ?Sized + Sync,
        T: DeserializeOwned,
    {
        let body = serde_json::to_vec(body)?;
        let response = self
            .send(reqwest::Method::POST, path, Some(body), Billing::Billable)
            .await?;
        decode_json(response).await
    }

    /// Sends an idempotent JSON `GET`.
    ///
    /// # Errors
    ///
    /// [`Error::Auth`], [`Error::Http`], [`Error::Transport`] or
    /// [`Error::Decode`] after retries are exhausted.
    pub async fn get_json<T: DeserializeOwned>(&self, path: &str) -> Result<T> {
        let response = self
            .send(reqwest::Method::GET, path, None, Billing::Idempotent)
            .await?;
        decode_json(response).await
    }

    /// Downloads a binary body with an idempotent `GET`, enforcing the media
    /// size cap. Returns the bytes and the response `Content-Type`, if any.
    ///
    /// # Errors
    ///
    /// [`Error::TooLarge`] when the body exceeds the cap, plus the errors of
    /// [`MediaTransport::get_json`].
    pub async fn get_bytes(&self, path: &str) -> Result<(Bytes, Option<String>)> {
        let token = self.auth.token()?;
        let mut response = self
            .send(reqwest::Method::GET, path, None, Billing::Idempotent)
            .await?;
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        if response
            .content_length()
            .is_some_and(|length| length > self.max_media_bytes as u64)
        {
            return Err(Error::TooLarge {
                limit: self.max_media_bytes,
            });
        }
        let mut body = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| Error::Transport(self.scrub(&error.to_string(), Some(&token))))?
        {
            if body.len() + chunk.len() > self.max_media_bytes {
                return Err(Error::TooLarge {
                    limit: self.max_media_bytes,
                });
            }
            body.extend_from_slice(&chunk);
        }
        Ok((Bytes::from(body), content_type))
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    fn scrub(&self, text: &str, token: Option<&str>) -> String {
        let mut text = text.to_owned();
        if let Some(token) = token
            && !token.is_empty()
        {
            text = text.replace(token, "[REDACTED]");
        }
        sanitize_api_error(&text)
    }

    async fn send(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<Vec<u8>>,
        billing: Billing,
    ) -> Result<reqwest::Response> {
        let url = self.url(path);
        let mut attempt = 0u32;
        loop {
            let token = self.auth.token()?;
            let mut request = self
                .client
                .request(method.clone(), &url)
                .headers(self.headers.clone())
                .header(AUTHORIZATION, format!("Bearer {token}"));
            if let Some(body) = &body {
                request = request
                    .header(CONTENT_TYPE, "application/json")
                    .body(body.clone());
            }
            tracing::debug!(
                method = %method,
                path,
                attempt,
                "[tinyinference-image] media request"
            );
            let outcome = request.send().await;
            let (retry, retry_after, error) = match outcome {
                Ok(response) if response.status().is_success() => return Ok(response),
                Ok(response) => {
                    let status = response.status().as_u16();
                    let retry_after = response
                        .headers()
                        .get(reqwest::header::RETRY_AFTER)
                        .and_then(|value| value.to_str().ok())
                        .map(str::to_owned);
                    let message = self.error_message(response, &token).await;
                    let error = match status {
                        401 | 403 => Error::Auth(format!("HTTP {status}: {message}")),
                        _ => Error::Http { status, message },
                    };
                    let retry = match billing {
                        Billing::Billable => status == 429,
                        Billing::Idempotent => status == 429 || status >= 500,
                    };
                    (retry, retry_after, error)
                }
                Err(error) => {
                    let error = Error::Transport(self.scrub(&error.to_string(), Some(&token)));
                    (billing == Billing::Idempotent, None, error)
                }
            };
            if !retry || attempt >= self.max_retries {
                tracing::warn!(
                    method = %method,
                    path,
                    attempt,
                    error = %error,
                    "[tinyinference-image] media request failed"
                );
                return Err(error);
            }
            let delay = backoff_ms_for_attempt(attempt, retry_after.as_deref());
            tracing::debug!(
                method = %method,
                path,
                attempt,
                delay_ms = delay,
                "[tinyinference-image] retrying media request"
            );
            tokio::time::sleep(Duration::from_millis(delay)).await;
            attempt += 1;
        }
    }

    async fn error_message(&self, mut response: reqwest::Response, token: &str) -> String {
        let mut body = Vec::new();
        while let Ok(Some(chunk)) = response.chunk().await {
            let room = MAX_ERROR_BODY_BYTES.saturating_sub(body.len());
            body.extend_from_slice(&chunk[..chunk.len().min(room)]);
            if body.len() >= MAX_ERROR_BODY_BYTES {
                break;
            }
        }
        let text = String::from_utf8_lossy(&body).into_owned();
        let message = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|value| {
                value
                    .pointer("/error/message")
                    .or_else(|| value.get("message"))
                    .or_else(|| value.get("error"))
                    .and_then(|message| message.as_str().map(str::to_owned))
            })
            .unwrap_or(text);
        self.scrub(&message, Some(token))
    }
}

impl std::fmt::Debug for MediaTransport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MediaTransport")
            .field("base_url", &self.base_url)
            .field("auth", &self.auth)
            .field(
                "headers",
                &self
                    .headers
                    .keys()
                    .map(HeaderName::as_str)
                    .collect::<Vec<_>>(),
            )
            .field("max_retries", &self.max_retries)
            .field("max_media_bytes", &self.max_media_bytes)
            .finish()
    }
}

async fn decode_json<T: DeserializeOwned>(response: reqwest::Response) -> Result<T> {
    decode_json_with_limit(response, DEFAULT_MAX_MEDIA_BYTES).await
}

async fn decode_json_with_limit<T: DeserializeOwned>(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<T> {
    let mut body = Vec::new();
    loop {
        match response.chunk().await {
            Ok(Some(chunk)) => {
                let room = limit.saturating_sub(body.len());
                if room == 0 {
                    return Err(Error::TooLarge { limit });
                }
                body.extend_from_slice(&chunk[..chunk.len().min(room)]);
            }
            Ok(None) => break,
            Err(error) => {
                return Err(Error::Transport(format!(
                    "response stream error: {error}"
                )))
            }
        }
    }
    let value: serde_json::Value = serde_json::from_slice(&body).map_err(|error| {
        Error::Decode(format!(
            "unexpected response body ({} bytes): {error}",
            body.len()
        ))
    })?;
    serde_json::from_value(unwrap_envelope(value)?)
        .map_err(|error| Error::Decode(format!("unexpected response shape: {error}")))
}

/// Unwraps a proxying backend's `{"success": bool, "data": …}` envelope.
///
/// OpenRouter's own media responses never carry a top-level boolean
/// `success`, so its presence identifies the envelope unambiguously; any other
/// body passes through untouched.
///
/// # Errors
///
/// [`Error::Http`] for a `success: false` envelope delivered with a 2xx
/// status, carrying the envelope's sanitized error message.
pub fn unwrap_envelope(value: serde_json::Value) -> Result<serde_json::Value> {
    let serde_json::Value::Object(mut map) = value else {
        return Ok(value);
    };
    match map.get("success").and_then(serde_json::Value::as_bool) {
        Some(true) => Ok(map.remove("data").unwrap_or(serde_json::Value::Null)),
        Some(false) => {
            let message = map
                .get("error")
                .and_then(|error| {
                    error.as_str().map(str::to_owned).or_else(|| {
                        error
                            .pointer("/message")
                            .and_then(|m| m.as_str())
                            .map(str::to_owned)
                    })
                })
                .or_else(|| {
                    map.get("message")
                        .and_then(|m| m.as_str())
                        .map(str::to_owned)
                })
                .unwrap_or_else(|| "request failed".to_owned());
            Err(Error::Http {
                status: 200,
                message: sanitize_api_error(&message),
            })
        }
        None => Ok(serde_json::Value::Object(map)),
    }
}

/// Strips an `openrouter/` routing prefix from a model id.
///
/// Hosts often qualify OpenRouter slugs (`openrouter/bytedance/seedance-2.0-mini`)
/// to say which provider serves them; the wire format wants the bare slug.
#[must_use]
pub fn wire_model_id(model: &str) -> &str {
    let model = model.trim();
    model.strip_prefix("openrouter/").unwrap_or(model)
}
