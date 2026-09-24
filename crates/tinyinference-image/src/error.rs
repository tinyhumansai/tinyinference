//! Error type shared by image generation and the OpenRouter media transport.

use thiserror::Error;

/// Result returned by TinyInference image APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// A normalized media-generation failure.
///
/// Variants are split by what a caller can do about them: fix the request
/// ([`Error::Validation`], [`Error::Unsupported`]), fix the credential
/// ([`Error::Auth`]), retry later ([`Error::Http`] with a retryable status,
/// [`Error::Transport`]), or report a billed non-delivery
/// ([`Error::NoMedia`]) without retrying.
#[derive(Debug, Error)]
pub enum Error {
    /// Caller input or configuration was invalid before any request was sent.
    #[error("validation error: {0}")]
    Validation(String),
    /// A request parameter is not supported by the selected model.
    #[error("model '{model}' does not support {field}={value}; supported: {}", allowed.join(", "))]
    Unsupported {
        /// Model id the capability check ran against.
        model: String,
        /// Request field that failed the check (for example `aspect_ratio`).
        field: String,
        /// The rejected value.
        value: String,
        /// Values the model advertises for `field` (empty when unsupported).
        allowed: Vec<String>,
    },
    /// No usable credential was available, or the provider rejected it.
    #[error("authentication error: {0}")]
    Auth(String),
    /// The provider answered with a non-success HTTP status.
    #[error("provider returned HTTP {status}: {message}")]
    Http {
        /// HTTP status code.
        status: u16,
        /// Sanitized provider error message.
        message: String,
    },
    /// The request never produced an HTTP response (DNS, TLS, timeout, reset).
    #[error("transport error: {0}")]
    Transport(String),
    /// A provider payload or media body could not be decoded.
    #[error("decode error: {0}")]
    Decode(String),
    /// The provider accepted (and billed) the request but returned no media.
    ///
    /// Retrying submits and bills a new generation, so callers should report
    /// this rather than retry automatically.
    #[error(
        "generation was accepted and billed but returned no media{}; do not retry automatically — report this to the user",
        request_id.as_deref().map(|id| format!(" (request_id: {id})")).unwrap_or_default()
    )]
    NoMedia {
        /// Provider request or job id, when one was returned.
        request_id: Option<String>,
    },
    /// A media body exceeded the configured size cap.
    #[error("media body exceeds the {limit}-byte limit")]
    TooLarge {
        /// The cap that was exceeded, in bytes.
        limit: usize,
    },
    /// Reading a local reference or writing an artifact failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// A JSON payload could not be encoded or decoded.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl Error {
    /// Whether the failure is transient and the same call may succeed later.
    ///
    /// Only rate limits, upstream 5xx responses and transport failures are
    /// retryable. A billed non-delivery ([`Error::NoMedia`]) is deliberately
    /// not: a retry is a new, separately billed generation.
    #[must_use]
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Http { status, .. } => *status == 429 || (500..=599).contains(status),
            Self::Transport(_) => true,
            _ => false,
        }
    }
}
