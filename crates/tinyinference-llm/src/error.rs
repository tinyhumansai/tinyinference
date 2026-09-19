//! Error types for language-model inference and provider transport.

use thiserror::Error;

use crate::model::ProviderError;

/// Result returned by TinyInference APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// A normalized inference failure.
#[derive(Debug, Error)]
pub enum Error {
    /// A model transport or response failed without structured provider detail.
    #[error("model error: {0}")]
    Model(String),
    /// A provider returned structured failure detail.
    #[error("model error: {0}")]
    Provider(Box<ProviderError>),
    /// Caller input or configuration was invalid.
    #[error("validation error: {0}")]
    Validation(String),
    /// A provider payload could not be encoded or decoded.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// A provider model catalog used an invalid response envelope.
    #[error("catalog error: {0}")]
    Catalog(String),
    /// The requested operation is not supported by this adapter.
    #[error("unsupported operation: {0}")]
    Unsupported(String),
}
