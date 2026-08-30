//! Error types for inference, provider transport, and embeddings.

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
    /// Embedding generation or vector-store behavior failed.
    #[error("embedding error: {0}")]
    Embedding(String),
}
