//! Error types for embedding providers and vector stores.

use thiserror::Error;

/// Result returned by TinyInference embedding APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// A normalized embedding failure.
#[derive(Debug, Error)]
pub enum Error {
    /// Caller input or configuration was invalid.
    #[error("validation error: {0}")]
    Validation(String),
    /// A provider payload could not be encoded or decoded.
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    /// Embedding generation or vector-store behavior failed.
    #[error("embedding error: {0}")]
    Embedding(String),
    /// The caller cancelled an embedding request.
    #[error("embedding request cancelled")]
    Cancelled,
}
