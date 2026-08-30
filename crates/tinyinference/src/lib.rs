//! Provider-neutral language-model inference for Rust.
//!
//! TinyInference owns the reusable API boundary between agent runtimes and
//! model vendors: typed messages, model requests and responses, asynchronous
//! streaming, tool-call wire shapes, normalized usage, hosted providers, and
//! embedding clients. It deliberately contains no agent loop, graph runtime,
//! middleware stack, registry, or workspace policy.

pub mod cache;
pub mod embeddings;
pub mod error;
pub mod failure;
pub mod message;
pub mod model;
pub mod providers;
pub mod tool;
pub mod usage;

pub use error::{Error, Result};

pub use embeddings::{
    EmbeddingModel, InMemoryVectorStore, MockEmbeddingModel, Retriever, ScoredDoc, VectorStore,
    cosine_similarity,
};
pub use message::{AssistantMessage, ContentBlock, Message, MessageDelta};
pub use model::{ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem};
pub use providers::{MockModel, ProviderKind, ProviderSpec};
pub use tool::{ToolCall, ToolDelta, ToolFormat, ToolSchema};
pub use usage::{Usage, UsageTotals};
