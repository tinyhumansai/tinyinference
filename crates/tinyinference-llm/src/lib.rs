//! Provider-neutral language-model inference for Rust.
//!
//! This crate owns typed chat messages, model requests and responses,
//! asynchronous streaming, tool-call wire shapes, normalized usage, hosted
//! providers, model catalogs, caching, and LLM-oriented helpers.

pub mod cache;
pub mod catalog;
pub mod classification;
pub mod completion;
pub mod error;
pub mod failure;
pub mod message;
pub mod model;
pub mod prompt_tools;
pub mod providers;
pub mod sentiment;
pub mod tool;
pub mod usage;

pub use error::{Error, Result};
pub use failure::{
    ProviderFailureClass, classify_provider_error, classify_provider_failure, parse_retry_after_ms,
    provider_error_is_retryable, structured_http_status,
};
pub use message::{AssistantMessage, ContentBlock, Message, MessageDelta};
pub use model::{
    ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
    context_window_for_model_id, model_id_supports_vision,
};
pub use providers::{MockModel, ProviderKind, ProviderSpec};
pub use tool::{ToolCall, ToolDelta, ToolFormat, ToolSchema};
pub use usage::{Usage, UsageTotals};
