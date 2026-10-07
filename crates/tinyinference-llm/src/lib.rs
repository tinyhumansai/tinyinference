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
mod network_guard;
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
    AnnotatedContent, ExecutionStatus, FetchedContent, FinanceResult, HostedToolUsage,
    ImageSearchResult, ModelExecution, ModelFile, ModelFileContent, ModelOutputEvent,
    ModelOutputItem, ModelOutputKind, ModelProgress, OutputAnnotation, RemoteToolDefinition,
    RemoteToolNamespace, ReportedCost, SearchSource,
};
pub use model::{
    ChatModel, InputModality, InputSource, ModelRequest, ModelResponse, ModelStream,
    ModelStreamItem, context_window_for_model_id, model_id_supports_vision,
};
pub use network_guard::{allow_network_models, deny_network_models, network_models_denied};
pub use providers::perplexity::{
    PerplexityConfig, PerplexityImageFilters, PerplexityLocation, PerplexityModel,
    PerplexityOptions, PerplexityRemoteCredentials, PerplexitySearchFilters, PerplexitySelection,
    PerplexityTool, PerplexityToolChoice, PerplexityWebSearch, build_perplexity_model,
};
pub use providers::{MockModel, ProviderKind, ProviderSpec};
pub use tool::{ToolCall, ToolDelta, ToolFormat, ToolSchema};
pub use tool::{ToolCallReplay, ToolResultContext};
pub use usage::{Usage, UsageTotals};
