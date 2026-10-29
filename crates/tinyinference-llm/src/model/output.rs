//! Ordered model output and execution metadata.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::ContentBlock;
use crate::tool::ToolCall;

/// A generated file owned by a provider response, not a local filesystem path.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelFile {
    /// Owning response identifier.
    pub response_id: String,
    /// File identifier, distinct from response ID and filename.
    pub id: String,
    /// Provider filename, treated as untrusted display metadata.
    pub filename: String,
    /// Reported byte length.
    pub bytes: u64,
    /// Provider creation timestamp in Unix seconds.
    pub created_at: u64,
}

/// Downloaded artifact content. The library never writes it to disk.
#[derive(Clone, PartialEq, Eq)]
pub struct ModelFileContent {
    /// Owning response identifier.
    pub response_id: String,
    /// Downloaded file identifier.
    pub file_id: String,
    /// Content type reported by the provider.
    pub media_type: Option<String>,
    /// Raw file bytes, bounded by the client's configured download limit.
    pub bytes: Vec<u8>,
}

impl std::fmt::Debug for ModelFileContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelFileContent")
            .field("byte_length", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

/// Provider execution state, preserving future states without guessing success.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub enum ExecutionStatus {
    /// The operation completed.
    Completed,
    /// The operation failed.
    Failed,
    /// The operation stopped with partial output.
    Incomplete,
    /// Accepted but not yet running.
    Queued,
    /// Work is in progress.
    InProgress,
    /// Cancellation was requested but not yet confirmed.
    Cancelling,
    /// Cancellation is confirmed.
    Cancelled,
    /// A provider state not yet understood by this version.
    Other(String),
}

impl From<String> for ExecutionStatus {
    fn from(value: String) -> Self {
        match value.as_str() {
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            "incomplete" => Self::Incomplete,
            "queued" => Self::Queued,
            "in_progress" => Self::InProgress,
            "cancelling" => Self::Cancelling,
            "cancelled" => Self::Cancelled,
            _ => Self::Other(value),
        }
    }
}

impl From<ExecutionStatus> for String {
    fn from(value: ExecutionStatus) -> Self {
        match value {
            ExecutionStatus::Completed => "completed".into(),
            ExecutionStatus::Failed => "failed".into(),
            ExecutionStatus::Incomplete => "incomplete".into(),
            ExecutionStatus::Queued => "queued".into(),
            ExecutionStatus::InProgress => "in_progress".into(),
            ExecutionStatus::Cancelling => "cancelling".into(),
            ExecutionStatus::Cancelled => "cancelled".into(),
            ExecutionStatus::Other(value) => value,
        }
    }
}

impl ExecutionStatus {
    /// Whether the provider has reported a known terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Incomplete | Self::Cancelled
        )
    }
}

/// Structured execution metadata, separate from the assistant message identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelExecution {
    /// Provider response/run identifier used for retrieval and continuation.
    pub id: String,
    /// Actual model returned by the provider, not a preset name.
    /// May be empty in a provisional stream snapshot before model selection.
    pub model: String,
    /// Current run status.
    pub status: ExecutionStatus,
    /// Explanation for an incomplete response, when supplied.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incomplete_reason: Option<String>,
    /// Last provider event sequence number for explicit reconnect.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence_number: Option<u64>,
    /// Provider-reported costs, without locally computed prices.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost: Option<ReportedCost>,
    /// Invocation counts and optional measured charges per hosted tool.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tool_usage: BTreeMap<String, HostedToolUsage>,
    /// Stream-only progress, retained even when absent from the final provider snapshot.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub progress: Vec<ModelProgress>,
}

/// Reported decimal cost amounts. Strings preserve sub-micro-unit precision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReportedCost {
    /// Provider currency identifier.
    pub currency: String,
    /// Reported components: input, output, cache_read, cache_creation, tools, total.
    pub components: BTreeMap<String, String>,
    /// Reported charges attributed to individual tools.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub tools: BTreeMap<String, String>,
}

/// Measured usage for one provider-executed tool.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HostedToolUsage {
    /// Invocation count, when supplied.
    pub invocations: Option<u64>,
    /// Provider-reported decimal USD charge, when supplied.
    pub cost_usd: Option<String>,
}

/// A source annotation attached to its original content part.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OutputAnnotation {
    /// A provider-supplied URL citation. Offsets are provider character units,
    /// not UTF-8 bytes; callers must not use them to slice Rust strings.
    UrlCitation {
        /// `provider_characters` until the provider specifies an exact offset encoding.
        offset_unit: String,
        /// Source URL.
        url: String,
        /// Source title.
        title: Option<String>,
        /// Original start offset, if supplied.
        start: Option<u64>,
        /// Original end offset, if supplied.
        end: Option<u64>,
    },
    /// An annotation this adapter cannot normalize without losing information.
    Extension {
        /// Provider identifier.
        provider: String,
        /// Original annotation data.
        data: Value,
    },
}

/// Content and its annotations, without merging citation coordinate spaces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnnotatedContent {
    /// Normalized text, reasoning, media, or extension block.
    pub block: ContentBlock,
    /// Annotations on this content part only.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub annotations: Vec<OutputAnnotation>,
}

/// Source returned by web or people search.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchSource {
    /// Provider source identifier, scoped to its output item.
    pub id: String,
    /// Source URL.
    pub url: String,
    /// Source title.
    pub title: String,
    /// Retrieved snippet.
    pub snippet: String,
    /// Source category, when supplied.
    pub source: Option<String>,
    /// Publication date in the provider's original representation.
    pub date: Option<String>,
    /// Last-update date in the provider's original representation.
    pub last_updated: Option<String>,
}

/// A hosted image-search match, not a downloaded image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSearchResult {
    /// Image URL.
    pub image_url: String,
    /// Page the image came from.
    pub origin_url: String,
    /// Reported pixel width.
    pub width: u64,
    /// Reported pixel height.
    pub height: u64,
    /// Image title, when supplied.
    pub title: Option<String>,
}

/// Content fetched by a hosted URL tool.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FetchedContent {
    /// Requested URL.
    pub url: String,
    /// Page title.
    pub title: String,
    /// Retrieved content.
    pub snippet: String,
}

/// A finance-search result; original tables/content remain unmodified.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinanceResult {
    /// Requested data category.
    pub category: String,
    /// Provider-rendered table or snippet.
    pub content: String,
    /// Associated ticker symbols.
    #[serde(default)]
    pub tickers: Vec<String>,
    /// Supporting source URLs.
    #[serde(default)]
    pub sources: Vec<String>,
}

/// A tool definition discovered remotely, never a request for local execution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemoteToolDefinition {
    /// Tool name.
    pub name: String,
    /// Tool description when supplied.
    pub description: Option<String>,
    /// JSON Schema for the tool arguments; null means the provider omitted it.
    pub parameters: Value,
}

/// Hosted tool namespace returned by tool discovery/search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RemoteToolNamespace {
    /// Namespace name.
    pub name: String,
    /// Namespace description.
    pub description: Option<String>,
    /// Available tools.
    pub tools: Vec<RemoteToolDefinition>,
}

/// A provider output item, preserving ordering and item identity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelOutputItem {
    /// Original zero-based output position.
    pub index: usize,
    /// Provider item ID, distinct from run and function call IDs.
    pub id: Option<String>,
    /// Provider item status, when present.
    pub status: Option<ExecutionStatus>,
    /// Normalized item content.
    pub kind: ModelOutputKind,
}

/// Normalized output families. Hosted calls never become local function calls.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelOutputKind {
    /// Assistant or provider message.
    Message {
        /// Original role.
        role: String,
        /// Ordered, individually annotated content parts.
        content: Vec<AnnotatedContent>,
    },
    /// Supplied reasoning content and opaque replay signatures.
    Reasoning {
        /// Ordered reasoning content.
        content: Vec<ContentBlock>,
    },
    /// Function request for the consuming application to execute.
    FunctionCall {
        /// Valid or explicitly invalid model call.
        call: ToolCall,
    },
    /// Web or people search results, grouped by invocation.
    Search {
        /// Search tool identity.
        tool: String,
        /// Queries used by the provider.
        queries: Vec<String>,
        /// Sources from this invocation.
        results: Vec<SearchSource>,
    },
    /// Hosted image search.
    ImageSearch {
        /// Search queries.
        queries: Vec<String>,
        /// Returned images.
        results: Vec<ImageSearchResult>,
        /// Tool-local error, independent of overall response success.
        error: Option<String>,
    },
    /// Hosted URL retrieval.
    Fetch {
        /// Retrieved pages.
        contents: Vec<FetchedContent>,
    },
    /// Hosted finance retrieval.
    Finance {
        /// Requested categories.
        categories: Vec<String>,
        /// Requested symbols.
        tickers: Vec<String>,
        /// Returned financial content and sources.
        results: Vec<FinanceResult>,
    },
    /// Execution inside the provider's sandbox.
    Sandbox {
        /// Code that was run, when supplied.
        code: Option<String>,
        /// Standard output.
        stdout: Option<String>,
        /// Standard error.
        stderr: Option<String>,
        /// Process exit status.
        exit_code: Option<i32>,
        /// Provider-reported runtime.
        duration_ms: Option<u64>,
        /// Tool execution state, preserving unrecognized values.
        status: String,
    },
    /// Tools discovered on a remote server.
    ToolDiscovery {
        /// Remote server label.
        server: String,
        /// Managed connector ID, when applicable.
        connector_id: Option<String>,
        /// Discovered definitions.
        tools: Vec<RemoteToolDefinition>,
        /// Discovery failure when supplied.
        error: Option<String>,
    },
    /// A tool already executed by the provider on a remote server.
    RemoteCall {
        /// Remote server label.
        server: String,
        /// Managed connector ID, when applicable.
        connector_id: Option<String>,
        /// Tool name.
        name: String,
        /// Original argument text, including malformed text if supplied.
        arguments: String,
        /// Returned tool content.
        output: Option<String>,
        /// Tool-local failure.
        error: Option<String>,
    },
    /// Hosted search for tool definitions.
    ToolSearch {
        /// Execution location as reported by the provider.
        execution: String,
        /// Call identifier when supplied.
        call_id: Option<String>,
        /// Original search arguments.
        arguments: Option<String>,
        /// Discovered namespaces.
        tools: Vec<RemoteToolNamespace>,
    },
    /// Notice of generated files. List files by response ID for authoritative descriptors.
    GeneratedFiles {
        /// Owning response ID.
        response_id: String,
        /// Unspecified provider notice fields, preserved without guessing meaning.
        metadata: Value,
    },
    /// Future provider output that cannot yet be normalized.
    Extension {
        /// Provider identifier.
        provider: String,
        /// Original item type.
        item_type: String,
        /// Original item data.
        data: Value,
    },
}

/// Rich stream updates, parallel to the existing text/tool/usage delta channels.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
pub enum ModelOutputEvent {
    /// Progress that is not an indexed response item.
    Progress(ModelProgress),
    /// An output item opened before its final required fields were available.
    ItemStarted {
        /// Original output index.
        index: usize,
        /// Provider item ID, when supplied.
        id: Option<String>,
        /// Original item type.
        item_type: String,
    },
    /// Current execution snapshot, including response ID and reconnect cursor.
    Execution(Box<ModelExecution>),
    /// Current item snapshot; replaces the same output index, never appends twice.
    Item(Box<ModelOutputItem>),
}

/// Non-item progress retained by stream collection and partial failures.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "content", rename_all = "snake_case")]
pub enum ModelProgress {
    /// Reasoning note supplied alongside a hosted-tool progress event.
    ReasoningText {
        /// Supplied text; never inferred from hidden reasoning.
        text: String,
    },
    /// Hosted results reported without an output index to upsert.
    HostedActivity(Box<ModelOutputKind>),
    /// Queries issued by a hosted tool before results arrive.
    Queries {
        /// Hosted tool name.
        tool: String,
        /// Queries in this event.
        queries: Vec<String>,
    },
    /// Provider reasoning phase; no hidden reasoning text is inferred.
    ReasoningPhase {
        /// Whether reasoning started or stopped.
        active: bool,
    },
    /// Unrecognized provider event, preserved for observers.
    Extension {
        /// Provider event discriminator.
        event_type: String,
        /// Original event data.
        data: Value,
    },
}

#[cfg(test)]
#[path = "output_tests.rs"]
mod tests;
