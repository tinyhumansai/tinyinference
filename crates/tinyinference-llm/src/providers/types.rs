//! Providers module types.
//!
//! All public and internal types for the `providers` module live here.
//! Implementations and trait-impls are in `mod.rs`.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::model::ModelResponse;

// ---------------------------------------------------------------------------
// Provider request options
// ---------------------------------------------------------------------------

/// Host-supplied hooks and transport override applied around a single
/// provider adapter's HTTP call.
///
/// Set on an adapter at construction time (for example
/// `OpenAiModel::with_request_options`) rather than per [`crate::model::ModelRequest`]:
/// [`ModelRequest`](crate::model::ModelRequest) is a serializable, provider-neutral value and
/// cannot carry closures. `on_payload` runs immediately before the adapter
/// serializes and sends the wire request body, so a host can inject
/// provider-specific fields (for example a beta header's JSON companion, or
/// an organization id) without the harness knowing about them. `on_response`
/// runs after a successful response is parsed into JSON, before the adapter
/// normalizes it, so a host can log or inspect the raw payload. `http`
/// overrides the adapter's own [`reqwest::Client`] (for a custom proxy,
/// timeout, or TLS configuration) when set.
///
/// Neither hook may fail: they observe or mutate in place. A hook that needs
/// to reject a request should be implemented as request validation before the
/// call is made instead.
#[derive(Clone, Default)]
pub struct ProviderRequestOptions {
    /// Invoked with the mutable wire payload immediately before it is sent.
    pub on_payload: Option<PayloadHook>,
    /// Invoked with the raw response payload after a successful call.
    pub on_response: Option<ResponseHook>,
    /// HTTP client to use in place of the adapter's own, when set.
    pub http: Option<reqwest::Client>,
}

/// A payload-mutation hook: see [`ProviderRequestOptions::on_payload`].
pub type PayloadHook = Arc<dyn Fn(&mut Value) + Send + Sync>;

/// A response-observation hook: see [`ProviderRequestOptions::on_response`].
pub type ResponseHook = Arc<dyn Fn(&Value) + Send + Sync>;

impl ProviderRequestOptions {
    /// Runs [`Self::on_payload`], when set.
    pub fn apply_payload(&self, payload: &mut Value) {
        if let Some(hook) = &self.on_payload {
            hook(payload);
        }
    }

    /// Runs [`Self::on_response`], when set.
    pub fn observe_response(&self, response: &Value) {
        if let Some(hook) = &self.on_response {
            hook(response);
        }
    }
}

impl std::fmt::Debug for ProviderRequestOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderRequestOptions")
            .field("on_payload", &self.on_payload.as_ref().map(|_| "<fn>"))
            .field("on_response", &self.on_response.as_ref().map(|_| "<fn>"))
            .field("http", &self.http.as_ref().map(|_| "<reqwest::Client>"))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Provider selection types
// ---------------------------------------------------------------------------

/// Common chat-model providers that can be selected through a uniform factory.
///
/// This enum mirrors LangChain's pragmatic provider registry: it covers popular
/// providers directly while leaving [`ProviderKind::Compatible`] for any
/// endpoint that implements the OpenAI Chat Completions wire protocol.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    /// Hosted OpenAI API.
    OpenAi,
    /// Anthropic via its OpenAI-compatible Chat Completions endpoint.
    Anthropic,
    /// Local Ollama server exposing `/v1/chat/completions`.
    Ollama,
    /// Local LM Studio server exposing `/v1/chat/completions`.
    LmStudio,
    /// Local llama.cpp server exposing `/v1/chat/completions`.
    LlamaCpp,
    /// Local vLLM server exposing `/v1/chat/completions`.
    Vllm,
    /// DeepSeek OpenAI-compatible endpoint.
    DeepSeek,
    /// Groq OpenAI-compatible endpoint.
    Groq,
    /// xAI OpenAI-compatible endpoint.
    Xai,
    /// OpenRouter OpenAI-compatible endpoint.
    OpenRouter,
    /// Fireworks AI OpenAI-compatible endpoint.
    Fireworks,
    /// TinyHumans OpenAI-compatible gateway.
    TinyHumans,
    /// Together AI OpenAI-compatible endpoint.
    Together,
    /// Mistral OpenAI-compatible endpoint.
    Mistral,
    /// Any user-supplied OpenAI-compatible endpoint.
    Compatible,
}

impl ProviderKind {
    /// Stable provider identifier used in profiles, errors, and registry names.
    pub fn as_str(&self) -> &'static str {
        match self {
            ProviderKind::OpenAi => "openai",
            ProviderKind::Anthropic => "anthropic",
            ProviderKind::Ollama => "ollama",
            ProviderKind::LmStudio => "lmstudio",
            ProviderKind::LlamaCpp => "llama_cpp",
            ProviderKind::Vllm => "vllm",
            ProviderKind::DeepSeek => "deepseek",
            ProviderKind::Groq => "groq",
            ProviderKind::Xai => "xai",
            ProviderKind::OpenRouter => "openrouter",
            ProviderKind::Fireworks => "fireworks",
            ProviderKind::TinyHumans => "tinyhumans",
            ProviderKind::Together => "together",
            ProviderKind::Mistral => "mistral",
            ProviderKind::Compatible => "compatible",
        }
    }

    /// Best-effort inference from a LangChain-style model string.
    ///
    /// Supports explicit prefixes like `openai:gpt-4.1-mini` as well as common
    /// bare model prefixes. Inference is intentionally conservative; pass an
    /// explicit [`ProviderSpec`] when ambiguity matters.
    pub fn infer(model: &str) -> Option<Self> {
        let lower = model.to_ascii_lowercase();
        if let Some((prefix, _)) = lower.split_once(':') {
            return match prefix {
                "openai" => Some(ProviderKind::OpenAi),
                "anthropic" => Some(ProviderKind::Anthropic),
                "ollama" => Some(ProviderKind::Ollama),
                "lmstudio" | "lm_studio" | "lm-studio" => Some(ProviderKind::LmStudio),
                "llamacpp" | "llama_cpp" | "llama-cpp" | "llamaserver" => {
                    Some(ProviderKind::LlamaCpp)
                }
                "vllm" => Some(ProviderKind::Vllm),
                "deepseek" => Some(ProviderKind::DeepSeek),
                "groq" => Some(ProviderKind::Groq),
                "xai" => Some(ProviderKind::Xai),
                "openrouter" => Some(ProviderKind::OpenRouter),
                "fireworks" => Some(ProviderKind::Fireworks),
                "tinyhumans" | "tiny_humans" | "tiny-humans" => Some(ProviderKind::TinyHumans),
                "together" => Some(ProviderKind::Together),
                "mistral" | "mistralai" => Some(ProviderKind::Mistral),
                _ => None,
            };
        }
        if lower.starts_with("gpt-")
            || lower.starts_with("o1")
            || lower.starts_with("o3")
            || lower.starts_with("o4")
        {
            Some(ProviderKind::OpenAi)
        } else if lower.starts_with("claude") {
            Some(ProviderKind::Anthropic)
        } else if lower.starts_with("deepseek") {
            Some(ProviderKind::DeepSeek)
        } else if lower.starts_with("grok") {
            Some(ProviderKind::Xai)
        } else if lower.starts_with("mistral") || lower.starts_with("mixtral") {
            Some(ProviderKind::Mistral)
        } else {
            None
        }
    }
}

/// Provider configuration used to construct a chat model adapter.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderSpec {
    /// Provider family.
    pub kind: ProviderKind,
    /// Provider id written to profiles and normalized errors.
    pub provider: String,
    /// Default provider model id.
    pub model: String,
    /// API base URL without a trailing slash.
    pub base_url: String,
    /// Environment variable containing the API key, when one is required.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Whether the provider requires a real API key.
    #[serde(default)]
    pub requires_api_key: bool,
}

impl ProviderSpec {
    /// Returns the default provider spec for a known provider.
    pub fn for_kind(kind: ProviderKind) -> Self {
        match kind {
            ProviderKind::OpenAi => Self::new(
                kind,
                "gpt-4.1-mini",
                "https://api.openai.com/v1",
                Some("OPENAI_API_KEY"),
                true,
            ),
            ProviderKind::Anthropic => Self::new(
                kind,
                "claude-3-5-sonnet-latest",
                "https://api.anthropic.com/v1",
                Some("ANTHROPIC_API_KEY"),
                true,
            ),
            ProviderKind::Ollama => {
                Self::new(kind, "llama3.2", "http://localhost:11434/v1", None, false)
            }
            ProviderKind::LmStudio => Self::new(kind, "", "http://localhost:1234/v1", None, false),
            ProviderKind::LlamaCpp => Self::new(kind, "", "http://localhost:8080/v1", None, false),
            ProviderKind::Vllm => Self::new(kind, "", "http://localhost:8000/v1", None, false),
            ProviderKind::DeepSeek => Self::new(
                kind,
                "deepseek-chat",
                "https://api.deepseek.com/v1",
                Some("DEEPSEEK_API_KEY"),
                true,
            ),
            ProviderKind::Groq => Self::new(
                kind,
                "llama-3.3-70b-versatile",
                "https://api.groq.com/openai/v1",
                Some("GROQ_API_KEY"),
                true,
            ),
            ProviderKind::Xai => Self::new(
                kind,
                "grok-2-latest",
                "https://api.x.ai/v1",
                Some("XAI_API_KEY"),
                true,
            ),
            ProviderKind::OpenRouter => Self::new(
                kind,
                "openai/gpt-4o-mini",
                "https://openrouter.ai/api/v1",
                Some("OPENROUTER_API_KEY"),
                true,
            ),
            ProviderKind::Fireworks => Self::new(
                kind,
                "accounts/fireworks/models/llama-v3p1-8b-instruct",
                "https://api.fireworks.ai/inference/v1",
                Some("FIREWORKS_API_KEY"),
                true,
            ),
            // The gateway supports many upstream models, so callers must pick
            // one explicitly rather than silently routing to an arbitrary
            // default. Its prompt-cache key is still lowered by the shared
            // OpenAI-compatible adapter.
            ProviderKind::TinyHumans => Self::new(
                kind,
                "",
                "https://api.tinyhumans.ai/openai/v1",
                Some("TINYHUMANS_AUTH_TOKEN"),
                true,
            ),
            ProviderKind::Together => Self::new(
                kind,
                "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                "https://api.together.xyz/v1",
                Some("TOGETHER_API_KEY"),
                true,
            ),
            ProviderKind::Mistral => Self::new(
                kind,
                "mistral-small-latest",
                "https://api.mistral.ai/v1",
                Some("MISTRAL_API_KEY"),
                true,
            ),
            ProviderKind::Compatible => Self::new(kind, "", "", None, true),
        }
    }

    fn new(
        kind: ProviderKind,
        model: impl Into<String>,
        base_url: impl Into<String>,
        api_key_env: Option<&str>,
        requires_api_key: bool,
    ) -> Self {
        let provider = kind.as_str().to_string();
        Self {
            kind,
            provider,
            model: model.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key_env: api_key_env.map(str::to_string),
            requires_api_key,
        }
    }

    /// Overrides the default model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Overrides the base URL.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Overrides the provider id.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self
    }

    /// Overrides the API-key environment variable.
    pub fn with_api_key_env(mut self, env: impl Into<String>) -> Self {
        self.api_key_env = Some(env.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Internal behavior enum
// ---------------------------------------------------------------------------

/// The scripted behavior that drives a [`MockModel`] invocation.
///
/// This is an internal type — callers interact with [`MockModel`]'s named
/// constructors instead.
pub(crate) enum MockBehavior {
    /// Echoes the text of the last [`Message::User`][crate::message::Message]
    /// in the request back as the assistant reply.
    Echo,

    /// Always returns a fixed assistant text string, regardless of input.
    Constant(String),

    /// Returns responses from a pre-loaded vector in order, cycling back to
    /// the start when all responses have been consumed. See
    /// [`MockModel::with_responses`] for details.
    Scripted(Vec<ModelResponse>),

    /// Returns a single tool-call request for the named tool.  The
    /// `AssistantMessage` carries the call in its `tool_calls` field and the
    /// `finish_reason` is `"tool_calls"`.
    ToolCall {
        /// Name of the tool the model is requesting.
        name: String,
        /// JSON arguments to supply to the tool.
        arguments: Value,
    },

    /// Emits a caller-provided list of [`ModelStreamItem`]s verbatim when
    /// streamed, so tests exercise truly incremental streaming (fine-grained
    /// text/reasoning/tool-call deltas). See [`MockModel::streaming_script`].
    /// `invoke` folds the same items through a
    /// [`StreamAccumulator`][crate::model::StreamAccumulator] to
    /// produce the equivalent unary response.
    StreamScript(Vec<crate::model::ModelStreamItem>),
}

// ---------------------------------------------------------------------------
// Internal mutable state (behind a Mutex for Send + Sync)
// ---------------------------------------------------------------------------

/// Mutable runtime state for [`MockModel`], protected by a `Mutex`.
#[derive(Default)]
pub(crate) struct MockInner {
    /// Total number of [`ChatModel::invoke`][crate::model::ChatModel]
    /// calls made so far (not counting `stream` calls that delegate to invoke).
    pub(crate) call_count: u64,
    /// Next index into the scripted response list (used by [`MockBehavior::Scripted`]).
    pub(crate) scripted_index: usize,
}

// ---------------------------------------------------------------------------
// MockModel
// ---------------------------------------------------------------------------

/// A deterministic, in-process chat model for tests and harness development.
///
/// `MockModel` implements [`ChatModel<State>`][crate::model::ChatModel]
/// generically for *any* `State: Send + Sync`.  It never makes network calls
/// and has no external dependencies.
///
/// # Constructors
///
/// | Constructor | Behaviour |
/// |---|---|
/// | [`MockModel::echo`] | Echoes the last user message text back. |
/// | [`MockModel::constant`] | Always returns the same fixed string. |
/// | [`MockModel::with_responses`] | Returns scripted [`ModelResponse`]s in order, cycling when exhausted. |
/// | [`MockModel::with_tool_call`] | Always returns one tool-call request. |
///
/// # Streaming
///
/// The [`ChatModel::stream`][crate::model::ChatModel] override
/// internally calls [`crate::model::ChatModel::invoke`] and replays the response as a real
/// [`ModelStream`][crate::model::ModelStream]: a
/// [`Started`][crate::model::ModelStreamItem::Started] item, one or two
/// [`MessageDelta`][crate::model::ModelStreamItem::MessageDelta] items
/// (text split into two equal-sized halves by Unicode scalar value), and a
/// terminal [`Completed`][crate::model::ModelStreamItem::Completed]
/// item carrying the full response. This lets downstream streaming consumers be
/// exercised without any real streaming infrastructure. When the response
/// contains no text (e.g. a tool-call response), a single empty text delta is
/// emitted before completion.
///
/// # Usage estimates
///
/// Every response carries a deterministic [`Usage`][crate::usage::Usage]
/// derived from character counts:
/// - `input_tokens` ≈ total characters in all request messages ÷ 4
/// - `output_tokens` ≈ total characters in the response text ÷ 4 (minimum 1)
///
/// This gives cost-accounting code realistic non-zero values to work with.
///
/// # Placement of real providers
///
/// Real network-backed providers live in sub-modules alongside this one. The
/// OpenAI (and OpenAI-compatible) adapter is always compiled; providers with a
/// different wire protocol would be gated behind their own Cargo feature:
///
/// ```text
/// pub mod openai;                          // always compiled
/// // #[cfg(feature = "anthropic")] pub mod anthropic;
/// // #[cfg(feature = "ollama")]   pub mod ollama;
/// ```
///
/// Add the feature flag to `Cargo.toml` and implement
/// [`ChatModel`][crate::model::ChatModel] in the corresponding module.
/// No changes to `mod.rs` or `harness/mod.rs` are needed beyond enabling the
/// `pub mod` declaration.
pub struct MockModel {
    pub(crate) behavior: MockBehavior,
    pub(crate) inner: Mutex<MockInner>,
}

impl std::fmt::Debug for MockModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let call_count = self
            .inner
            .lock()
            .map(|inner| inner.call_count)
            .unwrap_or_default();
        formatter
            .debug_struct("MockModel")
            .field("call_count", &call_count)
            .finish_non_exhaustive()
    }
}
