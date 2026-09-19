//! Harness model layer types.
//!
//! These provider-neutral shapes are the common currency of the recursive
//! harness: the same [`ModelRequest`] / [`ModelResponse`] / [`ModelStream`]
//! types describe a call whether it originates from a top-level agent, a nested
//! sub-agent, or a graph node, so model-calls-model recursion is expressed in
//! one uniform vocabulary regardless of depth or provider.
//!
//! These are the rich, harness-internal request/response shapes. They carry
//! tool declarations, tool-choice policy, structured-output formats, capability
//! profiles ([`ModelProfile`]/[`CapabilitySet`]), model-resolution inputs, and
//! prompt-cache layout metadata.

use std::pin::Pin;

use async_trait::async_trait;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Result;
use crate::cache::CachePolicy;
use crate::message::{AssistantMessage, ContentBlock, Message, MessageDelta};
use crate::tool::{ToolDelta, ToolSchema};
use crate::usage::Usage;

/// Policy controlling whether and how the model may call tools.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoice {
    /// The model decides whether to call a tool.
    #[default]
    Auto,
    /// The model must not call any tool.
    None,
    /// The model must call some tool.
    Required,
    /// The model must call the named tool.
    Tool(String),
}

/// The requested output format for a model response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Free-form text.
    Text,
    /// Any JSON object.
    JsonObject,
    /// JSON constrained to a named schema.
    JsonSchema {
        /// Schema name advertised to the provider.
        name: String,
        /// JSON Schema document.
        schema: Value,
    },
    /// Request structured JSON output but let the harness pick the extraction
    /// strategy from the resolved model's [`ModelProfile`].
    ///
    /// The consuming harness resolves this into either provider-native schema
    /// mode when the model advertises native structured output, or a tool-call
    /// fallback otherwise. When no profile is
    /// available it falls back to provider-native schema mode.
    ///
    Auto {
        /// Schema name advertised to the provider or used as the fallback tool
        /// name.
        name: String,
        /// JSON Schema document describing the desired structure.
        schema: Value,
    },
}

/// Provider-neutral reasoning effort for one model call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    /// Smallest available effort.
    Minimal,
    /// Below-default effort.
    Low,
    /// Provider default effort.
    #[default]
    Medium,
    /// Above-default effort.
    High,
    /// Explicitly disable reasoning.
    None,
}

impl ReasoningEffort {
    /// Returns the common provider wire token.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::None => "none",
        }
    }
}

/// Provider-neutral reasoning configuration.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Requested reasoning effort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<ReasoningEffort>,
    /// Explicit thinking-token budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
    /// Requested reasoning-summary verbosity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

impl ReasoningConfig {
    /// Creates a config containing only `effort`.
    pub fn effort(effort: ReasoningEffort) -> Self {
        Self {
            effort: Some(effort),
            ..Self::default()
        }
    }

    /// Returns whether no reasoning option is set.
    pub fn is_empty(&self) -> bool {
        self.effort.is_none() && self.budget_tokens.is_none() && self.summary.is_none()
    }
}

/// Lifecycle status of a model, used by [`ModelProfile`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelStatus {
    /// Generally available and supported.
    #[default]
    Stable,
    /// Preview/beta; behavior may change.
    Preview,
    /// Slated for removal; callers should migrate.
    Deprecated,
    /// No longer served by the provider.
    Retired,
}

/// The input/output modalities a model supports.
///
/// [`Default`] enables text in and text out only; all media modalities are
/// disabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Modalities {
    /// Accepts text input.
    pub text_in: bool,
    /// Produces text output.
    pub text_out: bool,
    /// Accepts image input (vision).
    pub image_in: bool,
    /// Produces image output.
    pub image_out: bool,
    /// Accepts audio input.
    pub audio_in: bool,
    /// Produces audio output.
    pub audio_out: bool,
    /// Accepts video input.
    pub video_in: bool,
    /// Produces video output.
    pub video_out: bool,
    /// Accepts document input (PDF and similar).
    pub document_in: bool,
}

impl Default for Modalities {
    fn default() -> Self {
        Self {
            text_in: true,
            text_out: true,
            image_in: false,
            image_out: false,
            audio_in: false,
            audio_out: false,
            video_in: false,
            video_out: false,
            document_in: false,
        }
    }
}

/// A capability profile describing what a model can do.
///
/// Profiles let the harness reject impossible requests before a network call,
/// choose native structured output versus tool-based structured output, decide
/// whether tool-call chunks can stream, and select fallbacks that satisfy
/// required capabilities. Profiles are not a pricing table; prices live in the
/// cost feature.
///
/// [`Default`] is conservative: text-only modalities and every optional
/// capability disabled. Providers should override with what they actually
/// support.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelProfile {
    /// Provider family identifier (for example `openai`), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// Provider model id this profile describes, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Human-readable display name, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub display_name: Option<String>,
    /// Lifecycle status.
    #[serde(default)]
    pub status: ModelStatus,
    /// Release date in ISO-8601 (`YYYY-MM-DD`), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub release_date: Option<String>,
    /// Supported input/output modalities.
    #[serde(default)]
    pub modalities: Modalities,
    /// Supports tool/function calling.
    #[serde(default)]
    pub tool_calling: bool,
    /// Supports multiple tool calls in a single response.
    #[serde(default)]
    pub parallel_tool_calls: bool,
    /// Supports streaming responses.
    #[serde(default)]
    pub streaming: bool,
    /// Streams tool-call fragments incrementally (versus reconstructing them
    /// from a final response).
    #[serde(default)]
    pub streaming_tool_chunks: bool,
    /// Supports provider-native structured output (constrained JSON).
    #[serde(default)]
    pub native_structured_output: bool,
    /// Honors a JSON Schema in the response-format request.
    #[serde(default)]
    pub json_schema: bool,
    /// Emits reasoning/thinking output.
    #[serde(default)]
    pub reasoning: bool,
    /// Accepts a configurable reasoning effort.
    #[serde(default)]
    pub reasoning_effort: bool,
    /// Maximum input (context) tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_input_tokens: Option<u64>,
    /// Maximum output tokens, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    /// Whether the provider accepts a `system`/developer-role message
    /// anywhere in the transcript's message list, not only as a single
    /// leading block.
    ///
    /// This is `false` by default and must be opted into explicitly, because
    /// getting it wrong silently drops content: an OpenAI-style chat API
    /// (`Message::System` translated 1:1 to a `role: "system"` wire message
    /// at its transcript position — see `providers::openai::convert::
    /// translate_message`) genuinely honors a system message wherever it
    /// appears, so `true` is correct there. Anthropic's Messages API has no
    /// such slot: every `Message::System` in the transcript, wherever it
    /// occurs, is collected into one top-level `system` array ahead of the
    /// `messages` list (see `providers::anthropic::request`), so a "mid-
    /// conversation" system message is actually hoisted to the front on the
    /// wire — `false` here is what tells a caller (the transcript-carried
    /// system-patch mechanism, `docs/runtime-comparison/plan.md`'s B6) to
    /// fold a patch into the leading system message instead of inserting it
    /// in place.
    #[serde(default)]
    pub mid_conversation_system_messages: bool,
    /// JSON-schema transform this model's adapter must apply before sending a
    /// schema to the provider (for example stripping `$defs` a provider
    /// rejects, or forcing `additionalProperties: false`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema_transform: Option<SchemaTransform>,
    /// Structured-output strategy the harness should default to for this
    /// model when the caller does not pin one explicitly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_structured_mode: Option<StructuredMode>,
    /// Prompt template used when `default_structured_mode` (or an explicit
    /// override) resolves to [`StructuredMode::Prompted`]. Implementations
    /// should substitute the target schema into this template.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompted_output_template: Option<String>,
    /// Open/close tag pair (for example `("<think>", "</think>")`) that this
    /// model emits around chain-of-thought text. Response normalization
    /// should extract tagged spans into a thinking content block rather than
    /// leaving them inline in the visible text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_tags: Option<(String, String)>,
    /// Whether leading whitespace on the first streamed text delta is an
    /// artifact of this provider's wire format and should be dropped rather
    /// than surfaced to the caller.
    #[serde(default)]
    pub ignore_streamed_leading_whitespace: bool,
    /// Maps a named reasoning/thinking level (for example `"low"`,
    /// `"high"`, or a provider-specific label) to the [`ReasoningConfig`] it
    /// expands to for this model.
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub thinking_level_map: std::collections::BTreeMap<String, ReasoningConfig>,
    /// Provider-family compatibility quirks that do not fit the capability
    /// model above.
    #[serde(default)]
    pub compat: ProviderCompat,
}

/// A named, serializable JSON-schema transform applied before a schema is
/// sent to a provider.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum SchemaTransform {
    /// Removes the top-level `$defs`/`definitions` map from the schema.
    StripDefs,
    /// Resolves `$ref` pointers into `$defs`/`definitions` inline, then
    /// drops the now-unused definitions map.
    InlineRefs,
    /// Recursively sets `additionalProperties: false` on every object
    /// schema that does not already specify it.
    NoAdditionalProperties,
    /// Applies the subset of adjustments Gemini's schema dialect requires:
    /// strips `additionalProperties`, `$schema`, `default`, and `examples`
    /// keywords it does not accept.
    GeminiCompat,
    /// Applies OpenAI "strict" JSON-schema mode: inlines refs, forces
    /// `additionalProperties: false`, and marks every property required.
    OpenAiStrict,
    /// Applies a sequence of transforms in order.
    Chain(Vec<SchemaTransform>),
}

impl SchemaTransform {
    /// Applies this transform to `schema`, returning the transformed value.
    /// The input is never mutated in place.
    #[must_use]
    pub fn apply(&self, schema: &Value) -> Value {
        let mut out = schema.clone();
        match self {
            Self::StripDefs => {
                strip_defs(&mut out);
            }
            Self::InlineRefs => {
                let defs = collect_defs(&out);
                inline_refs(&mut out, &defs);
                strip_defs(&mut out);
            }
            Self::NoAdditionalProperties => {
                set_no_additional_properties(&mut out);
            }
            Self::GeminiCompat => {
                strip_keys(
                    &mut out,
                    &["additionalProperties", "$schema", "default", "examples"],
                );
            }
            Self::OpenAiStrict => {
                let defs = collect_defs(&out);
                inline_refs(&mut out, &defs);
                strip_defs(&mut out);
                set_no_additional_properties(&mut out);
                require_all_properties(&mut out);
            }
            Self::Chain(steps) => {
                for step in steps {
                    out = step.apply(&out);
                }
            }
        }
        out
    }
}

fn strip_defs(value: &mut Value) {
    if let Value::Object(map) = value {
        map.remove("$defs");
        map.remove("definitions");
        for v in map.values_mut() {
            strip_defs(v);
        }
    } else if let Value::Array(items) = value {
        for v in items {
            strip_defs(v);
        }
    }
}

fn strip_keys(value: &mut Value, keys: &[&str]) {
    if let Value::Object(map) = value {
        for key in keys {
            map.remove(*key);
        }
        for v in map.values_mut() {
            strip_keys(v, keys);
        }
    } else if let Value::Array(items) = value {
        for v in items {
            strip_keys(v, keys);
        }
    }
}

fn collect_defs(value: &Value) -> serde_json::Map<String, Value> {
    let mut defs = serde_json::Map::new();
    if let Value::Object(map) = value {
        if let Some(Value::Object(d)) = map.get("$defs") {
            defs.extend(d.clone());
        }
        if let Some(Value::Object(d)) = map.get("definitions") {
            defs.extend(d.clone());
        }
    }
    defs
}

fn inline_refs(value: &mut Value, defs: &serde_json::Map<String, Value>) {
    match value {
        Value::Object(map) => {
            if let Some(Value::String(reference)) = map.get("$ref").cloned() {
                let name = reference.rsplit('/').next().unwrap_or(reference.as_str());
                if let Some(resolved) = defs.get(name) {
                    let mut resolved = resolved.clone();
                    inline_refs(&mut resolved, defs);
                    *value = resolved;
                    return;
                }
            }
            for v in map.values_mut() {
                inline_refs(v, defs);
            }
        }
        Value::Array(items) => {
            for v in items {
                inline_refs(v, defs);
            }
        }
        _ => {}
    }
}

fn set_no_additional_properties(value: &mut Value) {
    if let Value::Object(map) = value {
        let is_object_schema = matches!(map.get("type"), Some(Value::String(t)) if t == "object")
            || map.contains_key("properties");
        if is_object_schema && !map.contains_key("additionalProperties") {
            map.insert("additionalProperties".into(), Value::Bool(false));
        }
        for v in map.values_mut() {
            set_no_additional_properties(v);
        }
    } else if let Value::Array(items) = value {
        for v in items {
            set_no_additional_properties(v);
        }
    }
}

fn require_all_properties(value: &mut Value) {
    if let Value::Object(map) = value {
        if let Some(Value::Object(props)) = map.get("properties").cloned() {
            let required: Vec<Value> = props.keys().cloned().map(Value::String).collect();
            map.insert("required".into(), Value::Array(required));
        }
        for v in map.values_mut() {
            require_all_properties(v);
        }
    } else if let Value::Array(items) = value {
        for v in items {
            require_all_properties(v);
        }
    }
}

/// The structured-output extraction strategy the harness should use for a
/// model call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredMode {
    /// Extract structured output via a synthetic tool call.
    Tool,
    /// Use the provider's native constrained-JSON output mode.
    Native,
    /// Ask for structured output via a prompt template and parse the
    /// resulting text.
    Prompted,
}

/// Provider-family compatibility quirks that affect how the harness builds a
/// request, independent of the model's advertised capabilities.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderCompat {
    /// Supports a `system` message appearing anywhere in the conversation,
    /// not only as the first message.
    #[serde(default)]
    pub mid_conversation_system_messages: bool,
    /// Supports strict tool-schema validation (every property required,
    /// `additionalProperties: false` enforced by the provider).
    #[serde(default)]
    pub strict_tools: bool,
    /// Supports explicit prompt-cache retention control.
    #[serde(default)]
    pub cache_retention: bool,
    /// Requires calls for one logical session to land on the same backend
    /// instance (sticky routing) to benefit from caching.
    #[serde(default)]
    pub session_affinity: bool,
    /// Maximum length, in bytes, of a tool name this provider accepts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_name_length: Option<usize>,
    /// Regex pattern tool-call ids from this provider must match.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_id_pattern: Option<String>,
}

/// A set of required capabilities used to validate a request against a
/// [`ModelProfile`] and to filter candidate models during resolution.
///
/// Every boolean field is a *requirement*: `true` means the capability must be
/// present, `false` means "don't care". The token fields require a minimum
/// advertised capacity. [`Default`] requires nothing, so it is satisfied by any
/// profile.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySet {
    /// Requires tool/function calling.
    #[serde(default)]
    pub tool_calling: bool,
    /// Requires parallel tool calls.
    #[serde(default)]
    pub parallel_tool_calls: bool,
    /// Requires streaming responses.
    #[serde(default)]
    pub streaming: bool,
    /// Requires incremental tool-call streaming.
    #[serde(default)]
    pub streaming_tool_chunks: bool,
    /// Requires provider-native structured output.
    #[serde(default)]
    pub native_structured_output: bool,
    /// Requires JSON Schema support.
    #[serde(default)]
    pub json_schema: bool,
    /// Requires reasoning output.
    #[serde(default)]
    pub reasoning: bool,
    /// Requires configurable reasoning effort.
    #[serde(default)]
    pub reasoning_effort: bool,
    /// Requires image input (vision).
    #[serde(default)]
    pub image_in: bool,
    /// Requires image output.
    #[serde(default)]
    pub image_out: bool,
    /// Requires audio input.
    #[serde(default)]
    pub audio_in: bool,
    /// Requires audio output.
    #[serde(default)]
    pub audio_out: bool,
    /// Requires at least this many input (context) tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_input_tokens: Option<u64>,
    /// Requires at least this many output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_output_tokens: Option<u64>,
}

/// The role a [`PromptSegment`] plays in the assembled prompt. Earlier roles
/// form the stable, cacheable prefix; [`SegmentRole::Volatile`] marks the tail
/// that changes every turn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentRole {
    /// System prompt.
    System,
    /// Tool declarations.
    Tools,
    /// Stable instructions.
    Instructions,
    /// Conversation history.
    History,
    /// Volatile, per-turn content that must stay out of stable prefixes.
    Volatile,
}

/// A labeled segment of the prompt used to reason about provider prompt/KV
/// cache stability.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptSegment {
    /// Stable identifier for the segment.
    pub id: String,
    /// The role the segment plays in the prompt.
    pub role: SegmentRole,
    /// Whether this segment is part of the cacheable stable prefix.
    pub cacheable: bool,
}

/// Runtime-supplied model candidate metadata.
///
/// TinyInference carries this serializable value without registering, ranking,
/// or resolving models; consuming runtimes own those policies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelHint {
    /// Runtime registry name or provider model id.
    pub model: String,
    /// Higher values indicate stronger runtime preference.
    #[serde(default)]
    pub priority: i32,
    /// Optional runtime explanation for observability.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Runtime-owned source that selected a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelResolutionSource {
    /// Explicit request-level override.
    RequestOverride,
    /// Reused from durable runtime state.
    StateReuse,
    /// Chosen from runtime hints.
    Hint,
    /// Default declared by an agent.
    AgentDefault,
    /// Default declared by a consuming registry.
    RegistryDefault,
}

/// Durable metadata describing a runtime-selected model.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModel {
    /// Runtime registry name or provider model id.
    pub name: String,
    /// Originally requested name, when different.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested: Option<String>,
    /// Runtime selection source.
    pub source: ModelResolutionSource,
}

/// Stable identifiers correlating one model call with its owning run.
///
/// Hosts allocate these identifiers before dispatch. Providers and generic
/// decorators only preserve them; neither generates product run identifiers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelCallCorrelation {
    /// Stable identifier for the containing run.
    pub run_id: String,
    /// Stable identifier for this model call within the run.
    pub call_id: String,
}

impl ModelCallCorrelation {
    /// Creates a run/model-call correlation pair.
    #[must_use]
    pub fn new(run_id: impl Into<String>, call_id: impl Into<String>) -> Self {
        Self {
            run_id: run_id.into(),
            call_id: call_id.into(),
        }
    }
}

/// The concrete provider, provider model, and host route that handled a call.
///
/// A route is an opaque host-selected label. TinyInference records it for
/// observability but does not perform route or fallback policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedModelRoute {
    /// Concrete provider family, for example `openai`.
    pub provider: String,
    /// Concrete provider model identifier.
    pub model: String,
    /// Host route or registry key that selected this provider/model pair.
    pub route: String,
}

impl ResolvedModelRoute {
    /// Creates a resolved model-route identity.
    #[must_use]
    pub fn new(
        provider: impl Into<String>,
        model: impl Into<String>,
        route: impl Into<String>,
    ) -> Self {
        Self {
            provider: provider.into(),
            model: model.into(),
            route: route.into(),
        }
    }
}

/// A provider-neutral chat model request.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelRequest {
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Tool declarations exposed for this call.
    #[serde(default)]
    pub tools: Vec<ToolSchema>,
    /// Tool-choice policy.
    #[serde(default)]
    pub tool_choice: ToolChoice,
    /// Requested response format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
    /// Model id or registry alias override.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Host route requested for this call, distinct from a provider model id.
    ///
    /// This is observability metadata used to identify an actual fallback. It
    /// must not be inferred from [`Self::model`], because provider model ids
    /// and host route names occupy different namespaces.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_route: Option<String>,
    /// Ordered runtime model-selection hints carried without interpretation.
    #[serde(default)]
    pub model_hints: Vec<ModelHint>,
    /// Whether a consuming runtime may reuse its prior selected model.
    #[serde(default)]
    pub reuse_previous_model: bool,
    /// Sampling temperature.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Nucleus sampling probability mass, when supported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Maximum output tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Stop sequences that should terminate generation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
    /// Deterministic generation seed, when supported by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Per-call timeout in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Free-form request metadata.
    #[serde(default)]
    pub metadata: Value,
    /// Tags propagated to events and traces.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Declared prompt cache segments for KV-cache stability.
    #[serde(default)]
    pub cache_segments: Vec<PromptSegment>,
    /// Fingerprint of the stable prompt prefix, when computed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_fingerprint: Option<String>,
    /// Capabilities the resolved model must satisfy. Used to validate the
    /// request before a provider call and to filter resolution candidates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required_capabilities: Option<CapabilitySet>,
    /// Provider-specific options passed through untouched (for example OpenAI
    /// Responses API knobs, Anthropic thinking config, Ollama local `options`,
    /// or provider-specific controls such as `hotness`). Defaults to JSON null.
    #[serde(default)]
    pub provider_options: Value,
    /// Optional caching policy for this call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_policy: Option<CachePolicy>,
    /// Optional provider continuation/response id for stateful follow-ups.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuation_id: Option<String>,
    /// Provider-neutral reasoning configuration.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfig>,
    /// Optional stable run/model-call correlation supplied by the host.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation: Option<ModelCallCorrelation>,
}

/// A provider-neutral chat model response.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelResponse {
    /// The assistant message produced by the model.
    pub message: AssistantMessage,
    /// Token usage, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// Provider finish reason (for example `stop`, `tool_calls`, `length`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Raw provider metadata preserved for callers who need it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
    /// Durable model-selection metadata attached by a consuming runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_model: Option<ResolvedModel>,
    /// Runtime nudge indicating that this response does not end the turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continue_turn: Option<String>,
    /// Whether a consuming runtime served this response from local cache.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub served_from_cache: bool,
    /// Stable run/model-call correlation carried from the request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation: Option<ModelCallCorrelation>,
    /// Concrete provider/model/route identity for this completed call.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_route: Option<ResolvedModelRoute>,
}

/// Metadata shared by every item in one model stream.
///
/// Streaming transports generally emit deltas before a provider can produce a
/// terminal response. Keeping immutable call metadata on the stream makes the
/// correlation and resolved route available from the first delta through the
/// terminal item without repeating it in every chunk.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelStreamMetadata {
    /// Stable run/model-call correlation for every stream item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correlation: Option<ModelCallCorrelation>,
    /// Concrete provider/model/route identity for every stream item.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resolved_route: Option<ResolvedModelRoute>,
}

/// An incremental streamed chunk of a model response.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ModelDelta {
    /// Id of the model call this delta belongs to.
    pub call_id: String,
    /// Incremental text content.
    #[serde(default)]
    pub content: String,
    /// Incremental reasoning/thinking content, kept separate from visible text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,
    /// Incremental tool-call fragment, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<ToolDelta>,
}

/// Normalized provider failure details.
///
/// Provider adapters use this shape for HTTP failures, stream error events, and
/// terminal stream failures so callers can reason about errors without parsing
/// provider-specific JSON. The original provider payload can still be retained
/// in [`ProviderError::raw`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProviderError {
    /// Provider family identifier, for example `openai` or `ollama`.
    pub provider: String,
    /// Provider model id, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Transport status code, when the failure came from HTTP.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    /// Provider error code or type, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Human-readable error message.
    pub message: String,
    /// Whether retrying the same request may succeed.
    #[serde(default)]
    pub retryable: bool,
    /// Parsed provider Retry-After delay in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
    /// Raw provider payload, when available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
    /// The assistant message accumulated from stream items before the
    /// failure, when any content had arrived. Lets a caller keep (or discard)
    /// partial work instead of losing every block a mid-stream failure
    /// interrupted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub partial_message: Option<AssistantMessage>,
    /// The stop/finish reason reported before the failure, when the provider
    /// sent one (for example Anthropic's `message_delta.stop_reason`) prior
    /// to the error that ended the stream.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<String>,
}

/// The syntactic category of a streamed content block, established when the
/// block opens ([`ModelStreamItem::BlockStart`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum BlockKind {
    /// Visible assistant text.
    Text,
    /// Model reasoning/thinking content.
    Thinking,
    /// A tool call. The id and name are known as soon as the block opens
    /// (Anthropic's `content_block_start`; OpenAI's first `tool_calls[]`
    /// fragment for a wire index), before any argument fragments arrive.
    ToolCall {
        /// Provider-assigned call identifier.
        id: String,
        /// Tool name.
        name: String,
    },
}

/// An incremental fragment belonging to the open block named in the
/// accompanying [`ModelStreamItem::BlockDelta::index`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
pub enum BlockDelta {
    /// Visible text fragment.
    Text(String),
    /// Thinking/reasoning fragment.
    Thinking(String),
    /// Incremental tool-call argument JSON fragment.
    ToolArgs(String),
}

/// A single item produced by a real, asynchronous model stream.
///
/// A well-behaved stream begins with [`ModelStreamItem::Started`], emits zero or
/// more [`ModelStreamItem::MessageDelta`] / [`ModelStreamItem::ToolCallDelta`] /
/// [`ModelStreamItem::UsageDelta`] items as the provider produces output, and
/// terminates with exactly one of [`ModelStreamItem::Completed`] (carrying the
/// fully merged response), [`ModelStreamItem::Failed`] (carrying an error
/// message), or [`ModelStreamItem::ProviderFailed`] (carrying normalized
/// provider details). Providers that can build an authoritative final response — such as
/// the OpenAI adapter — emit it via [`ModelStreamItem::Completed`] so the
/// merged response preserves tool-call names and ids that individual deltas may
/// omit.
///
/// Use [`crate::model::StreamAccumulator`] (or the
/// [`crate::model::collect_model_stream`] helper) to fold a
/// stream of these items back into a [`ModelResponse`].
/// # Serialization
///
/// This enum is **adjacently tagged** (`{"type": …, "content": …}`) rather than
/// internally tagged. Several variants wrap a non-struct payload
/// ([`ModelStreamItem::Failed`] wraps a `String`); an internally tagged enum
/// cannot serialize those (serde errors, or silently corrupts scalar JSON into
/// `{}`), so adjacent tagging is required for every variant to round-trip.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type", content = "content")]
#[allow(
    clippy::large_enum_variant,
    reason = "the serialized stream contract keeps Completed inline; boxing it would be a needless allocation on every terminal response"
)]
pub enum ModelStreamItem {
    /// The stream has opened; no content has arrived yet.
    Started,
    /// An incremental message fragment (text and/or a tool-call fragment).
    ///
    /// Kept alongside [`ModelStreamItem::BlockDelta`] for backward
    /// compatibility: every fragment a block-aware adapter emits as a
    /// `BlockDelta` is also folded into a `MessageDelta` on the same channel
    /// (see [`crate::model::block_delta_to_message_delta`]), so consumers that
    /// only understand the flat delta shape keep working unchanged.
    MessageDelta(MessageDelta),
    /// An incremental tool-call argument fragment correlated by call id. When
    /// the provider reports block-indexed content, [`ToolDelta::content_index`]
    /// names the block this fragment belongs to.
    ToolCallDelta(ToolDelta),
    /// A usage update. Providers may send cumulative usage; the accumulator
    /// keeps the most recent value.
    UsageDelta(Usage),
    /// A new content block has opened at `index`. Anthropic's
    /// `content_block_start` maps to this 1:1; the OpenAI adapters derive it
    /// from a delta's shape changing (text starting, or a new `tool_calls[]`
    /// wire index appearing).
    BlockStart {
        /// Zero-based position of this block within the assistant message,
        /// stable for the life of the block.
        index: usize,
        /// The block's syntactic category.
        kind: BlockKind,
    },
    /// An incremental fragment for the open block at `index`.
    BlockDelta {
        /// Position of the block this fragment belongs to.
        index: usize,
        /// The fragment payload.
        delta: BlockDelta,
    },
    /// The block at `index` has closed; `block` is its fully assembled
    /// content, ready to append to an [`AssistantMessage::content`] in order.
    BlockEnd {
        /// Position of the closed block.
        index: usize,
        /// The finished content block.
        block: ContentBlock,
    },
    /// Terminal success: the fully merged response.
    Completed(ModelResponse),
    /// Terminal failure with a human-readable error message.
    Failed(String),
    /// Terminal failure with normalized provider details.
    ProviderFailed(ProviderError),
    /// Terminal deferral: the provider accepted the request but will finish
    /// it asynchronously (for example an OpenAI batch or background
    /// response). The caller polls or otherwise resolves the response later
    /// via [`ChatModel::fetch_deferred`].
    Deferred(DeferredHandle),
}

/// An opaque, provider-issued handle to a model call whose response is not
/// yet available (for example a queued batch job or a background response).
///
/// The handle is deliberately provider-neutral and serializable so a host can
/// persist it and resume polling after a process restart. `id` is the only
/// field callers must treat as meaningful to the provider; `kind` and
/// `metadata` are informational.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeferredHandle {
    /// Provider family identifier (for example `openai`).
    pub provider: String,
    /// Provider-issued identifier for the deferred call (batch id, response
    /// id, or similar).
    pub id: String,
    /// Provider-specific deferral kind (for example `"batch"` or
    /// `"background"`), when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    /// Additional provider-specific metadata needed to resolve the handle.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, Value>,
}

impl DeferredHandle {
    /// Creates a handle for `provider`/`id` with no kind or metadata.
    #[must_use]
    pub fn new(provider: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            provider: provider.into(),
            id: id.into(),
            kind: None,
            metadata: serde_json::Map::new(),
        }
    }
}

/// The current status of a previously deferred model call.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum DeferredStatus {
    /// Still queued or in progress; not yet ready.
    Pending,
    /// Finished successfully.
    Completed(Box<ModelResponse>),
    /// Finished with a failure.
    Failed(String),
}

/// A cancellation guard owned by a model stream.
///
/// Detached producer tasks should hand their [`tokio::task::AbortHandle`] to
/// the consumer stream. Dropping an unfinished stream then aborts the producer
/// instead of allowing a billed provider request to outlive its caller.
#[derive(Debug)]
pub struct AbortOnDrop {
    abort_handle: tokio::task::AbortHandle,
    armed: bool,
}

impl AbortOnDrop {
    /// Creates a guard for a producer's abort handle.
    #[must_use]
    pub fn new(abort_handle: tokio::task::AbortHandle) -> Self {
        Self {
            abort_handle,
            armed: true,
        }
    }

    /// Creates a guard from a spawned producer task.
    #[must_use]
    pub fn from_join_handle<T>(handle: &tokio::task::JoinHandle<T>) -> Self {
        Self::new(handle.abort_handle())
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.abort_handle.abort();
        }
    }
}

/// A pinned, boxed, `Send` stream of [`ModelStreamItem`]s plus call metadata.
///
/// This is the return type of [`ChatModel::stream`]. It is runtime-agnostic:
/// the caller's executor drives it. Producers may opt into
/// [`ModelStream::abort_on_drop`] when they run in a detached Tokio task.
pub struct ModelStream {
    inner: Pin<Box<dyn Stream<Item = ModelStreamItem> + Send>>,
    metadata: ModelStreamMetadata,
    abort_on_drop: Option<AbortOnDrop>,
}

impl std::fmt::Debug for ModelStream {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ModelStream")
            .field("metadata", &self.metadata)
            .field("abort_on_drop", &self.abort_on_drop.is_some())
            .finish_non_exhaustive()
    }
}

impl ModelStream {
    /// Wraps a stream with empty call metadata.
    #[must_use]
    pub fn new(inner: Pin<Box<dyn Stream<Item = ModelStreamItem> + Send>>) -> Self {
        Self {
            inner,
            metadata: ModelStreamMetadata::default(),
            abort_on_drop: None,
        }
    }

    /// Returns immutable metadata shared by all emitted stream items.
    #[must_use]
    pub fn metadata(&self) -> &ModelStreamMetadata {
        &self.metadata
    }

    /// Replaces metadata shared by all emitted stream items.
    ///
    /// A terminal [`ModelStreamItem::Completed`] inherits any absent
    /// correlation or resolved route from this metadata when it is polled.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ModelStreamMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the stable run/model-call correlation for this stream.
    ///
    /// Repeated calls replace the previous value. A terminal
    /// [`ModelStreamItem::Completed`] inherits the current value only when its
    /// provider response did not set one.
    #[must_use]
    pub fn with_correlation(mut self, correlation: ModelCallCorrelation) -> Self {
        self.metadata.correlation = Some(correlation);
        self
    }

    /// Sets the concrete provider/model/route identity for this stream.
    ///
    /// Repeated calls replace the previous value. A terminal
    /// [`ModelStreamItem::Completed`] inherits the current value only when its
    /// provider response did not set one.
    #[must_use]
    pub fn with_resolved_route(mut self, route: ResolvedModelRoute) -> Self {
        self.metadata.resolved_route = Some(route);
        self
    }

    /// Aborts a detached producer if this stream is dropped before completion.
    #[must_use]
    pub fn abort_on_drop(mut self, guard: AbortOnDrop) -> Self {
        self.abort_on_drop = Some(guard);
        self
    }

    /// Maps emitted items while retaining stream metadata and cancellation.
    #[must_use]
    pub fn map_items(
        self,
        mapper: impl FnMut(ModelStreamItem) -> ModelStreamItem + Send + 'static,
    ) -> Self {
        Self {
            inner: Box::pin(self.inner.map(mapper)),
            metadata: self.metadata,
            abort_on_drop: self.abort_on_drop,
        }
    }
}

impl Stream for ModelStream {
    type Item = ModelStreamItem;

    fn poll_next(
        mut self: Pin<&mut Self>,
        context: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let mut item = self.inner.as_mut().poll_next(context);
        if let std::task::Poll::Ready(Some(ModelStreamItem::Completed(response))) = &mut item {
            if response.correlation.is_none() {
                response.correlation.clone_from(&self.metadata.correlation);
            }
            if response.resolved_route.is_none() {
                response
                    .resolved_route
                    .clone_from(&self.metadata.resolved_route);
            }
        }
        if matches!(
            &item,
            std::task::Poll::Ready(Some(
                ModelStreamItem::Completed(_)
                    | ModelStreamItem::Failed(_)
                    | ModelStreamItem::ProviderFailed(_)
            ))
        ) && let Some(guard) = self.abort_on_drop.as_mut()
        {
            guard.disarm();
        }
        item
    }
}

/// A provider-neutral chat model.
///
/// Generic over the application `State`.
#[async_trait]
pub trait ChatModel<State: Send + Sync>: Send + Sync {
    /// Returns the model's capability [`ModelProfile`], when known.
    ///
    /// The default returns `None`; providers that know their capabilities
    /// should override this. The harness consults the profile to choose a
    /// structured-output strategy for [`ResponseFormat::Auto`] and to validate
    /// [`ModelRequest::required_capabilities`].
    fn profile(&self) -> Option<&ModelProfile> {
        None
    }

    /// Returns a stable, credential-safe identity for response-cache scoping.
    ///
    /// The default declines identity. Implementations must never include raw
    /// credentials in the returned value.
    fn cache_identity(&self) -> Option<String> {
        None
    }

    /// Invokes the model and returns a complete response.
    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse>;

    /// Streams the model response as a real asynchronous [`ModelStream`].
    ///
    /// The default implementation calls [`ChatModel::invoke`] and replays the
    /// complete response as three items: [`ModelStreamItem::Started`], a single
    /// [`ModelStreamItem::MessageDelta`] carrying the full text, and a terminal
    /// [`ModelStreamItem::Completed`] carrying the response. Providers that talk
    /// to a streaming endpoint (for example the OpenAI adapter) override this to
    /// emit incremental deltas as bytes arrive.
    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        let correlation = request.correlation.clone();
        let response = self.invoke(state, request).await?;
        let delta = MessageDelta {
            text: response.text(),
            reasoning: String::new(),
            tool_call: None,
        };
        let items = vec![
            ModelStreamItem::Started,
            ModelStreamItem::MessageDelta(delta),
            ModelStreamItem::Completed(response),
        ];
        let stream = ModelStream::new(Box::pin(futures::stream::iter(items)));
        Ok(match correlation {
            Some(correlation) => stream.with_correlation(correlation),
            None => stream,
        })
    }

    /// Resolves a previously issued [`DeferredHandle`] (see
    /// [`ModelStreamItem::Deferred`]), returning the current
    /// [`DeferredStatus`].
    ///
    /// The default implementation returns [`Error::Unsupported`]; only
    /// adapters that can actually issue deferred calls (for example an
    /// OpenAI batch/background adapter) should override this.
    async fn fetch_deferred(&self, _handle: &DeferredHandle) -> Result<DeferredStatus> {
        Err(crate::Error::Unsupported(
            "this model adapter does not support deferred calls".to_string(),
        ))
    }
}
