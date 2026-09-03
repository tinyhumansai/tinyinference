//! HTTP transport: `OpenAiModel` construction, provider presets, request
//! building, and the `ChatModel` impl (`invoke`/`stream`).
//!
//! Split out of `openai/mod.rs`; see that module's doc comment for the
//! full provider overview.

use super::responses;
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

/// How the provider expects the API credential to be sent on each request.
///
/// OpenAI-compatible endpoints diverge on auth: hosted OpenAI and most gateways
/// use `Bearer`, some providers use a bare `x-api-key`, Anthropic's compat
/// endpoint pairs `x-api-key` with an `anthropic-version` header, and a few need
/// an arbitrary custom header carrying the raw credential. Defaults to
/// [`AuthStyle::Bearer`].
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AuthStyle {
    /// No authentication header (e.g. a local Ollama server).
    None,
    /// `Authorization: Bearer <key>` — hosted OpenAI and most gateways.
    #[default]
    Bearer,
    /// `x-api-key: <key>` — used by some OpenAI-compatible providers.
    XApiKey,
    /// `x-api-key: <key>` + `anthropic-version: 2023-06-01` (Anthropic compat).
    Anthropic,
    /// A custom header name carrying the raw credential.
    Custom(String),
}

/// A [`ChatModel`] backed by the hosted OpenAI Chat Completions API.
///
/// Construct one with [`OpenAiModel::new`] (plus the `with_*` builders) or
/// [`OpenAiModel::from_env`]. The model holds a reusable [`reqwest::Client`] so
/// repeated calls share a connection pool.
pub struct OpenAiModel {
    /// Shared HTTP client.
    client: reqwest::Client,
    /// API credential; how it is sent is governed by [`Self::auth`].
    api_key: String,
    /// How `api_key` is attached to each request (default [`AuthStyle::Bearer`]).
    auth: AuthStyle,
    /// Extra static headers attached to every request (e.g. provider
    /// attribution headers). Applied after the auth header.
    extra_headers: Vec<(String, String)>,
    /// Model-id glob patterns (`*` wildcard) whose targets reject a `temperature`
    /// parameter; matching requests omit `temperature` entirely. Empty by default.
    temperature_unsupported: Vec<String>,
    /// When set, overrides the request's sampling temperature for every call
    /// (unless the model is in [`Self::temperature_unsupported`]).
    temperature_override: Option<f64>,
    /// When `true`, system messages are folded into the first user message and
    /// the `system` role is dropped — for OpenAI-compatible endpoints that reject
    /// a `system` role. `false` by default (system messages pass through).
    merge_system_into_user: bool,
    /// Whether the endpoint accepts a **named** `tool_choice`
    /// (`{"type":"function","function":{"name":…}}`). `true` by default. When
    /// `false`, a [`ToolChoice::Tool`] request is degraded to
    /// `tool_choice:"required"` with the `tools` array filtered to the named tool
    /// — some local runtimes (LM Studio, llama.cpp server) 400 on the object form.
    /// See [`Self::with_named_tool_choice`].
    named_tool_choice_supported: bool,
    /// Whether the endpoint accepts `response_format:{"type":"json_object"}`.
    /// `true` by default. When `false`, a [`ResponseFormat::JsonObject`] request is
    /// degraded to a permissive `json_schema` wire form — some local runtimes 400
    /// on `json_object`. See [`Self::with_json_object_format`].
    json_object_format_supported: bool,
    /// Default model id used when a request does not override it.
    model: String,
    /// Provider family identifier used in profiles and normalized errors.
    provider: String,
    /// API base URL (no trailing slash); `/chat/completions` is appended.
    base_url: String,
    /// Capability profile derived from the default model id, optionally adjusted
    /// by [`Self::with_native_tool_calling`] / [`Self::with_vision`].
    profile: ModelProfile,
    /// Provider-specific options baked onto every request (e.g. a local model's
    /// `{"options": {"num_ctx": 8192}}`). Merged under each request's own
    /// `provider_options`, which win on key conflicts. `Value::Null` by default
    /// (no baked options). See [`Self::with_default_provider_options`].
    default_provider_options: Value,
    /// When `true`, calls go to the OpenAI **Responses API** (`/v1/responses`)
    /// instead of Chat Completions. See [`Self::with_responses_api_primary`].
    responses_api_primary: bool,
    /// When `true` (Responses path only), omit `max_output_tokens` from the wire
    /// body — the OpenAI Codex OAuth backend rejects it. See
    /// [`Self::with_responses_omit_max_output_tokens`].
    responses_omit_max_output_tokens: bool,
    /// Static query parameters appended to every request URL (e.g. the Codex
    /// `client_version`). See [`Self::with_extra_query_param`].
    extra_query_params: Vec<(String, String)>,
    /// A `User-Agent` header override (e.g. the Codex CLI UA). `None` uses
    /// reqwest's default. See [`Self::with_user_agent`].
    user_agent: Option<String>,
    /// Inline `<think>…</think>` reasoning-tag extraction config. `Some` moves
    /// inline chain-of-thought onto the reasoning channel; `None` passes content
    /// through untouched. Until [`Self::with_reasoning_tag_extraction`] is
    /// called (`reasoning_tags_overridden == false`) this default only takes
    /// effect for non-hosted base URLs — see
    /// [`Self::effective_reasoning_tags`].
    reasoning_tags: Option<ReasoningTagExtraction>,
    /// Whether [`Self::with_reasoning_tag_extraction`] was called explicitly.
    /// When `false`, extraction auto-enables only for OpenAI-compatible
    /// endpoints (`base_url != DEFAULT_BASE_URL`): hosted OpenAI never emits
    /// inline `<think>` reasoning, and unconditional extraction would silently
    /// strip legitimate content that mentions a literal `<think>` tag.
    reasoning_tags_overridden: bool,
    local_runtime: Option<LocalRuntimeKind>,
    keep_alive: Option<String>,
    json_schema_strict: AtomicBool,
    native_tools_on_wire: AtomicBool,
}

impl std::fmt::Debug for OpenAiModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let extra_header_names: Vec<&str> = self
            .extra_headers
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        formatter
            .debug_struct("OpenAiModel")
            .field("auth", &self.auth)
            .field("extra_header_names", &extra_header_names)
            .field("model", &self.model)
            .field("provider", &self.provider)
            .field("base_url", &self.base_url)
            .field("profile", &self.profile)
            .field("responses_api_primary", &self.responses_api_primary)
            .finish_non_exhaustive()
    }
}

/// The auth headers `(name, value)` for a given [`AuthStyle`] + credential.
///
/// Pure (no request), so the header mapping is unit-testable without a network
/// round-trip. Applied by [`OpenAiModel::authorized`].
pub(super) fn auth_headers(auth: &AuthStyle, api_key: &str) -> Vec<(String, String)> {
    match auth {
        AuthStyle::None => Vec::new(),
        AuthStyle::Bearer => vec![("Authorization".to_string(), format!("Bearer {api_key}"))],
        AuthStyle::XApiKey => vec![("x-api-key".to_string(), api_key.to_string())],
        AuthStyle::Anthropic => vec![
            ("x-api-key".to_string(), api_key.to_string()),
            ("anthropic-version".to_string(), "2023-06-01".to_string()),
        ],
        AuthStyle::Custom(name) => vec![(name.clone(), api_key.to_string())],
    }
}

/// Case-insensitive glob match supporting the `*` wildcard (the only metacharacter
/// used by model-id patterns). `"o1*"` matches `"o1-mini"`; `"*turbo"` matches
/// `"gpt-4-turbo"`; a pattern with no `*` matches exactly.
pub(super) fn glob_match(pattern: &str, value: &str) -> bool {
    let pattern = pattern.to_ascii_lowercase();
    let value = value.to_ascii_lowercase();
    let segments: Vec<&str> = pattern.split('*').collect();
    if segments.len() == 1 {
        return pattern == value;
    }
    let mut cursor = 0usize;
    for (idx, segment) in segments.iter().enumerate() {
        if segment.is_empty() {
            continue;
        }
        if idx == 0 {
            // A non-empty leading segment must be a prefix.
            if !value[cursor..].starts_with(segment) {
                return false;
            }
            cursor += segment.len();
        } else if idx == segments.len() - 1 {
            // A non-empty trailing segment must be a suffix.
            return value[cursor..].ends_with(segment);
        } else {
            match value[cursor..].find(segment) {
                Some(offset) => cursor += offset + segment.len(),
                None => return false,
            }
        }
    }
    // Trailing `*` (empty last segment) matches the remainder.
    true
}

/// The `temperature` to send for `model`: `None` (omitted) when the model matches
/// a temperature-unsupported pattern, else the override if set, else the request's
/// temperature. Pure, so the policy is unit-testable without a request.
pub(super) fn effective_temperature(
    model: &str,
    request_temperature: Option<f64>,
    temperature_override: Option<f64>,
    temperature_unsupported: &[String],
) -> Option<f64> {
    if temperature_unsupported.iter().any(|p| glob_match(p, model)) {
        return None;
    }
    temperature_override.or(request_temperature)
}

/// Concatenates the text of a message's content blocks (non-text blocks — images,
/// json, thinking — are ignored). Used to gather system-prompt text for merging.
fn content_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Folds all `system` messages into the first `user` message and drops the
/// `system` role, for endpoints that reject a `system` role. The concatenated
/// system text is prefixed to the first user message (`"{system}\n\n{user}"` —
/// matching the common host behavior); image/other user content blocks are
/// preserved. When there is no user message, the system text is promoted to a
/// user turn. Pure, so the transform is unit-testable.
pub(super) fn merge_system_into_user(messages: &[Message]) -> Vec<Message> {
    use crate::message::UserMessage;

    let system_text = messages
        .iter()
        .filter_map(|m| match m {
            Message::System(s) => {
                let t = content_text(&s.content);
                (!t.is_empty()).then_some(t)
            }
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n");

    if system_text.is_empty() {
        // Nothing to fold; drop any (empty) system messages if present, else pass
        // through untouched.
        if messages.iter().any(|m| matches!(m, Message::System(_))) {
            return messages
                .iter()
                .filter(|m| !matches!(m, Message::System(_)))
                .cloned()
                .collect();
        }
        return messages.to_vec();
    }

    let mut merged: Vec<Message> = Vec::with_capacity(messages.len());
    let mut folded = false;
    for msg in messages {
        match msg {
            Message::System(_) => {} // dropped
            Message::User(user) if !folded => {
                folded = true;
                let mut content = Vec::with_capacity(user.content.len() + 1);
                match user.content.split_first() {
                    Some((ContentBlock::Text(first), rest)) => {
                        content.push(ContentBlock::Text(format!("{system_text}\n\n{first}")));
                        content.extend(rest.iter().cloned());
                    }
                    _ => {
                        content.push(ContentBlock::Text(format!("{system_text}\n\n")));
                        content.extend(user.content.iter().cloned());
                    }
                }
                merged.push(Message::User(UserMessage { content }));
            }
            other => merged.push(other.clone()),
        }
    }

    if !folded {
        // No user message to fold into — promote the system text to a user turn.
        merged.insert(0, Message::user(system_text));
    }

    merged
}

/// Returns `true` for OpenAI o-series reasoning models (`o1`/`o3`/`o4`), which
/// reject `max_tokens` and require `max_completion_tokens` instead.
pub(super) fn is_reasoning_model(model: &str) -> bool {
    let lower = model.to_ascii_lowercase();
    lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("gpt-5")
        || lower.starts_with("gpt5")
        || lower.contains("/gpt-5")
}

/// Derives a static [`ModelProfile`] for an OpenAI(-compatible) model id.
///
/// All targets support tool calling, streaming (including tool-call chunks),
/// and JSON Schema response formats. Modern OpenAI-family models additionally
/// advertise native structured output and (for the o-series) reasoning output.
///
/// The context window is populated from the provider-neutral
/// [`context_window_for_model_id`][crate::model::context_window_for_model_id]
/// hint table when the id is recognized (`None` otherwise), so downstream
/// context-window-aware consumers can use a real window instead of silently
/// falling back to a fixed threshold.
pub(super) fn derive_profile(provider: &str, model: &str) -> ModelProfile {
    let lower = model.to_ascii_lowercase();
    let reasoning = is_reasoning_model(model);
    let native_structured = lower.contains("gpt-4o") || lower.contains("gpt-4.1") || reasoning;
    ModelProfile {
        provider: Some(provider.to_string()),
        model: Some(model.to_string()),
        status: ModelStatus::Stable,
        modalities: Modalities {
            image_in: true,
            ..Modalities::default()
        },
        tool_calling: true,
        parallel_tool_calls: true,
        streaming: true,
        streaming_tool_chunks: true,
        native_structured_output: native_structured,
        json_schema: true,
        reasoning,
        reasoning_effort: reasoning,
        max_input_tokens: crate::model::context_window_for_model_id(model),
        ..ModelProfile::default()
    }
}

impl OpenAiModel {
    /// Creates a model with the given API key, the default model
    /// (`gpt-4.1-mini`), and the default base URL (`https://api.openai.com/v1`).
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
                .build()
                .expect("default reqwest client builds"),
            api_key: api_key.into(),
            auth: AuthStyle::Bearer,
            extra_headers: Vec::new(),
            temperature_unsupported: Vec::new(),
            temperature_override: None,
            merge_system_into_user: false,
            named_tool_choice_supported: true,
            json_object_format_supported: true,
            model: DEFAULT_MODEL.to_string(),
            provider: "openai".to_string(),
            base_url: DEFAULT_BASE_URL.to_string(),
            profile: derive_profile("openai", DEFAULT_MODEL),
            default_provider_options: Value::Null,
            responses_api_primary: false,
            responses_omit_max_output_tokens: false,
            extra_query_params: Vec::new(),
            user_agent: None,
            // Inline `<think>` extraction defaults ON, but until overridden it
            // only takes effect for non-hosted base URLs (see
            // `effective_reasoning_tags`): unhandled leakage is the common
            // local-model failure, while hosted OpenAI never emits inline
            // `<think>` and must not strip literal mentions of the tag.
            reasoning_tags: Some(ReasoningTagExtraction::default()),
            reasoning_tags_overridden: false,
            local_runtime: None,
            keep_alive: None,
            json_schema_strict: AtomicBool::new(true),
            native_tools_on_wire: AtomicBool::new(true),
        }
    }

    /// Routes calls to the OpenAI **Responses API** (`/v1/responses`) instead of
    /// Chat Completions. Required for the OpenAI Codex OAuth backend; pair with
    /// [`with_extra_query_param`](Self::with_extra_query_param) +
    /// [`with_user_agent`](Self::with_user_agent) for Codex.
    pub fn with_responses_api_primary(mut self) -> Self {
        self.responses_api_primary = true;
        self
    }

    /// Omits `max_output_tokens` from Responses requests (the Codex OAuth backend
    /// rejects it). No effect on the Chat Completions path.
    pub fn with_responses_omit_max_output_tokens(mut self) -> Self {
        self.responses_omit_max_output_tokens = true;
        self
    }

    /// Appends a static query parameter to every request URL (repeatable) — e.g.
    /// the Codex `client_version`.
    pub fn with_extra_query_param(
        mut self,
        name: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.extra_query_params.push((name.into(), value.into()));
        self
    }

    /// Overrides the `User-Agent` header sent on every request (e.g. the Codex
    /// CLI user agent).
    pub fn with_user_agent(mut self, user_agent: impl Into<String>) -> Self {
        self.user_agent = Some(user_agent.into());
        self
    }

    /// Overrides how the API credential is sent (default [`AuthStyle::Bearer`]).
    ///
    /// Use this for OpenAI-compatible endpoints that authenticate with
    /// `x-api-key`, the Anthropic header pair, or a custom header instead of a
    /// bearer token.
    pub fn with_auth_style(mut self, auth: AuthStyle) -> Self {
        self.auth = auth;
        self
    }

    /// Attaches a static header to every request (repeatable). Applied after the
    /// auth header — e.g. provider attribution headers.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.push((name.into(), value.into()));
        self
    }

    /// Sets model-id glob patterns (`*` wildcard, case-insensitive) whose targets
    /// reject a `temperature` parameter. A request whose model matches any pattern
    /// omits `temperature` from the wire body (e.g. OpenAI o-series / some hosted
    /// reasoning models that 400 on an explicit temperature).
    pub fn with_temperature_unsupported_models(
        mut self,
        patterns: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        self.temperature_unsupported = patterns.into_iter().map(Into::into).collect();
        self
    }

    /// Overrides the sampling temperature for every request (unless the model is
    /// temperature-unsupported). Useful for endpoints that require a fixed
    /// temperature regardless of the caller's request.
    pub fn with_temperature_override(mut self, temperature: Option<f64>) -> Self {
        self.temperature_override = temperature;
        self
    }

    /// Folds system messages into the first user message and drops the `system`
    /// role, for OpenAI-compatible endpoints that reject a `system` role.
    pub fn with_merge_system_into_user(mut self) -> Self {
        self.merge_system_into_user = true;
        self
    }

    /// Declares whether the endpoint accepts a **named** `tool_choice`
    /// (`{"type":"function","function":{"name":…}}`). `true` by default.
    ///
    /// Pass `false` for local OpenAI-compatible runtimes (LM Studio, llama.cpp
    /// server, and others) that only accept the string forms `none`/`auto`/
    /// `required` and 400 on the object form. A [`ToolChoice::Tool`] request is
    /// then degraded to `tool_choice:"required"` with the wire `tools` array
    /// filtered down to just the named tool, preserving the "must call *this*
    /// tool" semantics. Independent of this flag, a 400 whose body implicates
    /// `tool_choice` triggers the same degraded retry automatically (once).
    pub fn with_named_tool_choice(mut self, supported: bool) -> Self {
        self.named_tool_choice_supported = supported;
        self
    }

    /// Declares whether the endpoint accepts
    /// `response_format:{"type":"json_object"}`. `true` by default.
    ///
    /// Pass `false` for local OpenAI-compatible runtimes that only accept
    /// `json_schema`/`text` and 400 on `json_object`. A
    /// [`ResponseFormat::JsonObject`] request is then degraded to a permissive
    /// `json_schema` wire form (an empty object schema with `strict:false`).
    /// Independent of this flag, a 400 whose body implicates `response_format`
    /// triggers the same degraded retry automatically (once).
    pub fn with_json_object_format(mut self, supported: bool) -> Self {
        self.json_object_format_supported = supported;
        self
    }

    /// Configures inline `<think>…</think>` reasoning-tag extraction for
    /// OpenAI-compatible reasoning models that embed chain-of-thought in the
    /// visible `content` string (qwen3, deepseek-r1 distills via Ollama `/v1`,
    /// LM Studio, llama.cpp) rather than on the `reasoning_content` /
    /// `reasoning` side-channel.
    ///
    /// Until this method is called, extraction defaults ON with the plain
    /// `think` tag for **non-hosted base URLs only** (hosted OpenAI never emits
    /// inline `<think>` reasoning, and extraction there would silently strip
    /// legitimate content that mentions a literal tag). Pass `None` to disable
    /// it everywhere (content passes through verbatim), or `Some(config)` to
    /// force it on — including for the hosted base URL — and customize the tag
    /// name, separator, or DeepSeek-R1 template mode via
    /// [`ReasoningTagExtraction`]. Extracted reasoning surfaces as a leading
    /// [`ContentBlock::Thinking`](crate::message::ContentBlock::Thinking)
    /// block on both the unary and streamed paths, consistent with the
    /// side-channel normalization.
    pub fn with_reasoning_tag_extraction(mut self, config: Option<ReasoningTagExtraction>) -> Self {
        self.reasoning_tags = config;
        self.reasoning_tags_overridden = true;
        self
    }

    /// The reasoning-tag extraction config in effect for a call: the explicit
    /// [`Self::with_reasoning_tag_extraction`] override when one was given,
    /// otherwise the built-in `think` default gated to OpenAI-compatible
    /// endpoints (`base_url != DEFAULT_BASE_URL`).
    pub(super) fn effective_reasoning_tags(&self) -> Option<&ReasoningTagExtraction> {
        if !self.reasoning_tags_overridden && self.base_url == DEFAULT_BASE_URL {
            return None;
        }
        self.reasoning_tags.as_ref()
    }

    /// Overrides whether this model advertises **native** tool calling on its
    /// [`profile`](ChatModel::profile). Many self-hosted / local OpenAI-compatible
    /// runtimes (Ollama and others) reject the OpenAI `tools` parameter with an
    /// HTTP 400; passing `false` lets a harness detect that and embed tool specs
    /// in the prompt instead of sending native `tools`. Disabling native tools
    /// also clears `parallel_tool_calls` and `streaming_tool_chunks`, which are
    /// meaningless without them.
    ///
    /// This mutates the derived profile, so apply it **after**
    /// [`with_model`](Self::with_model) / [`with_provider`](Self::with_provider)
    /// (which re-derive the profile).
    pub fn with_native_tool_calling(mut self, enabled: bool) -> Self {
        self.profile.tool_calling = enabled;
        if !enabled {
            self.profile.parallel_tool_calls = false;
            self.profile.streaming_tool_chunks = false;
        }
        self
    }

    /// Overrides whether this model advertises image input (vision) on its
    /// [`profile`](ChatModel::profile). The OpenAI wire preset defaults to `true`;
    /// pass `false` for text-only local/self-hosted models so a harness does not
    /// route image content to an endpoint that cannot accept it.
    ///
    /// This mutates the derived profile, so apply it **after**
    /// [`with_model`](Self::with_model) / [`with_provider`](Self::with_provider).
    pub fn with_vision(mut self, enabled: bool) -> Self {
        self.profile.modalities.image_in = enabled;
        self
    }

    /// Bakes provider-specific options onto every request (e.g. a local model's
    /// `{"options": {"num_ctx": 8192}}`). These are merged **under** each
    /// request's own [`ModelRequest::provider_options`], so a per-call option of
    /// the same key wins. Reserved OpenAI fields are still stripped downstream by
    /// `provider_extra_options`. Passing `Value::Null` clears the baked options.
    pub fn with_default_provider_options(mut self, options: Value) -> Self {
        self.default_provider_options = options;
        self
    }

    /// Overrides the default model id.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.profile = derive_profile(&self.provider, &self.model);
        self
    }

    /// Overrides the provider family id used in profiles and normalized errors.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = provider.into();
        self.profile = derive_profile(&self.provider, &self.model);
        self
    }

    /// Overrides the API base URL. A trailing slash is trimmed so the joined
    /// endpoint is always `{base_url}/chat/completions`.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Builds a model from environment variables.
    ///
    /// Reads `OPENAI_API_KEY` (required), `OPENAI_MODEL` (optional, defaults to
    /// `gpt-4.1-mini`), and `OPENAI_BASE_URL` (optional, defaults to
    /// `https://api.openai.com/v1`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Validation`] when `OPENAI_API_KEY` is missing
    /// or empty.
    pub fn from_env() -> Result<Self> {
        let api_key = Self::api_key_from_env("OPENAI_API_KEY")?;

        let mut model = Self::new(api_key);
        if let Ok(name) = std::env::var("OPENAI_MODEL")
            && !name.trim().is_empty()
        {
            model = model.with_model(name);
        }
        if let Ok(url) = std::env::var("OPENAI_BASE_URL")
            && !url.trim().is_empty()
        {
            model = model.with_base_url(url);
        }
        Ok(model)
    }

    pub(super) fn api_key_from_env(name: &str) -> Result<String> {
        std::env::var(name)
            .ok()
            .filter(|k| !k.trim().is_empty())
            .ok_or_else(|| Error::Validation(format!("{name} is not set or is empty")))
    }

    /// Builds an OpenAI-compatible model from a provider spec and explicit API
    /// key.
    pub fn from_spec(spec: ProviderSpec, api_key: impl Into<String>) -> Result<Self> {
        if spec.model.trim().is_empty() {
            return Err(Error::Validation(
                "provider spec model must not be empty".to_string(),
            ));
        }
        if spec.base_url.trim().is_empty() {
            return Err(Error::Validation(
                "provider spec base_url must not be empty".to_string(),
            ));
        }
        let kind = match spec.kind {
            crate::providers::ProviderKind::Ollama => Some(LocalRuntimeKind::Ollama),
            crate::providers::ProviderKind::LmStudio => Some(LocalRuntimeKind::LmStudio),
            crate::providers::ProviderKind::LlamaCpp => Some(LocalRuntimeKind::LlamaCpp),
            crate::providers::ProviderKind::Vllm => Some(LocalRuntimeKind::Vllm),
            _ => None,
        };
        if let Some(kind) = kind {
            return Self::local_runtime(kind, &spec.provider, spec.base_url, api_key, spec.model);
        }
        Ok(Self::compatible_provider(
            spec.provider,
            api_key,
            spec.base_url,
            spec.model,
        ))
    }

    /// Builds an OpenAI-compatible model from a provider spec, reading the API
    /// key from the spec's environment variable when required.
    pub fn from_spec_env(spec: ProviderSpec) -> Result<Self> {
        let api_key = if spec.requires_api_key {
            let env = spec.api_key_env.as_deref().ok_or_else(|| {
                Error::Validation(format!(
                    "{} requires an api_key_env in ProviderSpec",
                    spec.provider
                ))
            })?;
            std::env::var(env)
                .ok()
                .filter(|k| !k.trim().is_empty())
                .ok_or_else(|| {
                    Error::Validation(format!(
                        "{env} is not set; export it or provide an explicit API key"
                    ))
                })?
        } else {
            "local".to_string()
        };
        Self::from_spec(spec, api_key)
    }

    /// Lists the models the provider advertises via `GET {base_url}/models`.
    ///
    /// This is a provider/account-level capability (independent of which model
    /// this handle is bound to), so it uses the same credentials and base URL as
    /// chat calls. Every OpenAI-compatible endpoint (Ollama, Together, Groq,
    /// OpenRouter, …) serves the same shape, so this doubles as runtime model
    /// discovery for local/self-hosted providers. Returned ids can be fed to
    /// [`with_model`](Self::with_model).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Model`] on transport failure or a non-2xx
    /// status, and [`Error::Serialization`] when the body cannot be
    /// decoded.
    pub async fn list_models(&self) -> Result<Vec<ModelListing>> {
        let url = format!("{}/models", self.base_url);

        let response = self
            .send_checked(self.authorized(self.client.get(&url)), "request", &url)
            .await?;

        let text = response
            .text()
            .await
            .map_err(|e| Error::Model(format!("openai response body read failed: {e}")))?;

        let listing: ModelListWire = serde_json::from_str(&text)?;
        Ok(listing.data)
    }

    // -----------------------------------------------------------------------
    // OpenAI-compatible provider presets
    //
    // DeepSeek, Groq, xAI, OpenRouter, Together, Mistral, Ollama, and
    // Anthropic's compatibility endpoint all accept the same Chat Completions
    // wire format, so the same [`OpenAiModel`] talks to all of them — only the
    // base URL and default model differ. Each preset is a thin wrapper over
    // [`OpenAiModel::new`] + [`with_base_url`][Self::with_base_url] +
    // [`with_model`][Self::with_model]; override the model with `with_model`.
    // -----------------------------------------------------------------------

    /// Points at an arbitrary OpenAI-compatible endpoint with an explicit base
    /// URL and default model.
    ///
    /// Use this for any provider that implements the Chat Completions API but is
    /// not covered by a named preset below.
    pub fn compatible(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::new(api_key).with_base_url(base_url).with_model(model)
    }

    /// Points at an arbitrary OpenAI-compatible endpoint with an explicit
    /// provider id, base URL, and default model.
    pub fn compatible_provider(
        provider: impl Into<String>,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::new(api_key)
            .with_provider(provider)
            .with_base_url(base_url)
            .with_model(model)
    }

    /// DeepSeek (`https://api.deepseek.com/v1`), default model `deepseek-chat`.
    pub fn deepseek(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "deepseek",
            api_key,
            "https://api.deepseek.com/v1",
            "deepseek-chat",
        )
    }

    /// Anthropic's OpenAI-compatible endpoint (`https://api.anthropic.com/v1`),
    /// default model `claude-3-5-sonnet-latest`.
    pub fn anthropic(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "anthropic",
            api_key,
            "https://api.anthropic.com/v1",
            "claude-3-5-sonnet-latest",
        )
    }

    /// Groq (`https://api.groq.com/openai/v1`), default model
    /// `llama-3.3-70b-versatile`.
    pub fn groq(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "groq",
            api_key,
            "https://api.groq.com/openai/v1",
            "llama-3.3-70b-versatile",
        )
    }

    /// xAI (`https://api.x.ai/v1`), default model `grok-2-latest`.
    pub fn xai(api_key: impl Into<String>) -> Self {
        Self::compatible_provider("xai", api_key, "https://api.x.ai/v1", "grok-2-latest")
    }

    /// OpenRouter (`https://openrouter.ai/api/v1`), default model
    /// `openai/gpt-4o-mini`.
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "openrouter",
            api_key,
            "https://openrouter.ai/api/v1",
            "openai/gpt-4o-mini",
        )
    }

    /// Together AI (`https://api.together.xyz/v1`), default model
    /// `meta-llama/Llama-3.3-70B-Instruct-Turbo`.
    pub fn together(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "together",
            api_key,
            "https://api.together.xyz/v1",
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        )
    }

    /// Mistral (`https://api.mistral.ai/v1`), default model
    /// `mistral-small-latest`.
    pub fn mistral(api_key: impl Into<String>) -> Self {
        Self::compatible_provider(
            "mistral",
            api_key,
            "https://api.mistral.ai/v1",
            "mistral-small-latest",
        )
    }

    /// A local Ollama server (`http://localhost:11434/v1`), default model
    /// `llama3.2`. Ollama ignores the API key, so a placeholder is used.
    pub fn ollama() -> Self {
        Self::ollama_at("http://localhost:11434", "llama3.2")
            .expect("the built-in Ollama URL is valid")
    }

    /// Constructs an Ollama model at a custom server root.
    pub fn ollama_at(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Self::local_runtime(LocalRuntimeKind::Ollama, "ollama", base_url, "", model)
    }

    /// Constructs a llama.cpp model at a custom server root.
    pub fn llama_cpp(base_url: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        Self::local_runtime(LocalRuntimeKind::LlamaCpp, "llama_cpp", base_url, "", model)
    }

    /// Constructs a vLLM model at a custom server root.
    pub fn vllm(
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        Self::local_runtime(LocalRuntimeKind::Vllm, "vllm", base_url, api_key, model)
    }

    fn local_runtime(
        kind: LocalRuntimeKind,
        provider: &str,
        base_url: impl Into<String>,
        api_key: impl Into<String>,
        model: impl Into<String>,
    ) -> Result<Self> {
        let base_url = normalize_local_v1_base_url(base_url.into(), kind.default_root())?;
        let mut output = Self::compatible_provider(provider, api_key, base_url, model)
            .with_auth_style(AuthStyle::None)
            .with_vision(false)
            .with_named_tool_choice(false)
            .with_json_object_format(false);
        output.local_runtime = Some(kind);
        output.json_schema_strict.store(false, Ordering::Relaxed);
        output.profile.max_input_tokens = None;
        Ok(output)
    }

    /// Sets native Ollama context options and the advertised context window.
    pub fn with_local_num_ctx(mut self, num_ctx: u32) -> Self {
        self.default_provider_options = merge_provider_options(
            &self.default_provider_options,
            &json!({"options": {"num_ctx": num_ctx}}),
        );
        self.profile.max_input_tokens = Some(u64::from(num_ctx));
        self
    }

    /// Sets the native local-runtime model residency hint.
    pub fn with_keep_alive(mut self, keep_alive: impl Into<String>) -> Self {
        self.keep_alive = Some(keep_alive.into());
        self
    }

    /// Returns the configured local runtime kind.
    pub fn local_runtime_kind(&self) -> Option<LocalRuntimeKind> {
        self.local_runtime
    }

    /// Probes a local runtime for its loaded model profile.
    pub async fn probe_local_profile(&self) -> Result<LocalProbe> {
        let Some(kind) = self.local_runtime else {
            return Err(Error::Validation(format!(
                "probe_local_profile is only meaningful for a local runtime; `{}` at {} is not one",
                self.provider, self.base_url
            )));
        };
        let root = kind.native_root(&self.base_url);
        let (endpoint, builder) = match kind {
            LocalRuntimeKind::Ollama => {
                let endpoint = format!("{root}/api/show");
                let builder = self
                    .authorized(self.client.post(&endpoint))
                    .json(&ollama_show_body(&self.model));
                (endpoint, builder)
            }
            LocalRuntimeKind::LmStudio => {
                let endpoint = format!("{root}/api/v0/models");
                let builder = self.authorized(self.client.get(&endpoint));
                (endpoint, builder)
            }
            LocalRuntimeKind::LlamaCpp | LocalRuntimeKind::Vllm => return Ok(LocalProbe::default()),
        };
        let response = builder
            .timeout(PROBE_TIMEOUT)
            .send()
            .await
            .map_err(|error| probe_error(&endpoint, error))?;
        if !response.status().is_success() {
            return Ok(LocalProbe::default());
        }
        let body: Value = response
            .json()
            .await
            .map_err(|error| probe_error(&endpoint, error))?;
        Ok(match kind {
            LocalRuntimeKind::Ollama => probe_from_ollama_show(&body),
            LocalRuntimeKind::LmStudio => probe_from_lm_studio_models(&body, &self.model),
            _ => LocalProbe::default(),
        })
    }

    /// Applies a probed local profile.
    pub fn apply_local_probe(mut self, probe: &LocalProbe) -> Self {
        if let Some(window) = probe.effective_context_window() {
            self.profile.max_input_tokens = Some(window);
        }
        if let Some(value) = probe.tool_calling {
            self.profile.tool_calling = value;
            self.profile.parallel_tool_calls = value;
            self.profile.streaming_tool_chunks = value;
        }
        if let Some(value) = probe.vision {
            self.profile.modalities.image_in = value;
        }
        if let Some(value) = probe.reasoning {
            self.profile.reasoning = value;
        }
        self
    }

    /// Probes and applies local capabilities.
    pub async fn probed(self) -> Result<Self> {
        let probe = self.probe_local_profile().await?;
        Ok(self.apply_local_probe(&probe))
    }

    /// Validates that the configured model is advertised by the provider.
    pub async fn validate_model(&self) -> Result<()> {
        let listed = self.list_models().await?;
        if listed.is_empty() || listed.iter().any(|entry| entry.id == self.model) {
            return Ok(());
        }
        let mut available: Vec<&str> = listed.iter().map(|entry| entry.id.as_str()).collect();
        available.sort_unstable();
        let remediation = if self.local_runtime == Some(LocalRuntimeKind::Ollama) {
            format!(" Run `ollama pull {}` to install it.", self.model)
        } else {
            String::new()
        };
        Err(Error::Validation(format!(
            "{} at {} does not serve model `{}`.{} Available: {}",
            self.provider,
            self.base_url,
            self.model,
            remediation,
            available.join(", ")
        )))
    }

    /// Loads an Ollama model through its native warm-up API.
    pub async fn warm_up(&self) -> Result<()> {
        let Some(kind) = self.local_runtime.filter(|kind| kind.has_native_api()) else {
            return Ok(());
        };
        let url = format!("{}/api/chat", kind.native_root(&self.base_url));
        let body = ollama_load_body(
            &self.model,
            local_options_object(&self.default_provider_options),
            self.keep_alive.as_deref(),
        );
        self.authorized(self.client.post(&url))
            .json(&body)
            .send()
            .await
            .map_err(|error| Error::Model(format!("openai warm-up of {url} failed: {error}")))?;
        Ok(())
    }

    /// Returns the default model id this instance will request.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns the configured provider family id.
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// Returns the configured API base URL.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// The baseline request-shape degradations to apply for this instance,
    /// derived from its capability knobs. A `true` field means "degrade this
    /// shape on the wire".
    pub(super) fn baseline_degrade(&self) -> Degrade {
        Degrade {
            named_tool_choice: !self.named_tool_choice_supported,
            json_object: !self.json_object_format_supported,
            json_schema_strict: !self.json_schema_strict.load(Ordering::Relaxed),
            native_tools: !self.native_tools_on_wire.load(Ordering::Relaxed),
        }
    }

    /// Translates a provider-neutral [`ModelRequest`] into the OpenAI wire
    /// request body, applying this instance's baseline request-shape
    /// degradations (see [`Self::baseline_degrade`]).
    ///
    /// Test-only: production paths go through [`Self::build_chat_body`] /
    /// [`Self::post_chat_with_degrade`], which thread an explicit [`Degrade`].
    #[cfg(test)]
    pub(super) fn translate_request(
        &self,
        request: &ModelRequest,
    ) -> Result<ChatCompletionRequest> {
        self.translate_request_with(request, self.baseline_degrade())
    }

    /// Translates a provider-neutral [`ModelRequest`] into the OpenAI wire
    /// request body, applying the given request-shape `degrade`. The per-request
    /// `model` override wins over the instance default.
    pub(super) fn translate_request_with(
        &self,
        request: &ModelRequest,
        degrade: Degrade,
    ) -> Result<ChatCompletionRequest> {
        // Prompt-guided tools: a model without native tool calling that is still
        // handed tools gets the tool protocol embedded in its system prompt and no
        // native `tools` on the wire (many local runtimes 400 on `tools`). The
        // model's `<tool_call>` blocks are parsed back in [`Self::invoke`]/stream.
        let prompt_guided_tools = (!self.profile.tool_calling || degrade.native_tools)
            && !request.tools.is_empty()
            && request.tool_choice != ToolChoice::None;
        let prompt_tool_schemas = match &request.tool_choice {
            ToolChoice::Tool(name) => request
                .tools
                .iter()
                .filter(|tool| tool.name == *name)
                .cloned()
                .collect::<Vec<_>>(),
            _ => request.tools.clone(),
        };
        let instructed_messages;
        let base_messages: &[Message] = if prompt_guided_tools {
            instructed_messages = prompt_tools::with_tool_instructions(
                &request.messages,
                &prompt_tool_schemas,
                &request.tool_choice,
            );
            &instructed_messages
        } else {
            &request.messages
        };

        // Optionally fold system messages into the first user turn (for endpoints
        // that reject a `system` role) before wire translation.
        let merged_messages;
        let source_messages: &[Message] = if self.merge_system_into_user {
            merged_messages = merge_system_into_user(base_messages);
            &merged_messages
        } else {
            base_messages
        };
        let messages = source_messages
            .iter()
            .map(translate_message)
            .collect::<Result<Vec<_>>>()?;

        let mut tools: Vec<ToolWire> = if prompt_guided_tools {
            Vec::new()
        } else {
            request
                .tools
                .iter()
                .map(|schema| ToolWire {
                    kind: "function".to_string(),
                    function: FunctionSchemaWire {
                        name: schema.name.clone(),
                        description: schema.description.clone(),
                        parameters: schema.parameters.clone(),
                    },
                })
                .collect()
        };

        // tool_choice is only meaningful when tools are declared.
        let tool_choice = if tools.is_empty() {
            None
        } else if let (true, ToolChoice::Tool(name)) =
            (degrade.named_tool_choice, &request.tool_choice)
        {
            // The endpoint rejects a named `tool_choice` object. Preserve the
            // "must call *this* tool" semantics by sending `"required"` and, when
            // the named tool is actually declared, filtering the wire `tools`
            // down to just it so the model has no other tool to pick. If the named
            // tool is absent, leave `tools` intact (mirrors the un-degraded path,
            // which would also send an unmatched name) and still send "required".
            if tools.iter().any(|t| t.function.name == *name) {
                tools.retain(|t| t.function.name == *name);
            }
            Some(json!("required"))
        } else {
            Some(translate_tool_choice(&request.tool_choice))
        };

        let mut response_format = request.response_format.as_ref().and_then(|format| {
            if degrade.json_object && matches!(format, ResponseFormat::JsonObject) {
                // The endpoint rejects `{"type":"json_object"}`; use a permissive
                // `json_schema` that still guarantees a JSON object.
                Some(degraded_json_object_format())
            } else {
                translate_response_format(format)
            }
        });
        if let Some(format) = response_format.as_mut()
            && let Some(json_schema) = format.get_mut("json_schema")
        {
            let strict = !degrade.json_schema_strict && !degrade.json_object;
            json_schema["strict"] = json!(strict);
            if strict && let Some(schema) = json_schema.get_mut("schema") {
                make_schema_strict(schema);
            }
        }

        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        // The o-series reasoning models reject `max_tokens` and require
        // `max_completion_tokens`; classic Chat Completions models use
        // `max_tokens`. Route the request's cap to whichever field the target
        // model accepts.
        let (max_tokens, max_completion_tokens) = if is_reasoning_model(&model) {
            (None, request.max_tokens)
        } else {
            (request.max_tokens, None)
        };

        let temperature = effective_temperature(
            &model,
            request.temperature,
            self.temperature_override,
            &self.temperature_unsupported,
        );
        let merged_provider_options =
            merge_provider_options(&self.default_provider_options, &request.provider_options);
        let reasoning_effort = if merged_provider_options
            .get("reasoning_effort")
            .is_some_and(|value| !value.is_null())
        {
            None
        } else {
            request
                .reasoning
                .as_ref()
                .and_then(|reasoning| reasoning.effort)
        };

        Ok(ChatCompletionRequest {
            model: model.clone(),
            messages,
            tools,
            tool_choice,
            response_format,
            temperature,
            top_p: request.top_p,
            max_tokens,
            max_completion_tokens,
            reasoning_effort,
            stop: request.stop_sequences.clone(),
            seed: request.seed,
            stream: false,
            stream_options: None,
            extra: provider_extra_options(&merged_provider_options)?,
        })
    }

    /// Attaches the provider's credential (per [`Self::auth`]) plus any static
    /// [`Self::extra_headers`] to an outbound request.
    ///
    /// The single place auth is applied, shared by the chat and model-listing
    /// calls. The header mapping itself lives in the pure [`auth_headers`] helper
    /// so it is unit-testable without a network round-trip.
    fn authorized(&self, mut builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        for (name, value) in auth_headers(&self.auth, &self.api_key) {
            builder = builder.header(name, value);
        }
        for (name, value) in &self.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        if let Some(user_agent) = &self.user_agent {
            builder = builder.header(reqwest::header::USER_AGENT, user_agent.as_str());
        }
        if !self.extra_query_params.is_empty() {
            builder = builder.query(&self.extra_query_params);
        }
        builder
    }

    /// The `/responses` endpoint URL — a sibling of `/chat/completions` under the
    /// same base URL, tolerating a base that already ends in `/responses` or a
    /// `…/v1` chat base.
    fn responses_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        if base.ends_with("/responses") {
            base.to_string()
        } else {
            format!("{base}/responses")
        }
    }

    /// Builds the `/v1/responses` request body from a provider-neutral request.
    pub(super) fn translate_responses_request(
        &self,
        request: &ModelRequest,
    ) -> Result<responses::ResponsesRequest> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let (instructions, input) = responses::build_responses_input(&request.messages);
        let max_output_tokens = if self.responses_omit_max_output_tokens {
            None
        } else {
            request.max_tokens
        };
        let provider_options =
            merge_provider_options(&self.default_provider_options, &request.provider_options);
        let extra = provider_extra_options(&provider_options)?;
        let reasoning = if extra.contains_key("reasoning") {
            None
        } else {
            request
                .reasoning
                .as_ref()
                .and_then(responses::translate_reasoning)
        };
        let include = if reasoning.is_some() {
            vec![responses::INCLUDE_ENCRYPTED_REASONING.to_string()]
        } else {
            Vec::new()
        };
        let tools: Vec<Value> = if self.native_tools_on_wire.load(Ordering::Relaxed) {
            request
                .tools
                .iter()
                .map(responses::translate_tool)
                .collect()
        } else {
            Vec::new()
        };
        let tool_choice = (!tools.is_empty()).then(|| translate_tool_choice(&request.tool_choice));
        let strict = self.json_schema_strict.load(Ordering::Relaxed);
        Ok(responses::ResponsesRequest {
            model: model.clone(),
            input,
            instructions,
            stream: Some(true),
            store: Some(false),
            max_output_tokens,
            tools,
            tool_choice,
            previous_response_id: request.continuation_id.clone(),
            text: request
                .response_format
                .as_ref()
                .and_then(|format| responses::translate_text_format(format, strict)),
            temperature: effective_temperature(
                &model,
                request.temperature,
                self.temperature_override,
                &self.temperature_unsupported,
            ),
            top_p: request.top_p,
            seed: request.seed,
            stop: request.stop_sequences.clone(),
            reasoning,
            include,
            extra,
        })
    }

    /// Issues a `POST` to the Responses endpoint and maps the body onto a
    /// [`ModelResponse`]. On a 400 whose body implicates `max_output_tokens`, the
    /// request is retried once without that field (some `/responses` backends
    /// reject the cap outright).
    async fn invoke_responses(&self, request: &ModelRequest) -> Result<ModelResponse> {
        let url = self.responses_url();
        let body = self.translate_responses_request(request)?;
        let response = match self.send_responses(&body, request.timeout_ms, &url).await {
            Ok(r) => r,
            Err(Error::Provider(err))
                if err.status == Some(400)
                    && body.max_output_tokens.is_some()
                    && err.message.contains("max_output_tokens") =>
            {
                // Retry once without the cap.
                let retry = responses::ResponsesRequest {
                    max_output_tokens: None,
                    ..body
                };
                self.send_responses(&retry, request.timeout_ms, &url)
                    .await?
            }
            Err(e) => return Err(e),
        };
        let text = response
            .text()
            .await
            .map_err(|e| Error::Model(format!("openai responses body read failed: {e}")))?;
        let value = responses::parse_responses_wire(&text)?;
        Ok(responses::parse_responses_response(value))
    }

    /// Shared `POST {responses_url}` with auth, query params, and timeout, mapped
    /// through the same checked-transport tail as chat calls.
    async fn send_responses(
        &self,
        body: &responses::ResponsesRequest,
        timeout_ms: Option<u64>,
        url: &str,
    ) -> Result<reqwest::Response> {
        let mut builder = self.authorized(self.client.post(url)).json(body);
        if let Some(timeout) = request_timeout(timeout_ms, body.stream == Some(true)) {
            builder = builder.timeout(timeout);
        }
        self.send_checked(builder, "responses request", url).await
    }

    /// Sends `builder` and returns the checked (2xx) [`reqwest::Response`].
    ///
    /// The shared transport tail for every OpenAI call: a send/transport failure
    /// is mapped to a structured [`Error::Provider`] describing `what` (e.g.
    /// `"request"`, `"stream request"`) against `url`, and any non-2xx status is
    /// decoded through [`Self::parse_error_body`] into a structured
    /// [`Error::Provider`]. On success the raw response is handed back
    /// so the caller can read it as text (unary/list) or stream its body.
    async fn send_checked(
        &self,
        builder: reqwest::RequestBuilder,
        what: &str,
        url: &str,
    ) -> Result<reqwest::Response> {
        let response = builder.send().await.map_err(|e| {
            let error =
                self.provider_error(format!("{what} to {url} failed: {e}"), None, None, None);
            Error::Provider(Box::new(error))
        })?;

        let status = response.status();
        if !status.is_success() {
            let text = response.text().await.unwrap_or_default();
            let error = self.parse_error_body(status.as_u16(), &text);
            return Err(Error::Provider(Box::new(error)));
        }
        Ok(response)
    }

    /// Issues an authenticated `POST {base_url}/chat/completions` with `body`,
    /// applying the resolved per-request timeout, and returns the checked
    /// response.
    ///
    /// Shared by the unary ([`Self::invoke`]) and streaming ([`Self::stream`])
    /// paths so URL construction, auth, timeout selection, and transport/status
    /// handling live in exactly one place.
    async fn post_json(
        &self,
        body: &ChatCompletionRequest,
        timeout_ms: Option<u64>,
        streaming: bool,
        what: &str,
    ) -> Result<reqwest::Response> {
        let url = format!("{}/chat/completions", self.base_url);
        let mut builder = self.authorized(self.client.post(&url)).json(body);
        if let Some(timeout) = request_timeout(timeout_ms, streaming) {
            builder = builder.timeout(timeout);
        }
        self.send_checked(builder, what, &url).await
    }

    /// Builds the chat-completions wire body for `request` under the given
    /// `degrade`, setting the streaming fields when `streaming` is `true`.
    fn build_chat_body(
        &self,
        request: &ModelRequest,
        degrade: Degrade,
        streaming: bool,
    ) -> Result<ChatCompletionRequest> {
        let mut body = self.translate_request_with(request, degrade)?;
        if streaming {
            body.stream = true;
            body.stream_options = Some(json!({ "include_usage": true }));
        }
        Ok(body)
    }

    /// Posts a chat-completions request with automatic single-shot degraded
    /// retry for local-server request-shape rejections.
    ///
    /// The first attempt applies this instance's baseline degradations. If it
    /// fails with an HTTP 400 whose body implicates a named `tool_choice` or a
    /// `json_object` `response_format` that the request actually used — and that
    /// shape was not already degraded — the request is rebuilt with that shape
    /// degraded and sent exactly once more. Any other error (or a request that
    /// used neither shape) surfaces unchanged. Shared by [`Self::invoke`] and
    /// [`Self::stream`], so the retry covers the streaming path too.
    async fn post_chat_with_degrade(
        &self,
        request: &ModelRequest,
        streaming: bool,
        what: &str,
    ) -> Result<reqwest::Response> {
        let baseline = self.baseline_degrade();
        let body = self.build_chat_body(request, baseline, streaming)?;
        match self
            .post_json(&body, request.timeout_ms, streaming, what)
            .await
        {
            Ok(response) => Ok(response),
            Err(Error::Provider(err)) if err.status == Some(400) => {
                if let Some(degrade) = degrade_for_400(&err.message, request, baseline) {
                    if degrade.native_tools {
                        self.native_tools_on_wire.store(false, Ordering::Relaxed);
                    }
                    if degrade.json_schema_strict {
                        self.json_schema_strict.store(false, Ordering::Relaxed);
                    }
                    let retry = self.build_chat_body(request, degrade, streaming)?;
                    self.post_json(&retry, request.timeout_ms, streaming, what)
                        .await
                } else {
                    Err(Error::Provider(err))
                }
            }
            Err(e) => Err(e),
        }
    }

    fn provider_error(
        &self,
        message: impl Into<String>,
        status: Option<u16>,
        code: Option<String>,
        raw: Option<Value>,
    ) -> ProviderError {
        let message = message.into();
        let retryable =
            crate::failure::classify_provider_failure(status, code.as_deref(), &message)
                .is_retryable();
        ProviderError {
            provider: self.provider.clone(),
            model: Some(self.model.clone()),
            status,
            code,
            message,
            retryable,
            raw,
            retry_after_ms: None,
        }
    }

    pub(super) fn parse_error_body(&self, status: u16, text: &str) -> ProviderError {
        let raw = serde_json::from_str::<Value>(text).ok();
        let error_obj = raw.as_ref().and_then(|value| value.get("error"));
        let message = error_obj
            .and_then(|error| error.get("message"))
            .and_then(Value::as_str)
            .or_else(|| {
                raw.as_ref()
                    .and_then(|value| value.get("message"))
                    .and_then(Value::as_str)
            })
            .filter(|message| !message.trim().is_empty())
            .unwrap_or(text)
            .to_string();
        let code = error_obj
            .and_then(|error| error.get("code").or_else(|| error.get("type")))
            .and_then(Value::as_str)
            .map(str::to_string);
        let code = if is_context_overflow(status, &message) {
            Some(CONTEXT_OVERFLOW_CODE.to_string())
        } else {
            code
        };
        let message = self
            .local_runtime
            .and_then(|kind| {
                missing_model_remediation(kind, status, &message, &self.model, &self.base_url)
            })
            .unwrap_or(message);
        self.provider_error(message, Some(status), code, raw)
    }
}

/// Request-shape degradations to apply when building an OpenAI wire body.
///
/// Each field, when `true`, replaces a request shape that some local
/// OpenAI-compatible servers reject with an equivalent one they accept. The
/// baseline comes from the instance's capability knobs
/// ([`OpenAiModel::baseline_degrade`]) from the instance's capability knobs; a
/// 400 whose error body implicates one of these shapes turns the corresponding
/// field on for a single degraded retry (see [`degrade_for_400`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Degrade {
    /// Degrade a named `tool_choice` to `"required"` + a filtered `tools` array.
    pub named_tool_choice: bool,
    /// Degrade `response_format:{"type":"json_object"}` to a permissive
    /// `json_schema`.
    pub json_object: bool,
    pub json_schema_strict: bool,
    pub native_tools: bool,
}

fn make_schema_strict(schema: &mut Value) {
    match schema {
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("object") {
                object.insert("additionalProperties".to_string(), Value::Bool(false));
                if let Some(properties) = object.get("properties").and_then(Value::as_object) {
                    let required = properties.keys().cloned().map(Value::String).collect();
                    object.insert("required".to_string(), Value::Array(required));
                }
            }
            for value in object.values_mut() {
                make_schema_strict(value);
            }
        }
        Value::Array(values) => values.iter_mut().for_each(make_schema_strict),
        _ => {}
    }
}

/// Computes the additional degradation to apply after an HTTP 400, or `None`
/// when the failure is not an auto-degradable request-shape rejection.
///
/// Returns `Some(degrade)` only when the 400 error body implicates a shape the
/// request actually used *and* that shape was not already degraded on the
/// original attempt — so a degraded retry is issued at most once and only when
/// it could plausibly help. The returned [`Degrade`] is the union of `already`
/// and the newly implicated shape, so the retry keeps any baseline degradations.
///
/// Pure, so the 400-detection policy is unit-testable without a network call.
pub(super) fn degrade_for_400(
    message: &str,
    request: &ModelRequest,
    already: Degrade,
) -> Option<Degrade> {
    let lower = message.to_ascii_lowercase();
    let mut degrade = already;

    if !already.named_tool_choice
        && lower.contains("tool_choice")
        && matches!(request.tool_choice, ToolChoice::Tool(_))
    {
        degrade.named_tool_choice = true;
    }
    if !already.json_object
        && lower.contains("response_format")
        && matches!(request.response_format, Some(ResponseFormat::JsonObject))
    {
        degrade.json_object = true;
    }
    if !already.json_schema_strict
        && lower.contains("strict")
        && matches!(
            request.response_format,
            Some(ResponseFormat::JsonSchema { .. } | ResponseFormat::Auto { .. })
        )
    {
        degrade.json_schema_strict = true;
    }
    if !already.native_tools
        && !request.tools.is_empty()
        && (lower.contains("does not support tools") || lower.contains("tools unsupported"))
    {
        degrade.native_tools = true;
    }

    (degrade != already).then_some(degrade)
}

/// Resolves the per-request timeout to apply to an outbound HTTP call.
///
/// An explicit [`ModelRequest::timeout_ms`] always wins. Otherwise a unary call
/// falls back to [`DEFAULT_REQUEST_TIMEOUT_SECS`], while a streaming call gets no
/// overall cap (a total-request timeout would truncate a legitimately
/// long-running stream mid-flight).
pub(super) fn request_timeout(timeout_ms: Option<u64>, streaming: bool) -> Option<Duration> {
    match timeout_ms {
        Some(ms) => Some(Duration::from_millis(ms)),
        None if streaming => None,
        None => Some(Duration::from_secs(DEFAULT_REQUEST_TIMEOUT_SECS)),
    }
}

/// Merges baked `defaults` under a request's own `overrides` provider options.
///
/// Keys present in `overrides` win over `defaults`. A `Null` on either side
/// contributes nothing; when neither side is an object the result is
/// `Value::Null`, so [`provider_extra_options`] short-circuits to an empty map.
///
/// A **non-null, non-object** `overrides` is invalid caller input and is returned
/// untouched (never merged) so [`provider_extra_options`] still rejects it with
/// its clear validation error instead of the merge silently dropping it. Pure, so
/// the merge policy is unit-testable without a request.
pub(super) fn merge_provider_options(defaults: &Value, overrides: &Value) -> Value {
    if !overrides.is_null() && !overrides.is_object() {
        return overrides.clone();
    }
    match (defaults.as_object(), overrides.as_object()) {
        (None, None) => Value::Null,
        (Some(base), None) => Value::Object(base.clone()),
        (None, Some(over)) => Value::Object(over.clone()),
        (Some(base), Some(over)) => {
            let mut merged = base.clone();
            for (key, value) in over {
                merged.insert(key.clone(), value.clone());
            }
            Value::Object(merged)
        }
    }
}

/// Returns provider-specific top-level fields to flatten into the request body.
///
/// Core OpenAI-compatible fields are intentionally reserved so normalized
/// TinyAgents fields remain the source of truth. Callers that need local-model
/// controls should use distinct provider fields such as Ollama's `options`.
pub(super) fn provider_extra_options(options: &Value) -> Result<Map<String, Value>> {
    if options.is_null() {
        return Ok(Map::new());
    }
    let Some(object) = options.as_object() else {
        return Err(Error::Validation(
            "provider_options for OpenAI-compatible providers must be a JSON object".to_string(),
        ));
    };

    const RESERVED: &[&str] = &[
        "model",
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "seed",
        "stream",
        "stream_options",
    ];

    Ok(object
        .iter()
        .filter(|(key, _)| !RESERVED.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for OpenAiModel {
    /// Returns the capability profile derived from the configured model id.
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    /// Invokes the OpenAI Chat Completions endpoint and maps the response into a
    /// [`ModelResponse`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Model`] on transport failure or a non-2xx
    /// status (the message includes the status code and response body), and
    /// [`Error::Serialization`] when the response cannot be decoded.
    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        if self.responses_api_primary {
            return self.invoke_responses(&request).await;
        }
        let response = self
            .post_chat_with_degrade(&request, false, "request")
            .await?;

        let text = response
            .text()
            .await
            .map_err(|e| Error::Model(format!("openai response body read failed: {e}")))?;

        let value: Value = serde_json::from_str(&text)?;
        let response = parse_chat_response(value, self.effective_reasoning_tags())?;
        // Prompt-guided tools: recover the model's `<tool_call>` blocks into
        // `message.tool_calls` when native tool calling was suppressed.
        if !self.profile.tool_calling
            && !request.tools.is_empty()
            && request.tool_choice != ToolChoice::None
        {
            return Ok(prompt_tools::apply_to_response(response));
        }
        Ok(response)
    }

    /// Streams the OpenAI Chat Completions response as a real [`ModelStream`].
    ///
    /// Sends the request with `stream: true` (and `stream_options.include_usage`
    /// so a usage chunk is delivered), reads the Server-Sent-Events body
    /// incrementally with [`reqwest::Response::bytes_stream`], and parses each
    /// `data:` line into [`ModelStreamItem`]s: a leading
    /// [`ModelStreamItem::Started`], a [`ModelStreamItem::MessageDelta`] per text
    /// fragment, a [`ModelStreamItem::ToolCallDelta`] per tool-call argument
    /// fragment, a [`ModelStreamItem::UsageDelta`] when usage arrives, and a
    /// terminal [`ModelStreamItem::Completed`] carrying the fully merged
    /// response (with reassembled tool-call names and ids). Transport errors
    /// surface as a terminal [`ModelStreamItem::ProviderFailed`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Model`] when the initial request fails or the
    /// endpoint returns a non-2xx status (per-chunk transport errors are
    /// surfaced as [`ModelStreamItem::ProviderFailed`] inside the stream
    /// instead).
    async fn stream(&self, _state: &State, request: ModelRequest) -> Result<ModelStream> {
        // The Responses path is text-in/text-out in this port: do the unary call
        // and surface it as a single terminal `Completed` (a leading `Started` +
        // one `MessageDelta` carrying the text so a UI still renders it). True
        // Responses SSE is a follow-up.
        if self.responses_api_primary {
            let response = self.invoke_responses(&request).await?;
            let delta = crate::message::MessageDelta {
                text: response.text(),
                reasoning: String::new(),
                tool_call: None,
            };
            let items = vec![
                ModelStreamItem::Started,
                ModelStreamItem::MessageDelta(delta),
                ModelStreamItem::Completed(response),
            ];
            return Ok(Box::pin(futures::stream::iter(items)));
        }
        let response = self
            .post_chat_with_degrade(&request, true, "stream request")
            .await?;

        // Leniency: some OpenAI-compatible endpoints (and test mocks) ignore
        // `stream: true` and return a single non-streaming JSON body. Detect a
        // non-`text/event-stream` content-type and parse it as one completion,
        // surfaced as `Started` + one `MessageDelta` + `Completed` — matching the
        // tolerant behaviour of hand-rolled OpenAI-compatible clients so a
        // non-streaming upstream still drives a streaming turn.
        let is_event_stream = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|ct| ct.to_ascii_lowercase().contains("text/event-stream"))
            .unwrap_or(false);
        if !is_event_stream {
            let text = response.text().await.map_err(|e| {
                Error::Model(format!("openai non-stream stream-body read failed: {e}"))
            })?;
            let value: Value = serde_json::from_str(&text)?;
            let mut parsed = parse_chat_response(value, self.effective_reasoning_tags())?;
            if !self.profile.tool_calling
                && !request.tools.is_empty()
                && request.tool_choice != ToolChoice::None
            {
                parsed = prompt_tools::apply_to_response(parsed);
            }
            let delta = crate::message::MessageDelta {
                text: parsed.text(),
                reasoning: String::new(),
                tool_call: None,
            };
            let items = vec![
                ModelStreamItem::Started,
                ModelStreamItem::MessageDelta(delta),
                ModelStreamItem::Completed(parsed),
            ];
            return Ok(Box::pin(futures::stream::iter(items)));
        }

        // Forward each raw chunk as the `bytes::Bytes` buffer reqwest already
        // produced (a cheap refcount clone, no per-chunk copy); only the error
        // type is mapped onto the crate error. `SseState` is crate-internal,
        // so `bytes` never leaks into the public API.
        let bytes = response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|e| Error::Model(format!("stream chunk failed: {e}"))));

        let state = SseState {
            bytes: Box::pin(bytes),
            buf: Vec::new(),
            pending: VecDeque::new(),
            acc: OpenAiStreamAcc::new(self.effective_reasoning_tags().cloned()),
            provider: self.provider.clone(),
            model: self.model.clone(),
            started: false,
            finished: false,
            completion_seen: false,
            terminal_emitted: false,
        };

        let stream = futures::stream::unfold(state, sse_next);
        // Prompt-guided tools: recover `<tool_call>` blocks from the terminal
        // `Completed` response into `message.tool_calls` (the streamed text deltas
        // still carry the raw markup — cleaning them mid-stream is a follow-up).
        if !self.profile.tool_calling
            && !request.tools.is_empty()
            && request.tool_choice != ToolChoice::None
        {
            return Ok(Box::pin(stream.map(|item| match item {
                ModelStreamItem::Completed(response) => {
                    ModelStreamItem::Completed(prompt_tools::apply_to_response(response))
                }
                other => other,
            })));
        }
        Ok(Box::pin(stream))
    }
}
