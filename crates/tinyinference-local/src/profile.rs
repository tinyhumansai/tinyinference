//! Local provider profiles — capability metadata for local inference runtimes.
//!
//! Instead of treating all local OpenAI-compatible providers identically, each
//! provider type (Ollama, LM Studio, MLX-compatible, generic local OpenAI) gets
//! a profile that declares its capabilities, quirks, and default context window.
//! The factory and agent harness consult these profiles for:
//!
//! - Tool dispatch strategy (native vs prompt-guided)
//! - Context window defaults for unknown model names
//! - Request body extras (`options.num_ctx`, `think` field suppression)
//! - Temperature handling

use serde::{Deserialize, Serialize};

/// Resolve a preferred context window with a local runtime profile fallback.
///
/// Local runtimes must always receive a pre-dispatch budget, even when neither
/// model metadata nor the runtime profile declares one. In that final case a
/// conservative 4,096-token floor avoids dispatching a prompt that is certain
/// to overflow an unknown loaded context. Non-local callers retain `None`.
pub fn context_window_with_local_fallback(
    model: &str,
    preferred_window: Option<u64>,
    local_kind: Option<LocalProviderKind>,
) -> Option<u64> {
    if let Some(window) = preferred_window {
        return Some(window);
    }
    let kind = local_kind?;
    let profile = profile_for_kind(kind);
    if matches!(kind, LocalProviderKind::Mlx | LocalProviderKind::Omlx)
        && let Some(window) = model_dir_context_window(model)
    {
        tracing::debug!(
            model,
            provider = kind.as_str(),
            context_window = window,
            "using context window from the model directory's config.json"
        );
        return Some(window);
    }
    if let Some(window) = profile.default_context_window {
        tracing::debug!(
            model,
            provider = kind.as_str(),
            context_window = window,
            "using local provider profile context window"
        );
        return Some(window);
    }
    const CONSERVATIVE_LOCAL_CONTEXT_FLOOR: u64 = 4_096;
    tracing::debug!(
        model,
        provider = kind.as_str(),
        context_window = CONSERVATIVE_LOCAL_CONTEXT_FLOOR,
        "local provider has no context default; using conservative floor"
    );
    Some(CONSERVATIVE_LOCAL_CONTEXT_FLOOR)
}

/// Context window a local model directory declares in its `config.json`.
///
/// MLX servers (`mlx_lm`, `mlx_vlm`) report no limits, and their model id is
/// the directory the weights were loaded from. When `model` is such a
/// directory, reads `max_position_embeddings` from the top level or from a
/// nested `text_config` (multimodal checkpoints). Returns `None` for anything
/// that is not a readable directory with a positive value.
///
/// shortcut: the model's trained maximum, not what the host's memory can hold;
/// a host that knows better sets an explicit override.
pub fn model_dir_context_window(model: &str) -> Option<u64> {
    let path = std::path::Path::new(model.trim()).join("config.json");
    if !path.is_absolute() {
        return None;
    }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    [&config, &config["text_config"]]
        .iter()
        .find_map(|section| section["max_position_embeddings"].as_u64())
        .filter(|window| *window > 0)
}

/// Identifies a local provider type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalProviderKind {
    /// Ollama native runtime.
    Ollama,
    /// LM Studio runtime.
    LmStudio,
    /// MLX-compatible local server (e.g. `mlx_lm.server`).
    Mlx,
    /// OMLX — OpenAI v1-compatible MLX server that requires an API key.
    Omlx,
    /// Generic local OpenAI-compatible endpoint (llama.cpp, vLLM, etc.).
    LocalOpenai,
}

impl LocalProviderKind {
    /// Returns the stable provider identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LmStudio => "lmstudio",
            Self::Mlx => "mlx",
            Self::Omlx => "omlx",
            Self::LocalOpenai => "local-openai",
        }
    }

    /// Returns the user-facing provider name.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Ollama => "Ollama",
            Self::LmStudio => "LM Studio",
            Self::Mlx => "MLX",
            Self::Omlx => "OMLX",
            Self::LocalOpenai => "Local OpenAI",
        }
    }

    /// Parse a provider kind from a string, accepting common aliases.
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "lmstudio" | "lm-studio" | "lm_studio" => Some(Self::LmStudio),
            "mlx" | "mlx-server" | "mlx_lm" => Some(Self::Mlx),
            "omlx" | "omlx-server" => Some(Self::Omlx),
            "openai" | "local-openai" | "local_openai" | "custom-openai" | "custom_openai"
            | "llamacpp" | "llama.cpp" | "vllm" => Some(Self::LocalOpenai),
            _ => None,
        }
    }
}

/// How the provider handles tool calling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolSupport {
    /// Provider reliably supports native OpenAI-style tool calling.
    Native,
    /// Provider does NOT support native tools — use prompt-guided dispatch.
    PromptGuided,
    /// Support depends on the specific model; probe or consult model profile.
    ModelDependent,
}

/// Extra request body options for a local provider.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestQuirks {
    /// Ollama `options.num_ctx` override. When set, injected into the
    /// request body as `{"options": {"num_ctx": <value>}}`.
    pub num_ctx: Option<u32>,
    /// When true, suppress reasoning/thinking fields in the request.
    /// Some Ollama models reject requests containing `think` parameters.
    pub suppress_thinking: bool,
    /// When true, omit the `temperature` field entirely (model uses its
    /// own default). Distinct from `temperature_unsupported_models` which
    /// is pattern-based — this is a blanket provider-level override.
    pub omit_temperature: bool,
    /// When true, merge system messages into user messages (provider
    /// rejects `role: system`).
    pub merge_system_into_user: bool,
}

/// Static capability profile for a local provider type.
#[derive(Debug, Clone)]
pub struct LocalProviderProfile {
    /// Provider kind described by this profile.
    pub kind: LocalProviderKind,
    /// Default tool support level for this provider type.
    pub tool_support: ToolSupport,
    /// Default context window (tokens) when the model name is not
    /// recognized by `context_window_for_model`. `None` means "unknown,
    /// skip preflight trimming".
    pub default_context_window: Option<u64>,
    /// Whether the provider supports the Responses API (`/v1/responses`).
    pub supports_responses_api: bool,
    /// Whether the provider supports SSE streaming.
    pub supports_streaming: bool,
    /// Default request quirks for this provider type.
    pub default_quirks: RequestQuirks,
    /// Default base URL when none is configured.
    pub default_base_url: &'static str,
    /// Environment variable name for base URL override.
    pub base_url_env: &'static str,
}

/// Ollama profile: conservative defaults, no native tools.
pub const OLLAMA_PROFILE: LocalProviderProfile = LocalProviderProfile {
    kind: LocalProviderKind::Ollama,
    tool_support: ToolSupport::PromptGuided,
    default_context_window: Some(8_192),
    supports_responses_api: false,
    supports_streaming: true,
    default_quirks: RequestQuirks {
        num_ctx: None,
        suppress_thinking: false,
        omit_temperature: false,
        merge_system_into_user: false,
    },
    default_base_url: "http://127.0.0.1:11434",
    base_url_env: "OLLAMA_HOST",
};

/// LM Studio profile: conservative defaults, no native tools.
pub const LM_STUDIO_PROFILE: LocalProviderProfile = LocalProviderProfile {
    kind: LocalProviderKind::LmStudio,
    tool_support: ToolSupport::PromptGuided,
    default_context_window: Some(8_192),
    supports_responses_api: false,
    supports_streaming: true,
    default_quirks: RequestQuirks {
        num_ctx: None,
        suppress_thinking: false,
        omit_temperature: false,
        merge_system_into_user: false,
    },
    default_base_url: "http://127.0.0.1:1234/v1",
    base_url_env: "LM_STUDIO_BASE_URL",
};

/// MLX-compatible server profile (mlx_lm.server, etc.).
pub const MLX_PROFILE: LocalProviderProfile = LocalProviderProfile {
    kind: LocalProviderKind::Mlx,
    tool_support: ToolSupport::PromptGuided,
    default_context_window: Some(4_096),
    supports_responses_api: false,
    supports_streaming: true,
    default_quirks: RequestQuirks {
        num_ctx: None,
        suppress_thinking: false,
        omit_temperature: false,
        merge_system_into_user: false,
    },
    default_base_url: "http://127.0.0.1:8080/v1",
    base_url_env: "MLX_SERVER_URL",
};

/// OMLX profile: OpenAI v1-compatible MLX server, default port 8000, key required.
pub const OMLX_PROFILE: LocalProviderProfile = LocalProviderProfile {
    kind: LocalProviderKind::Omlx,
    tool_support: ToolSupport::PromptGuided,
    default_context_window: Some(4_096),
    supports_responses_api: false,
    supports_streaming: true,
    default_quirks: RequestQuirks {
        num_ctx: None,
        suppress_thinking: false,
        omit_temperature: false,
        merge_system_into_user: false,
    },
    default_base_url: "http://127.0.0.1:8000/v1",
    base_url_env: "OMLX_SERVER_URL",
};

/// Generic local OpenAI-compatible server (llama.cpp, vLLM, etc.).
pub const LOCAL_OPENAI_PROFILE: LocalProviderProfile = LocalProviderProfile {
    kind: LocalProviderKind::LocalOpenai,
    tool_support: ToolSupport::PromptGuided,
    default_context_window: None,
    supports_responses_api: false,
    supports_streaming: true,
    default_quirks: RequestQuirks {
        num_ctx: None,
        suppress_thinking: false,
        omit_temperature: false,
        merge_system_into_user: false,
    },
    default_base_url: "http://127.0.0.1:8080/v1",
    base_url_env: "LOCAL_OPENAI_URL",
};

/// Look up the static profile for a provider kind.
pub fn profile_for_kind(kind: LocalProviderKind) -> &'static LocalProviderProfile {
    match kind {
        LocalProviderKind::Ollama => &OLLAMA_PROFILE,
        LocalProviderKind::LmStudio => &LM_STUDIO_PROFILE,
        LocalProviderKind::Mlx => &MLX_PROFILE,
        LocalProviderKind::Omlx => &OMLX_PROFILE,
        LocalProviderKind::LocalOpenai => &LOCAL_OPENAI_PROFILE,
    }
}

/// Resolve the provider kind from a provider string prefix.
///
/// Returns `None` for cloud/openhuman/unknown providers.
pub fn kind_from_provider_string(provider: &str) -> Option<LocalProviderKind> {
    let p = provider.trim().to_ascii_lowercase();
    LocalProviderKind::from_str_loose(&p).or_else(|| {
        p.split_once(':').and_then(|(prefix, _)| {
            (prefix != "openai")
                .then(|| LocalProviderKind::from_str_loose(prefix))
                .flatten()
        })
    })
}

/// Returns `true` when the provider string resolves to any local provider.
pub fn is_local_provider_string(provider: &str) -> bool {
    kind_from_provider_string(provider).is_some()
}

#[cfg(test)]
#[path = "profile_tests.rs"]
mod tests;
