//! [`LocalRuntime`]: one enum that reconciles the four local-runtime enums the
//! ecosystem already has, by conversion and never by replacement.
//!
//! | Existing enum | Crate | Relationship |
//! |---|---|---|
//! | `ProviderKind` (local subset) | llm | [`LocalRuntime::from_provider_kind`] |
//! | `LocalRuntimeKind` | llm | `From` / `TryFrom` |
//! | `LocalProviderKind` | local | `From` both ways (feature `local-bridge`) |
//! | `LocalAiProvider` | local | `From` / `TryFrom` (feature `local-bridge`) |
//!
//! No existing enum is changed, deprecated or aliased: their variant sets
//! differ, so a `pub type` would lie.

use std::fmt;

use serde::{Deserialize, Serialize};
use tinyinference_llm::ProviderKind;
use tinyinference_llm::providers::openai::LocalRuntimeKind;

/// A local model runtime.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalRuntime {
    /// Ollama.
    Ollama,
    /// LM Studio.
    #[serde(alias = "lmstudio", alias = "lm-studio")]
    LmStudio,
    /// llama.cpp's `llama-server`.
    #[serde(alias = "llamacpp", alias = "llama.cpp", alias = "llama-cpp")]
    LlamaCpp,
    /// vLLM's OpenAI-compatible server.
    Vllm,
    /// An MLX-compatible server (`mlx_lm.server`).
    #[serde(alias = "mlx-server", alias = "mlx_lm")]
    Mlx,
    /// OMLX, an OpenAI-v1-compatible MLX server with an optional key.
    #[serde(alias = "omlx-server")]
    Omlx,
    /// Any other local OpenAI-compatible endpoint.
    #[serde(
        rename = "openai_compatible",
        alias = "open_ai_compatible",
        alias = "local-openai",
        alias = "local_openai",
        alias = "custom-openai",
        alias = "custom_openai"
    )]
    OpenAiCompatible,
}

/// A runtime that has no equivalent in a narrower enum.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnsupportedRuntime(pub LocalRuntime);

impl fmt::Display for UnsupportedRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} has no equivalent in that enum", self.0.as_str())
    }
}

impl std::error::Error for UnsupportedRuntime {}

impl LocalRuntime {
    /// Every runtime, in declaration order.
    pub const ALL: [LocalRuntime; 7] = [
        Self::Ollama,
        Self::LmStudio,
        Self::LlamaCpp,
        Self::Vllm,
        Self::Mlx,
        Self::Omlx,
        Self::OpenAiCompatible,
    ];

    /// The canonical wire spelling (identical to the serde form).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LmStudio => "lm_studio",
            Self::LlamaCpp => "llama_cpp",
            Self::Vllm => "vllm",
            Self::Mlx => "mlx",
            Self::Omlx => "omlx",
            Self::OpenAiCompatible => "openai_compatible",
        }
    }

    /// The catalogue slug of the row that represents this runtime.
    pub fn catalogue_slug(self) -> &'static str {
        match self {
            Self::Ollama => "ollama",
            Self::LmStudio => "lmstudio",
            Self::Mlx => "mlx",
            Self::Omlx => "omlx",
            Self::LlamaCpp | Self::Vllm | Self::OpenAiCompatible => "local-openai",
        }
    }

    /// Parses any spelling in use today, case-insensitively.
    ///
    /// **A bare `openai` is never a local runtime.** OpenHuman's loose parser
    /// accepts it as `LocalOpenai` while its `openai:` prefix form does not; the
    /// hub refuses the trap and returns `None`, so a hosted OpenAI row cannot be
    /// mistaken for a local server.
    pub fn parse_loose(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "ollama" => Some(Self::Ollama),
            "lmstudio" | "lm-studio" | "lm_studio" => Some(Self::LmStudio),
            "llamacpp" | "llama.cpp" | "llama_cpp" | "llama-cpp" => Some(Self::LlamaCpp),
            "vllm" => Some(Self::Vllm),
            "mlx" | "mlx-server" | "mlx_lm" => Some(Self::Mlx),
            "omlx" | "omlx-server" => Some(Self::Omlx),
            "local-openai" | "local_openai" | "custom-openai" | "custom_openai"
            | "openai_compatible" | "open_ai_compatible" => Some(Self::OpenAiCompatible),
            _ => None,
        }
    }

    /// The local runtime a llm `ProviderKind` denotes, if it denotes one.
    pub fn from_provider_kind(kind: &ProviderKind) -> Option<Self> {
        match kind {
            ProviderKind::Ollama => Some(Self::Ollama),
            ProviderKind::LmStudio => Some(Self::LmStudio),
            ProviderKind::LlamaCpp => Some(Self::LlamaCpp),
            ProviderKind::Vllm => Some(Self::Vllm),
            _ => None,
        }
    }
}

impl fmt::Display for LocalRuntime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<LocalRuntimeKind> for LocalRuntime {
    fn from(kind: LocalRuntimeKind) -> Self {
        match kind {
            LocalRuntimeKind::Ollama => Self::Ollama,
            LocalRuntimeKind::LmStudio => Self::LmStudio,
            LocalRuntimeKind::LlamaCpp => Self::LlamaCpp,
            LocalRuntimeKind::Vllm => Self::Vllm,
        }
    }
}

impl TryFrom<LocalRuntime> for LocalRuntimeKind {
    type Error = UnsupportedRuntime;

    fn try_from(runtime: LocalRuntime) -> Result<Self, Self::Error> {
        match runtime {
            LocalRuntime::Ollama => Ok(Self::Ollama),
            LocalRuntime::LmStudio => Ok(Self::LmStudio),
            LocalRuntime::LlamaCpp => Ok(Self::LlamaCpp),
            LocalRuntime::Vllm => Ok(Self::Vllm),
            other => Err(UnsupportedRuntime(other)),
        }
    }
}

#[cfg(feature = "local-bridge")]
mod bridge {
    //! Conversions to and from `tinyinference-local`'s enums.

    use tinyinference_local::profile::LocalProviderKind;
    use tinyinference_local::provider::LocalAiProvider;

    use super::{LocalRuntime, UnsupportedRuntime};

    impl From<LocalProviderKind> for LocalRuntime {
        fn from(kind: LocalProviderKind) -> Self {
            match kind {
                LocalProviderKind::Ollama => Self::Ollama,
                LocalProviderKind::LmStudio => Self::LmStudio,
                LocalProviderKind::Mlx => Self::Mlx,
                LocalProviderKind::Omlx => Self::Omlx,
                LocalProviderKind::LocalOpenai => Self::OpenAiCompatible,
            }
        }
    }

    /// `LlamaCpp` and `Vllm` collapse to `LocalOpenai`, exactly as the local
    /// crate's own loose parser does.
    impl From<LocalRuntime> for LocalProviderKind {
        fn from(runtime: LocalRuntime) -> Self {
            match runtime {
                LocalRuntime::Ollama => Self::Ollama,
                LocalRuntime::LmStudio => Self::LmStudio,
                LocalRuntime::Mlx => Self::Mlx,
                LocalRuntime::Omlx => Self::Omlx,
                LocalRuntime::LlamaCpp | LocalRuntime::Vllm | LocalRuntime::OpenAiCompatible => {
                    Self::LocalOpenai
                }
            }
        }
    }

    impl From<LocalAiProvider> for LocalRuntime {
        fn from(provider: LocalAiProvider) -> Self {
            match provider {
                LocalAiProvider::Ollama => Self::Ollama,
                LocalAiProvider::LmStudio => Self::LmStudio,
            }
        }
    }

    impl TryFrom<LocalRuntime> for LocalAiProvider {
        type Error = UnsupportedRuntime;

        fn try_from(runtime: LocalRuntime) -> Result<Self, Self::Error> {
            match runtime {
                LocalRuntime::Ollama => Ok(Self::Ollama),
                LocalRuntime::LmStudio => Ok(Self::LmStudio),
                other => Err(UnsupportedRuntime(other)),
            }
        }
    }
}
