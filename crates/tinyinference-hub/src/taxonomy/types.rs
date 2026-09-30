//! The simple taxonomy enums.

use std::fmt;

use serde::{Deserialize, Serialize};

/// What the operator supplies to use a provider, and therefore how the hub
/// treats it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderGroup {
    /// The first-party TinyHumans provider; the credential comes from a
    /// host-built chain (D1, D2).
    Managed,
    /// A hosted vendor; the operator supplies a key and the endpoint is a
    /// preset.
    Cloud,
    /// A runtime on this machine; the operator supplies an endpoint.
    Local,
    /// A CLI login; readiness is decided by launching the binary.
    Cli,
    /// An operator-named OpenAI-compatible endpoint.
    Custom,
    /// A provider reached through a browser login (feature `oauth`; types
    /// only).
    #[serde(rename = "oauth_backed", alias = "o_auth_backed")]
    OAuthBacked,
}

impl ProviderGroup {
    /// The stable wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Managed => "managed",
            Self::Cloud => "cloud",
            Self::Local => "local",
            Self::Cli => "cli",
            Self::Custom => "custom",
            Self::OAuthBacked => "oauth_backed",
        }
    }
}

impl fmt::Display for ProviderGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How the hub reaches a provider (D9).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Transport {
    /// HTTP(S) to an endpoint.
    Http,
    /// A subprocess (CLI logins); there is no endpoint.
    Subprocess,
}

/// The chat wire protocol a provider speaks.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Protocol {
    /// OpenAI Chat Completions.
    OpenAiChat,
    /// OpenAI Responses API.
    OpenAiResponses,
    /// Anthropic native Messages.
    AnthropicMessages,
    /// A CLI's streamed output.
    CliStream,
}

/// How a credential is presented to a provider (D2).
///
/// `Custom` carries the header name for the rare provider (Azure's `api-key`)
/// that wants the bare key in a header of its own.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthStyle {
    /// `Authorization: Bearer <key>`, the OpenAI-compatible default.
    Bearer,
    /// `x-api-key: <key>` with no version header.
    XApiKey,
    /// `x-api-key: <key>` plus `anthropic-version: 2023-06-01`.
    Anthropic,
    /// A rotating session token (OpenHuman's JWT). Older data spells it
    /// `openhuman_jwt` or `openhumanjwt`; both are accepted.
    #[serde(alias = "openhuman_jwt", alias = "openhumanjwt")]
    SessionJwt,
    /// No auth header at all.
    None,
    /// The bare key in the named header.
    Custom(String),
}

impl AuthStyle {
    /// The stable wire spelling; a [`AuthStyle::Custom`] style reports
    /// `"custom"` (its header name is data, not a spelling).
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Bearer => "bearer",
            Self::XApiKey => "x_api_key",
            Self::Anthropic => "anthropic",
            Self::SessionJwt => "session_jwt",
            Self::None => "none",
            Self::Custom(_) => "custom",
        }
    }

    /// The lowercase names of the request headers that carry this style's
    /// credential (empty for [`AuthStyle::None`]). A custom style reports the
    /// header it was configured with.
    pub fn credential_headers(&self) -> Vec<String> {
        match self {
            Self::Bearer | Self::SessionJwt => vec!["authorization".to_string()],
            Self::XApiKey | Self::Anthropic => vec!["x-api-key".to_string()],
            Self::None => Vec::new(),
            Self::Custom(header) => {
                let name = header.trim().to_ascii_lowercase();
                if name.is_empty() {
                    Vec::new()
                } else {
                    vec![name]
                }
            }
        }
    }

    /// Whether a request needs a credential to authenticate.
    pub fn needs_credential(&self) -> bool {
        !matches!(self, Self::None)
    }
}

/// The shape a provider's model listing answers in.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogShape {
    /// `{"data":[{"id":...}]}`, one response.
    OpenAi,
    /// `{"success":true,"data":{"data":[...],"total":N,"limit":L,"offset":O}}`,
    /// paged (500 per page, at most 20 pages).
    PagedEnvelope,
    /// Ollama's `/api/tags`.
    OllamaTags,
    /// LM Studio's richer `/api/v0/models`.
    LmStudioV0,
    /// The provider publishes no listing.
    None,
}

/// A CLI-login kind (feature `cli` adds readiness; the type is always
/// available).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CliKind {
    /// The `claude` binary. The Claude subscription is used only by running
    /// this binary (D12), never by replaying a token.
    ClaudeCode,
    /// The `codex` binary; its credential is stored under the `openai` slug.
    Codex,
}

impl CliKind {
    /// The option slug a picker offers (`claude-code`, `codex`).
    pub fn option_slug(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
        }
    }

    /// The slug the login's credential is stored under; Codex has no row of
    /// its own and reuses `openai`.
    pub fn stored_slug(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "openai",
        }
    }

    /// Parses an option slug.
    pub fn from_option_slug(slug: &str) -> Option<Self> {
        match slug.trim() {
            "claude-code" => Some(Self::ClaudeCode),
            "codex" => Some(Self::Codex),
            _ => None,
        }
    }
}

/// How deep a provider test goes (D12): each depth costs more and proves more.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestDepth {
    /// A cheap key validation call (for example OpenRouter `GET /key`).
    KeyOnly,
    /// Read the model listing.
    Catalog,
    /// A one-token completion (`max_tokens` about 16).
    Completion,
}

impl TestDepth {
    /// The stable wire spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::KeyOnly => "key_only",
            Self::Catalog => "catalog",
            Self::Completion => "completion",
        }
    }
}

impl fmt::Display for TestDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
