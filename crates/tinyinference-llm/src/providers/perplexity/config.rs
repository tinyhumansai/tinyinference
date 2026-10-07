//! Perplexity selection and request configuration.

use crate::{Error, Result, model::ModelRequest};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fmt, time::Duration};

/// Explicit model or server-managed preset selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "type",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum PerplexitySelection {
    /// Provider-qualified model ID.
    Model(String),
    /// Provider-managed fallback chain, in priority order (one to five IDs).
    Models(Vec<String>),
    /// A dynamic preset with optional model override.
    Preset {
        /// Preset name; new provider presets do not require a library update.
        name: String,
        /// Optional explicit model override.
        model: Option<String>,
        /// Optional fallback override, mutually exclusive with `model`.
        models: Option<Vec<String>>,
    },
}

impl PerplexitySelection {
    /// Selects an unmodified server-managed preset.
    pub fn preset(name: impl Into<String>) -> Self {
        Self::Preset {
            name: name.into(),
            model: None,
            models: None,
        }
    }

    pub(super) fn validate(&self) -> Result<()> {
        let valid_model = |model: &str| {
            !model.trim().is_empty()
                && model
                    .split_once('/')
                    .is_some_and(|(a, b)| !a.is_empty() && !b.is_empty())
        };
        let valid_models = |models: &[String]| {
            (1..=5).contains(&models.len()) && models.iter().all(|model| valid_model(model))
        };
        let valid = match self {
            Self::Model(model) => valid_model(model),
            Self::Models(models) => valid_models(models),
            Self::Preset {
                name,
                model,
                models,
            } => {
                !name.trim().is_empty()
                    && !(model.is_some() && models.is_some())
                    && model.as_deref().is_none_or(valid_model)
                    && models.as_deref().is_none_or(valid_models)
            }
        };
        if valid {
            Ok(())
        } else {
            Err(Error::Validation(
                "invalid Perplexity model/preset selection".into(),
            ))
        }
    }
}

/// Search filtering; include and exclude domain modes cannot be mixed.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerplexitySearchFilters {
    /// At most twenty domains or URLs, with `-` prefixes for exclusion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_domain_filter: Option<Vec<String>>,
    /// Publication start date in provider MM/DD/YYYY format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_after_date_filter: Option<String>,
    /// Publication end date.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_before_date_filter: Option<String>,
    /// Last-update start date.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated_after_filter: Option<String>,
    /// Last-update end date.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated_before_filter: Option<String>,
    /// Relative age: day, week, month, or year.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_recency_filter: Option<String>,
}

/// Geographic context for hosted web search.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerplexityLocation {
    /// Two-letter country code; required with coordinates.
    pub country: String,
    /// Optional region.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    /// Optional city.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// Latitude, paired with longitude.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latitude: Option<f64>,
    /// Longitude, paired with latitude.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longitude: Option<f64>,
}

/// Web-search settings; unset fields inherit the provider/preset defaults.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerplexityWebSearch {
    /// Standard `web` search or lower-latency `fast` search.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_type: Option<String>,
    /// Named context budget: low, medium, high.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_context_size: Option<String>,
    /// Maximum results per invocation, from 1 to 50.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<u32>,
    /// Total search context token budget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Token budget per page.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens_per_page: Option<u32>,
    /// Domain and date filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<PerplexitySearchFilters>,
    /// Geographic context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_location: Option<PerplexityLocation>,
}

/// Image-search filters.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerplexityImageFilters {
    /// Domain allowlist or exclusion list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain_filter: Option<Vec<String>>,
    /// Permitted image formats.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format_filter: Option<Vec<String>>,
    /// Safe-search setting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub safe_search: Option<bool>,
}

/// Provider-executed tools. Custom local functions use `ModelRequest.tools`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum PerplexityTool {
    /// Hosted web search.
    WebSearch {
        /// Optional search settings.
        #[serde(flatten)]
        options: Box<PerplexityWebSearch>,
    },
    /// Hosted image search.
    ImageSearch {
        /// Maximum results, from 1 to 30.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_results: Option<u32>,
        /// Optional filters.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        filters: Option<PerplexityImageFilters>,
    },
    /// Hosted URL content retrieval.
    FetchUrl {
        /// Maximum URLs per invocation, from 1 to 10.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        max_urls: Option<u32>,
    },
    /// Financial data retrieval.
    FinanceSearch,
    /// People search.
    PeopleSearch,
    /// Hosted code execution, never executed locally by this crate.
    Sandbox,
    /// Remote MCP execution. Credentials are resolved by server label from config.
    Mcp {
        /// Unique label, 1–64 ASCII letters/digits/underscores/hyphens.
        server_label: String,
        /// HTTPS Streamable HTTP server URL.
        server_url: String,
        /// Explicit allowed tools; empty exposes all discovered tools.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_tools: Vec<String>,
        /// Ask the provider to load tool definitions on demand.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        defer_loading: Option<bool>,
    },
    /// An existing provider-managed connector; this does not create one.
    Connector {
        /// Existing connector ID.
        id: String,
        /// Unique tool namespace label.
        server_label: String,
        /// Optional namespace description.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        server_description: Option<String>,
        /// Explicit allowed tools; empty exposes all live tools.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        allowed_tools: Vec<String>,
    },
}

/// Explicit override of a preset's tool-choice default.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", content = "name", rename_all = "snake_case")]
pub enum PerplexityToolChoice {
    /// Provider chooses.
    Auto,
    /// Disable tool calls.
    None,
    /// Require a tool call.
    Required,
    /// Require a named client-executed function.
    Function(String),
    /// Require a hosted tool type, such as image_search.
    Hosted(String),
}

/// Typed per-call overrides. Credentials never belong in this serializable type.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerplexityOptions {
    /// Optional per-call model/preset selection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selection: Option<PerplexitySelection>,
    /// Hosted-tool overrides, merged by the provider with preset tools.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<PerplexityTool>>,
    /// Explicit tool-choice override; absence preserves preset defaults.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<PerplexityToolChoice>,
    /// Maximum agent steps, from 1 to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_steps: Option<u32>,
    /// Storage visibility. False does not disable persistence or continuation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub store: Option<bool>,
    /// Background execution; use submit_background or streaming for this mode.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background: Option<bool>,
    /// Optional output cap; the common request's max_tokens wins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    /// ISO 639-1 language preference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language_preference: Option<String>,
    /// Provider reasoning effort, including its additional `max` level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
}

impl PerplexityOptions {
    /// Sets the provider options on a common model request.
    ///
    /// # Errors
    /// Returns validation errors for invalid hosted tools, or a serialization
    /// error if options cannot be encoded. Leaves the request unchanged on error.
    pub fn apply_to(&self, request: &mut ModelRequest) -> Result<()> {
        super::request::validate_options_before_serialization(self)?;
        request.provider_options = serde_json::json!({"perplexity": serde_json::to_value(self)?});
        Ok(())
    }
}

impl PerplexityTool {
    /// Creates a hosted web-search tool with explicit optional settings.
    pub fn web_search(options: PerplexityWebSearch) -> Self {
        Self::WebSearch {
            options: Box::new(options),
        }
    }
}

/// Remote-server secrets, kept out of model request serialization and Debug.
#[derive(Clone, Default)]
pub struct PerplexityRemoteCredentials {
    /// HTTPS server URL these credentials are authorized for.
    pub server_url: String,
    /// Raw remote access token, not a prefixed bearer header.
    pub authorization: Option<String>,
    /// Extra remote request headers.
    pub headers: BTreeMap<String, String>,
}

impl fmt::Debug for PerplexityRemoteCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("PerplexityRemoteCredentials([REDACTED])")
    }
}

/// Immutable settings for a Perplexity Agent client.
#[derive(Clone)]
pub struct PerplexityConfig {
    /// Required model or preset selection.
    pub selection: PerplexitySelection,
    /// API base URL, without the create route.
    pub base_url: String,
    /// Use `/responses` for creation instead of the canonical `/agent` route.
    pub responses_alias: bool,
    /// Default options; typed per-call values override them.
    pub options: PerplexityOptions,
    /// Total request/stream budget, including retries.
    pub timeout: Duration,
    /// Maximum inactivity while waiting for stream bytes.
    pub idle_timeout: Duration,
    /// Connection-establishment budget.
    pub connect_timeout: Duration,
    /// Bounded retry count; zero disables retries.
    pub max_retries: u32,
    /// Maximum request, JSON response, or reconstructed output bytes.
    pub max_body_bytes: usize,
    /// Maximum bytes in one SSE event.
    pub max_event_bytes: usize,
    /// Maximum bytes downloaded for one generated file.
    pub max_file_bytes: usize,
    /// Remote secrets by server label, injected only into outgoing requests.
    pub remote_credentials: BTreeMap<String, PerplexityRemoteCredentials>,
    /// Optional last-mile payload callback. The result is validated before sending.
    pub on_payload: Option<super::super::PayloadHook>,
    /// Optional observer of sanitized provider replies.
    pub on_response: Option<super::super::ResponseHook>,
}

impl PerplexityConfig {
    /// Creates configuration with bounded defaults and explicit selection.
    pub fn new(selection: PerplexitySelection) -> Self {
        Self {
            selection,
            base_url: "https://api.perplexity.ai/v1".into(),
            responses_alias: false,
            options: PerplexityOptions::default(),
            timeout: Duration::from_secs(600),
            idle_timeout: Duration::from_secs(60),
            connect_timeout: Duration::from_secs(30),
            max_retries: 3,
            max_body_bytes: 32 * 1024 * 1024,
            max_event_bytes: 4 * 1024 * 1024,
            max_file_bytes: 64 * 1024 * 1024,
            remote_credentials: BTreeMap::new(),
            on_payload: None,
            on_response: None,
        }
    }
}

impl fmt::Debug for PerplexityConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PerplexityConfig")
            .field("responses_alias", &self.responses_alias)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}
