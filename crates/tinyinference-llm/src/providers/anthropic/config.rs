//! Complete construction policy for Anthropic Messages models.

use std::sync::Arc;

use crate::model::ChatModel;

use super::AnthropicModel;

/// Resolved configuration for an Anthropic Messages provider.
#[derive(Clone)]
pub struct AnthropicConfig<'a> {
    /// Base URL for the Messages API.
    pub endpoint: &'a str,
    /// API credential sent as x-api-key.
    pub api_key: &'a str,
    /// Default model identifier.
    pub model: &'a str,
    /// Fixed temperature override.
    pub temperature_override: Option<f64>,
    /// Model-id glob patterns whose targets reject temperature.
    pub temperature_unsupported_models: &'a [String],
    /// Static headers attached to every request.
    pub extra_headers: &'a [(String, String)],
}

impl std::fmt::Debug for AnthropicConfig<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let endpoint = tinyinference_core::sanitize::redact_url(self.endpoint);
        formatter
            .debug_struct("AnthropicConfig")
            .field("endpoint", &endpoint)
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("temperature_override", &self.temperature_override)
            .field(
                "temperature_unsupported_models",
                &self.temperature_unsupported_models,
            )
            .field(
                "extra_header_names",
                &self
                    .extra_headers
                    .iter()
                    .map(|(name, _)| name)
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// Returns whether an endpoint is Anthropic's first-party Messages API.
pub fn endpoint_is_anthropic_messages(endpoint: &str) -> bool {
    reqwest::Url::parse(endpoint.trim())
        .ok()
        .and_then(|url| url.host_str().map(str::to_ascii_lowercase))
        .is_some_and(|host| host == "api.anthropic.com")
}

/// Builds an Anthropic Messages model from fully resolved configuration.
pub fn build_anthropic_model(config: AnthropicConfig<'_>) -> Arc<dyn ChatModel<()>> {
    let mut model = AnthropicModel::with_base_url(config.api_key, config.endpoint)
        .with_model(config.model)
        .with_temperature_override(config.temperature_override)
        .with_temperature_unsupported_models(config.temperature_unsupported_models.iter().cloned());
    for (name, value) in config.extra_headers {
        model = model.with_header(name.clone(), value.clone());
    }
    Arc::new(model)
}
