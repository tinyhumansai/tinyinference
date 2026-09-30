//! How the model behind a resolved turn is built.

use std::fmt::{self, Debug};
use std::sync::Arc;

use tinyinference_llm::model::ChatModel;
use tinyinference_llm::providers::anthropic::{
    AnthropicConfig, build_anthropic_model, endpoint_is_anthropic_messages,
};
use tinyinference_llm::providers::openai::{
    AuthStyle as LlmAuth, OpenAiConfig, build_local_runtime_chat_model, build_openai_model,
};

use crate::error::{HubError, Operation};
use crate::route::ResolvedTurn;
use crate::taxonomy::{AuthStyle, Protocol, ProviderGroup};

/// What a factory needs to build one model.
#[non_exhaustive]
pub struct ModelSpec<'a> {
    /// The resolved turn.
    pub turn: &'a ResolvedTurn,
    /// The credential resolved for this call, if any.
    pub key: Option<&'a str>,
    /// Headers to send on every request: the kind's own (OpenRouter's
    /// attribution) and the product header when the endpoint is first-party.
    pub extra_headers: &'a [(String, String)],
    /// Whether the endpoint is OpenAI's own, which serves the Responses API.
    pub responses_api: bool,
}

impl Debug for ModelSpec<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelSpec")
            .field("turn", self.turn)
            .field("key", &self.key.map(|_| "<redacted>"))
            .field("responses_api", &self.responses_api)
            .finish_non_exhaustive()
    }
}

/// Builds the model a turn runs on. The default ([`LlmModelFactory`]) uses the
/// `tinyinference-llm` builders; a host or a test may supply its own.
pub trait ModelFactory: Send + Sync + Debug {
    /// Builds a model for `spec`.
    ///
    /// # Errors
    ///
    /// [`HubError::Unsupported`] when the protocol cannot be built here.
    fn build(&self, spec: &ModelSpec<'_>) -> Result<Arc<dyn ChatModel<()>>, HubError>;
}

/// The default factory, over `tinyinference-llm`'s builders.
///
/// The builder signatures it targets (read from `tinyinference-llm` on
/// 2026-09-30):
///
/// * `providers::openai::build_openai_model(OpenAiConfig<'_>) -> Arc<dyn ChatModel<()>>`
///   (every field of the config is set explicitly, so a field added upstream
///   fails this crate's build instead of silently defaulting);
/// * `providers::openai::build_local_runtime_chat_model(provider_name, endpoint,
///   api_key, AuthStyle, model, &[String], Option<f64>, Option<u32>)`;
/// * `providers::anthropic::build_anthropic_model(AnthropicConfig<'_>)` and
///   `endpoint_is_anthropic_messages(&str)`.
///
/// * OpenAI-compatible endpoints use `build_openai_model` (the Responses API
///   only on OpenAI's own host), local runtimes `build_local_runtime_chat_model`
///   (conservative capabilities);
/// * Anthropic uses the native Messages API **only** on the first-party
///   endpoint (open question Q13); on any other endpoint it is spoken to as an
///   OpenAI-compatible server with Anthropic's headers;
/// * a CLI login is not an HTTP model: [`HubError::Unsupported`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LlmModelFactory;

impl LlmModelFactory {
    /// The default factory.
    pub fn new() -> Self {
        Self
    }
}

fn llm_auth(auth: &AuthStyle) -> LlmAuth {
    match auth {
        AuthStyle::Bearer | AuthStyle::SessionJwt => LlmAuth::Bearer,
        AuthStyle::XApiKey => LlmAuth::XApiKey,
        AuthStyle::Anthropic => LlmAuth::Anthropic,
        AuthStyle::None => LlmAuth::None,
        AuthStyle::Custom(name) => LlmAuth::Custom(name.clone()),
    }
}

impl ModelFactory for LlmModelFactory {
    fn build(&self, spec: &ModelSpec<'_>) -> Result<Arc<dyn ChatModel<()>>, HubError> {
        let turn = spec.turn;
        let unsupported = || HubError::Unsupported {
            op: Operation::ChatModel,
            kind: turn.kind.clone(),
        };
        let model = turn.model.as_ref().ok_or_else(unsupported)?.as_str();
        let key = spec.key.unwrap_or("");
        let temperature = turn.temperature.map(|t| f64::from(t.get()));
        match turn.protocol {
            Protocol::CliStream => Err(unsupported()),
            Protocol::AnthropicMessages if endpoint_is_anthropic_messages(&turn.base_url) => {
                Ok(build_anthropic_model(AnthropicConfig {
                    endpoint: &turn.base_url,
                    api_key: key,
                    model,
                    temperature_override: temperature,
                    temperature_unsupported_models: &[],
                    extra_headers: spec.extra_headers,
                }))
            }
            _ if turn.group == ProviderGroup::Local => Ok(build_local_runtime_chat_model(
                turn.kind.as_str(),
                &turn.base_url,
                key,
                llm_auth(&turn.auth),
                model,
                &[],
                temperature,
                None,
            )),
            _ => Ok(build_openai_model(OpenAiConfig {
                provider_name: turn.kind.as_str(),
                endpoint: &turn.base_url,
                api_key: key,
                auth_style: llm_auth(&turn.auth),
                model,
                temperature_unsupported_models: &[],
                temperature_override: temperature,
                merge_system_into_user: false,
                extra_headers: spec.extra_headers,
                native_tool_calling: None,
                vision: None,
                default_provider_options: None,
                responses_api_primary: spec.responses_api,
                responses_omit_max_output_tokens: false,
                extra_query_params: &[],
                user_agent: None,
                explicit_cache_control: false,
            })),
        }
    }
}
