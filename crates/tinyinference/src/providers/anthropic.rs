//! Anthropic Messages API provider with explicit prompt-cache breakpoints.
//!
//! Unlike OpenAI-compatible APIs, Anthropic enables prompt caching by attaching
//! `{"type":"ephemeral"}` as `cache_control` to a system/content block. This
//! adapter turns TinyInference's cacheable prompt segments into that wire shape
//! and maps the provider's cache usage counters back into [`Usage`].

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::message::{AssistantMessage, ContentBlock, Message};
use crate::model::{ChatModel, ModelProfile, ModelRequest, ModelResponse};
use crate::usage::Usage;
use crate::{Error, Result};

const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_MODEL: &str = "claude-sonnet-4-20250514";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// A chat model backed by Anthropic's native Messages API.
#[derive(Debug)]
pub struct AnthropicModel {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
    profile: ModelProfile,
}

impl AnthropicModel {
    /// Creates an Anthropic model using the default Messages API endpoint.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    /// Creates an Anthropic model targeting a Messages-API-compatible endpoint.
    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let model = DEFAULT_MODEL.to_string();
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            profile: ModelProfile {
                provider: Some("anthropic".to_string()),
                model: Some(model.clone()),
                ..ModelProfile::default()
            },
            model,
        }
    }

    /// Overrides the default model id used when a request does not specify one.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self.profile.model = Some(self.model.clone());
        self
    }

    /// Reads `ANTHROPIC_API_KEY`, plus optional `ANTHROPIC_BASE_URL` and
    /// `ANTHROPIC_MODEL`, from the environment.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .map_err(|_| Error::Model("ANTHROPIC_API_KEY is not set".to_string()))?;
        let base_url =
            std::env::var("ANTHROPIC_BASE_URL").unwrap_or_else(|_| DEFAULT_BASE_URL.into());
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.into());
        Ok(Self::with_base_url(key, base_url).with_model(model))
    }

    fn endpoint(&self) -> String {
        if self.base_url.ends_with("/messages") {
            self.base_url.clone()
        } else {
            format!("{}/messages", self.base_url)
        }
    }
}

/// Builds the native Messages API body. Kept pure so cache-control placement is
/// exhaustively testable without a network server.
pub(crate) fn request_body(request: &ModelRequest, default_model: &str) -> Value {
    let cache_enabled = request
        .cache_policy
        .as_ref()
        .is_some_and(|policy| policy.protect_prompt_prefix)
        && request
            .cache_segments
            .iter()
            .any(|segment| segment.cacheable);
    let mut system = Vec::new();
    let mut messages = Vec::new();
    for message in &request.messages {
        let role = match message {
            Message::System(system_message) => {
                system.push(text_block(
                    system_message
                        .content
                        .iter()
                        .filter_map(ContentBlock::as_text)
                        .collect(),
                ));
                continue;
            }
            Message::User(_) | Message::Tool(_) => "user",
            Message::Assistant(_) => "assistant",
        };
        messages.push(json!({ "role": role, "content": message.text() }));
    }
    if cache_enabled {
        if let Some(block) = system.last_mut() {
            block["cache_control"] = json!({ "type": "ephemeral" });
        } else if let Some(message) = messages.first_mut() {
            message["cache_control"] = json!({ "type": "ephemeral" });
        }
    }
    let mut body = json!({
        "model": request.model.as_deref().unwrap_or(default_model),
        "max_tokens": request.max_tokens.unwrap_or(1024),
        "messages": messages,
    });
    if !system.is_empty() {
        body["system"] = Value::Array(system);
    }
    body
}

fn text_block(text: String) -> Value {
    json!({ "type": "text", "text": text })
}

fn parse_response(body: Value) -> Result<ModelResponse> {
    let text = body["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| {
            (block["type"].as_str() == Some("text"))
                .then(|| block["text"].as_str())
                .flatten()
        })
        .collect::<String>();
    let usage = body.get("usage").map(|usage| Usage {
        input_tokens: usage["input_tokens"].as_u64().unwrap_or(0),
        output_tokens: usage["output_tokens"].as_u64().unwrap_or(0),
        cache_read_tokens: usage["cache_read_input_tokens"].as_u64().unwrap_or(0),
        cache_creation_tokens: usage["cache_creation_input_tokens"].as_u64().unwrap_or(0),
        ..Usage::default()
    });
    Ok(ModelResponse {
        message: AssistantMessage {
            id: body["id"].as_str().map(str::to_string),
            content: vec![ContentBlock::Text(text)],
            tool_calls: Vec::new(),
            usage,
        },
        usage,
        finish_reason: body["stop_reason"].as_str().map(str::to_string),
        raw: Some(body),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
    })
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for AnthropicModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    fn cache_identity(&self) -> Option<String> {
        Some(format!("anthropic:{}:{}", self.base_url, self.model))
    }

    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        let response = self
            .client
            .post(self.endpoint())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&request_body(&request, &self.model))
            .send()
            .await
            .map_err(|error| Error::Model(format!("anthropic request failed: {error}")))?;
        let status = response.status();
        let body: Value = response
            .json()
            .await
            .map_err(|error| Error::Model(format!("anthropic response was not JSON: {error}")))?;
        if !status.is_success() {
            return Err(Error::Model(format!(
                "anthropic returned HTTP {status}: {}",
                body["error"]["message"].as_str().unwrap_or("unknown error")
            )));
        }
        parse_response(body)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::cache::CachePolicy;
    use crate::model::{PromptSegment, SegmentRole};

    #[test]
    fn cacheable_system_prefix_becomes_an_anthropic_cache_breakpoint() {
        let request = ModelRequest::new(vec![
            Message::system("stable instructions"),
            Message::user("hello"),
        ])
        .with_cache_segments(vec![PromptSegment {
            id: "system".into(),
            role: SegmentRole::System,
            cacheable: true,
        }])
        .with_cache_policy(CachePolicy {
            protect_prompt_prefix: true,
            ..CachePolicy::default()
        });
        let body = request_body(&request, "test-model");
        assert_eq!(
            body["system"][0]["cache_control"],
            json!({ "type": "ephemeral" })
        );
        assert_eq!(body["messages"][0]["role"], "user");
    }

    #[test]
    fn cache_usage_is_mapped_from_anthropic_response() {
        let response = parse_response(json!({
            "id": "msg_1", "content": [{ "type": "text", "text": "hello" }], "stop_reason": "end_turn",
            "usage": { "input_tokens": 100, "output_tokens": 5, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 10 }
        })).unwrap();
        assert_eq!(response.text(), "hello");
        let usage = response.usage.unwrap();
        assert_eq!(usage.cache_read_tokens, 90);
        assert_eq!(usage.cache_creation_tokens, 10);
    }
}
