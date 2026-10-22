//! Native chat response parsing (`ChatResult` with a GENERIC or COHERE
//! `chatResponse`), shared by the unary and streaming paths.

use serde_json::Value;

use crate::message::{AssistantMessage, ContentBlock};
use crate::model::ModelResponse;
use crate::tool::ToolCall;
use crate::usage::Usage;
use crate::{Error, Result};

fn malformed(message: &str) -> Error {
    Error::Model(format!("invalid OCI GenAI response: {message}"))
}

/// Maps an OCI `usage` object (`promptTokens`, `completionTokens`,
/// `totalTokens`, `promptTokensDetails.cachedTokens`,
/// `completionTokensDetails.reasoningTokens`) onto [`Usage`].
pub(super) fn parse_usage(usage: &Value) -> Usage {
    let count = |value: &Value| value.as_u64().unwrap_or(0);
    let input_tokens = count(&usage["promptTokens"]);
    let output_tokens = count(&usage["completionTokens"]);
    let total = count(&usage["totalTokens"]);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: if total > 0 {
            total
        } else {
            input_tokens + output_tokens
        },
        cache_read_tokens: count(&usage["promptTokensDetails"]["cachedTokens"]),
        reasoning_tokens: count(&usage["completionTokensDetails"]["reasoningTokens"]),
        ..Usage::default()
    }
}

/// Builds a [`ToolCall`], keeping malformed argument JSON as an invalid call
/// the agent loop can feed back instead of failing the turn.
pub(super) fn tool_call(id: &str, name: &str, arguments: &str) -> ToolCall {
    if arguments.trim().is_empty() {
        return ToolCall::new(id, name, Value::Object(Default::default()));
    }
    match serde_json::from_str::<Value>(arguments) {
        Ok(value) => ToolCall::new(id, name, value),
        Err(error) => ToolCall::invalid(id, name, arguments.to_string(), error.to_string()),
    }
}

/// Parses a unary `ChatResult` body.
pub(crate) fn parse_response(body: Value) -> Result<ModelResponse> {
    let chat = body
        .get("chatResponse")
        .filter(|value| value.is_object())
        .ok_or_else(|| malformed("chatResponse is required"))?;
    let (content, tool_calls, finish_reason, usage) = match chat["apiFormat"].as_str() {
        Some("COHERE") => parse_cohere(chat),
        Some("GENERIC") | None => parse_generic(chat)?,
        Some(other) => return Err(malformed(&format!("unsupported apiFormat `{other}`"))),
    };
    Ok(ModelResponse {
        message: AssistantMessage {
            id: None,
            content,
            tool_calls,
            usage,
            origin: None,
        },
        usage,
        finish_reason,
        raw: Some(body),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    })
}

type Parsed = (
    Vec<ContentBlock>,
    Vec<ToolCall>,
    Option<String>,
    Option<Usage>,
);

fn parse_generic(chat: &Value) -> Result<Parsed> {
    let choice = chat["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .ok_or_else(|| malformed("choices must be a non-empty array"))?;
    let message = &choice["message"];
    let mut content = Vec::new();
    if let Some(reasoning) = message["reasoningContent"]
        .as_str()
        .filter(|t| !t.is_empty())
    {
        content.push(ContentBlock::Thinking {
            text: reasoning.to_string(),
            signature: None,
        });
    }
    for part in message["content"].as_array().into_iter().flatten() {
        if part["type"].as_str() == Some("TEXT")
            && let Some(text) = part["text"].as_str().filter(|t| !t.is_empty())
        {
            content.push(ContentBlock::Text(text.to_string()));
        }
    }
    let tool_calls = message["toolCalls"]
        .as_array()
        .into_iter()
        .flatten()
        .map(|call| {
            tool_call(
                call["id"].as_str().unwrap_or_default(),
                call["name"].as_str().unwrap_or_default(),
                call["arguments"].as_str().unwrap_or_default(),
            )
        })
        .collect();
    // Usage is on the response; some deployments put it on the choice.
    let usage = chat
        .get("usage")
        .or_else(|| choice.get("usage"))
        .filter(|value| value.is_object())
        .map(parse_usage);
    Ok((
        content,
        tool_calls,
        choice["finishReason"].as_str().map(str::to_string),
        usage,
    ))
}

fn parse_cohere(chat: &Value) -> Parsed {
    let mut content = Vec::new();
    if let Some(text) = chat["text"].as_str().filter(|t| !t.is_empty()) {
        content.push(ContentBlock::Text(text.to_string()));
    }
    // Cohere v1 tool calls carry no id; synthesize stable ones.
    let tool_calls = chat["toolCalls"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
        .map(|(index, call)| {
            ToolCall::new(
                format!("call_{index}"),
                call["name"].as_str().unwrap_or_default(),
                call.get("parameters")
                    .filter(|value| value.is_object())
                    .cloned()
                    .unwrap_or_else(|| Value::Object(Default::default())),
            )
        })
        .collect();
    (
        content,
        tool_calls,
        chat["finishReason"].as_str().map(str::to_string),
        chat.get("usage")
            .filter(|value| value.is_object())
            .map(parse_usage),
    )
}

#[cfg(test)]
#[path = "response_tests.rs"]
mod tests;
