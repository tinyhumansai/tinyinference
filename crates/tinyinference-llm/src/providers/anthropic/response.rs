//! Messages API response parsing shared by the unary and streaming paths.

use serde_json::Value;

use crate::message::{AssistantMessage, ContentBlock};
use crate::model::ModelResponse;
use crate::tool::ToolCall;
use crate::usage::Usage;
use crate::{Error, Result};

/// Maps a `usage` object onto [`Usage`]. Anthropic reports the three input
/// classes separately; `input_tokens` here is their sum so it stays the "size
/// of the prompt" every other adapter reports, with the cache split preserved
/// alongside.
pub(super) fn parse_usage(usage: &Value) -> Usage {
    let uncached_input_tokens = usage["input_tokens"].as_u64().unwrap_or(0);
    let cache_read_tokens = usage["cache_read_input_tokens"].as_u64().unwrap_or(0);
    let cache_creation_tokens = usage["cache_creation_input_tokens"].as_u64().unwrap_or(0);
    let input_tokens = uncached_input_tokens + cache_read_tokens + cache_creation_tokens;
    let output_tokens = usage["output_tokens"].as_u64().unwrap_or(0);
    Usage {
        input_tokens,
        output_tokens,
        total_tokens: input_tokens + output_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        ..Usage::default()
    }
}

pub(crate) fn parse_response(body: Value) -> Result<ModelResponse> {
    let object = body
        .as_object()
        .ok_or_else(|| malformed("response must be an object"))?;
    let id = required_string(object.get("id"), "id")?.to_string();
    let stop_reason = required_string(object.get("stop_reason"), "stop_reason")?.to_string();
    let blocks = object
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("content must be an array"))?;
    let usage_value = object
        .get("usage")
        .ok_or_else(|| malformed("usage is required"))?;
    let usage = parse_usage_checked(usage_value)?;
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();
    for block in blocks {
        let block_type = required_string(block.get("type"), "content[].type")?;
        match block_type {
            "text" => content.push(ContentBlock::Text(
                required_string(block.get("text"), "content[].text")?.to_string(),
            )),
            "tool_use" => {
                let arguments = block
                    .get("input")
                    .filter(|value| value.is_object())
                    .ok_or_else(|| malformed("content[].input must be an object"))?;
                tool_calls.push(ToolCall::new(
                    required_string(block.get("id"), "content[].id")?,
                    required_string(block.get("name"), "content[].name")?,
                    arguments.clone(),
                ));
            }
            "thinking" => content.push(ContentBlock::Thinking {
                text: required_string(block.get("thinking"), "content[].thinking")?.to_string(),
                signature: Some(
                    required_string(block.get("signature"), "content[].signature")?.to_string(),
                ),
            }),
            "redacted_thinking" => content.push(ContentBlock::RedactedThinking {
                data: required_string(block.get("data"), "content[].data")?.to_string(),
            }),
            other => {
                return Err(malformed(&format!(
                    "unsupported content block type: {other}"
                )));
            }
        }
    }
    Ok(ModelResponse {
        message: AssistantMessage {
            id: Some(id),
            content,
            tool_calls,
            usage: Some(usage),
        },
        usage: Some(usage),
        finish_reason: Some(stop_reason),
        raw: Some(body),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    })
}

fn malformed(message: &str) -> Error {
    Error::Model(format!("invalid Anthropic response: {message}"))
}

fn required_string<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a str> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| malformed(&format!("{field} must be a non-empty string")))
}

fn parse_usage_checked(usage: &Value) -> Result<Usage> {
    let object = usage
        .as_object()
        .ok_or_else(|| malformed("usage must be an object"))?;
    let counter = |name: &str, required: bool| -> Result<u64> {
        match object.get(name) {
            Some(value) => value
                .as_u64()
                .ok_or_else(|| malformed(&format!("usage.{name} must be an unsigned integer"))),
            None if required => Err(malformed(&format!("usage.{name} is required"))),
            None => Ok(0),
        }
    };
    let uncached = counter("input_tokens", true)?;
    let cache_read_tokens = counter("cache_read_input_tokens", false)?;
    let cache_creation_tokens = counter("cache_creation_input_tokens", false)?;
    let output_tokens = counter("output_tokens", true)?;
    let input_tokens = uncached
        .checked_add(cache_read_tokens)
        .and_then(|value| value.checked_add(cache_creation_tokens))
        .ok_or_else(|| malformed("usage input-token total overflowed"))?;
    let total_tokens = input_tokens
        .checked_add(output_tokens)
        .ok_or_else(|| malformed("usage total_tokens overflowed"))?;
    Ok(Usage {
        input_tokens,
        output_tokens,
        total_tokens,
        cache_read_tokens,
        cache_creation_tokens,
        ..Usage::default()
    })
}
