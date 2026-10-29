//! Conversion from portable provider responses into the normalized model API.

use crate::message::{AssistantMessage, ContentBlock};
use crate::model::{ModelRequest, ModelResponse, canonical_finish_reason};
use crate::providers::response::ChatResponse;
use crate::tool::ToolCall;
use crate::usage::{BilledUsage, billed_usage_report};
use tinytools_agent::dialect::NativeToolCall;

/// Converts a provider response to the normalized response shape.
///
/// Native tool calls take precedence. If none are present, text fallback is
/// attempted only when the request advertised tools. The advertised schemas
/// provide positional parameter names for P-Format calls. Unknown tool names
/// are retained for the harness's policy to handle.
#[must_use]
pub fn response_to_model_response(
    response: &ChatResponse<BilledUsage, NativeToolCall>,
    request: &ModelRequest,
) -> ModelResponse {
    let text = response.text.as_deref().unwrap_or_default();
    let (visible_text, tool_calls) = if !response.tool_calls.is_empty() {
        let calls = response
            .tool_calls
            .iter()
            .map(|call| {
                // Zero-argument calls often arrive as `""`: an empty object.
                let arguments = match call.arguments.trim() {
                    "" => "{}",
                    _ => &call.arguments,
                };
                match serde_json::from_str(arguments) {
                    Ok(arguments) => ToolCall::new(call.id.clone(), call.name.clone(), arguments),
                    Err(error) => ToolCall::invalid(
                        call.id.clone(),
                        call.name.clone(),
                        call.arguments.clone(),
                        error.to_string(),
                    ),
                }
            })
            .collect();
        (text.to_owned(), calls)
    } else if request.tools.is_empty() {
        (text.to_owned(), Vec::new())
    } else {
        let registry = request
            .tools
            .iter()
            .map(|tool| {
                (
                    tool.name.clone(),
                    tinytools_agent::PFormatToolParams::from_schema(&tool.parameters),
                )
            })
            .collect();
        let (prose, parsed) = tinytools_agent::parse_tool_calls_with_pformat(text, &registry);
        if parsed.is_empty() {
            (text.to_owned(), Vec::new())
        } else {
            let calls = parsed
                .into_iter()
                .enumerate()
                .map(|(index, call)| {
                    ToolCall::new(
                        call.id.unwrap_or_else(|| format!("call_{index}")),
                        call.name,
                        call.arguments,
                    )
                })
                .collect();
            (prose, calls)
        }
    };

    let mut content = Vec::new();
    if !visible_text.is_empty() {
        content.push(ContentBlock::Text(visible_text));
    }
    if let Some(reasoning) = response
        .reasoning_content
        .as_deref()
        .filter(|reasoning| !reasoning.trim().is_empty())
    {
        content.push(ContentBlock::Thinking {
            text: reasoning.to_owned(),
            signature: None,
        });
    }

    let report = response.usage.as_ref().map(billed_usage_report);
    let usage = report.as_ref().map(|report| report.to_usage());
    let finish_reason = match response.finish_reason.clone() {
        Some(reported) => canonical_finish_reason(reported),
        None if tool_calls.is_empty() => "stop".to_owned(),
        None => "tool_calls".to_owned(),
    };
    ModelResponse {
        output: Vec::new(),
        execution: None,
        message: AssistantMessage {
            id: None,
            content,
            tool_calls,
            usage,
            origin: None,
        },
        usage,
        finish_reason: Some(finish_reason),
        raw: report.and_then(|report| report.raw_meta()),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

#[cfg(test)]
#[path = "response_conversion_tests.rs"]
mod tests;
