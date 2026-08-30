//! OpenAI **Responses API** (`/v1/responses`) request/response translation.
//!
//! A second wire shape [`OpenAiModel`](super::OpenAiModel) can speak, selected
//! by [`with_responses_api_primary`](super::OpenAiModel::with_responses_api_primary).
//! Where Chat Completions uses `messages` / `choices`, the Responses API uses
//! `input` / `instructions` (the system prompt) / `output`. It is the wire the
//! OpenAI Codex OAuth path requires (paired with `with_extra_query_param` +
//! `with_user_agent`).
//!
//! This first port is **text-in / text-out**: system messages fold into
//! `instructions`, user/assistant/tool turns become `input` items, and the
//! terminal `output_text` (or the first `output_text` content part) becomes the
//! assistant reply. Native tool calls over `/responses` and true SSE streaming
//! are follow-ups; the harness embeds tool specs in the prompt for this path
//! (its [`profile`](super::OpenAiModel) advertises the caller's chosen
//! `tool_calling`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::message::{AssistantMessage, ContentBlock, Message};
use crate::model::{ModelResponse, ToolChoice};
use crate::tool::{ToolCall, ToolSchema};
use crate::usage::Usage;
use crate::{Error, Result};

/// The `/v1/responses` request body.
#[derive(Debug, Serialize)]
pub(super) struct ResponsesRequest {
    pub(super) model: String,
    pub(super) input: Vec<ResponsesInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) store: Option<bool>,
    /// `max_output_tokens` — the Responses-API output cap, carrying the request's
    /// `max_tokens`. Omitted for the Codex OAuth backend, which rejects it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) tools: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) tool_choice: Option<Value>,
}

#[derive(Debug, Serialize)]
pub(super) struct ResponsesInput {
    pub(super) role: String,
    pub(super) content: Vec<ResponsesContentPart>,
}

#[derive(Debug, Serialize)]
pub(super) struct ResponsesContentPart {
    #[serde(rename = "type")]
    pub(super) kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) image_url: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesResponse {
    #[serde(default)]
    pub(super) output: Vec<ResponsesOutput>,
    #[serde(default)]
    pub(super) output_text: Option<String>,
    #[serde(default)]
    pub(super) usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesOutput {
    #[serde(rename = "type", default)]
    pub(super) kind: Option<String>,
    #[serde(default)]
    pub(super) content: Vec<ResponsesContent>,
    #[serde(default)]
    pub(super) call_id: Option<String>,
    #[serde(default)]
    pub(super) name: Option<String>,
    #[serde(default)]
    pub(super) arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesContent {
    #[serde(rename = "type")]
    pub(super) kind: Option<String>,
    pub(super) text: Option<String>,
}

/// Responses-API usage block (`input_tokens` / `output_tokens`).
#[derive(Debug, Deserialize)]
pub(super) struct ResponsesUsage {
    #[serde(default)]
    pub(super) input_tokens: Option<u64>,
    #[serde(default)]
    pub(super) output_tokens: Option<u64>,
}

/// Concatenates the visible text of a message's content blocks.
fn message_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(ContentBlock::as_text)
        .collect::<Vec<_>>()
        .join("")
}

fn input_parts(message: &Message) -> Result<Vec<ResponsesContentPart>> {
    let role = normalize_role(message);
    let content = match message {
        Message::System(message) => &message.content,
        Message::User(message) => &message.content,
        Message::Assistant(message) => &message.content,
        Message::Tool(message) => &message.content,
    };
    let mut parts = Vec::new();
    for block in content {
        match block {
            ContentBlock::Text(text) if !text.trim().is_empty() => {
                parts.push(ResponsesContentPart {
                    kind: if role == "assistant" {
                        "output_text".into()
                    } else {
                        "input_text".into()
                    },
                    text: Some(text.clone()),
                    image_url: None,
                });
            }
            ContentBlock::Json(value) => parts.push(ResponsesContentPart {
                kind: if role == "assistant" {
                    "output_text".into()
                } else {
                    "input_text".into()
                },
                text: Some(value.to_string()),
                image_url: None,
            }),
            ContentBlock::Image(image) if matches!(message, Message::User(_)) => {
                parts.push(ResponsesContentPart {
                    kind: "input_image".into(),
                    text: None,
                    image_url: Some(image.url.clone()),
                });
            }
            ContentBlock::Image(_) => {
                return Err(Error::Validation(
                    "Responses API images are supported only in user messages".into(),
                ));
            }
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::ProviderExtension(_) => {
                return Err(Error::Validation(
                    "provider extension content cannot be represented by the Responses API".into(),
                ));
            }
            ContentBlock::Text(_) => {}
        }
    }
    Ok(parts)
}

/// Normalizes a message role for the Responses API: assistant + tool turns fold
/// into `assistant` (which the API keys to `output_text`), everything else to
/// `user` (`input_text`). Mirrors the host `normalize_responses_role`.
fn normalize_role(message: &Message) -> &'static str {
    match message {
        Message::Assistant(_) | Message::Tool(_) => "assistant",
        _ => "user",
    }
}

/// Splits a provider-neutral message list into the Responses `instructions`
/// (concatenated system text) and `input` items. Empty-text turns are skipped;
/// the content-part `kind` tracks the *normalized* role (`output_text` for
/// assistant/tool, `input_text` otherwise) — the API rejects `input_text` on an
/// assistant item.
pub(super) fn build_responses_input(
    messages: &[Message],
) -> Result<(Option<String>, Vec<ResponsesInput>)> {
    let mut instructions_parts = Vec::new();
    let mut input = Vec::new();

    for message in messages {
        match message {
            Message::System(m) => {
                let t = message_text(&m.content);
                if !t.trim().is_empty() {
                    instructions_parts.push(t);
                }
                continue;
            }
            Message::User(_) | Message::Assistant(_) | Message::Tool(_) => {}
        }
        let content = input_parts(message)?;
        if content.is_empty() {
            continue;
        }
        let role = normalize_role(message);
        input.push(ResponsesInput {
            role: role.to_string(),
            content,
        });
    }

    let instructions = (!instructions_parts.is_empty()).then(|| instructions_parts.join("\n\n"));
    Ok((instructions, input))
}

pub(super) fn responses_tools(tools: &[ToolSchema]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            })
        })
        .collect()
}

pub(super) fn responses_tool_choice(choice: &ToolChoice, has_tools: bool) -> Option<Value> {
    if !has_tools {
        return None;
    }
    Some(match choice {
        ToolChoice::Auto => Value::String("auto".into()),
        ToolChoice::None => Value::String("none".into()),
        ToolChoice::Required => Value::String("required".into()),
        ToolChoice::Tool(name) => serde_json::json!({"type": "function", "name": name}),
    })
}

/// Extracts the assistant text from a Responses body: the convenience
/// `output_text` field first, else the first `output_text` content part.
pub(super) fn extract_responses_text(response: &ResponsesResponse) -> Option<String> {
    if let Some(text) = response
        .output_text
        .as_deref()
        .map(str::trim)
        .filter(|t| !t.is_empty())
    {
        return Some(text.to_string());
    }
    for item in &response.output {
        for content in &item.content {
            if content.kind.as_deref() == Some("output_text")
                && let Some(text) = content
                    .text
                    .as_deref()
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
            {
                return Some(text.to_string());
            }
        }
    }
    None
}

/// Parses a raw `/v1/responses` JSON body into a [`ModelResponse`] (text reply).
pub(super) fn parse_responses_response(value: Value) -> Result<ModelResponse> {
    let parsed: ResponsesResponse = serde_json::from_value(value.clone())?;
    let text = extract_responses_text(&parsed).unwrap_or_default();
    let tool_calls = parsed
        .output
        .iter()
        .filter(|item| item.kind.as_deref() == Some("function_call"))
        .map(|item| {
            let id = item.call_id.clone().ok_or_else(|| {
                Error::Serialization(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Responses function_call missing call_id",
                )))
            })?;
            let name = item.name.clone().ok_or_else(|| {
                Error::Serialization(serde_json::Error::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Responses function_call missing name",
                )))
            })?;
            let raw = item.arguments.clone().unwrap_or_else(|| "{}".into());
            Ok(match serde_json::from_str(&raw) {
                Ok(arguments) => ToolCall::new(id, name, arguments),
                Err(error) => ToolCall::invalid(id, name, raw, error.to_string()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let usage = parsed.usage.as_ref().map(|u| Usage {
        input_tokens: u.input_tokens.unwrap_or(0),
        output_tokens: u.output_tokens.unwrap_or(0),
        total_tokens: u.input_tokens.unwrap_or(0) + u.output_tokens.unwrap_or(0),
        ..Usage::default()
    });
    Ok(ModelResponse {
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text(text)],
            tool_calls,
            usage,
        },
        usage,
        finish_reason: Some(
            if parsed
                .output
                .iter()
                .any(|item| item.kind.as_deref() == Some("function_call"))
            {
                "tool_calls".to_string()
            } else {
                "stop".to_string()
            },
        ),
        raw: Some(value),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::Message;
    use serde_json::json;

    #[test]
    fn build_input_folds_system_into_instructions_and_keys_roles() {
        let messages = vec![
            Message::system("be terse"),
            Message::system("and correct"),
            Message::user("hi"),
            Message::assistant("hello"),
            Message::user("  "), // empty → skipped
        ];
        let (instructions, input) = build_responses_input(&messages).unwrap();
        assert_eq!(instructions.as_deref(), Some("be terse\n\nand correct"));
        assert_eq!(input.len(), 2);
        assert_eq!(input[0].role, "user");
        assert_eq!(input[0].content[0].kind, "input_text");
        assert_eq!(input[0].content[0].text.as_deref(), Some("hi"));
        // Assistant items must use `output_text`, not `input_text`.
        assert_eq!(input[1].role, "assistant");
        assert_eq!(input[1].content[0].kind, "output_text");
        assert_eq!(input[1].content[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn extract_text_prefers_output_text_then_scans_content() {
        let with_convenience = ResponsesResponse {
            output: Vec::new(),
            output_text: Some("  final  ".to_string()),
            usage: None,
        };
        assert_eq!(
            extract_responses_text(&with_convenience).as_deref(),
            Some("final")
        );

        let via_content = ResponsesResponse {
            output: vec![ResponsesOutput {
                kind: Some("message".into()),
                content: vec![
                    ResponsesContent {
                        kind: Some("reasoning".into()),
                        text: Some("...".into()),
                    },
                    ResponsesContent {
                        kind: Some("output_text".into()),
                        text: Some("answer".into()),
                    },
                ],
                call_id: None,
                name: None,
                arguments: None,
            }],
            output_text: None,
            usage: None,
        };
        assert_eq!(
            extract_responses_text(&via_content).as_deref(),
            Some("answer")
        );

        let empty = ResponsesResponse {
            output: Vec::new(),
            output_text: None,
            usage: None,
        };
        assert_eq!(extract_responses_text(&empty), None);
    }

    #[test]
    fn parse_maps_text_and_usage_onto_model_response() {
        let body = json!({
            "output_text": "the answer",
            "usage": { "input_tokens": 12, "output_tokens": 5 }
        });
        let resp = parse_responses_response(body).unwrap();
        assert_eq!(resp.text(), "the answer");
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        let usage = resp.usage.expect("usage mapped");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.total_tokens, 17);
    }

    #[test]
    fn parse_tolerates_a_body_without_output() {
        let resp = parse_responses_response(json!({ "id": "resp_1" })).unwrap();
        assert_eq!(resp.text(), "");
    }

    #[test]
    fn parse_rejects_incompatible_output_shape() {
        assert!(parse_responses_response(json!({"output": "not-an-array"})).is_err());
    }

    #[test]
    fn build_input_preserves_user_images() {
        use crate::message::{ImageRef, UserMessage};

        let message = Message::User(UserMessage {
            content: vec![
                ContentBlock::Text("inspect".into()),
                ContentBlock::Image(ImageRef {
                    url: "https://example.test/image.png".into(),
                    mime_type: Some("image/png".into()),
                }),
            ],
        });
        let (_, input) = build_responses_input(&[message]).unwrap();
        assert_eq!(input[0].content.len(), 2);
        assert_eq!(input[0].content[1].kind, "input_image");
        assert_eq!(
            input[0].content[1].image_url.as_deref(),
            Some("https://example.test/image.png")
        );
    }

    #[test]
    fn responses_request_preserves_tools_and_choice() {
        let tools = vec![ToolSchema::new(
            "lookup",
            "Look up a value",
            json!({"type": "object"}),
        )];
        assert_eq!(responses_tools(&tools)[0]["name"], "lookup");
        assert_eq!(
            responses_tool_choice(&ToolChoice::Tool("lookup".into()), true).unwrap()["name"],
            "lookup"
        );
    }
}
