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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) previous_response_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) text: Option<Value>,
    #[serde(flatten, skip_serializing_if = "serde_json::Map::is_empty")]
    pub(super) extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Serialize)]
pub(super) struct ResponsesInput {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) role: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub(super) kind: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) content: Vec<ResponsesContentPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) arguments: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) output: Option<String>,
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
    pub(super) status: Option<String>,
    #[serde(default)]
    pub(super) incomplete_details: Option<ResponsesIncompleteDetails>,
    #[serde(default)]
    pub(super) output: Vec<ResponsesOutput>,
    #[serde(default)]
    pub(super) output_text: Option<String>,
    #[serde(default)]
    pub(super) usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize)]
pub(super) struct ResponsesIncompleteDetails {
    #[serde(default)]
    pub(super) reason: Option<String>,
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

fn message_output_text(content: &[ContentBlock]) -> Result<String> {
    let mut output = String::new();
    for block in content {
        match block {
            ContentBlock::Text(text) => output.push_str(text),
            ContentBlock::Json(value) => output.push_str(&value.to_string()),
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => {}
            ContentBlock::Image(_) | ContentBlock::ProviderExtension(_) => {
                return Err(Error::Validation(
                    "tool output cannot be represented by the Responses API".into(),
                ));
            }
        }
    }
    Ok(output)
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
            Message::Tool(tool) => {
                input.push(ResponsesInput {
                    role: None,
                    kind: Some("function_call_output".into()),
                    content: Vec::new(),
                    call_id: Some(tool.tool_call_id.clone()),
                    name: None,
                    arguments: None,
                    output: Some(message_output_text(&tool.content)?),
                });
                continue;
            }
            Message::Assistant(assistant) => {
                let content = input_parts(message)?;
                if !content.is_empty() {
                    input.push(ResponsesInput {
                        role: Some("assistant".into()),
                        kind: None,
                        content,
                        call_id: None,
                        name: None,
                        arguments: None,
                        output: None,
                    });
                }
                for call in &assistant.tool_calls {
                    input.push(ResponsesInput {
                        role: None,
                        kind: Some("function_call".into()),
                        content: Vec::new(),
                        call_id: Some(call.id.clone()),
                        name: Some(call.name.clone()),
                        arguments: Some(serde_json::to_string(&call.arguments)?),
                        output: None,
                    });
                }
                continue;
            }
            Message::User(_) => {}
        }
        let content = input_parts(message)?;
        if content.is_empty() {
            continue;
        }
        let role = normalize_role(message);
        input.push(ResponsesInput {
            role: Some(role.to_string()),
            kind: None,
            content,
            call_id: None,
            name: None,
            arguments: None,
            output: None,
        });
    }

    let instructions = (!instructions_parts.is_empty()).then(|| instructions_parts.join("\n\n"));
    Ok((instructions, input))
}

pub(super) fn responses_text_format(
    format: Option<&crate::model::ResponseFormat>,
) -> Option<Value> {
    use crate::model::ResponseFormat;

    format.map(|format| match format {
        ResponseFormat::Text => serde_json::json!({"format": {"type": "text"}}),
        ResponseFormat::JsonObject => {
            serde_json::json!({"format": {"type": "json_object"}})
        }
        ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
            serde_json::json!({
                "format": {
                    "type": "json_schema",
                    "name": name,
                    "schema": schema,
                    "strict": true
                }
            })
        }
    })
}

pub(super) fn responses_extra_options(options: &Value) -> Result<serde_json::Map<String, Value>> {
    if options.is_null() {
        return Ok(serde_json::Map::new());
    }
    let object = options.as_object().ok_or_else(|| {
        Error::Validation("provider_options for Responses must be a JSON object".into())
    })?;
    const RESERVED: &[&str] = &[
        "model",
        "input",
        "instructions",
        "stream",
        "store",
        "max_output_tokens",
        "tools",
        "tool_choice",
        "previous_response_id",
        "text",
    ];
    Ok(object
        .iter()
        .filter(|(key, _)| !RESERVED.contains(&key.as_str()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
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
        finish_reason: Some(if parsed.status.as_deref() == Some("incomplete") {
            match parsed
                .incomplete_details
                .as_ref()
                .and_then(|details| details.reason.as_deref())
            {
                Some("max_output_tokens") => "length".to_string(),
                Some(reason) => reason.to_string(),
                None => "incomplete".to_string(),
            }
        } else if parsed
            .output
            .iter()
            .any(|item| item.kind.as_deref() == Some("function_call"))
        {
            "tool_calls".to_string()
        } else {
            "stop".to_string()
        }),
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
        assert_eq!(input[0].role.as_deref(), Some("user"));
        assert_eq!(input[0].content[0].kind, "input_text");
        assert_eq!(input[0].content[0].text.as_deref(), Some("hi"));
        // Assistant items must use `output_text`, not `input_text`.
        assert_eq!(input[1].role.as_deref(), Some("assistant"));
        assert_eq!(input[1].content[0].kind, "output_text");
        assert_eq!(input[1].content[0].text.as_deref(), Some("hello"));
    }

    #[test]
    fn extract_text_prefers_output_text_then_scans_content() {
        let with_convenience = ResponsesResponse {
            status: Some("completed".into()),
            incomplete_details: None,
            output: Vec::new(),
            output_text: Some("  final  ".to_string()),
            usage: None,
        };
        assert_eq!(
            extract_responses_text(&with_convenience).as_deref(),
            Some("final")
        );

        let via_content = ResponsesResponse {
            status: Some("completed".into()),
            incomplete_details: None,
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
            status: Some("completed".into()),
            incomplete_details: None,
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

    #[test]
    fn build_input_correlates_function_calls_and_outputs() {
        let messages = vec![
            Message::Assistant(AssistantMessage {
                id: None,
                content: Vec::new(),
                tool_calls: vec![ToolCall::new("call-1", "lookup", json!({"query": "rust"}))],
                usage: None,
            }),
            Message::Tool(crate::message::ToolMessage {
                tool_call_id: "call-1".into(),
                content: vec![ContentBlock::Json(json!({"answer": 42}))],
            }),
        ];
        let (_, input) = build_responses_input(&messages).unwrap();
        assert_eq!(input.len(), 2);
        assert_eq!(input[0].kind.as_deref(), Some("function_call"));
        assert_eq!(input[0].call_id.as_deref(), Some("call-1"));
        assert_eq!(input[0].name.as_deref(), Some("lookup"));
        assert_eq!(input[0].arguments.as_deref(), Some("{\"query\":\"rust\"}"));
        assert_eq!(input[1].kind.as_deref(), Some("function_call_output"));
        assert_eq!(input[1].call_id.as_deref(), Some("call-1"));
        assert_eq!(input[1].output.as_deref(), Some("{\"answer\":42}"));
    }

    #[test]
    fn structured_formats_map_to_responses_text_configuration() {
        let schema = json!({"type": "object"});
        let format = responses_text_format(Some(&crate::model::ResponseFormat::json_schema(
            "answer",
            schema.clone(),
        )))
        .unwrap();
        assert_eq!(format["format"]["type"], "json_schema");
        assert_eq!(format["format"]["name"], "answer");
        assert_eq!(format["format"]["schema"], schema);
    }

    #[test]
    fn incomplete_response_preserves_length_finish_reason() {
        let response = parse_responses_response(json!({
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output_text": "partial"
        }))
        .unwrap();
        assert_eq!(response.finish_reason.as_deref(), Some("length"));
    }
}
