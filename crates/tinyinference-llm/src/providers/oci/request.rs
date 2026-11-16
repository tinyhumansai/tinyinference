//! Native `POST /20231130/actions/chat` request construction (GENERIC and
//! COHERE API formats). Pure, so every mapping is testable without a network.
//!
//! Field names follow the OCI Go SDK `generativeaiinference` package
//! (`ChatDetails`, `GenericChatRequest`, `CohereChatRequest`, ...).

use serde_json::{Map, Value, json};

use super::config::{OciApiFormat, OciServing};
use crate::message::{ContentBlock, Message};
use crate::model::{ModelRequest, ReasoningEffort, ToolChoice};

/// Body keys the adapter owns; the same key in `provider_options` is dropped.
const RESERVED_OPTIONS: &[&str] = &[
    "compartmentId",
    "servingMode",
    "chatRequest",
    "apiFormat",
    "messages",
    "message",
    "chatHistory",
    "isStream",
    "tools",
    "toolChoice",
    "toolResults",
    "preambleOverride",
    "maxTokens",
    "maxCompletionTokens",
    "temperature",
    "topP",
    "stop",
    "stopSequences",
];

/// Whether `key` can be a native `chatRequest` field.
///
/// Native OCI field names are camelCase (Go SDK `GenericChatRequest`,
/// `CohereChatRequest`). A snake_case key such as `prompt_cache_key` is an
/// OpenAI-wire option a host set for OpenAI-compatible providers; the native
/// endpoint rejects any unknown field with "Please pass in correct format of
/// request", so those keys are dropped rather than forwarded.
fn is_native_field_name(key: &str) -> bool {
    !key.is_empty() && !key.contains('_')
}

/// Builds the full `ChatDetails` body.
pub(super) fn chat_details(
    request: &ModelRequest,
    default_model: &str,
    serving: &OciServing,
    compartment_id: &str,
    format: OciApiFormat,
    streaming: bool,
) -> Value {
    let model_id = request.model.as_deref().unwrap_or(default_model);
    let serving_mode = match serving {
        OciServing::OnDemand => json!({ "servingType": "ON_DEMAND", "modelId": model_id }),
        OciServing::Dedicated { endpoint_id } => {
            json!({ "servingType": "DEDICATED", "endpointId": endpoint_id })
        }
    };
    let mut chat_request = match format {
        OciApiFormat::Generic => generic_request(request, model_id),
        OciApiFormat::Cohere => cohere_request(request),
    };
    if streaming {
        chat_request["isStream"] = Value::Bool(true);
        chat_request["streamOptions"] = json!({ "isIncludeUsage": true });
    }
    if let (Some(extra), Some(object)) = (
        request.provider_options.as_object(),
        chat_request.as_object_mut(),
    ) {
        for (key, value) in extra {
            if RESERVED_OPTIONS.contains(&key.as_str()) {
                continue;
            }
            if !is_native_field_name(key) {
                tracing::debug!(key = %key, "[oci] dropping non-native provider option");
                continue;
            }
            object.insert(key.clone(), value.clone());
        }
    }
    json!({
        "compartmentId": compartment_id,
        "servingMode": serving_mode,
        "chatRequest": chat_request,
    })
}

fn text_part(text: &str) -> Option<Value> {
    (!text.is_empty()).then(|| json!({ "type": "TEXT", "text": text }))
}

fn generic_parts(content: &[ContentBlock], allow_images: bool) -> Vec<Value> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => text_part(text),
            ContentBlock::Json(value) => text_part(&value.to_string()),
            ContentBlock::Image(image) if allow_images => {
                Some(json!({ "type": "IMAGE", "imageUrl": { "url": image.url } }))
            }
            _ => None,
        })
        .collect()
}

fn plain_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => Some(text.clone()),
            ContentBlock::Json(value) => Some(value.to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether the GENERIC token cap must be sent as `maxCompletionTokens`.
///
/// OCI forwards OpenAI models to the OpenAI API, where the reasoning models
/// (gpt-5 family, o-series) reject `max_tokens` with `unsupported_parameter`.
/// Every current OpenAI chat model accepts `max_completion_tokens`, so the
/// whole `openai.` family uses it; other vendors keep `maxTokens`.
fn uses_max_completion_tokens(model_id: &str) -> bool {
    model_id.starts_with("openai.")
}

/// Whether the model accepts only its default sampling parameters.
///
/// OpenAI reasoning models reject any non-default `temperature` (and
/// `top_p`), so the adapter omits both for them rather than failing the call.
fn rejects_sampling_parameters(model_id: &str) -> bool {
    model_id
        .strip_prefix("openai.")
        .is_some_and(crate::providers::openai::is_reasoning_model)
}

fn generic_request(request: &ModelRequest, model_id: &str) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    for message in &request.messages {
        match message {
            Message::System(m) => {
                messages
                    .push(json!({ "role": "SYSTEM", "content": generic_parts(&m.content, false) }));
            }
            Message::User(m) => {
                messages
                    .push(json!({ "role": "USER", "content": generic_parts(&m.content, true) }));
            }
            Message::Assistant(m) => {
                let mut value = json!({ "role": "ASSISTANT" });
                let parts = generic_parts(&m.content, false);
                if !parts.is_empty() {
                    value["content"] = Value::Array(parts);
                }
                if !m.tool_calls.is_empty() {
                    value["toolCalls"] = Value::Array(
                        m.tool_calls
                            .iter()
                            .map(|call| {
                                json!({
                                    "type": "FUNCTION",
                                    "id": call.id,
                                    "name": call.name,
                                    "arguments": call.arguments.to_string(),
                                })
                            })
                            .collect(),
                    );
                }
                messages.push(value);
            }
            Message::Tool(m) => {
                let mut parts = generic_parts(&m.content, false);
                if parts.is_empty() {
                    parts.push(json!({ "type": "TEXT", "text": "" }));
                }
                messages.push(json!({
                    "role": "TOOL",
                    "toolCallId": m.tool_call_id,
                    "content": parts,
                }));
            }
            Message::Custom(_) => {}
        }
    }
    let mut body = Map::new();
    body.insert("apiFormat".into(), json!("GENERIC"));
    body.insert("messages".into(), Value::Array(messages));
    if let Some(max_tokens) = request.max_tokens {
        let key = if uses_max_completion_tokens(model_id) {
            "maxCompletionTokens"
        } else {
            "maxTokens"
        };
        body.insert(key.into(), json!(max_tokens));
    }
    if rejects_sampling_parameters(model_id) {
        if request.temperature.is_some() || request.top_p.is_some() {
            tracing::debug!(
                model = model_id,
                "[oci] omitting temperature/top_p for reasoning model"
            );
        }
    } else {
        if let Some(temperature) = request.temperature {
            body.insert("temperature".into(), json!(temperature));
        }
        if let Some(top_p) = request.top_p {
            body.insert("topP".into(), json!(top_p));
        }
    }
    if !request.stop_sequences.is_empty() {
        body.insert("stop".into(), json!(request.stop_sequences));
    }
    if !request.tools.is_empty() {
        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|tool| {
                json!({
                    "type": "FUNCTION",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                })
            })
            .collect();
        body.insert("tools".into(), Value::Array(tools));
        body.insert(
            "toolChoice".into(),
            match &request.tool_choice {
                ToolChoice::Auto => json!({ "type": "AUTO" }),
                ToolChoice::None => json!({ "type": "NONE" }),
                ToolChoice::Required => json!({ "type": "REQUIRED" }),
                ToolChoice::Tool(name) => json!({ "type": "FUNCTION", "name": name }),
            },
        );
    }
    if let Some(effort) = request.reasoning.as_ref().and_then(|r| r.effort) {
        body.insert(
            "reasoningEffort".into(),
            json!(match effort {
                ReasoningEffort::None => "NONE",
                ReasoningEffort::Minimal => "MINIMAL",
                ReasoningEffort::Low => "LOW",
                ReasoningEffort::Medium => "MEDIUM",
                ReasoningEffort::High | ReasoningEffort::XHigh => "HIGH",
            }),
        );
    }
    Value::Object(body)
}

/// Maps a JSON-schema `properties` object to Cohere `parameterDefinitions`.
fn cohere_tool(tool: &crate::tool::ToolSchema) -> Value {
    let required: Vec<&str> = tool
        .parameters
        .get("required")
        .and_then(Value::as_array)
        .map(|names| names.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut definitions = Map::new();
    if let Some(properties) = tool.parameters.get("properties").and_then(Value::as_object) {
        for (name, schema) in properties {
            let mut definition = json!({
                "type": schema.get("type").and_then(Value::as_str).unwrap_or("string"),
                "isRequired": required.contains(&name.as_str()),
            });
            if let Some(description) = schema.get("description").and_then(Value::as_str) {
                definition["description"] = json!(description);
            }
            definitions.insert(name.clone(), definition);
        }
    }
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameterDefinitions": definitions,
    })
}

fn cohere_request(request: &ModelRequest) -> Value {
    let mut preamble: Vec<String> = Vec::new();
    let mut history: Vec<Value> = Vec::new();
    let mut tool_results: Vec<Value> = Vec::new();
    let mut known_calls: Vec<(String, String, Value)> = Vec::new();
    let mut message = String::new();

    let count = request.messages.len();
    // The trailing run of Tool messages become `toolResults`; the final user
    // message (when last) becomes `message`; everything else is history.
    let mut trailing_tools_start = count;
    while trailing_tools_start > 0
        && matches!(request.messages[trailing_tools_start - 1], Message::Tool(_))
    {
        trailing_tools_start -= 1;
    }
    for (index, entry) in request.messages.iter().enumerate() {
        match entry {
            Message::System(m) => preamble.push(plain_text(&m.content)),
            Message::User(m) => {
                let text = plain_text(&m.content);
                if index + 1 == count {
                    message = text;
                } else {
                    history.push(json!({ "role": "USER", "message": text }));
                }
            }
            Message::Assistant(m) => {
                let mut value = json!({ "role": "CHATBOT", "message": plain_text(&m.content) });
                if !m.tool_calls.is_empty() {
                    value["toolCalls"] = Value::Array(
                        m.tool_calls
                            .iter()
                            .map(|c| json!({ "name": c.name, "parameters": c.arguments }))
                            .collect(),
                    );
                }
                for call in &m.tool_calls {
                    known_calls.push((call.id.clone(), call.name.clone(), call.arguments.clone()));
                }
                history.push(value);
            }
            Message::Tool(m) => {
                let (name, parameters) = known_calls
                    .iter()
                    .find(|(id, _, _)| *id == m.tool_call_id)
                    .map(|(_, name, args)| (name.clone(), args.clone()))
                    .unwrap_or_else(|| (m.tool_call_id.clone(), json!({})));
                let result = json!({
                    "call": { "name": name, "parameters": parameters },
                    "outputs": [{ "output": plain_text(&m.content) }],
                });
                if index >= trailing_tools_start {
                    tool_results.push(result);
                } else {
                    history.push(json!({ "role": "TOOL", "toolResults": [result] }));
                }
            }
            Message::Custom(_) => {}
        }
    }
    let mut body = Map::new();
    body.insert("apiFormat".into(), json!("COHERE"));
    body.insert("message".into(), json!(message));
    if !history.is_empty() {
        body.insert("chatHistory".into(), Value::Array(history));
    }
    if !preamble.is_empty() {
        body.insert("preambleOverride".into(), json!(preamble.join("\n\n")));
    }
    if !tool_results.is_empty() {
        body.insert("toolResults".into(), Value::Array(tool_results));
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".into(),
            Value::Array(request.tools.iter().map(cohere_tool).collect()),
        );
    }
    if let Some(max_tokens) = request.max_tokens {
        body.insert("maxTokens".into(), json!(max_tokens));
    }
    if let Some(temperature) = request.temperature {
        body.insert("temperature".into(), json!(temperature));
    }
    if let Some(top_p) = request.top_p {
        body.insert("topP".into(), json!(top_p));
    }
    if !request.stop_sequences.is_empty() {
        body.insert("stopSequences".into(), json!(request.stop_sequences));
    }
    Value::Object(body)
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
