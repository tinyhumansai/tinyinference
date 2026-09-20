//! Messages API request construction, kept pure so cache-control placement is
//! exhaustively testable without a network server.

use serde_json::{Value, json};

use crate::message::{ContentBlock, ImageRef, Message};
use crate::model::{ModelRequest, ReasoningEffort, ToolChoice};

use super::DEFAULT_MAX_TOKENS;

/// Top-level body fields the adapter owns. A matching key in
/// `provider_options` is dropped rather than allowed to overwrite the
/// normalized request.
const RESERVED_OPTIONS: &[&str] = &[
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "max_tokens",
    "temperature",
    "top_p",
    "stop_sequences",
    "stream",
    // The harness's provider-neutral routing hint (`prompt_cache_key`, named
    // after OpenAI's). Anthropic's cache is keyed on content alone and the API
    // rejects unknown top-level fields, so it is consumed here, not forwarded.
    "prompt_cache_key",
];

/// Builds the native Messages API body.
pub(crate) fn request_body(request: &ModelRequest, default_model: &str) -> Value {
    let cache_enabled = request.wants_prompt_cache_breakpoints();

    let mut system = Vec::new();
    let mut messages: Vec<Value> = Vec::new();
    for message in &request.messages {
        match message {
            Message::System(system_message) => {
                system.extend(text_only_blocks(&system_message.content));
            }
            Message::User(user_message) => {
                push_message(&mut messages, "user", content_blocks(&user_message.content));
            }
            Message::Assistant(assistant_message) => {
                let mut content = assistant_blocks(&assistant_message.content);
                content.extend(assistant_message.tool_calls.iter().map(|call| {
                    json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": if call.arguments.is_object() { call.arguments.clone() } else { json!({}) },
                    })
                }));
                push_message(&mut messages, "assistant", content);
            }
            Message::Tool(tool_message) => {
                let blocks = content_blocks(&tool_message.content);
                let content = if blocks.is_empty() {
                    Value::String(String::new())
                } else {
                    Value::Array(blocks)
                };
                push_message(
                    &mut messages,
                    "user",
                    vec![json!({
                        "type": "tool_result",
                        "tool_use_id": tool_message.tool_call_id,
                        "content": content,
                    })],
                );
            }
            // Host-side out-of-band record; never sent to the provider.
            Message::Custom(_) => {}
        }
    }

    let mut tools: Vec<Value> = request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.name,
                "description": tool.description,
                "input_schema": if tool.parameters.is_object() {
                    tool.parameters.clone()
                } else {
                    json!({ "type": "object", "properties": {} })
                },
            })
        })
        .collect();

    if cache_enabled {
        mark_ephemeral(tools.last_mut());
        mark_ephemeral(system.last_mut());
        mark_ephemeral(
            messages
                .last_mut()
                .and_then(|message| message["content"].as_array_mut())
                .and_then(|blocks| blocks.last_mut()),
        );
    }

    let mut body = json!({
        "model": request.model.as_deref().unwrap_or(default_model),
        "max_tokens": request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS),
        "messages": messages,
    });
    if !system.is_empty() {
        body["system"] = Value::Array(system);
    }
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
        body["tool_choice"] = match &request.tool_choice {
            ToolChoice::Auto => json!({ "type": "auto" }),
            ToolChoice::None => json!({ "type": "none" }),
            ToolChoice::Required => json!({ "type": "any" }),
            ToolChoice::Tool(name) => json!({ "type": "tool", "name": name }),
        };
    }
    if let Some(temperature) = request.temperature {
        body["temperature"] = json!(clamp_temperature(temperature));
    }
    if let Some(top_p) = request.top_p {
        body["top_p"] = json!(top_p);
    }
    if !request.stop_sequences.is_empty() {
        body["stop_sequences"] = json!(request.stop_sequences);
    }
    if let (Some(extra), Some(object)) =
        (request.provider_options.as_object(), body.as_object_mut())
    {
        for (key, value) in extra {
            if !RESERVED_OPTIONS.contains(&key.as_str()) {
                object.insert(key.clone(), value.clone());
            }
        }
    }
    if let Some(reasoning) = &request.reasoning {
        if let Some(budget_tokens) = reasoning.budget_tokens {
            body["thinking"] = json!({
                "type": "enabled",
                "budget_tokens": budget_tokens,
            });
        } else if let Some(effort) = reasoning.effort
            && effort != ReasoningEffort::None
        {
            body["thinking"] = json!({ "type": "adaptive" });
            body["output_config"] = json!({
                "effort": match effort {
                    ReasoningEffort::Minimal => "low",
                    ReasoningEffort::Low => "low",
                    ReasoningEffort::Medium => "medium",
                    ReasoningEffort::High => "high",
                    ReasoningEffort::XHigh => "max",
                    ReasoningEffort::None => unreachable!(),
                },
            });
        }
    }
    body
}

/// Anthropic accepts `0.0..=1.0`; OpenAI-style callers routinely pass up to
/// `2.0`, which the API rejects outright. Clamping keeps a shared temperature
/// setting usable across providers instead of failing the call.
pub(super) fn clamp_temperature(temperature: f64) -> f64 {
    temperature.clamp(0.0, 1.0)
}

/// Appends `content` as a message with `role`, merging into the previous
/// message when it has the same role. The Messages API requires strictly
/// alternating roles, and parallel tool calls produce consecutive tool results
/// that must travel as one user message. Empty content is dropped: the API
/// rejects a message with no blocks.
fn push_message(messages: &mut Vec<Value>, role: &str, content: Vec<Value>) {
    if content.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut()
        && last["role"].as_str() == Some(role)
        && let Some(blocks) = last["content"].as_array_mut()
    {
        blocks.extend(content);
        return;
    }
    messages.push(json!({ "role": role, "content": content }));
}

fn mark_ephemeral(block: Option<&mut Value>) {
    if let Some(block) = block {
        block["cache_control"] = json!({ "type": "ephemeral" });
    }
}

fn text_block(text: &str) -> Option<Value> {
    // The API rejects empty text blocks.
    (!text.is_empty()).then(|| json!({ "type": "text", "text": text }))
}

/// System content: text only. Anthropic's `system` accepts text blocks alone.
fn text_only_blocks(content: &[ContentBlock]) -> Vec<Value> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => text_block(text),
            ContentBlock::Json(value) => text_block(&value.to_string()),
            ContentBlock::Thinking { .. }
            | ContentBlock::RedactedThinking { .. }
            | ContentBlock::Image(_)
            | ContentBlock::ProviderExtension(_)
            | ContentBlock::Audio(_)
            | ContentBlock::Video(_)
            | ContentBlock::Document(_) => None,
        })
        .collect()
}

/// User-side content: text, images, and documents (Anthropic's native
/// `document` block). Audio and video have no Messages API representation
/// and are rendered as placeholder text rather than silently dropped.
/// Thinking blocks never appear in user content; provider extensions have no
/// faithful representation and are dropped.
fn content_blocks(content: &[ContentBlock]) -> Vec<Value> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => text_block(text),
            ContentBlock::Json(value) => text_block(&value.to_string()),
            ContentBlock::Image(image) => Some(image_block(image)),
            ContentBlock::ProviderExtension(value) => provider_extension_block(value),
            ContentBlock::Document(media) => Some(document_block(media)),
            ContentBlock::Audio(media) => Some(unsupported_media_placeholder("audio", media)),
            ContentBlock::Video(media) => Some(unsupported_media_placeholder("video", media)),
            ContentBlock::Thinking { .. } | ContentBlock::RedactedThinking { .. } => None,
        })
        .collect()
}

/// Assistant-side content: text plus the thinking blocks Anthropic requires
/// to be replayed verbatim ahead of a tool result. An unsigned thinking block
/// cannot be replayed (the API rejects it) and is dropped rather than leaked
/// into visible text.
fn assistant_blocks(content: &[ContentBlock]) -> Vec<Value> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text(text) => text_block(text),
            ContentBlock::Json(value) => text_block(&value.to_string()),
            ContentBlock::Thinking {
                text,
                signature: Some(signature),
            } => Some(json!({
                "type": "thinking",
                "thinking": text,
                "signature": signature,
            })),
            ContentBlock::RedactedThinking { data } => {
                Some(json!({ "type": "redacted_thinking", "data": data }))
            }
            ContentBlock::ProviderExtension(value) => provider_extension_block(value),
            ContentBlock::Thinking {
                signature: None, ..
            }
            | ContentBlock::Image(_) => None,
            ContentBlock::Audio(_) | ContentBlock::Video(_) | ContentBlock::Document(_) => None,
        })
        .collect()
}

/// Returns an opaque Anthropic content block when it has the object shape the
/// Messages API requires. Keeping the full object intact lets hosts persist and
/// replay newer block types without waiting for a TinyInference release.
fn provider_extension_block(value: &Value) -> Option<Value> {
    let object = value.as_object()?;
    object
        .get("type")
        .and_then(Value::as_str)
        .map(|_| value.clone())
}

/// Renders an image reference: a `data:` URI becomes an inline base64 source,
/// anything else a URL source.
fn image_block(image: &ImageRef) -> Value {
    if let Some(rest) = image.url.strip_prefix("data:")
        && let Some((header, data)) = rest.split_once(',')
    {
        let media_type = header
            .split(';')
            .next()
            .filter(|media| !media.is_empty())
            .map(str::to_string)
            .or_else(|| image.mime_type.clone())
            .unwrap_or_else(|| "image/png".to_string());
        return json!({
            "type": "image",
            "source": { "type": "base64", "media_type": media_type, "data": data },
        });
    }
    json!({
        "type": "image",
        "source": { "type": "url", "url": image.url },
    })
}

/// Renders a document reference as Anthropic's `document` content block.
/// `MediaRef::Path` has no wire representation (the harness never reads
/// local files) and falls back to a placeholder text block instead of being
/// silently dropped.
fn document_block(media: &crate::message::MediaRef) -> Value {
    use crate::message::MediaRef;
    match media {
        MediaRef::Base64 { data, media_type } => json!({
            "type": "document",
            "source": { "type": "base64", "media_type": media_type, "data": data },
        }),
        MediaRef::Url { url, .. } => json!({
            "type": "document",
            "source": { "type": "url", "url": url },
        }),
        MediaRef::Path { path, .. } => json!({
            "type": "text",
            "text": format!("[document attachment omitted: local path {path} was not resolved]"),
        }),
    }
}

/// Renders an audio or video reference as a placeholder text block: neither
/// has a wire representation in Anthropic's Messages API.
fn unsupported_media_placeholder(kind: &str, media: &crate::message::MediaRef) -> Value {
    let descriptor = match media {
        crate::message::MediaRef::Url { url, .. } => url.clone(),
        crate::message::MediaRef::Base64 { media_type, .. } => {
            format!("inline {media_type} data")
        }
        crate::message::MediaRef::Path { path, .. } => path.clone(),
    };
    json!({
        "type": "text",
        "text": format!("[{kind} attachment omitted: {descriptor}]"),
    })
}
