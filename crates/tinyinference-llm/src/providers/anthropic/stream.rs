//! Messages API server-sent-events streaming.
//!
//! The wire is `event:`/`data:` pairs; every `data:` payload carries its own
//! `type`, so only `data:` lines are read. Events folded here:
//!
//! | event                 | effect                                              |
//! |-----------------------|-----------------------------------------------------|
//! | `message_start`       | message id, prompt-side usage (incl. cache counters) |
//! | `content_block_start` | opens a text / tool_use / thinking / redacted block  |
//! | `content_block_delta` | `text_delta`, `input_json_delta`, `thinking_delta`, `signature_delta` |
//! | `message_delta`       | `stop_reason`, output-side usage                     |
//! | `message_stop`        | completion signal                                   |
//! | `error`               | provider failure mid-stream                         |
//!
//! Text and thinking fragments are emitted as they arrive; tool-call argument
//! fragments are emitted as [`ModelStreamItem::ToolCallDelta`] and also
//! accumulated so the terminal [`ModelStreamItem::Completed`] carries fully
//! parsed [`ToolCall`]s. Usage is cumulative: `message_start` reports the
//! input side, `message_delta` the output side, and the accumulator keeps the
//! merged value.

use std::collections::VecDeque;
use std::pin::Pin;

use futures::{Stream, StreamExt};
use serde_json::Value;

use crate::message::{AssistantMessage, ContentBlock, MessageDelta};
use crate::model::{
    BlockDelta, BlockKind, ModelResponse, ModelStream, ModelStreamItem, ProviderError,
};
use crate::tool::{ToolCall, ToolDelta};
use crate::usage::Usage;
use crate::{Error, Result};

use super::PROVIDER;
use super::response::parse_usage;

/// Anthropic emits content blocks densely from zero. This generous bound keeps
/// a malicious or corrupt event from turning one wire index into an unbounded
/// allocation while remaining far above practical response sizes.
const MAX_CONTENT_BLOCK_INDEX: usize = 1023;

/// One open content block, keyed by its wire `index`.
#[derive(Clone, Debug)]
enum OpenBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        partial_json: String,
    },
    Thinking {
        text: String,
        signature: Option<String>,
    },
    Redacted(String),
    ProviderExtension {
        value: Value,
        partial_json: String,
    },
}

impl OpenBlock {
    /// Converts a closed block into the [`ContentBlock`] carried on
    /// [`ModelStreamItem::BlockEnd`].
    ///
    /// [`ContentBlock`] has no dedicated tool-call variant (tool calls live on
    /// [`AssistantMessage::tool_calls`]), so a closed tool-use block is
    /// represented as [`ContentBlock::Json`] carrying `{id, name, arguments}`;
    /// consumers that want the parsed [`crate::tool::ToolCall`] already saw the
    /// id and name on the matching [`ModelStreamItem::BlockStart`].
    fn into_content_block(self) -> ContentBlock {
        match self {
            OpenBlock::Text(text) => ContentBlock::Text(text),
            OpenBlock::Thinking { text, signature } => ContentBlock::Thinking { text, signature },
            OpenBlock::Redacted(data) => ContentBlock::RedactedThinking { data },
            OpenBlock::ToolUse {
                id,
                name,
                partial_json,
            } => {
                let arguments = if partial_json.trim().is_empty() {
                    Value::Object(Default::default())
                } else {
                    serde_json::from_str(&partial_json).unwrap_or(Value::String(partial_json))
                };
                ContentBlock::Json(serde_json::json!({
                    "id": id,
                    "name": name,
                    "arguments": arguments,
                }))
            }
            OpenBlock::ProviderExtension {
                value,
                partial_json,
            } => ContentBlock::ProviderExtension(provider_extension_value(value, partial_json)),
        }
    }
}

fn provider_extension_value(mut value: Value, partial_json: String) -> Value {
    if !partial_json.trim().is_empty()
        && let Ok(input) = serde_json::from_str::<Value>(&partial_json)
    {
        value["input"] = input;
    }
    value
}

/// Provider-side accumulator rebuilding the terminal [`ModelResponse`].
#[derive(Debug, Default)]
struct AnthropicStreamAcc {
    id: Option<String>,
    /// Blocks in wire order, `None` while a slot's start event has not arrived.
    blocks: Vec<Option<OpenBlock>>,
    usage: Option<Usage>,
    stop_reason: Option<String>,
}

impl AnthropicStreamAcc {
    fn slot(&mut self, index: usize) -> &mut Option<OpenBlock> {
        if self.blocks.len() <= index {
            self.blocks.resize(index + 1, None);
        }
        &mut self.blocks[index]
    }

    fn merge_usage(&mut self, incoming: Usage) -> Usage {
        let merged = match self.usage {
            Some(existing) => Usage {
                input_tokens: existing.input_tokens.max(incoming.input_tokens),
                output_tokens: existing.output_tokens.max(incoming.output_tokens),
                cache_read_tokens: existing.cache_read_tokens.max(incoming.cache_read_tokens),
                cache_creation_tokens: existing
                    .cache_creation_tokens
                    .max(incoming.cache_creation_tokens),
                reasoning_tokens: existing.reasoning_tokens.max(incoming.reasoning_tokens),
                total_tokens: 0,
                charged_amount: incoming.charged_amount.or(existing.charged_amount),
                context_window_tokens: incoming
                    .context_window_tokens
                    .or(existing.context_window_tokens),
            },
            None => incoming,
        };
        let merged = Usage {
            total_tokens: merged.input_tokens + merged.output_tokens,
            ..merged
        };
        self.usage = Some(merged);
        merged
    }

    /// Folds one parsed event and queues the neutral items it produces.
    fn ingest(&mut self, event: Value, pending: &mut VecDeque<ModelStreamItem>) -> Result<bool> {
        match event["type"].as_str() {
            Some("message_start") => {
                let message = &event["message"];
                if let Some(id) = message["id"].as_str() {
                    self.id = Some(id.to_string());
                }
                if let Some(usage) = message.get("usage") {
                    let usage = self.merge_usage(parse_usage(usage));
                    pending.push_back(ModelStreamItem::UsageDelta(usage));
                }
            }
            Some("content_block_start") => {
                let index = event_index(&event)?;
                let block = &event["content_block"];
                let open = match block["type"].as_str() {
                    Some("tool_use") => {
                        let id = block["id"].as_str().unwrap_or_default().to_string();
                        let name = block["name"].as_str().unwrap_or_default().to_string();
                        pending.push_back(ModelStreamItem::BlockStart {
                            index,
                            kind: BlockKind::ToolCall {
                                id: id.clone(),
                                name: name.clone(),
                            },
                        });
                        pending.push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                            call_id: id.clone(),
                            content: String::new(),
                            tool_name: Some(name.clone()),
                            content_index: Some(index),
                        }));
                        OpenBlock::ToolUse {
                            id,
                            name,
                            partial_json: String::new(),
                        }
                    }
                    Some("thinking") => {
                        pending.push_back(ModelStreamItem::BlockStart {
                            index,
                            kind: BlockKind::Thinking,
                        });
                        let text = block["thinking"].as_str().unwrap_or_default().to_string();
                        if !text.is_empty() {
                            pending.push_back(ModelStreamItem::BlockDelta {
                                index,
                                delta: BlockDelta::Thinking(text.clone()),
                            });
                            pending.push_back(ModelStreamItem::MessageDelta(
                                MessageDelta::reasoning(text.clone()),
                            ));
                        }
                        OpenBlock::Thinking {
                            text,
                            signature: None,
                        }
                    }
                    Some("redacted_thinking") => {
                        pending.push_back(ModelStreamItem::BlockStart {
                            index,
                            kind: BlockKind::Thinking,
                        });
                        OpenBlock::Redacted(block["data"].as_str().unwrap_or_default().to_string())
                    }
                    Some("text") => {
                        pending.push_back(ModelStreamItem::BlockStart {
                            index,
                            kind: BlockKind::Text,
                        });
                        let text = block["text"].as_str().unwrap_or_default().to_string();
                        if !text.is_empty() {
                            pending.push_back(ModelStreamItem::BlockDelta {
                                index,
                                delta: BlockDelta::Text(text.clone()),
                            });
                            pending.push_back(ModelStreamItem::MessageDelta(MessageDelta::text(
                                text.clone(),
                            )));
                        }
                        OpenBlock::Text(text)
                    }
                    _ => {
                        pending.push_back(ModelStreamItem::BlockStart {
                            index,
                            kind: BlockKind::ProviderExtension {
                                block_type: block["type"].as_str().unwrap_or_default().to_string(),
                            },
                        });
                        OpenBlock::ProviderExtension {
                            value: block.clone(),
                            partial_json: String::new(),
                        }
                    }
                };
                *self.slot(index) = Some(open);
            }
            Some("content_block_delta") => {
                let index = event_index(&event)?;
                let delta = &event["delta"];
                let slot = self.slot(index);
                match (delta["type"].as_str(), slot.as_mut()) {
                    (Some("text_delta"), Some(OpenBlock::Text(text))) => {
                        let fragment = delta["text"].as_str().unwrap_or_default();
                        text.push_str(fragment);
                        pending.push_back(ModelStreamItem::BlockDelta {
                            index,
                            delta: BlockDelta::Text(fragment.to_string()),
                        });
                        pending.push_back(ModelStreamItem::MessageDelta(MessageDelta::text(
                            fragment.to_string(),
                        )));
                    }
                    (
                        Some("input_json_delta"),
                        Some(OpenBlock::ToolUse {
                            id,
                            name,
                            partial_json,
                        }),
                    ) => {
                        let fragment = delta["partial_json"].as_str().unwrap_or_default();
                        partial_json.push_str(fragment);
                        pending.push_back(ModelStreamItem::BlockDelta {
                            index,
                            delta: BlockDelta::ToolArgs(fragment.to_string()),
                        });
                        pending.push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                            call_id: id.clone(),
                            content: fragment.to_string(),
                            tool_name: Some(name.clone()),
                            content_index: Some(index),
                        }));
                    }
                    (
                        Some("input_json_delta"),
                        Some(OpenBlock::ProviderExtension { partial_json, .. }),
                    ) => {
                        let fragment = delta["partial_json"].as_str().unwrap_or_default();
                        partial_json.push_str(fragment);
                    }
                    (Some("thinking_delta"), Some(OpenBlock::Thinking { text, .. })) => {
                        let fragment = delta["thinking"].as_str().unwrap_or_default();
                        text.push_str(fragment);
                        pending.push_back(ModelStreamItem::BlockDelta {
                            index,
                            delta: BlockDelta::Thinking(fragment.to_string()),
                        });
                        pending.push_back(ModelStreamItem::MessageDelta(MessageDelta {
                            text: String::new(),
                            reasoning: fragment.to_string(),
                            tool_call: None,
                        }));
                    }
                    (Some("signature_delta"), Some(OpenBlock::Thinking { signature, .. })) => {
                        if let Some(fragment) = delta["signature"].as_str() {
                            signature.get_or_insert_with(String::new).push_str(fragment);
                        }
                    }
                    // A delta for a block whose start we never saw, or of a kind
                    // this adapter does not model: ignore rather than fail the
                    // turn.
                    _ => {}
                }
            }
            Some("content_block_stop") => {
                let index = event_index(&event)?;
                if let Some(block) = self.slot(index).clone() {
                    pending.push_back(ModelStreamItem::BlockEnd {
                        index,
                        block: block.into_content_block(),
                    });
                }
            }
            Some("message_delta") => {
                if let Some(stop_reason) = event["delta"]["stop_reason"].as_str() {
                    self.stop_reason = Some(stop_reason.to_string());
                }
                if let Some(usage) = event.get("usage") {
                    let usage = self.merge_usage(parse_usage(usage));
                    pending.push_back(ModelStreamItem::UsageDelta(usage));
                }
            }
            Some("message_stop") => return Ok(true),
            Some("error") => {
                let message = event["error"]["message"]
                    .as_str()
                    .unwrap_or("anthropic stream reported an error")
                    .to_string();
                let code = event["error"]["type"].as_str().map(str::to_string);
                let retryable =
                    crate::failure::classify_provider_failure(None, code.as_deref(), &message)
                        .is_retryable();
                return Err(Error::Provider(Box::new(ProviderError {
                    provider: PROVIDER.to_string(),
                    code,
                    message,
                    retryable,
                    raw: Some(event),
                    ..ProviderError::default()
                })));
            }
            // `ping`, `content_block_stop`, and anything newer than this adapter.
            _ => {}
        }
        Ok(false)
    }

    fn into_response(self) -> ModelResponse {
        let mut content = Vec::new();
        let mut tool_calls = Vec::new();
        for block in self.blocks.into_iter().flatten() {
            match block {
                OpenBlock::Text(text) => {
                    if !text.is_empty() {
                        content.push(ContentBlock::Text(text));
                    }
                }
                OpenBlock::Thinking { text, signature } => {
                    content.push(ContentBlock::Thinking { text, signature });
                }
                OpenBlock::Redacted(data) => content.push(ContentBlock::RedactedThinking { data }),
                OpenBlock::ProviderExtension {
                    value,
                    partial_json,
                } => {
                    content.push(ContentBlock::ProviderExtension(provider_extension_value(
                        value,
                        partial_json,
                    )));
                }
                OpenBlock::ToolUse {
                    id,
                    name,
                    partial_json,
                } => {
                    // Anthropic sends `{}` for argument-less tools as an empty
                    // fragment stream; treat that as an empty object rather
                    // than a parse failure. Malformed JSON becomes an invalid
                    // call the loop can feed back, never a stream failure — the
                    // same contract as the OpenAI adapter and a full response's
                    // `parse_content_block`.
                    let call = if partial_json.trim().is_empty() {
                        ToolCall::new(id, name, Value::Object(Default::default()))
                    } else {
                        match serde_json::from_str::<Value>(&partial_json) {
                            Ok(arguments) => ToolCall::new(id, name, arguments),
                            Err(error) => {
                                ToolCall::invalid(id, name, partial_json, error.to_string())
                            }
                        }
                    };
                    tool_calls.push(call);
                }
            }
        }
        // Reconstruct the wire shape so `raw` matches a unary response closely
        // enough for callers that read it.
        let raw = serde_json::json!({
            "id": self.id,
            "stop_reason": self.stop_reason,
            "content": content.iter().filter_map(|block| match block {
                ContentBlock::Text(text) => Some(serde_json::json!({"type": "text", "text": text})),
                ContentBlock::ProviderExtension(value) => Some(value.clone()),
                _ => None,
            }).chain(tool_calls.iter().map(|call| serde_json::json!({
                "type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments,
            }))).collect::<Vec<_>>(),
        });
        let usage = self.usage;
        ModelResponse {
            message: AssistantMessage {
                id: self.id,
                content,
                tool_calls,
                usage,
                // Stamped by the `sse_next` call site (which owns
                // `SseState::model`); this accumulator has no provider/model
                // context of its own.
                origin: None,
            },
            usage,
            finish_reason: self.stop_reason,
            raw: Some(raw),
            resolved_model: None,
            continue_turn: None,
            served_from_cache: false,
            correlation: None,
            resolved_route: None,
        }
    }
}

fn event_index(event: &Value) -> Result<usize> {
    let index = event["index"]
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
        .ok_or_else(|| Error::Model("invalid Anthropic SSE content-block index".to_string()))?;
    if index > MAX_CONTENT_BLOCK_INDEX {
        return Err(Error::Model(format!(
            "Anthropic SSE content-block index {index} exceeds limit {MAX_CONTENT_BLOCK_INDEX}"
        )));
    }
    Ok(index)
}

struct SseState {
    bytes: Pin<Box<dyn Stream<Item = Result<bytes::Bytes>> + Send>>,
    buf: Vec<u8>,
    pending: VecDeque<ModelStreamItem>,
    acc: AnthropicStreamAcc,
    model: String,
    started: bool,
    finished: bool,
    completion_seen: bool,
    terminal_emitted: bool,
}

impl SseState {
    /// Builds the terminal failure item and drops anything still queued: a
    /// failure must be the last item a consumer sees, not followed by deltas
    /// parsed before the error surfaced.
    fn provider_failure(&mut self, error: Error) -> ModelStreamItem {
        self.pending.clear();
        self.finished = true;
        self.terminal_emitted = true;
        let mut provider_error = match error {
            Error::Provider(error) => *error,
            error => {
                let message = error.to_string();
                let retryable =
                    crate::failure::classify_provider_failure(None, None, &message).is_retryable();
                ProviderError {
                    provider: PROVIDER.to_string(),
                    message,
                    retryable,
                    ..ProviderError::default()
                }
            }
        };
        provider_error.provider = PROVIDER.to_string();
        provider_error.model = Some(self.model.clone());
        provider_error.stop_reason = self.acc.stop_reason.clone();
        let partial = std::mem::take(&mut self.acc).into_response().message;
        if !partial.content.is_empty() || !partial.tool_calls.is_empty() {
            provider_error.partial_message = Some(partial);
        }
        ModelStreamItem::ProviderFailed(provider_error)
    }

    /// Folds one SSE line. Returns an error only for a provider-reported
    /// `error` event or a malformed complete `data:` payload.
    fn fold_line(&mut self, line: &[u8]) -> Result<()> {
        let line = String::from_utf8_lossy(line);
        let line = line.trim_end_matches('\r');
        let Some(rest) = line.strip_prefix("data:") else {
            return Ok(());
        };
        let payload = rest.trim();
        if payload.is_empty() {
            return Ok(());
        }
        let event = serde_json::from_str::<Value>(payload).map_err(|error| {
            Error::Model(format!("invalid Anthropic SSE data payload: {error}"))
        })?;
        if self.acc.ingest(event, &mut self.pending)? {
            self.completion_seen = true;
            self.finished = true;
        }
        Ok(())
    }

    /// Drains complete lines from `buf`; a trailing partial line is kept so a
    /// payload split across chunks (even mid-UTF-8-character) is decoded once
    /// it is whole.
    fn drain_lines(&mut self) -> Result<()> {
        let mut start = 0;
        while let Some(offset) = self.buf[start..].iter().position(|byte| *byte == b'\n') {
            let end = start + offset;
            let line = self.buf[start..end].to_vec();
            start = end + 1;
            self.fold_line(&line)?;
            if self.finished {
                break;
            }
        }
        self.buf.drain(..start);
        Ok(())
    }

    fn drain_remaining(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let line = std::mem::take(&mut self.buf);
        self.fold_line(&line)
    }
}

async fn sse_next(mut state: SseState) -> Option<(ModelStreamItem, SseState)> {
    loop {
        if let Some(item) = state.pending.pop_front() {
            return Some((item, state));
        }
        if !state.started {
            state.started = true;
            return Some((ModelStreamItem::Started, state));
        }
        if state.finished {
            if state.terminal_emitted {
                return None;
            }
            state.terminal_emitted = true;
            let mut response = std::mem::take(&mut state.acc).into_response();
            response.message.origin = Some(crate::message::MessageOrigin {
                provider: PROVIDER.to_string(),
                api: super::MESSAGES_API.to_string(),
                model: state.model.clone(),
            });
            return Some((ModelStreamItem::Completed(response), state));
        }
        match state.bytes.next().await {
            Some(Ok(chunk)) => {
                state.buf.extend_from_slice(&chunk);
                if let Err(error) = state.drain_lines() {
                    let item = state.provider_failure(error);
                    return Some((item, state));
                }
            }
            Some(Err(error)) => {
                let item = state.provider_failure(error);
                return Some((item, state));
            }
            None => {
                if let Err(error) = state.drain_remaining() {
                    let item = state.provider_failure(error);
                    return Some((item, state));
                }
                state.finished = true;
                if !state.completion_seen {
                    let item = state.provider_failure(Error::Model(
                        "provider stream ended before a completion signal".to_string(),
                    ));
                    return Some((item, state));
                }
            }
        }
    }
}

/// Wraps a successful streaming HTTP response as a [`ModelStream`].
pub(super) fn into_model_stream(response: reqwest::Response, model: String) -> ModelStream {
    let bytes = response
        .bytes_stream()
        .map(|chunk| chunk.map_err(|error| Error::Model(format!("stream chunk failed: {error}"))));
    let state = SseState {
        bytes: Box::pin(bytes),
        buf: Vec::new(),
        pending: VecDeque::new(),
        acc: AnthropicStreamAcc::default(),
        model,
        started: false,
        finished: false,
        completion_seen: false,
        terminal_emitted: false,
    };
    ModelStream::new(Box::pin(futures::stream::unfold(state, sse_next)))
}

/// Test seam: feeds raw SSE bytes through the same state machine the network
/// path uses.
#[cfg(test)]
pub(super) fn stream_from_bytes(chunks: Vec<Vec<u8>>, model: &str) -> ModelStream {
    let bytes = futures::stream::iter(
        chunks
            .into_iter()
            .map(|chunk| Ok(bytes::Bytes::from(chunk))),
    );
    let state = SseState {
        bytes: Box::pin(bytes),
        buf: Vec::new(),
        pending: VecDeque::new(),
        acc: AnthropicStreamAcc::default(),
        model: model.to_string(),
        started: false,
        finished: false,
        completion_seen: false,
        terminal_emitted: false,
    };
    ModelStream::new(Box::pin(futures::stream::unfold(state, sse_next)))
}
