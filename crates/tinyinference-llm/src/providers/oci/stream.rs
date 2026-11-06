//! Native chat server-sent-events streaming.
//!
//! Only `data:` lines are read; each payload is one JSON chunk.
//!
//! * GENERIC chunks look like `{"index":0,"message":{"role":"ASSISTANT",
//!   "content":[{"type":"TEXT","text":"..."}],"toolCalls":[...],
//!   "reasoningContent":"..."}}`, followed by a chunk carrying
//!   `finishReason`, and possibly a trailing chunk with `usage`.
//! * COHERE chunks look like `{"apiFormat":"COHERE","text":"..."}`; the final
//!   chunk repeats the full `text` alongside `finishReason`, so text on a
//!   chunk with a finish reason is not re-emitted.
//!
//! The stream ends at `data: [DONE]` (some models send it) or at EOF; either
//! counts as success only once a `finishReason` was seen.

use std::collections::VecDeque;
use std::pin::Pin;

use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;

use super::response::{parse_usage, tool_call};
use crate::message::{AssistantMessage, ContentBlock, MessageDelta, MessageOrigin};
use crate::model::{ModelResponse, ModelStream, ModelStreamItem, ProviderError};
use crate::tool::{ToolCall, ToolDelta};
use crate::usage::Usage;
use crate::{Error, Result};

use super::{API_NATIVE, PROVIDER};

#[derive(Debug, Default)]
struct CallAcc {
    id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Default)]
struct Acc {
    text: String,
    reasoning: String,
    calls: Vec<CallAcc>,
    usage: Option<Usage>,
    finish: Option<String>,
}

impl Acc {
    fn delta_tool(&mut self, id: &str, name: &str, args: &str, slot_hint: usize) -> usize {
        let existing = if id.is_empty() {
            // Continuation fragment: attach to the hinted slot, else the last.
            slot_hint.min(self.calls.len().saturating_sub(1))
        } else {
            match self.calls.iter().position(|call| call.id == id) {
                Some(position) => position,
                None => {
                    self.calls.push(CallAcc {
                        id: id.to_string(),
                        name: name.to_string(),
                        arguments: String::new(),
                    });
                    self.calls.len() - 1
                }
            }
        };
        if self.calls.is_empty() {
            self.calls.push(CallAcc::default());
        }
        let call = &mut self.calls[existing];
        if call.name.is_empty() {
            call.name = name.to_string();
        }
        // A repeated id carrying the cumulative arguments replaces rather
        // than duplicates what was already streamed.
        if !args.is_empty()
            && args.len() > call.arguments.len()
            && args.starts_with(&call.arguments)
        {
            call.arguments = args.to_string();
        } else {
            call.arguments.push_str(args);
        }
        existing
    }

    fn ingest(&mut self, event: &Value, pending: &mut VecDeque<ModelStreamItem>) -> Result<()> {
        if event["message"].is_string() && event.get("code").is_some() {
            return Err(Error::Provider(Box::new(ProviderError {
                provider: PROVIDER.to_string(),
                code: event["code"].as_str().map(str::to_string),
                message: event["message"].as_str().unwrap_or_default().to_string(),
                raw: Some(event.clone()),
                ..ProviderError::default()
            })));
        }
        let cohere = event["apiFormat"].as_str() == Some("COHERE")
            || (event.get("text").is_some() && event.get("message").is_none());
        let finishing = event["finishReason"].as_str().is_some();
        if let Some(reason) = event["finishReason"].as_str() {
            self.finish = Some(reason.to_string());
        }
        if cohere {
            if !finishing && let Some(text) = event["text"].as_str().filter(|t| !t.is_empty()) {
                self.text.push_str(text);
                pending.push_back(ModelStreamItem::MessageDelta(MessageDelta::text(
                    text.to_string(),
                )));
            }
            if finishing {
                for (index, call) in event["toolCalls"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    let arguments = call
                        .get("parameters")
                        .map(Value::to_string)
                        .unwrap_or_default();
                    let id = format!("call_{index}");
                    let name = call["name"].as_str().unwrap_or_default();
                    let slot = self.delta_tool(&id, name, &arguments, index);
                    pending.push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                        call_id: id,
                        content: arguments,
                        tool_name: Some(name.to_string()),
                        content_index: Some(slot),
                    }));
                }
            }
        } else {
            let message = &event["message"];
            for part in message["content"].as_array().into_iter().flatten() {
                if let Some(text) = part["text"].as_str().filter(|t| !t.is_empty()) {
                    self.text.push_str(text);
                    pending.push_back(ModelStreamItem::MessageDelta(MessageDelta::text(
                        text.to_string(),
                    )));
                }
            }
            if let Some(reasoning) = message["reasoningContent"]
                .as_str()
                .filter(|t| !t.is_empty())
            {
                self.reasoning.push_str(reasoning);
                pending.push_back(ModelStreamItem::MessageDelta(MessageDelta {
                    text: String::new(),
                    reasoning: reasoning.to_string(),
                    tool_call: None,
                }));
            }
            for (hint, call) in message["toolCalls"]
                .as_array()
                .into_iter()
                .flatten()
                .enumerate()
            {
                let id = call["id"].as_str().unwrap_or_default();
                let name = call["name"].as_str().unwrap_or_default();
                let args = call["arguments"].as_str().unwrap_or_default();
                let slot = self.delta_tool(id, name, args, hint);
                let resolved = &self.calls[slot];
                pending.push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                    call_id: resolved.id.clone(),
                    content: args.to_string(),
                    tool_name: Some(resolved.name.clone()),
                    content_index: Some(slot),
                }));
            }
        }
        if let Some(usage) = event.get("usage").filter(|value| value.is_object()) {
            let usage = parse_usage(usage);
            self.usage = Some(usage);
            pending.push_back(ModelStreamItem::UsageDelta(usage));
        }
        Ok(())
    }

    fn into_response(self) -> ModelResponse {
        let mut content = Vec::new();
        if !self.reasoning.is_empty() {
            content.push(ContentBlock::Thinking {
                text: self.reasoning,
                signature: None,
            });
        }
        if !self.text.is_empty() {
            content.push(ContentBlock::Text(self.text));
        }
        let tool_calls: Vec<ToolCall> = self
            .calls
            .iter()
            .map(|call| tool_call(&call.id, &call.name, &call.arguments))
            .collect();
        ModelResponse {
            message: AssistantMessage {
                id: None,
                content,
                tool_calls,
                usage: self.usage,
                origin: None,
            },
            usage: self.usage,
            finish_reason: self.finish,
            raw: None,
            resolved_model: None,
            continue_turn: None,
            served_from_cache: false,
            correlation: None,
            resolved_route: None,
        }
    }
}

struct SseState {
    bytes: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>,
    buf: Vec<u8>,
    pending: VecDeque<ModelStreamItem>,
    acc: Acc,
    model: String,
    started: bool,
    finished: bool,
    terminal_emitted: bool,
}

impl SseState {
    fn failure(&mut self, error: Error) -> ModelStreamItem {
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
        provider_error.stop_reason = self.acc.finish.clone();
        let partial = std::mem::take(&mut self.acc).into_response().message;
        if !partial.content.is_empty() || !partial.tool_calls.is_empty() {
            provider_error.partial_message = Some(partial);
        }
        ModelStreamItem::ProviderFailed(provider_error)
    }

    fn fold_line(&mut self, line: &[u8]) -> Result<()> {
        let line = String::from_utf8_lossy(line);
        let Some(rest) = line.trim_end_matches('\r').strip_prefix("data:") else {
            return Ok(());
        };
        let payload = rest.trim();
        if payload.is_empty() {
            return Ok(());
        }
        if payload == "[DONE]" {
            self.finished = true;
            return Ok(());
        }
        let event = serde_json::from_str::<Value>(payload)
            .map_err(|error| Error::Model(format!("invalid OCI SSE data payload: {error}")))?;
        self.acc.ingest(&event, &mut self.pending)
    }

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
            if state.acc.finish.is_none() {
                let item = state.failure(Error::Model(
                    "provider stream ended before a completion signal".to_string(),
                ));
                return Some((item, state));
            }
            let mut response = std::mem::take(&mut state.acc).into_response();
            response.message.origin = Some(MessageOrigin {
                provider: PROVIDER.to_string(),
                api: API_NATIVE.to_string(),
                model: state.model.clone(),
            });
            return Some((ModelStreamItem::Completed(response), state));
        }
        match state.bytes.next().await {
            Some(Ok(chunk)) => {
                state.buf.extend_from_slice(&chunk);
                if let Err(error) = state.drain_lines() {
                    let item = state.failure(error);
                    return Some((item, state));
                }
            }
            Some(Err(error)) => {
                let item = state.failure(error);
                return Some((item, state));
            }
            None => {
                if !state.buf.is_empty() {
                    let line = std::mem::take(&mut state.buf);
                    if let Err(error) = state.fold_line(&line) {
                        let item = state.failure(error);
                        return Some((item, state));
                    }
                }
                // EOF without `[DONE]` is a normal end for most OCI models.
                state.finished = true;
            }
        }
    }
}

fn build(bytes: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>, model: String) -> ModelStream {
    let state = SseState {
        bytes,
        buf: Vec::new(),
        pending: VecDeque::new(),
        acc: Acc::default(),
        model,
        started: false,
        finished: false,
        terminal_emitted: false,
    };
    ModelStream::new(Box::pin(futures::stream::unfold(state, sse_next)))
}

/// Wraps a successful streaming HTTP response as a [`ModelStream`].
pub(super) fn into_model_stream(response: reqwest::Response, model: String) -> ModelStream {
    let bytes = response
        .bytes_stream()
        .map(|chunk| chunk.map_err(|error| Error::Model(format!("stream chunk failed: {error}"))));
    build(Box::pin(bytes), model)
}

/// Test seam: feeds raw SSE bytes through the production state machine.
#[cfg(test)]
pub(super) fn stream_from_bytes(chunks: Vec<Vec<u8>>, model: &str) -> ModelStream {
    let bytes = futures::stream::iter(chunks.into_iter().map(|chunk| Ok(Bytes::from(chunk))));
    build(Box::pin(bytes), model.to_string())
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod tests;
