//! Prompt-guided (text-mode) tool calling over the provider-neutral message
//! model.
//!
//! A model without a native tool channel — a local runtime that rejects
//! `tools` with a 400, a profile with `tool_calling = false`, a CLI-driven
//! model — is still handed tools. This module is the bridge between this
//! crate's [`Message`] / [`ModelResponse`] types and the protocol crate that
//! owns *how* tools are spoken in text, [`tinytools_agent`]:
//!
//! * [`with_tool_instructions`] puts the protocol block and catalogue in the
//!   system prompt;
//! * [`coalesce_tool_results`] rewrites structured assistant calls and `tool`
//!   turns — which such a model cannot read — into the text forms it was
//!   taught;
//! * [`ensure_resolvable_user_turn`] satisfies chat templates that refuse a
//!   transcript with no user query;
//! * [`recover_tool_calls`] reads the model's answer back through every
//!   grammar in [`tinytools_agent::parse`] and mints the call ids;
//! * [`TextScrubber`] does the same for a live stream.
//!
//! Nothing here knows a specific model's markup. A format-specific string
//! belongs in a grammar file under `tinytools-agent`, where every consumer
//! gets it.

use std::sync::Arc;

use tinytools_agent::render;
use tinytools_agent::tinytools::ToolSpec;
use tinytools_agent::types::{ParseOptions, ParsedToolCall};
use tinytools_agent::{PFormatRegistry, StreamScrubber, StreamStep};

use crate::message::{ContentBlock, Message};
use crate::model::{ModelResponse, ToolChoice};
use crate::tool::{ToolCall, ToolSchema};

/// Leading marker of the synthetic user turn tool results are folded into;
/// the same constant the renderer writes, so a folded turn is recognisable.
pub use tinytools_agent::render::TOOL_RESULTS_PREFIX;

/// User turn synthesized by [`ensure_resolvable_user_turn`] when a request
/// carries none. Deliberately content-free: the actual task is in the system
/// prompt (or in the transcript that follows), and this exists to satisfy chat
/// templates that require a locatable user query, not to add instructions.
pub const CONTINUATION_USER_TURN: &str = "Continue with the task described above.";

/// A [`ToolSchema`] as the declaration the protocol crate renders.
fn to_spec(schema: &ToolSchema) -> ToolSpec {
    ToolSpec {
        name: schema.name.clone(),
        description: schema.description.clone(),
        parameters: schema.parameters.clone(),
    }
}

/// The tool-use protocol block for the system prompt: the JSON-in-tag
/// instructions plus the catalogue, followed by the sentence `choice` calls
/// for. Empty when `choice` is [`ToolChoice::None`].
#[must_use]
pub fn tool_instructions(tools: &[ToolSchema], choice: &ToolChoice) -> String {
    let specs: Vec<ToolSpec> = tools.iter().map(to_spec).collect();
    let mut out = render::json_instructions(&specs);
    match choice {
        ToolChoice::Required => out.push_str("\nYou must emit at least one tool call.\n"),
        ToolChoice::Tool(name) => {
            out.push_str(&format!("\nYou must call the `{name}` tool.\n"));
        }
        ToolChoice::Auto => {}
        ToolChoice::None => return String::new(),
    }
    out
}

/// Returns `messages` with the protocol block appended to the first system
/// message, or as a new leading system message when there is none. Empty
/// `tools` returns `messages` unchanged.
#[must_use]
pub fn with_tool_instructions(
    messages: &[Message],
    tools: &[ToolSchema],
    choice: &ToolChoice,
) -> Vec<Message> {
    if tools.is_empty() {
        return messages.to_vec();
    }
    let block = tool_instructions(tools, choice);
    if block.is_empty() {
        return messages.to_vec();
    }
    append_system_block(messages, &block)
}

/// Returns `messages` with `block` appended to the first system message as a
/// distinct text block (so the original prompt is intact), or as a new
/// leading system message when there is none. This is how any protocol
/// block — the JSON one above, a host's P-Format block — reaches the model.
#[must_use]
pub fn append_system_block(messages: &[Message], block: &str) -> Vec<Message> {
    let mut out = messages.to_vec();
    if let Some(Message::System(system)) = out.iter_mut().find(|m| matches!(m, Message::System(_)))
    {
        system
            .content
            .push(ContentBlock::Text(format!("\n\n{block}")));
    } else {
        out.insert(0, Message::system(block.to_string()));
    }
    out
}

/// Rewrites structured assistant tool calls and `tool`-role results into the
/// text forms a prompt-guided model can read.
///
/// Assistant calls are rendered back into `<tool_call>` markup appended to the
/// turn's text and cleared from the structured field; each run of consecutive
/// results is folded into one user turn under the `<tool_result>` envelope
/// the protocol block advertises, with the boundary-integrity rules the
/// protocol crate applies (a result body cannot forge a closing tag). Other
/// messages keep their order and type, images included.
#[must_use]
pub fn coalesce_tool_results(messages: &[Message]) -> Vec<Message> {
    use tinytools_agent::dialect::{ToolResultEntry, TranscriptEntry};

    let mut out = Vec::with_capacity(messages.len());
    let mut pending: Vec<ToolResultEntry> = Vec::new();

    fn flush(out: &mut Vec<Message>, pending: &mut Vec<ToolResultEntry>) {
        if pending.is_empty() {
            return;
        }
        let entries = std::mem::take(pending);
        for rendered in render::to_provider_messages(&[TranscriptEntry::ToolResults(entries)]) {
            out.push(Message::user(rendered.content));
        }
    }

    for message in messages {
        match message {
            Message::Tool(tool) => {
                let entry = ToolResultEntry::new(tool.tool_call_id.clone(), message.text());
                pending.push(if tool.trusted_verbatim {
                    entry.verbatim()
                } else {
                    entry
                });
            }
            Message::Assistant(assistant) if !assistant.tool_calls.is_empty() => {
                flush(&mut out, &mut pending);
                let mut assistant = assistant.clone();
                let mut rendered = String::new();
                if !message.text().trim().is_empty() {
                    rendered.push('\n');
                }
                rendered.push_str(&render::render_json_calls(
                    assistant
                        .tool_calls
                        .iter()
                        .map(|call| (call.name.as_str(), &call.arguments)),
                ));
                assistant.content.push(ContentBlock::Text(rendered));
                assistant.tool_calls.clear();
                out.push(Message::Assistant(assistant));
            }
            _ => {
                flush(&mut out, &mut pending);
                out.push(message.clone());
            }
        }
    }
    flush(&mut out, &mut pending);
    out
}

/// Whether this message is a user turn a chat template can resolve as "the
/// user query".
///
/// A folded tool-result turn does not count: it carries the
/// [`TOOL_RESULTS_PREFIX`], and templates that look for a user query want a
/// request to answer, not the transcript of a tool the model itself invoked —
/// Qwen 3's template makes the same distinction. Neither does an empty or
/// whitespace-only turn. Non-text content (JSON, an image, audio, video, or a
/// document) does count.
fn is_resolvable_user_query(message: &Message) -> bool {
    let Message::User(user) = message else {
        return false;
    };
    if message
        .text()
        .trim_start()
        .starts_with(TOOL_RESULTS_PREFIX.trim_end())
    {
        return false;
    }
    user.content.iter().any(|block| match block {
        ContentBlock::Text(text) => !text.trim().is_empty(),
        ContentBlock::Json(_)
        | ContentBlock::Image(_)
        | ContentBlock::Audio(_)
        | ContentBlock::Video(_)
        | ContentBlock::Document(_) => true,
        ContentBlock::Thinking { .. }
        | ContentBlock::RedactedThinking { .. }
        | ContentBlock::ProviderExtension(_) => false,
    })
}

/// Guarantees the outgoing list contains a user turn a chat template can
/// resolve, inserting one only when none is present.
///
/// Models without native tool calling are driven through their **own** chat
/// template by the serving runtime (LM Studio, llama.cpp, Ollama), and several
/// widely used templates hard-require a locatable user query — Qwen 3's raises
/// `No user query found in messages.` A prompt-guided loop reaches that state
/// legitimately once the real user turn has aged out of the window, leaving
/// only assistant continuations and folded tool results; the template then
/// aborts the request with a 400 before the model is ever called.
///
/// The inserted turn goes directly after any leading system messages.
#[must_use]
pub fn ensure_resolvable_user_turn(messages: &[Message]) -> Vec<Message> {
    if messages.iter().any(is_resolvable_user_query) {
        return messages.to_vec();
    }
    let mut out = messages.to_vec();
    let insert_at = out
        .iter()
        .position(|message| !matches!(message, Message::System(_)))
        .unwrap_or(out.len());
    out.insert(insert_at, Message::user(CONTINUATION_USER_TURN));
    out
}

/// Converts a recovered call into this crate's [`ToolCall`], minting a
/// process-unique id.
///
/// The protocol crate never mints ids; a per-response index would collide
/// across turns of one run (two assistant messages declaring `call_1`, two
/// results answering it, a pairing no provider can resolve). The `text-`
/// prefix keeps a recovered id visibly distinct from a provider's.
fn to_tool_call(call: ParsedToolCall, slot: usize) -> ToolCall {
    static SEQUENCE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = call.id.unwrap_or_else(|| {
        let sequence = SEQUENCE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        format!("text-{sequence}-{slot}")
    });
    ToolCall::new(id, call.name, call.arguments)
}

/// Reads text-mode tool calls out of a completed response into
/// `message.tool_calls`, replacing the visible text with the cleaned
/// narrative. Reasoning and other non-text blocks are kept in place. A
/// response with no recoverable call is returned unchanged.
///
/// `tools` are the schemas the model was offered: they enable name repair and
/// the known-tool gate on bare JSON.
#[must_use]
pub fn recover_tool_calls(mut response: ModelResponse, tools: &[ToolSchema]) -> ModelResponse {
    let known: Vec<String> = tools.iter().map(|tool| tool.name.clone()).collect();
    let options = ParseOptions::new().with_known_tools(&known);
    let text = response.text();
    let outcome = tinytools_agent::parse_text(&text, &options);
    if outcome.calls.is_empty() {
        return response;
    }
    for diagnostic in &outcome.diagnostics {
        tracing::debug!(?diagnostic, "prompt-guided tool-call recovery");
    }
    let calls = outcome
        .calls
        .into_iter()
        .enumerate()
        .map(|(index, call)| to_tool_call(call, index + 1));
    response.message.tool_calls.extend(calls);
    response.message.content =
        replace_text_blocks(std::mem::take(&mut response.message.content), outcome.text);
    response
}

/// Rebuilds a content vector keeping every non-text block in place and
/// substituting one cleaned text at the position of the first text block.
/// An empty `cleaned` emits no text block at all.
fn replace_text_blocks(content: Vec<ContentBlock>, cleaned: String) -> Vec<ContentBlock> {
    let mut out = Vec::with_capacity(content.len());
    let mut inserted = false;
    for block in content {
        match block {
            ContentBlock::Text(_) => {
                if !inserted {
                    if !cleaned.is_empty() {
                        out.push(ContentBlock::Text(cleaned.clone()));
                    }
                    inserted = true;
                }
            }
            other => out.push(other),
        }
    }
    if !inserted && !cleaned.is_empty() {
        out.push(ContentBlock::Text(cleaned));
    }
    out
}

/// Scrubs tool-call markup from streamed visible text as fragments arrive.
///
/// A thin wrapper over [`StreamScrubber`] that knows this crate's
/// [`ToolSchema`] and mints [`ToolCall`] ids for calls released mid-stream.
/// Feed each text delta through [`feed`](Self::feed); call
/// [`flush`](Self::flush) when the stream ends.
#[derive(Debug)]
pub struct TextScrubber {
    inner: StreamScrubber,
    released: usize,
}

impl TextScrubber {
    /// A scrubber that knows the offered tools.
    #[must_use]
    pub fn new(tools: &[ToolSchema]) -> Self {
        let known = tools.iter().map(|tool| tool.name.clone()).collect();
        Self {
            inner: StreamScrubber::new().with_known_tools(known),
            released: 0,
        }
    }

    /// Adds a P-Format registry.
    #[must_use]
    pub fn with_registry(mut self, registry: Arc<PFormatRegistry>) -> Self {
        self.inner = self.inner.with_registry(registry);
        self
    }

    /// Feeds one fragment; returns the text safe to show and any completed
    /// calls.
    pub fn feed(&mut self, fragment: &str) -> (String, Vec<ToolCall>) {
        let step = self.inner.feed(fragment);
        self.convert(step)
    }

    /// Drains the remainder at end of stream.
    pub fn flush(&mut self) -> (String, Vec<ToolCall>) {
        let step = self.inner.flush();
        self.convert(step)
    }

    fn convert(&mut self, step: StreamStep) -> (String, Vec<ToolCall>) {
        let calls = step
            .calls
            .into_iter()
            .map(|call| {
                self.released += 1;
                to_tool_call(call, self.released)
            })
            .collect();
        (step.text, calls)
    }
}

#[cfg(test)]
mod test;
