//! Rich internal message model for the harness.
//!
//! These typed values are the data that moves between recursion levels — parent
//! to sub-agent, node to sub-graph, model to REPL — so they are deliberately
//! structured rather than stringly typed.
//!
//! Raw strings appear only at API boundaries; internally the harness works
//! with structured [`Message`] values made of typed [`ContentBlock`]s.
//! Ergonomic constructors ([`Message::system`], [`Message::user`], …) and a
//! [`Message::text`] accessor keep the public surface easy to use.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::{ToolCall, ToolSchema};
use crate::usage::Usage;

/// A typed unit of message content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text.
    Text(String),
    /// Structured JSON content.
    Json(Value),
    /// A reference to an image input.
    Image(ImageRef),
    /// Model reasoning/thinking content.
    ///
    /// Kept out of visible assistant text ([`ContentBlock::as_text`] returns
    /// `None`) but preserved on the message so providers that require verbatim
    /// replay of the thinking turn preceding tool results (Anthropic) can round-
    /// trip it. Providers that reject thinking blocks (the OpenAI-compatible
    /// path, which serializes via [`crate::message::Message::text`])
    /// drop it naturally.
    Thinking {
        /// The reasoning text.
        text: String,
        /// Opaque provider signature required to replay the block verbatim.
        /// `None` when the provider does not sign thinking blocks.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    /// Redacted reasoning: an opaque, provider-encrypted thinking block whose
    /// content is not human-readable but which must still be replayed verbatim
    /// to preserve the provider's reasoning contract.
    RedactedThinking {
        /// Opaque provider payload, replayed verbatim.
        data: String,
    },
    /// An opaque provider-specific block preserved verbatim.
    ProviderExtension(Value),
}

/// A reference to an image, either by URL or inline base64 data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageRef {
    /// URL or data URI of the image.
    pub url: String,
    /// Optional MIME type (for example `image/png`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// A system/developer instruction message.
///
/// A transcript may carry more than one `SystemMessage`: the leading one
/// establishes the run's baseline instructions, and any later one is a
/// **patch** that layers additively onto everything before it (see
/// [`replay_system_state`]). This is what lets a mid-run change — a toolset
/// gaining or losing a tool, an instructions section being added or revised —
/// be expressed as a small delta appended to (or inserted into) the
/// transcript instead of rewriting the leading system message and busting a
/// provider's cached prefix.
///
/// The additional fields are all `#[serde(default)]` so a transcript
/// persisted before this type gained them deserializes unchanged (every
/// existing `SystemMessage` reads back with empty `sections`,
/// `tools_added`, and `tools_removed` — i.e. a no-op patch).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SystemMessage {
    /// Ordered content blocks.
    #[serde(default)]
    pub content: Vec<ContentBlock>,
    /// Named content sections this message contributes.
    ///
    /// Keyed by a stable section name (for example `"tool_changes"` or
    /// `"persona"`). `Some(text)` sets or replaces the section's content;
    /// `None` removes a section a prior message in the transcript
    /// established. [`BTreeMap`] keeps replay deterministic regardless of
    /// insertion order.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sections: BTreeMap<String, Option<String>>,
    /// Tool schemas this message adds to the effective tool set.
    ///
    /// A name already present is replaced (the newer declaration wins), so a
    /// patch can both add a brand-new tool and republish a changed schema for
    /// an existing one.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_added: Vec<ToolSchema>,
    /// Names of tools this message removes from the effective tool set.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools_removed: Vec<String>,
}

impl SystemMessage {
    /// Creates a plain-text system message with no sections or tool deltas.
    pub fn text(content: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::Text(content.into())],
            sections: BTreeMap::new(),
            tools_added: Vec::new(),
            tools_removed: Vec::new(),
        }
    }

    /// Returns `true` when this message carries no content, sections, or tool
    /// deltas — i.e. replaying it would be a pure no-op.
    pub fn is_empty_patch(&self) -> bool {
        self.content.is_empty()
            && self.sections.is_empty()
            && self.tools_added.is_empty()
            && self.tools_removed.is_empty()
    }
}

/// Walks `messages` and folds every [`SystemMessage`] in order into one
/// effective [`SystemState`]: the reconstructed named sections and tool set a
/// live run would have after processing the same sequence of patches.
///
/// This is the read-side counterpart to a `declare_tool_changes`-style
/// writer (see `tinyagents-harness::agent_loop`): given only the transcript,
/// it answers "what system prompt and tool set was actually in effect" —
/// which is what makes the transcript itself the durable record of tool
/// loadout changes, rather than requiring an out-of-band log.
///
/// Non-system messages are ignored. The leading free-form `content` text of
/// every `SystemMessage` (patches included) is concatenated in order,
/// separated by blank lines, ahead of the rendered named sections — a patch
/// that only adds `content` (no `sections`) still contributes its text.
pub fn replay_system_state(messages: &[Message]) -> (String, Vec<ToolSchema>) {
    let mut leading_content = String::new();
    let mut section_order: Vec<String> = Vec::new();
    let mut section_text: BTreeMap<String, String> = BTreeMap::new();
    let mut tool_order: Vec<String> = Vec::new();
    let mut tool_by_name: BTreeMap<String, ToolSchema> = BTreeMap::new();

    for message in messages {
        let Message::System(system) = message else {
            continue;
        };
        let text = super::concat_text(&system.content);
        if !text.is_empty() {
            if !leading_content.is_empty() {
                leading_content.push_str("\n\n");
            }
            leading_content.push_str(&text);
        }
        for (name, value) in &system.sections {
            match value {
                Some(text) => {
                    if !section_text.contains_key(name) {
                        section_order.push(name.clone());
                    }
                    section_text.insert(name.clone(), text.clone());
                }
                None => {
                    section_text.remove(name);
                    section_order.retain(|existing| existing != name);
                }
            }
        }
        for tool in &system.tools_added {
            if !tool_by_name.contains_key(&tool.name) {
                tool_order.push(tool.name.clone());
            }
            tool_by_name.insert(tool.name.clone(), tool.clone());
        }
        for name in &system.tools_removed {
            tool_by_name.remove(name);
            tool_order.retain(|existing| existing != name);
        }
    }

    let tools: Vec<ToolSchema> = tool_order
        .into_iter()
        .filter_map(|name| tool_by_name.remove(&name))
        .collect();

    let mut parts = Vec::new();
    if !leading_content.is_empty() {
        parts.push(leading_content);
    }
    for name in section_order {
        if let Some(text) = section_text.remove(&name)
            && !text.is_empty()
        {
            parts.push(format!("{name}\n\n{text}"));
        }
    }
    let prompt = parts.join("\n\n");

    (prompt, tools)
}

/// A user/human input message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UserMessage {
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
}

/// An assistant/model output message, possibly carrying tool calls and usage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AssistantMessage {
    /// Optional provider message id for continuation/resume.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
    /// Tool calls requested by the model.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Token usage reported for this message, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A tool result message correlated to a prior tool call.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolMessage {
    /// Id of the tool call this message answers.
    pub tool_call_id: String,
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
    /// Whether a consuming runtime must preserve the content byte-for-byte.
    #[serde(default, skip_serializing_if = "is_false")]
    pub trusted_verbatim: bool,
    /// Host-side structured payload that is never sent to the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Value>,
}

fn is_false(value: &bool) -> bool {
    !*value
}

/// A structured conversation message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Message {
    /// System/developer instructions.
    System(SystemMessage),
    /// User/human input.
    User(UserMessage),
    /// Assistant/model output.
    Assistant(AssistantMessage),
    /// Tool result.
    Tool(ToolMessage),
}

/// An incremental message update used for streaming model output.
///
/// The delta carries three provider-neutral channels so UI consumers can render
/// visible text, reasoning/thinking, and tool-call assembly from one stream:
/// [`text`](Self::text) (visible assistant output),
/// [`reasoning`](Self::reasoning) (thinking output, kept out of the final
/// message text), and [`tool_call`](Self::tool_call) (streamed tool-call
/// fragments).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Incremental visible text fragment.
    #[serde(default)]
    pub text: String,
    /// Incremental reasoning/thinking fragment, when the provider streams
    /// reasoning separately from visible text.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub reasoning: String,
    /// Incremental tool-call fragment, when the provider streams tool calls.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<crate::tool::ToolDelta>,
}

impl MessageDelta {
    /// Creates a delta carrying only a visible text fragment.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Creates a delta carrying only a reasoning/thinking fragment.
    pub fn reasoning(reasoning: impl Into<String>) -> Self {
        Self {
            reasoning: reasoning.into(),
            ..Self::default()
        }
    }
}
