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

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tool::ToolCall;
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
    /// A reference to an audio clip.
    Audio(MediaRef),
    /// A reference to a video clip.
    Video(MediaRef),
    /// A reference to a document (PDF and similar).
    Document(MediaRef),
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

/// A reference to a non-text media asset (audio, video, or document),
/// carried by [`ContentBlock::Audio`]/[`ContentBlock::Video`]/
/// [`ContentBlock::Document`].
///
/// The harness itself never fetches [`MediaRef::Url`] or
/// [`MediaRef::Path`] content; a host that needs to resolve those into bytes
/// (for example to inline them for a provider whose wire format requires
/// base64) owns that fetch and its safety policy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "source")]
pub enum MediaRef {
    /// A remote or data URL.
    Url {
        /// The URL (or data URI) to fetch.
        url: String,
        /// Optional MIME type (for example `audio/wav`), when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
    /// Inline base64-encoded content.
    Base64 {
        /// Base64-encoded bytes.
        data: String,
        /// MIME type of the decoded content (for example
        /// `application/pdf`).
        media_type: String,
    },
    /// A local filesystem path. Only meaningful to a host that has
    /// filesystem access and chooses to resolve it; providers never see raw
    /// paths and a host must inline the file's bytes before sending it.
    Path {
        /// The filesystem path.
        path: String,
        /// Optional MIME type, when known.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
    },
}

impl MediaRef {
    /// Creates a [`MediaRef::Url`] reference.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self::Url {
            url: url.into(),
            media_type: None,
        }
    }

    /// Creates a [`MediaRef::Base64`] reference.
    #[must_use]
    pub fn base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        Self::Base64 {
            data: data.into(),
            media_type: media_type.into(),
        }
    }

    /// Creates a [`MediaRef::Path`] reference.
    #[must_use]
    pub fn path(path: impl Into<String>) -> Self {
        Self::Path {
            path: path.into(),
            media_type: None,
        }
    }

    /// Returns the MIME type, when known.
    #[must_use]
    pub fn media_type(&self) -> Option<&str> {
        match self {
            Self::Url { media_type, .. } | Self::Path { media_type, .. } => media_type.as_deref(),
            Self::Base64 { media_type, .. } => Some(media_type.as_str()),
        }
    }
}

/// A system/developer instruction message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SystemMessage {
    /// Ordered content blocks.
    pub content: Vec<ContentBlock>,
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
