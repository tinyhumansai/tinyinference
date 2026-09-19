//! Unit tests for the rich internal message model.
//!
//! Covers the ergonomic [`Message`] constructors and the [`Message::text`]
//! accessor (including that it concatenates text blocks and ignores non-text
//! content), that assistant messages carry tool calls and usage, that tool
//! messages preserve their call id, and the [`MessageDelta`] default.

use super::*;
use crate::tool::{ToolCall, ToolSchema};
use crate::usage::Usage;
use serde_json::json;

#[test]
fn constructors_and_text_accessor() {
    assert_eq!(Message::system("sys").text(), "sys");
    assert_eq!(Message::user("hi").text(), "hi");
    assert_eq!(Message::assistant("yo").text(), "yo");
    assert_eq!(Message::tool("c-1", "result").text(), "result");
}

#[test]
fn text_concatenates_and_ignores_non_text() {
    let msg = Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("a".into()),
            ContentBlock::Json(json!({"k": "v"})),
            ContentBlock::Text("b".into()),
        ],
    });
    assert_eq!(msg.text(), "ab");
}

#[test]
fn char_len_matches_text_char_count_including_multibyte() {
    // Mixed text blocks with multi-byte scalar values; non-text blocks ignored.
    let msg = Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("héllo".into()),
            ContentBlock::Json(json!({"k": "v"})),
            ContentBlock::Text("🌍!".into()),
        ],
    });
    // char_len counts Unicode scalar values, matching text().chars().count()
    // without allocating the joined string.
    assert_eq!(msg.char_len(), msg.text().chars().count());
    assert_eq!(
        msg.char_len(),
        "héllo".chars().count() + "🌍!".chars().count()
    );
}

#[test]
fn estimated_char_weight_counts_non_text_blocks() {
    let text = "hello";
    let reasoning = "some private reasoning";
    let json_block = json!({"key": "value", "n": 42});
    let msg = Message::User(UserMessage {
        content: vec![
            ContentBlock::Text(text.into()),
            ContentBlock::Json(json_block.clone()),
            ContentBlock::thinking(reasoning),
        ],
    });

    // Visible-text accounting still ignores JSON and reasoning.
    assert_eq!(msg.char_len(), text.chars().count());

    // Token-estimation weight additionally counts JSON and reasoning, which
    // occupy real context, so it is strictly larger than the visible-text len.
    let expected =
        text.chars().count() + json_block.to_string().chars().count() + reasoning.chars().count();
    assert_eq!(msg.estimated_char_weight(), expected);
    assert!(msg.estimated_char_weight() > msg.char_len());
}

#[test]
fn estimated_char_weight_costs_images_that_carry_no_visible_text() {
    let msg = Message::User(UserMessage {
        content: vec![ContentBlock::Image(ImageRef {
            url: "data:image/png;base64,AAAA".into(),
            mime_type: Some("image/png".into()),
        })],
    });

    // An image contributes no visible text but a flat, non-trivial weight, so a
    // vision-heavy transcript can no longer under-count to zero and defeat
    // compaction gating.
    assert_eq!(msg.char_len(), 0);
    assert_eq!(msg.estimated_char_weight(), super::IMAGE_CHAR_WEIGHT);
}

#[test]
fn assistant_holds_tool_calls_and_usage() {
    let msg = Message::Assistant(AssistantMessage {
        id: Some("m-1".into()),
        content: vec![ContentBlock::Text("calling".into())],
        tool_calls: vec![ToolCall::new("c-1", "lookup", json!({}))],
        usage: Some(Usage::new(5, 5)),
    });
    if let Message::Assistant(a) = &msg {
        assert_eq!(a.tool_calls.len(), 1);
        assert_eq!(a.usage.unwrap().total_tokens, 10);
    } else {
        panic!("expected assistant");
    }
}

#[test]
fn tool_message_carries_call_id() {
    let msg = Message::tool("c-7", "done");
    match &msg {
        Message::Tool(t) => {
            assert_eq!(t.tool_call_id, "c-7");
            assert_eq!(msg.text(), "done");
        }
        _ => panic!("expected tool message"),
    }
}

#[test]
fn message_delta_default() {
    let delta = MessageDelta::default();
    assert_eq!(delta.text, "");
    assert!(delta.tool_call.is_none());
}

#[test]
fn text_ignores_thinking_blocks() {
    let msg = Message::Assistant(AssistantMessage {
        id: None,
        content: vec![
            ContentBlock::thinking("let me reason about this"),
            ContentBlock::Text("the answer is 42".into()),
            ContentBlock::RedactedThinking {
                data: "opaque".into(),
            },
        ],
        tool_calls: Vec::new(),
        usage: None,
    });
    // Reasoning blocks must never leak into visible text.
    assert_eq!(msg.text(), "the answer is 42");
}

#[test]
fn thinking_accessors() {
    let signed = ContentBlock::Thinking {
        text: "reasoning".into(),
        signature: Some("sig-abc".into()),
    };
    assert_eq!(signed.as_thinking(), Some(("reasoning", Some("sig-abc"))));
    assert!(signed.is_reasoning());
    assert!(signed.as_text().is_none());

    let unsigned = ContentBlock::thinking("just thinking");
    assert_eq!(unsigned.as_thinking(), Some(("just thinking", None)));
    assert!(unsigned.is_reasoning());

    let redacted = ContentBlock::RedactedThinking {
        data: "opaque".into(),
    };
    assert!(redacted.is_reasoning());
    assert!(redacted.as_thinking().is_none());

    assert!(!ContentBlock::Text("hi".into()).is_reasoning());
}

#[test]
fn thinking_block_serde_round_trips() {
    let signed = ContentBlock::Thinking {
        text: "step by step".into(),
        signature: Some("sig-1".into()),
    };
    let wire = serde_json::to_value(&signed).unwrap();
    assert_eq!(
        wire,
        json!({ "thinking": { "text": "step by step", "signature": "sig-1" } })
    );
    let back: ContentBlock = serde_json::from_value(wire).unwrap();
    assert_eq!(back, signed);

    // An unsigned thinking block omits the signature key entirely.
    let unsigned = ContentBlock::thinking("no sig");
    let wire = serde_json::to_value(&unsigned).unwrap();
    assert_eq!(wire, json!({ "thinking": { "text": "no sig" } }));
    let back: ContentBlock = serde_json::from_value(wire).unwrap();
    assert_eq!(back, unsigned);

    let redacted = ContentBlock::RedactedThinking {
        data: "opaque".into(),
    };
    let wire = serde_json::to_value(&redacted).unwrap();
    assert_eq!(wire, json!({ "redacted_thinking": { "data": "opaque" } }));
    let back: ContentBlock = serde_json::from_value(wire).unwrap();
    assert_eq!(back, redacted);
}

#[test]
fn legacy_content_without_thinking_still_parses() {
    // Additive tagging: transcripts serialized before thinking blocks existed
    // (only text/json/image/provider_extension) must deserialize unchanged.
    let legacy = json!([
        { "text": "hello" },
        { "json": { "k": "v" } }
    ]);
    let blocks: Vec<ContentBlock> = serde_json::from_value(legacy).unwrap();
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].as_text(), Some("hello"));
    assert!(!blocks[1].is_reasoning());
}

// ---------------------------------------------------------------------------
// SystemMessage sections/tool deltas, replay_system_state (B6)
// ---------------------------------------------------------------------------

fn tool(name: &str) -> ToolSchema {
    ToolSchema::new(name, format!("{name} tool"), json!({"type": "object"}))
}

#[test]
fn legacy_system_message_without_new_fields_deserializes_as_a_no_op_patch() {
    // A transcript persisted before SystemMessage gained sections/tool deltas
    // must still deserialize: the new fields are all `#[serde(default)]`.
    let legacy = json!({ "content": [{ "text": "you are a helpful assistant" }] });
    let msg: SystemMessage = serde_json::from_value(legacy).unwrap();
    assert_eq!(
        msg.content,
        vec![ContentBlock::Text("you are a helpful assistant".into())]
    );
    assert!(msg.sections.is_empty());
    assert!(msg.tools_added.is_empty());
    assert!(msg.tools_removed.is_empty());
}

#[test]
fn system_message_text_constructor_is_a_no_op_patch() {
    let msg = SystemMessage::text("hello");
    assert!(!msg.is_empty_patch());
    assert!(msg.sections.is_empty());
    assert!(msg.tools_added.is_empty());
    assert!(msg.tools_removed.is_empty());

    assert!(SystemMessage::default().is_empty_patch());
}

#[test]
fn replay_system_state_folds_sections_and_tool_deltas_in_order() {
    // Turn 1: baseline persona plus two tools.
    let base = SystemMessage {
        tools_added: vec![tool("search"), tool("read_file")],
        ..SystemMessage::text("You are Aria.")
    };
    let turn1 = vec![Message::System(base), Message::user("hi")];

    // Turn 2: a patch adds a "browse" tool, drops "read_file", and adds a
    // named instructions section.
    let mut sections = std::collections::BTreeMap::new();
    sections.insert(
        "tool_changes".to_string(),
        Some("browse is now available.".to_string()),
    );
    let patch = SystemMessage {
        tools_added: vec![tool("browse")],
        tools_removed: vec!["read_file".to_string()],
        sections,
        ..SystemMessage::default()
    };
    let mut messages = turn1;
    messages.push(Message::assistant("ok"));
    messages.push(Message::System(patch));
    messages.push(Message::user("go"));

    let (prompt, tools) = replay_system_state(&messages);

    let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
    assert_eq!(names, vec!["search", "browse"]);
    assert!(prompt.contains("You are Aria."));
    assert!(prompt.contains("tool_changes"));
    assert!(prompt.contains("browse is now available."));
}

#[test]
fn replay_system_state_section_removal_drops_it_from_the_effective_prompt() {
    let mut add_sections = std::collections::BTreeMap::new();
    add_sections.insert("scratch".to_string(), Some("temporary note".to_string()));
    let add = SystemMessage {
        sections: add_sections,
        ..SystemMessage::default()
    };
    let mut remove_sections = std::collections::BTreeMap::new();
    remove_sections.insert("scratch".to_string(), None);
    let remove = SystemMessage {
        sections: remove_sections,
        ..SystemMessage::default()
    };

    let messages = vec![Message::System(add), Message::System(remove)];
    let (prompt, tools) = replay_system_state(&messages);
    assert!(!prompt.contains("temporary note"));
    assert!(tools.is_empty());
}

#[test]
fn replay_system_state_later_tool_schema_for_same_name_wins() {
    let mut first = SystemMessage::default();
    first.tools_added = vec![ToolSchema::new("search", "v1", json!({"type": "object"}))];
    let mut second = SystemMessage::default();
    second.tools_added = vec![ToolSchema::new("search", "v2", json!({"type": "object"}))];

    let messages = vec![Message::System(first), Message::System(second)];
    let (_, tools) = replay_system_state(&messages);
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].description, "v2");
}

#[test]
fn replay_system_state_ignores_non_system_messages() {
    let messages = vec![
        Message::user("hello"),
        Message::assistant("hi"),
        Message::tool("c-1", "result"),
    ];
    let (prompt, tools) = replay_system_state(&messages);
    assert!(prompt.is_empty());
    assert!(tools.is_empty());
}

#[test]
fn model_profile_default_disallows_mid_conversation_system_messages() {
    // Conservative default: a caller must opt in per-provider (see
    // `providers::openai::transport::derive_profile`), because folding into
    // the leading system message is always correct while inserting one where
    // the provider does not actually honor it silently loses content.
    assert!(!crate::model::ModelProfile::default().mid_conversation_system_messages);
}
