//! Unit tests for the rich internal message model.
//!
//! Covers the ergonomic [`Message`] constructors and the [`Message::text`]
//! accessor (including that it concatenates text blocks and ignores non-text
//! content), that assistant messages carry tool calls and usage, that tool
//! messages preserve their call id, and the [`MessageDelta`] default.

use super::*;
use crate::tool::ToolCall;
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
        origin: None,
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
        origin: None,
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
fn custom_message_text_uses_display_and_carries_no_content_blocks() {
    let custom = Message::Custom(CustomMessage {
        kind: "compaction".into(),
        payload: json!({"summary": "..."}),
        display: Some("Compacted 40 turns".into()),
    });
    assert_eq!(custom.text(), "Compacted 40 turns");
    assert_eq!(custom.char_len(), "Compacted 40 turns".chars().count());
    assert_eq!(
        custom.estimated_char_weight(),
        "Compacted 40 turns".chars().count()
    );
    assert!(custom.artifact().is_none());

    let no_display = Message::Custom(CustomMessage {
        kind: "label".into(),
        payload: json!({"name": "checkpoint"}),
        display: None,
    });
    assert_eq!(no_display.text(), "");
    assert_eq!(no_display.char_len(), 0);
    assert_eq!(no_display.estimated_char_weight(), 0);
}

#[test]
fn custom_message_round_trips_through_serde() {
    let custom = Message::Custom(CustomMessage {
        kind: "audit".into(),
        payload: json!({"note": "reviewed"}),
        display: None,
    });
    let wire = serde_json::to_value(&custom).unwrap();
    assert_eq!(
        wire,
        json!({ "custom": { "kind": "audit", "payload": { "note": "reviewed" } } })
    );
    let back: Message = serde_json::from_value(wire).unwrap();
    assert_eq!(back, custom);
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

#[test]
fn message_origin_round_trips_through_serde() {
    let assistant = AssistantMessage {
        id: Some("msg_1".into()),
        content: vec![ContentBlock::Text("hi".into())],
        tool_calls: Vec::new(),
        usage: None,
        origin: Some(MessageOrigin {
            provider: "anthropic".into(),
            api: "messages".into(),
            model: "claude-sonnet-4-6".into(),
        }),
    };
    let wire = serde_json::to_value(&assistant).unwrap();
    assert_eq!(
        wire["origin"],
        json!({ "provider": "anthropic", "api": "messages", "model": "claude-sonnet-4-6" })
    );
    let back: AssistantMessage = serde_json::from_value(wire).unwrap();
    assert_eq!(back, assistant);
}

#[test]
fn message_origin_is_none_by_default_and_omitted_from_wire() {
    let assistant = AssistantMessage {
        id: None,
        content: vec![ContentBlock::Text("hi".into())],
        tool_calls: Vec::new(),
        usage: None,
        origin: None,
    };
    let wire = serde_json::to_value(&assistant).unwrap();
    assert!(
        wire.get("origin").is_none(),
        "a None origin must not appear on the wire"
    );
}

#[test]
fn legacy_assistant_message_without_origin_field_deserializes_with_none() {
    // Additive tagging: a journal serialized before `origin` existed must
    // still deserialize, with the field defaulting to `None`.
    let legacy = json!({
        "id": "msg_1",
        "content": [{ "text": "hi" }],
        "tool_calls": [],
    });
    let assistant: AssistantMessage = serde_json::from_value(legacy).unwrap();
    assert_eq!(assistant.origin, None);
}
