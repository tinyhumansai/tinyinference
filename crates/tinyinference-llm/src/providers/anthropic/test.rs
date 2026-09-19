use futures::StreamExt;
use serde_json::json;

use super::*;
use crate::cache::CachePolicy;
use crate::message::{ContentBlock, ImageRef, Message, ToolMessage};
use crate::model::{
    ModelStreamItem, PromptSegment, ReasoningConfig, ReasoningEffort, SegmentRole, ToolChoice,
};
use crate::tool::{ToolCall, ToolSchema};

fn cacheable_system() -> PromptSegment {
    PromptSegment {
        id: "system".into(),
        role: SegmentRole::System,
        cacheable: true,
    }
}

fn protecting_policy() -> CachePolicy {
    CachePolicy {
        protect_prompt_prefix: true,
        ..CachePolicy::default()
    }
}

fn echo_tool() -> ToolSchema {
    ToolSchema::new(
        "echo",
        "Echoes its input",
        json!({ "type": "object", "properties": { "text": { "type": "string" } } }),
    )
}

#[test]
fn cacheable_system_prefix_becomes_an_anthropic_cache_breakpoint() {
    let request = ModelRequest::new(vec![
        Message::system("stable instructions"),
        Message::user("hello"),
    ])
    .with_cache_segments(vec![cacheable_system()])
    .with_cache_policy(protecting_policy());
    let body = request_body(&request, "test-model");
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(body["messages"][0]["role"], "user");
}

#[test]
fn cacheable_user_prefix_becomes_a_content_block_breakpoint() {
    let request = ModelRequest::new(vec![Message::user("stable context")])
        .with_cache_segments(vec![PromptSegment {
            id: "history".into(),
            role: SegmentRole::History,
            cacheable: true,
        }])
        .with_cache_policy(protecting_policy());
    let body = request_body(&request, "test-model");
    assert_eq!(body["messages"][0]["content"][0]["type"], "text");
    assert_eq!(body["messages"][0]["content"][0]["text"], "stable context");
    assert_eq!(
        body["messages"][0]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
}

/// The regression this adapter exists to fix: a harness that protects the
/// prefix on its *run* policy never stamps `request.cache_policy`, so gating
/// breakpoints on the request-level policy produced none.
#[test]
fn cacheable_segments_alone_enable_breakpoints() {
    let request = ModelRequest::new(vec![Message::system("stable"), Message::user("hi")])
        .with_cache_segments(vec![cacheable_system()]);
    assert!(request.cache_policy.is_none());
    assert!(request.wants_prompt_cache_breakpoints());
    let body = request_body(&request, "m");
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
}

#[test]
fn a_request_policy_can_veto_breakpoints() {
    let request = ModelRequest::new(vec![Message::system("stable"), Message::user("hi")])
        .with_cache_segments(vec![cacheable_system()])
        .with_cache_policy(CachePolicy {
            protect_prompt_prefix: false,
            ..CachePolicy::default()
        });
    assert!(!request.wants_prompt_cache_breakpoints());
    assert!(
        request_body(&request, "m")["system"][0]
            .get("cache_control")
            .is_none()
    );
}

#[test]
fn no_cacheable_segment_means_no_breakpoints() {
    let request = ModelRequest::new(vec![Message::system("stable"), Message::user("hi")])
        .with_cache_policy(protecting_policy());
    assert!(!request.wants_prompt_cache_breakpoints());
}

/// Tools, system, and the tail of the conversation each carry a marker so a
/// growing tool loop reuses the previous iteration's cache instead of only the
/// system prompt.
#[test]
fn breakpoints_land_on_tools_system_and_the_final_message() {
    let request = ModelRequest::new(vec![
        Message::system("stable"),
        Message::user("first"),
        Message::assistant("reply"),
        Message::user("second"),
    ])
    .with_tools(vec![
        echo_tool(),
        ToolSchema::new("other", "Other", json!({"type":"object"})),
    ])
    .with_cache_segments(vec![cacheable_system()]);
    let body = request_body(&request, "m");
    assert!(body["tools"][0].get("cache_control").is_none());
    assert_eq!(
        body["tools"][1]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    assert_eq!(
        body["system"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert!(messages[0]["content"][0].get("cache_control").is_none());
    assert_eq!(
        messages[2]["content"][0]["cache_control"],
        json!({ "type": "ephemeral" })
    );
    // Three markers: within Anthropic's limit of four.
    let markers = body.to_string().matches("\"cache_control\"").count();
    assert_eq!(markers, 3);
}

#[test]
fn tools_are_declared_with_input_schema_and_tool_choice() {
    let request = ModelRequest::new(vec![Message::user("hi")])
        .with_tools(vec![echo_tool()])
        .with_tool_choice(ToolChoice::Tool("echo".into()));
    let body = request_body(&request, "m");
    assert_eq!(body["tools"][0]["name"], "echo");
    assert_eq!(body["tools"][0]["description"], "Echoes its input");
    assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
    assert_eq!(
        body["tool_choice"],
        json!({ "type": "tool", "name": "echo" })
    );

    let auto = request_body(
        &ModelRequest::new(vec![Message::user("hi")]).with_tools(vec![echo_tool()]),
        "m",
    );
    assert_eq!(auto["tool_choice"], json!({ "type": "auto" }));
    let required = request_body(
        &ModelRequest::new(vec![Message::user("hi")])
            .with_tools(vec![echo_tool()])
            .with_tool_choice(ToolChoice::Required),
        "m",
    );
    assert_eq!(required["tool_choice"], json!({ "type": "any" }));
    let none = request_body(&ModelRequest::new(vec![Message::user("hi")]), "m");
    assert!(none.get("tools").is_none());
    assert!(none.get("tool_choice").is_none());
}

#[test]
fn assistant_tool_calls_become_tool_use_blocks() {
    let mut assistant = Message::assistant("calling");
    if let Message::Assistant(message) = &mut assistant {
        message.tool_calls = vec![ToolCall::new("toolu_1", "echo", json!({"text": "x"}))];
    }
    let body = request_body(
        &ModelRequest::new(vec![Message::user("hi"), assistant]),
        "m",
    );
    let content = body["messages"][1]["content"].as_array().unwrap();
    assert_eq!(content[0], json!({ "type": "text", "text": "calling" }));
    assert_eq!(content[1]["type"], "tool_use");
    assert_eq!(content[1]["id"], "toolu_1");
    assert_eq!(content[1]["name"], "echo");
    assert_eq!(content[1]["input"], json!({"text": "x"}));
}

#[test]
fn tool_results_use_anthropic_tool_result_blocks() {
    let request = ModelRequest::new(vec![Message::Tool(ToolMessage {
        tool_call_id: "tool_1".into(),
        content: vec![ContentBlock::Text("42".into())],
        trusted_verbatim: false,
        artifact: None,
    })]);
    let body = request_body(&request, "test-model");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"][0]["type"], "tool_result");
    assert_eq!(body["messages"][0]["content"][0]["tool_use_id"], "tool_1");
    assert_eq!(
        body["messages"][0]["content"][0]["content"][0]["text"],
        "42"
    );
}

#[test]
fn custom_messages_are_never_sent_to_the_provider() {
    let request = ModelRequest::new(vec![
        Message::user("hi"),
        Message::Custom(crate::message::CustomMessage {
            kind: "compaction".into(),
            payload: serde_json::json!({"summary": "..."}),
            display: Some("Compacted".into()),
        }),
        Message::assistant("hello"),
    ]);
    let body = request_body(&request, "test-model");
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
}

/// Parallel tool calls answer as consecutive tool messages; the Messages API
/// requires them merged into one user turn.
#[test]
fn consecutive_tool_results_merge_into_one_user_message() {
    let tool = |id: &str| {
        Message::Tool(ToolMessage {
            tool_call_id: id.into(),
            content: vec![ContentBlock::Text("ok".into())],
            trusted_verbatim: false,
            artifact: None,
        })
    };
    let body = request_body(
        &ModelRequest::new(vec![
            Message::user("go"),
            Message::assistant("x"),
            tool("a"),
            tool("b"),
        ]),
        "m",
    );
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[2]["content"].as_array().unwrap().len(), 2);
    assert_eq!(messages[2]["content"][1]["tool_use_id"], "b");
}

#[test]
fn empty_tool_result_sends_an_empty_string() {
    let body = request_body(
        &ModelRequest::new(vec![Message::Tool(ToolMessage {
            tool_call_id: "t".into(),
            content: vec![],
            trusted_verbatim: false,
            artifact: None,
        })]),
        "m",
    );
    assert_eq!(body["messages"][0]["content"][0]["content"], "");
}

#[test]
fn signed_thinking_is_replayed_and_unsigned_thinking_is_dropped() {
    let assistant = Message::Assistant(crate::message::AssistantMessage {
        id: None,
        content: vec![
            ContentBlock::Thinking {
                text: "hmm".into(),
                signature: Some("sig".into()),
            },
            ContentBlock::Thinking {
                text: "unsigned".into(),
                signature: None,
            },
            ContentBlock::RedactedThinking {
                data: "blob".into(),
            },
            ContentBlock::Text("answer".into()),
        ],
        tool_calls: vec![],
        usage: None,
        origin: None,
    });
    let body = request_body(&ModelRequest::new(vec![Message::user("q"), assistant]), "m");
    let content = body["messages"][1]["content"].as_array().unwrap();
    assert_eq!(content.len(), 3);
    assert_eq!(
        content[0],
        json!({"type": "thinking", "thinking": "hmm", "signature": "sig"})
    );
    assert_eq!(
        content[1],
        json!({"type": "redacted_thinking", "data": "blob"})
    );
    assert_eq!(content[2]["text"], "answer");
    assert!(!body.to_string().contains("unsigned"));
}

#[test]
fn images_become_base64_or_url_sources() {
    let request = ModelRequest::new(vec![Message::User(crate::message::UserMessage {
        content: vec![
            ContentBlock::Text("look".into()),
            ContentBlock::Image(ImageRef {
                url: "data:image/jpeg;base64,AAAA".into(),
                mime_type: None,
            }),
            ContentBlock::Image(ImageRef {
                url: "https://example.com/a.png".into(),
                mime_type: None,
            }),
        ],
    })]);
    let body = request_body(&request, "m");
    let content = body["messages"][0]["content"].as_array().unwrap();
    assert_eq!(
        content[1]["source"],
        json!({ "type": "base64", "media_type": "image/jpeg", "data": "AAAA" })
    );
    assert_eq!(
        content[2]["source"],
        json!({ "type": "url", "url": "https://example.com/a.png" })
    );
}

#[test]
fn provider_options_flatten_except_reserved_and_the_routing_hint() {
    let request = ModelRequest::new(vec![Message::user("hi")]).with_provider_options(json!({
        "thinking": { "type": "enabled", "budget_tokens": 1024 },
        "metadata": { "user_id": "u1" },
        "prompt_cache_key": "tap-abc",
        "model": "must-not-override",
    }));
    let body = request_body(&request, "m");
    assert_eq!(body["thinking"]["budget_tokens"], 1024);
    assert_eq!(body["metadata"]["user_id"], "u1");
    assert!(body.get("prompt_cache_key").is_none());
    assert_eq!(body["model"], "m");
}

#[test]
fn normalized_reasoning_is_lowered_to_anthropic_thinking() {
    let budgeted = ModelRequest::new(vec![Message::user("hi")]).with_reasoning(ReasoningConfig {
        budget_tokens: Some(2048),
        ..ReasoningConfig::default()
    });
    let body = request_body(&budgeted, "m");
    assert_eq!(
        body["thinking"],
        json!({ "type": "enabled", "budget_tokens": 2048 })
    );

    let adaptive =
        ModelRequest::new(vec![Message::user("hi")]).with_reasoning_effort(ReasoningEffort::High);
    let body = request_body(&adaptive, "m");
    assert_eq!(body["thinking"], json!({ "type": "adaptive" }));
    assert_eq!(body["output_config"]["effort"], "high");
}

#[test]
fn default_model_is_current() {
    assert_eq!(AnthropicModel::new("key").model, "claude-sonnet-4-6");
}

#[test]
fn profile_advertises_tools_streaming_and_vision() {
    let model = AnthropicModel::new("key");
    let profile = model.profile.clone();
    assert!(profile.tool_calling);
    assert!(profile.streaming);
    assert!(profile.streaming_tool_chunks);
    assert!(profile.modalities.image_in);
}

#[test]
fn cache_usage_is_mapped_from_anthropic_response() {
    let response = parse_response(json!({
        "id": "msg_1", "content": [{ "type": "text", "text": "hello" }], "stop_reason": "end_turn",
        "usage": { "input_tokens": 100, "output_tokens": 5, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 10 }
    })).unwrap();
    assert_eq!(response.text(), "hello");
    let usage = response.usage.unwrap();
    assert_eq!(usage.input_tokens, 200);
    assert_eq!(usage.total_tokens, 205);
    assert_eq!(usage.cache_read_tokens, 90);
    assert_eq!(usage.cache_creation_tokens, 10);
}

#[test]
fn tool_use_blocks_parse_into_tool_calls() {
    let response = parse_response(json!({
        "id": "msg_2",
        "content": [
            { "type": "thinking", "thinking": "plan", "signature": "s" },
            { "type": "text", "text": "Let me check." },
            { "type": "tool_use", "id": "toolu_1", "name": "echo", "input": { "text": "x" } }
        ],
        "stop_reason": "tool_use",
        "usage": { "input_tokens": 2, "output_tokens": 3 }
    }))
    .unwrap();
    assert_eq!(response.text(), "Let me check.");
    assert_eq!(response.finish_reason.as_deref(), Some("tool_use"));
    let calls = response.message.tool_calls;
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "toolu_1");
    assert_eq!(calls[0].name, "echo");
    assert_eq!(calls[0].arguments, json!({ "text": "x" }));
    assert!(matches!(
        &response.message.content[0],
        ContentBlock::Thinking { signature: Some(s), .. } if s == "s"
    ));
}

#[test]
fn malformed_successful_responses_are_rejected() {
    let valid = json!({
        "id": "msg_1",
        "content": [{ "type": "text", "text": "hello" }],
        "stop_reason": "end_turn",
        "usage": { "input_tokens": 1, "output_tokens": 1 }
    });
    let mut cases = Vec::new();
    let mut missing_id = valid.clone();
    missing_id.as_object_mut().unwrap().remove("id");
    cases.push(missing_id);
    let mut bad_content = valid.clone();
    bad_content["content"] = json!({});
    cases.push(bad_content);
    let mut bad_usage = valid.clone();
    bad_usage["usage"]["output_tokens"] = json!("one");
    cases.push(bad_usage);
    let mut bad_tool = valid;
    bad_tool["content"] = json!([{
        "type": "tool_use", "id": "toolu_1", "name": "echo"
    }]);
    cases.push(bad_tool);

    for body in cases {
        assert!(parse_response(body).is_err());
    }
}

#[tokio::test]
async fn cleartext_anthropic_endpoint_requires_explicit_opt_in() {
    let model = AnthropicModel::with_base_url("secret", "http://example.com/v1");
    let error = model
        .post(&ModelRequest::new(vec![Message::user("hi")]), false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("HTTPS endpoint"));
}

#[test]
fn request_body_forwards_generation_controls() {
    let request = ModelRequest::new(vec![Message::user("hello")])
        .with_temperature(0.2)
        .with_top_p(0.8)
        .with_stop_sequences(["END"]);
    let body = request_body(&request, "test-model");
    assert_eq!(body["temperature"], 0.2);
    assert_eq!(body["top_p"], 0.8);
    assert_eq!(body["stop_sequences"], json!(["END"]));
    assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
}

#[test]
fn temperature_is_clamped_into_anthropics_range() {
    let body = request_body(
        &ModelRequest::new(vec![Message::user("hello")]).with_temperature(1.7),
        "m",
    );
    assert_eq!(body["temperature"], 1.0);
    let below = request_body(
        &ModelRequest::new(vec![Message::user("hello")]).with_temperature(-0.5),
        "m",
    );
    assert_eq!(below["temperature"], 0.0);
}

#[test]
fn temperature_override_is_recorded_on_the_model() {
    let model = AnthropicModel::new("key").with_temperature_override(Some(0.3));
    assert_eq!(model.temperature_override, Some(0.3));
    assert!(format!("{model:?}").contains("temperature_override: Some(0.3)"));
}

#[test]
fn temperature_policy_uses_the_effective_request_model() {
    let model = AnthropicModel::new("key")
        .with_model("claude-sonnet-4-6")
        .with_temperature_override(Some(0.3))
        .with_temperature_unsupported_models(["claude-3-5-*"]);
    let normal =
        model.request_body(&ModelRequest::new(vec![Message::user("hello")]).with_temperature(0.7));
    assert_eq!(normal["temperature"], 0.3);

    let suppressed = model.request_body(
        &ModelRequest::new(vec![Message::user("hello")])
            .with_model("claude-3-5-sonnet")
            .with_temperature(0.7),
    );
    assert!(suppressed.get("temperature").is_none());
}

#[test]
fn debug_redacts_the_api_key() {
    let model = AnthropicModel::new("secret-api-key");
    let debug = format!("{model:?}");
    assert!(debug.contains("[redacted]"));
    assert!(!debug.contains("secret-api-key"));
}

fn sse(events: &[serde_json::Value]) -> Vec<u8> {
    let mut out = String::new();
    for event in events {
        out.push_str(&format!(
            "event: {}\ndata: {}\n\n",
            event["type"].as_str().unwrap(),
            event
        ));
    }
    out.into_bytes()
}

#[tokio::test]
async fn streaming_reassembles_text_tool_calls_and_cache_usage() {
    let events = [
        json!({"type":"message_start","message":{"id":"msg_s","usage":{"input_tokens":3,"cache_read_input_tokens":900,"cache_creation_input_tokens":0,"output_tokens":1}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        json!({"type":"ping"}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_9","name":"echo","input":{}}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"text\":"}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"héllo\"}"}}),
        json!({"type":"content_block_stop","index":1}),
        json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}),
        json!({"type":"message_stop"}),
    ];
    let bytes = sse(&events);
    // Split on a multi-byte character boundary inside "héllo" to prove the
    // line buffer reassembles before decoding.
    let split = bytes.iter().position(|b| *b == 0xC3).unwrap() + 1;
    let chunks = vec![bytes[..split].to_vec(), bytes[split..].to_vec()];
    let items: Vec<ModelStreamItem> = stream::stream_from_bytes(chunks, "m").collect().await;

    assert!(matches!(items[0], ModelStreamItem::Started));
    let text: String = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello");
    let tool_fragments: String = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::ToolCallDelta(delta) => Some(delta.content.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_fragments, "{\"text\":\"héllo\"}");
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("stream must end in Completed, got {:?}", items.last());
    };
    assert_eq!(response.text(), "Hello");
    assert_eq!(response.message.id.as_deref(), Some("msg_s"));
    assert_eq!(response.finish_reason.as_deref(), Some("tool_use"));
    assert_eq!(response.message.tool_calls.len(), 1);
    assert_eq!(
        response.message.tool_calls[0].arguments,
        json!({"text": "héllo"})
    );
    let usage = response.usage.unwrap();
    assert_eq!(usage.cache_read_tokens, 900);
    assert_eq!(usage.input_tokens, 903);
    assert_eq!(usage.output_tokens, 12);
    assert_eq!(usage.total_tokens, 915);
}

#[tokio::test]
async fn streaming_thinking_arrives_on_the_reasoning_channel_with_its_signature() {
    let events = [
        json!({"type":"message_start","message":{"id":"m","usage":{"input_tokens":1,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}),
        json!({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}),
        json!({"type":"message_stop"}),
    ];
    let items: Vec<ModelStreamItem> = stream::stream_from_bytes(vec![sse(&events)], "m")
        .collect()
        .await;
    let reasoning: String = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::MessageDelta(delta) => Some(delta.reasoning.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "plan");
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("expected Completed");
    };
    assert_eq!(response.text(), "done");
    assert!(matches!(
        &response.message.content[0],
        ContentBlock::Thinking { text, signature: Some(s) } if text == "plan" && s == "sig"
    ));
}

#[tokio::test]
async fn streaming_without_message_stop_is_a_provider_failure() {
    let events = [
        json!({"type":"message_start","message":{"id":"m","usage":{"input_tokens":1,"output_tokens":0}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":"partial"}}),
    ];
    let items: Vec<ModelStreamItem> = stream::stream_from_bytes(vec![sse(&events)], "m")
        .collect()
        .await;
    assert!(
        matches!(items.last(), Some(ModelStreamItem::ProviderFailed(error)) if error.retryable)
    );
}

#[tokio::test]
async fn streaming_error_event_is_a_provider_failure() {
    let events = [
        json!({"type":"message_start","message":{"id":"m","usage":{"input_tokens":1,"output_tokens":0}}}),
        json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}),
    ];
    let items: Vec<ModelStreamItem> = stream::stream_from_bytes(vec![sse(&events)], "m")
        .collect()
        .await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(error))
            if error.message.contains("Overloaded")
                && error.code.as_deref() == Some("overloaded_error")
                && error.retryable
                && error.raw.is_some()
    ));
}

#[tokio::test]
async fn permanent_stream_error_is_not_retryable() {
    let events = [json!({
        "type":"error",
        "error":{"type":"authentication_error","message":"invalid api key"}
    })];
    let items: Vec<ModelStreamItem> = stream::stream_from_bytes(vec![sse(&events)], "m")
        .collect()
        .await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(error))
            if error.code.as_deref() == Some("authentication_error") && !error.retryable
    ));
}

#[tokio::test]
async fn malformed_sse_payload_terminates_with_provider_failure() {
    let bytes = b"data: {not-json}\n\ndata: {\"type\":\"message_stop\"}\n\n".to_vec();
    let items: Vec<ModelStreamItem> = stream::stream_from_bytes(vec![bytes], "m").collect().await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(error))
            if error.message.contains("invalid Anthropic SSE data payload")
    ));
    assert!(
        !items
            .iter()
            .any(|item| matches!(item, ModelStreamItem::Completed(_)))
    );
}

#[tokio::test]
async fn oversized_sse_content_block_index_is_rejected() {
    let events = [json!({
        "type":"content_block_start",
        "index": 1_000_000,
        "content_block":{"type":"text","text":""}
    })];
    let items: Vec<ModelStreamItem> = stream::stream_from_bytes(vec![sse(&events)], "m")
        .collect()
        .await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(error)) if error.message.contains("exceeds limit")
    ));
}

#[test]
fn origin_for_records_provider_api_and_effective_model() {
    let model = AnthropicModel::new("key").with_model("claude-opus-4-6");
    let request = ModelRequest::default();
    let origin = model.origin_for(&request);
    assert_eq!(origin.provider, "anthropic");
    assert_eq!(origin.api, "messages");
    assert_eq!(origin.model, "claude-opus-4-6");
}

#[test]
fn origin_for_prefers_a_per_request_model_override() {
    let model = AnthropicModel::new("key").with_model("claude-opus-4-6");
    let mut request = ModelRequest::default();
    request.model = Some("claude-sonnet-4-6".to_string());
    let origin = model.origin_for(&request);
    assert_eq!(origin.model, "claude-sonnet-4-6");
}

#[test]
fn default_profile_advertises_the_tool_call_id_shape() {
    let model = AnthropicModel::new("key");
    let profile = model.profile().expect("anthropic always has a profile");
    assert_eq!(
        profile.tool_call_id_pattern.as_deref(),
        Some("^[a-zA-Z0-9_-]{1,64}$")
    );
    assert_eq!(profile.max_tool_call_id_len, Some(64));
}

#[tokio::test]
async fn streamed_terminal_response_carries_origin() {
    let events = [
        json!({"type":"message_start","message":{"id":"msg_o","usage":{"input_tokens":1,"cache_read_input_tokens":0,"cache_creation_input_tokens":0,"output_tokens":1}}}),
        json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}),
        json!({"type":"content_block_stop","index":0}),
        json!({"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}),
        json!({"type":"message_stop"}),
    ];
    let items: Vec<ModelStreamItem> = stream::stream_from_bytes(vec![sse(&events)], "claude-opus-4-6")
        .collect()
        .await;
    let completed = items
        .into_iter()
        .find_map(|item| match item {
            ModelStreamItem::Completed(response) => Some(response),
            _ => None,
        })
        .expect("a terminal Completed item");
    let origin = completed
        .message
        .origin
        .expect("origin stamped on stream terminal");
    assert_eq!(origin.provider, "anthropic");
    assert_eq!(origin.api, "messages");
    assert_eq!(origin.model, "claude-opus-4-6");
}
