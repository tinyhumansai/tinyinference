use super::*;
use serde_json::json;

#[test]
fn generic_text_response_with_usage() {
    let body = json!({
        "modelId": "meta.llama",
        "modelVersion": "1.0",
        "chatResponse": {
            "apiFormat": "GENERIC",
            "timeCreated": "2025-01-01T00:00:00Z",
            "choices": [{
                "index": 0,
                "message": {"role": "ASSISTANT", "content": [{"type": "TEXT", "text": "Hello!"}]},
                "finishReason": "stop"
            }],
            "usage": {
                "promptTokens": 10, "completionTokens": 4, "totalTokens": 14,
                "promptTokensDetails": {"cachedTokens": 6},
                "completionTokensDetails": {"reasoningTokens": 2}
            }
        }
    });
    let response = parse_response(body).unwrap();
    assert_eq!(
        response.message.content,
        vec![ContentBlock::Text("Hello!".into())]
    );
    assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    let usage = response.usage.unwrap();
    assert_eq!(
        (usage.input_tokens, usage.output_tokens, usage.total_tokens),
        (10, 4, 14)
    );
    assert_eq!((usage.cache_read_tokens, usage.reasoning_tokens), (6, 2));
}

#[test]
fn generic_tool_calls_reasoning_and_malformed_arguments() {
    let body = json!({"chatResponse": {"apiFormat": "GENERIC", "choices": [{
        "index": 0,
        "message": {
            "role": "ASSISTANT",
            "content": null,
            "reasoningContent": "thinking...",
            "toolCalls": [
                {"type": "FUNCTION", "id": "c1", "name": "get_weather", "arguments": "{\"city\":\"Rome\"}"},
                {"type": "FUNCTION", "id": "c2", "name": "broken", "arguments": "{oops"}
            ]
        },
        "finishReason": "tool_calls"
    }]}});
    let response = parse_response(body).unwrap();
    assert!(matches!(
        &response.message.content[0],
        ContentBlock::Thinking { text, signature: None } if text == "thinking..."
    ));
    let calls = &response.message.tool_calls;
    assert_eq!(calls[0].arguments, json!({"city": "Rome"}));
    assert!(calls[0].invalid.is_none());
    assert!(calls[1].invalid.is_some());
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
}

#[test]
fn cohere_response_text_tool_calls_and_usage() {
    let body = json!({"chatResponse": {
        "apiFormat": "COHERE",
        "text": "Sure",
        "finishReason": "COMPLETE",
        "toolCalls": [{"name": "get_weather", "parameters": {"city": "Oslo"}}],
        "usage": {"promptTokens": 3, "completionTokens": 1, "totalTokens": 4}
    }});
    let response = parse_response(body).unwrap();
    assert_eq!(
        response.message.content,
        vec![ContentBlock::Text("Sure".into())]
    );
    assert_eq!(response.message.tool_calls[0].id, "call_0");
    assert_eq!(
        response.message.tool_calls[0].arguments,
        json!({"city": "Oslo"})
    );
    assert_eq!(response.finish_reason.as_deref(), Some("COMPLETE"));
    assert_eq!(response.usage.unwrap().total_tokens, 4);
}

#[test]
fn malformed_bodies_are_errors() {
    for body in [
        json!({}),
        json!({"chatResponse": {"apiFormat": "GENERIC", "choices": []}}),
        json!({"chatResponse": {"apiFormat": "WAT"}}),
    ] {
        assert!(parse_response(body).is_err());
    }
}

#[test]
fn total_tokens_falls_back_to_the_sum() {
    let usage = parse_usage(&json!({"promptTokens": 2, "completionTokens": 3}));
    assert_eq!(usage.total_tokens, 5);
}
