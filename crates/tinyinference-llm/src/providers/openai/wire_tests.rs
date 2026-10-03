//! Request-body and header tests for the OpenAI-compatible provider.
//!
//! Ported from OpenHuman's `tests/inference_provider_e2e.rs`, which drove
//! `OpenAiModel` against a wiremock server and no OpenHuman code. The
//! translation tests assert on the JSON body directly; the wire tests use a
//! one-shot local socket to check what is actually sent over HTTP.

use std::io::{Read, Write};
use std::net::TcpListener;

use serde_json::{Value, json};

use super::*;
use crate::message::Message;
use crate::model::{ChatModel, ModelRequest};

fn body_for(model: &OpenAiModel, request: &ModelRequest) -> Value {
    serde_json::to_value(model.translate_request(request).unwrap()).unwrap()
}

fn request(model: &str, temperature: f64) -> ModelRequest {
    ModelRequest::new(vec![Message::user("hi")])
        .with_model(model)
        .with_temperature(temperature)
}

fn unsupported_patterns() -> Vec<String> {
    ["o1*", "o3*", "o4*", "gpt-5*"]
        .into_iter()
        .map(String::from)
        .collect()
}

#[test]
fn temperature_is_sent_for_models_that_support_it() {
    let model = OpenAiModel::new("k").with_temperature_unsupported_models(unsupported_patterns());
    let body = body_for(&model, &request("gpt-4o-mini", 0.7));
    assert_eq!(body["temperature"].as_f64(), Some(0.7), "body={body}");
}

#[test]
fn temperature_is_omitted_for_unsupported_model_families() {
    let model = OpenAiModel::new("k").with_temperature_unsupported_models(unsupported_patterns());
    for name in [
        "o1-preview",
        "o3-mini",
        "o4-preview",
        "gpt-5",
        "gpt-5-turbo",
    ] {
        let body = body_for(&model, &request(name, 0.7));
        assert!(
            body.get("temperature").is_none(),
            "temperature must be absent for model={name}; body={body}"
        );
        assert_eq!(body["model"], json!(name));
    }
}

#[test]
fn temperature_is_kept_when_no_unsupported_patterns_are_configured() {
    let model = OpenAiModel::new("k");
    let body = body_for(&model, &request("o1-preview", 0.4));
    assert_eq!(body["temperature"].as_f64(), Some(0.4), "body={body}");
}

#[test]
fn request_model_overrides_the_configured_default_model() {
    let model = OpenAiModel::new("k").with_model("gpt-4.1-mini");
    let body = body_for(&model, &request("claude-3-sonnet", 0.5));
    assert_eq!(body["model"], json!("claude-3-sonnet"));

    let no_override = ModelRequest::new(vec![Message::user("hi")]);
    assert_eq!(
        body_for(&model, &no_override)["model"],
        json!("gpt-4.1-mini")
    );
}

fn budgeted_request(effort: Option<crate::model::ReasoningEffort>) -> ModelRequest {
    ModelRequest::new(vec![Message::user("hi")])
        .with_model("deepseek/deepseek-v4")
        .with_reasoning(crate::model::ReasoningConfig {
            effort,
            budget_tokens: Some(4505),
            summary: None,
        })
}

#[test]
fn openrouter_receives_the_reasoning_budget_as_reasoning_max_tokens() {
    let model = OpenAiModel::new("k").with_base_url("https://openrouter.ai/api/v1");
    let body = body_for(
        &model,
        &budgeted_request(Some(crate::model::ReasoningEffort::High)),
    );
    assert_eq!(
        body["reasoning"],
        json!({ "max_tokens": 4505 }),
        "body={body}"
    );
    // OpenRouter takes one of `effort` / `max_tokens`; the explicit budget wins.
    assert!(body.get("reasoning_effort").is_none(), "body={body}");
}

#[test]
fn non_openrouter_endpoints_never_receive_a_reasoning_object() {
    for base in ["https://api.openai.com/v1", "https://api.deepseek.com/v1"] {
        let model = OpenAiModel::new("k").with_base_url(base);
        let body = body_for(
            &model,
            &budgeted_request(Some(crate::model::ReasoningEffort::High)),
        );
        assert!(body.get("reasoning").is_none(), "base={base} body={body}");
        assert_eq!(body["reasoning_effort"], json!("high"), "base={base}");
    }
}

#[test]
fn explicit_reasoning_provider_option_wins_over_the_budget_on_openrouter() {
    let model = OpenAiModel::new("k").with_base_url("https://openrouter.ai/api/v1");
    let request =
        budgeted_request(None).with_provider_options(json!({ "reasoning": { "effort": "low" } }));
    let body = body_for(&model, &request);
    assert_eq!(body["reasoning"], json!({ "effort": "low" }), "body={body}");
}

/// Reads one HTTP/1.1 request (head plus a `content-length` body) off `sock`.
fn read_request(sock: &mut std::net::TcpStream) -> Vec<u8> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = sock.read(&mut tmp).unwrap();
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        let Some(split) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let head = String::from_utf8_lossy(&buf[..split]).to_ascii_lowercase();
        let len = head
            .lines()
            .find_map(|l| {
                l.strip_prefix("content-length:")?
                    .trim()
                    .parse::<usize>()
                    .ok()
            })
            .unwrap_or(0);
        if buf.len() >= split + 4 + len {
            break;
        }
    }
    buf
}

/// Serves one canned chat completion and returns the raw request it received.
fn serve_once() -> (String, std::thread::JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let buf = read_request(&mut sock);
        let payload = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "m",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "Hello!" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15 }
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{payload}",
            payload.len()
        );
        sock.write_all(response.as_bytes()).unwrap();
        String::from_utf8_lossy(&buf).into_owned()
    });
    (base, handle)
}

fn header_value(raw: &str, name: &str) -> Option<String> {
    raw.split("\r\n\r\n").next()?.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.trim()
            .eq_ignore_ascii_case(name)
            .then(|| value.trim().to_string())
    })
}

#[tokio::test]
async fn bearer_auth_is_sent_as_authorization_header_and_body_carries_the_model() {
    let (base, server) = serve_once();
    let model = OpenAiModel::new("secret-key")
        .with_base_url(&base)
        .with_auth_style(AuthStyle::Bearer);

    let response = model.invoke(&(), request("gpt-4o", 0.7)).await.unwrap();
    assert_eq!(response.text(), "Hello!");

    let raw = server.join().unwrap();
    assert!(raw.starts_with("POST /v1/chat/completions "), "{raw}");
    assert_eq!(
        header_value(&raw, "authorization").as_deref(),
        Some("Bearer secret-key")
    );
    let body: Value = serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap()).unwrap();
    assert_eq!(body["model"], json!("gpt-4o"));
}

#[tokio::test]
async fn anthropic_auth_sends_key_and_version_but_no_authorization_header() {
    let (base, server) = serve_once();
    let model = OpenAiModel::new("sk-ant-test")
        .with_base_url(&base)
        .with_auth_style(AuthStyle::Anthropic);

    model
        .invoke(&(), request("claude-3-haiku", 0.5))
        .await
        .unwrap();

    let raw = server.join().unwrap();
    assert_eq!(
        header_value(&raw, "x-api-key").as_deref(),
        Some("sk-ant-test")
    );
    assert_eq!(
        header_value(&raw, "anthropic-version").as_deref(),
        Some("2023-06-01")
    );
    assert_eq!(header_value(&raw, "authorization"), None, "{raw}");
}

#[tokio::test]
async fn no_auth_style_sends_no_credentials() {
    let (base, server) = serve_once();
    let model = OpenAiModel::new("")
        .with_provider("ollama")
        .with_base_url(&base)
        .with_auth_style(AuthStyle::None);

    let response = model.invoke(&(), request("llama3", 0.7)).await.unwrap();
    assert_eq!(response.text(), "Hello!");

    let raw = server.join().unwrap();
    assert_eq!(header_value(&raw, "authorization"), None, "{raw}");
    assert_eq!(header_value(&raw, "x-api-key"), None, "{raw}");
}

fn transcript_with_record_only_patch() -> ModelRequest {
    let patch = crate::message::SystemMessage {
        content: Vec::new(),
        sections: [(
            "tool_changes".to_string(),
            Some("Tools now available: x.".to_string()),
        )]
        .into_iter()
        .collect(),
        tools_added: Vec::new(),
        tools_removed: Vec::new(),
    };
    ModelRequest::new(vec![
        Message::system("You are helpful."),
        Message::user("hi"),
        Message::System(patch),
        Message::system("a real mid-turn instruction"),
    ])
}

fn wire_roles(body: &Value) -> Vec<String> {
    body["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|message| message["role"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn hoisting_model_drops_system_messages_that_render_no_text() {
    // A record-only patch (sections / tool deltas, no content) renders as an
    // empty `role: system` message. A route that hoists system content to the
    // front of the prompt would still rewrite its cached prefix for it (#6962).
    let model = OpenAiModel::new("k").with_model("deepseek/deepseek-v4.1-flash");
    let body = body_for(&model, &transcript_with_record_only_patch());
    assert_eq!(
        wire_roles(&body),
        ["system", "user", "system"],
        "body={body}"
    );
    assert_eq!(
        body["messages"][2]["content"],
        json!("a real mid-turn instruction")
    );
}

#[test]
fn non_hoisting_model_keeps_every_system_message_in_place() {
    let model = OpenAiModel::new("k").with_model("gpt-4.1-mini");
    let body = body_for(&model, &transcript_with_record_only_patch());
    assert_eq!(
        wire_roles(&body),
        ["system", "user", "system", "system"],
        "body={body}"
    );
}

#[test]
fn request_model_override_controls_record_only_system_filtering() {
    let model = OpenAiModel::new("k").with_model("gpt-4.1-mini");
    let request = transcript_with_record_only_patch().with_model("deepseek-chat");
    assert_eq!(
        wire_roles(&body_for(&model, &request)),
        ["system", "user", "system"]
    );

    let model = OpenAiModel::new("k").with_model("deepseek-chat");
    let request = transcript_with_record_only_patch().with_model("gpt-4.1-mini");
    assert_eq!(
        wire_roles(&body_for(&model, &request)),
        ["system", "user", "system", "system"]
    );
}

/// Serves one canned error response and returns the base URL (unique port, so
/// the process-wide limits cache key never collides across tests).
fn serve_error_once(status: u16, body: &'static str) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        read_request(&mut sock);
        let response = format!(
            "HTTP/1.1 {status} Error\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = sock.write_all(response.as_bytes());
    });
    base
}

const OVERFLOW_BODY: &str = r#"{"error":{"message":"This model's maximum context length is 131072 tokens. However, you requested 200000 tokens.","type":"invalid_request_error"}}"#;

#[tokio::test]
async fn chat_overflow_error_records_the_stated_window() {
    let base = serve_error_once(400, OVERFLOW_BODY);
    let model = OpenAiModel::new("k").with_base_url(&base);
    assert!(
        model
            .invoke(&(), request("overflow-chat-model", 0.5))
            .await
            .is_err()
    );
    let learned = crate::model::discover::cached_model_limits(&base, "overflow-chat-model")
        .expect("overflow window recorded");
    assert_eq!(learned.context_window, Some(131_072));
}

#[tokio::test]
async fn responses_overflow_error_records_the_stated_window() {
    let base = serve_error_once(400, OVERFLOW_BODY);
    let model = OpenAiModel::new("k")
        .with_base_url(&base)
        .with_responses_api_primary();
    assert!(
        model
            .invoke(&(), request("overflow-responses-model", 0.5))
            .await
            .is_err()
    );
    let learned = crate::model::discover::cached_model_limits(&base, "overflow-responses-model")
        .expect("overflow window recorded on the Responses path");
    assert_eq!(learned.context_window, Some(131_072));
}

#[tokio::test]
async fn non_overflow_error_records_nothing() {
    let base = serve_error_once(400, r#"{"error":{"message":"bad request"}}"#);
    let model = OpenAiModel::new("k").with_base_url(&base);
    assert!(
        model
            .invoke(&(), request("no-overflow-model", 0.5))
            .await
            .is_err()
    );
    assert_eq!(
        crate::model::discover::cached_model_limits(&base, "no-overflow-model"),
        None
    );
}

#[tokio::test]
async fn overflow_without_stamped_code_still_records_the_window() {
    // vLLM phrasing that `is_context_overflow` does not stamp with the code.
    let base = serve_error_once(
        400,
        r#"{"error":{"message":"input is longer than the maximum model length of 32768"}}"#,
    );
    let model = OpenAiModel::new("k").with_base_url(&base);
    assert!(
        model
            .invoke(&(), request("vllm-unstamped-model", 0.5))
            .await
            .is_err()
    );
    let learned = crate::model::discover::cached_model_limits(&base, "vllm-unstamped-model")
        .expect("window learned from the message itself");
    assert_eq!(learned.context_window, Some(32_768));
}

// ---------------------------------------------------------------------------
// Adaptive parameter omission (a 400 that names a parameter we sent)
// ---------------------------------------------------------------------------

/// A canned chat completion body, shared by the multi-response server tests.
fn completion_body() -> String {
    json!({
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "m",
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": "Hello!" },
            "finish_reason": "stop"
        }]
    })
    .to_string()
}

/// Serves `responses` in order, one connection each, and returns the raw
/// requests received. The listener closes after the last one, so an extra
/// attempt surfaces as a transport error rather than a canned answer.
fn serve_sequence(responses: Vec<(u16, String)>) -> (String, std::thread::JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}/v1", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let mut received = Vec::new();
        for (status, body) in responses {
            let (mut sock, _) = listener.accept().unwrap();
            received.push(String::from_utf8_lossy(&read_request(&mut sock)).into_owned());
            let response = format!(
                "HTTP/1.1 {status} Canned\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(response.as_bytes());
        }
        received
    });
    (base, handle)
}

fn request_body(raw: &str) -> Value {
    serde_json::from_str(raw.split("\r\n\r\n").nth(1).unwrap()).unwrap()
}

fn unsupported(parameter: &str) -> String {
    json!({
        "error": {
            "message": format!("Unsupported parameter: '{parameter}' is not supported with this model."),
            "type": "invalid_request_error",
            "param": parameter,
            "code": "unsupported_parameter"
        }
    })
    .to_string()
}

#[tokio::test]
async fn a_rejected_parameter_is_dropped_and_the_request_retried_once() {
    let (base, server) = serve_sequence(vec![
        (400, unsupported("temperature")),
        (200, completion_body()),
    ]);
    let model = OpenAiModel::new("k").with_base_url(&base);

    let response = model
        .invoke(&(), request("omission-retry-model", 0.7))
        .await
        .unwrap();
    assert_eq!(response.text(), "Hello!");

    let raw = server.join().unwrap();
    assert_eq!(raw.len(), 2, "one rejection, one retry");
    let first = request_body(&raw[0]);
    let retry = request_body(&raw[1]);
    assert_eq!(first["temperature"], json!(0.7));
    assert!(retry.get("temperature").is_none(), "retry={retry}");
    // Only the blamed parameter goes: the question asked is unchanged.
    assert_eq!(retry["model"], first["model"]);
    assert_eq!(retry["messages"], first["messages"]);
    assert!(super::super::omission::is_omitted(
        &base,
        "omission-retry-model",
        "temperature"
    ));
}

#[tokio::test]
async fn a_learned_omission_is_applied_up_front_on_the_next_request() {
    let (base, server) = serve_sequence(vec![
        (400, unsupported("max_tokens")),
        (200, completion_body()),
        (200, completion_body()),
    ]);
    let request = || request("omission-learned-model", 0.5).with_max_tokens(64);

    let first = OpenAiModel::new("k").with_base_url(&base);
    first.invoke(&(), request()).await.unwrap();
    // A separate instance: what was learnt belongs to the endpoint and model,
    // not to one adapter value.
    let second = OpenAiModel::new("k").with_base_url(&base);
    second.invoke(&(), request()).await.unwrap();

    let raw = server.join().unwrap();
    assert_eq!(raw.len(), 3, "no round-trip is spent re-learning");
    let later = request_body(&raw[2]);
    assert!(later.get("max_tokens").is_none(), "later={later}");
    assert_eq!(later["temperature"], json!(0.5), "nothing else is dropped");
}

#[tokio::test]
async fn only_one_omission_retry_is_spent() {
    let (base, server) = serve_sequence(vec![
        (400, unsupported("temperature")),
        (400, unsupported("seed")),
    ]);
    let model = OpenAiModel::new("k").with_base_url(&base);
    let error = model
        .invoke(
            &(),
            request("omission-single-retry-model", 0.3).with_seed(7),
        )
        .await
        .unwrap_err();

    // The second rejection is what surfaces: had a third attempt been made it
    // would have failed on the closed listener instead.
    match error {
        Error::Provider(error) => {
            assert_eq!(error.status, Some(400));
            assert!(error.message.contains("'seed'"), "{}", error.message);
        }
        other => panic!("expected a provider error, got {other:?}"),
    }
    assert_eq!(server.join().unwrap().len(), 2);
}

#[tokio::test]
async fn a_rejection_naming_nothing_we_sent_is_not_retried() {
    let (base, server) = serve_sequence(vec![(400, unsupported("logit_bias"))]);
    let model = OpenAiModel::new("k").with_base_url(&base);
    assert!(
        model
            .invoke(&(), request("omission-unrelated-model", 0.3))
            .await
            .is_err()
    );
    assert_eq!(server.join().unwrap().len(), 1);
    assert!(!super::super::omission::is_omitted(
        &base,
        "omission-unrelated-model",
        "temperature"
    ));
}

#[tokio::test]
async fn a_context_overflow_never_drops_the_output_cap() {
    // Overflow messages routinely name `max_tokens`; dropping (and
    // remembering) the cap over one would uncap every later request.
    let overflow = json!({
        "error": {
            "message": "This model's maximum context length is 4096 tokens. 'max_tokens' is unsupported at 9000.",
            "type": "invalid_request_error"
        }
    })
    .to_string();
    let (base, server) = serve_sequence(vec![(400, overflow)]);
    let model = OpenAiModel::new("k").with_base_url(&base);
    assert!(
        model
            .invoke(
                &(),
                request("omission-overflow-model", 0.3).with_max_tokens(9000)
            )
            .await
            .is_err()
    );
    assert_eq!(server.join().unwrap().len(), 1);
    assert!(!super::super::omission::is_omitted(
        &base,
        "omission-overflow-model",
        "max_tokens"
    ));
}

#[tokio::test]
async fn the_streaming_path_retries_a_rejected_parameter_too() {
    let (base, server) =
        serve_sequence(vec![(400, unsupported("top_p")), (200, completion_body())]);
    let model = OpenAiModel::new("k").with_base_url(&base);
    let mut stream = model
        .stream(&(), request("omission-stream-model", 0.3).with_top_p(0.9))
        .await
        .unwrap();
    let mut completed = None;
    while let Some(item) = stream.next().await {
        if let crate::model::ModelStreamItem::Completed(response) = item {
            completed = Some(response);
        }
    }
    assert_eq!(completed.unwrap().text(), "Hello!");

    let raw = server.join().unwrap();
    assert_eq!(raw.len(), 2);
    assert!(request_body(&raw[1]).get("top_p").is_none());
}

// ---------------------------------------------------------------------------
// Per-request credentials (BearerSource)
// ---------------------------------------------------------------------------

/// A rotating credential: yields `token-1`, `token-2`, ... and counts how
/// often it was told its value was rejected.
#[derive(Default)]
struct RotatingToken {
    reads: std::sync::atomic::AtomicUsize,
    invalidations: std::sync::atomic::AtomicUsize,
}

#[async_trait::async_trait]
impl crate::providers::BearerSource for RotatingToken {
    async fn current(&self) -> crate::Result<Option<String>> {
        let n = self.reads.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
        Ok(Some(format!("token-{n}")))
    }

    fn invalidate(&self) {
        self.invalidations
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// A source that has nothing to present (a keyless endpoint).
struct NoToken;

#[async_trait::async_trait]
impl crate::providers::BearerSource for NoToken {
    async fn current(&self) -> crate::Result<Option<String>> {
        Ok(None)
    }
}

/// A source that cannot produce a credential at all.
struct BrokenToken;

#[async_trait::async_trait]
impl crate::providers::BearerSource for BrokenToken {
    async fn current(&self) -> crate::Result<Option<String>> {
        Err(Error::Model("token file unreadable".to_string()))
    }
}

#[tokio::test]
async fn a_bearer_source_is_read_on_every_request() {
    let (base, server) = serve_sequence(vec![(200, completion_body()), (200, completion_body())]);
    let model = OpenAiModel::new("static-key")
        .with_base_url(&base)
        .with_bearer_source(std::sync::Arc::new(RotatingToken::default()));

    model.invoke(&(), request("gpt-4o", 0.2)).await.unwrap();
    model.invoke(&(), request("gpt-4o", 0.2)).await.unwrap();

    let raw = server.join().unwrap();
    assert_eq!(
        header_value(&raw[0], "authorization").as_deref(),
        Some("Bearer token-1")
    );
    assert_eq!(
        header_value(&raw[1], "authorization").as_deref(),
        Some("Bearer token-2"),
        "a rotated credential is picked up without rebuilding the model"
    );
}

#[tokio::test]
async fn a_bearer_source_is_sent_in_the_configured_auth_style() {
    let (base, server) = serve_sequence(vec![(200, completion_body())]);
    let model = OpenAiModel::new("static-key")
        .with_base_url(&base)
        .with_auth_style(AuthStyle::XApiKey)
        .with_header("x-extra", "kept")
        .with_bearer_source(std::sync::Arc::new(RotatingToken::default()));

    model.invoke(&(), request("gpt-4o", 0.2)).await.unwrap();

    let raw = server.join().unwrap();
    assert_eq!(
        header_value(&raw[0], "x-api-key").as_deref(),
        Some("token-1")
    );
    assert_eq!(header_value(&raw[0], "authorization"), None);
    assert_eq!(header_value(&raw[0], "x-extra").as_deref(), Some("kept"));
}

#[tokio::test]
async fn a_bearer_source_with_nothing_to_present_sends_no_credentials() {
    let (base, server) = serve_sequence(vec![(200, completion_body())]);
    let model = OpenAiModel::new("static-key")
        .with_base_url(&base)
        .with_bearer_source(std::sync::Arc::new(NoToken));

    model.invoke(&(), request("llama3", 0.2)).await.unwrap();

    let raw = server.join().unwrap();
    assert_eq!(header_value(&raw[0], "authorization"), None, "{}", raw[0]);
}

#[tokio::test]
async fn a_rejected_bearer_is_invalidated_at_its_source() {
    let (base, server) = serve_sequence(vec![(
        401,
        r#"{"error":{"message":"invalid token"}}"#.to_string(),
    )]);
    let source = std::sync::Arc::new(RotatingToken::default());
    let model = OpenAiModel::new("")
        .with_base_url(&base)
        .with_bearer_source(source.clone());

    assert!(model.invoke(&(), request("gpt-4o", 0.2)).await.is_err());
    server.join().unwrap();
    assert_eq!(
        source
            .invalidations
            .load(std::sync::atomic::Ordering::SeqCst),
        1,
        "the next request must re-read rather than re-present a refused token"
    );
}

#[tokio::test]
async fn a_failing_bearer_source_fails_the_call() {
    let model = OpenAiModel::new("static-key")
        .with_base_url("http://127.0.0.1:9/v1")
        .with_bearer_source(std::sync::Arc::new(BrokenToken));
    let error = model.invoke(&(), request("gpt-4o", 0.2)).await.unwrap_err();
    assert!(
        error.to_string().contains("token file unreadable"),
        "{error}"
    );
}

#[test]
fn debug_output_reports_a_bearer_source_without_reading_it() {
    let model = OpenAiModel::new("static-key")
        .with_bearer_source(std::sync::Arc::new(RotatingToken::default()));
    let debug = format!("{model:?}");
    assert!(debug.contains("bearer_source: true"), "{debug}");
    assert!(!debug.contains("static-key"), "{debug}");
}
