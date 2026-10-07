use super::*;
use crate::{
    message::Message,
    model::{ChatModel, ModelRequest},
};
use async_trait::async_trait;
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
};

pub(super) struct Scripted {
    pub requests: Mutex<Vec<reqwest::Request>>,
    replies: Mutex<VecDeque<crate::Result<reqwest::Response>>>,
}

#[async_trait]
impl transport::HttpTransport for Scripted {
    async fn send(&self, request: reqwest::Request) -> crate::Result<reqwest::Response> {
        self.requests.lock().unwrap().push(request);
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected network operation")
    }
}

pub(super) fn fixture(
    replies: Vec<crate::Result<reqwest::Response>>,
) -> (PerplexityModel, Arc<Scripted>) {
    let scripted = Arc::new(Scripted {
        requests: Mutex::new(vec![]),
        replies: Mutex::new(replies.into()),
    });
    let mut model = PerplexityModel::new("test-key", PerplexitySelection::preset("low")).unwrap();
    Arc::get_mut(&mut model.inner).unwrap().sender = scripted.clone();
    (model, scripted)
}

pub(super) fn json_reply(status: u16, body: serde_json::Value) -> crate::Result<reqwest::Response> {
    Ok(http::Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(body.to_string())
        .unwrap()
        .into())
}

pub(super) fn complete() -> serde_json::Value {
    serde_json::json!({"id":"resp_1","model":"openai/test","status":"completed",
        "output":[{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"output_text","text":"answer"}]}]})
}

#[tokio::test]
async fn public_invoke_uses_native_route_and_preserves_response_identity() {
    let (model, scripted) = fixture(vec![json_reply(200, complete())]);
    let result = model
        .invoke(&(), ModelRequest::new(vec![Message::user("question")]))
        .await
        .unwrap();
    assert_eq!(result.text(), "answer");
    assert_eq!(result.execution.unwrap().id, "resp_1");
    let requests = scripted.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(
        requests[0].url().as_str(),
        "https://api.perplexity.ai/v1/agent"
    );
    assert_eq!(requests[0].headers()["authorization"], "Bearer test-key");
}

#[tokio::test(start_paused = true)]
async fn creation_retries_only_explicit_rate_limits_and_bounds_the_attempts() {
    let throttled = http::Response::builder()
        .status(429)
        .header("retry-after", "2")
        .body("")
        .unwrap()
        .into();
    let (model, scripted) = fixture(vec![Ok(throttled), json_reply(200, complete())]);
    let start = tokio::time::Instant::now();
    assert_eq!(
        model
            .invoke(&(), ModelRequest::new(vec![Message::user("q")]))
            .await
            .unwrap()
            .text(),
        "answer"
    );
    assert_eq!(start.elapsed(), std::time::Duration::from_secs(2));
    assert_eq!(scripted.requests.lock().unwrap().len(), 2);
    let (model, scripted) = fixture(
        (0..4)
            .map(|_| json_reply(429, serde_json::json!({})))
            .collect(),
    );
    assert!(
        matches!(model.invoke(&(),ModelRequest::new(vec![Message::user("q")])).await,Err(Error::Provider(error)) if error.status==Some(429))
    );
    assert_eq!(scripted.requests.lock().unwrap().len(), 4);
    for status in [400, 401, 403, 500, 502, 503, 504] {
        let (model, scripted) = fixture(vec![json_reply(
            status,
            serde_json::json!({"error":{"message":"failed test-key"}}),
        )]);
        let error = model
            .invoke(&(), ModelRequest::new(vec![Message::user("q")]))
            .await
            .unwrap_err();
        assert!(
            matches!(&error,Error::Provider(error) if error.status==Some(status)&&!error.retryable)
        );
        assert!(!format!("{error:?} {error}").contains("test-key"));
        assert_eq!(scripted.requests.lock().unwrap().len(), 1);
    }
}

#[tokio::test(start_paused = true)]
async fn total_deadline_covers_header_and_body_waits() {
    let incoming = http::Response::new(reqwest::Body::wrap_stream(futures::stream::pending::<
        std::result::Result<String, std::io::Error>,
    >()))
    .into();
    let (model, scripted) = fixture(vec![Ok(incoming)]);
    let mut request = ModelRequest::new(vec![Message::user("q")]);
    request.timeout_ms = Some(10);
    assert!(
        matches!(model.invoke(&(),request).await,Err(Error::Provider(error)) if error.code.as_deref()==Some("timeout"))
    );
    assert_eq!(scripted.requests.lock().unwrap().len(), 1);
}

#[tokio::test(start_paused = true)]
async fn total_deadline_cancels_a_pending_transport_send() {
    struct DelayedTransport;
    #[async_trait]
    impl transport::HttpTransport for DelayedTransport {
        async fn send(&self, _request: reqwest::Request) -> crate::Result<reqwest::Response> {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            json_reply(200, complete())
        }
    }
    let (mut model, _) = fixture(vec![]);
    Arc::get_mut(&mut model.inner).unwrap().sender = Arc::new(DelayedTransport);
    let mut request = ModelRequest::new(vec![Message::user("question")]);
    request.timeout_ms = Some(10);
    let start = tokio::time::Instant::now();
    assert!(matches!(model.invoke(&(), request).await,
        Err(Error::Provider(error)) if error.code.as_deref() == Some("timeout")));
    assert_eq!(start.elapsed(), std::time::Duration::from_millis(10));
}

#[tokio::test]
async fn aliases_and_payload_hooks_are_validated_before_sending() {
    let (mut model, scripted) = fixture(vec![json_reply(200, complete())]);
    let config = &mut Arc::get_mut(&mut model.inner).unwrap().config;
    config.responses_alias = true;
    config.on_payload = Some(Arc::new(|body| body["max_steps"] = serde_json::json!(3)));
    model
        .invoke(&(), ModelRequest::new(vec![Message::user("q")]))
        .await
        .unwrap();
    assert_eq!(
        scripted.requests.lock().unwrap()[0].url().path(),
        "/v1/responses"
    );
    let (mut model, scripted) = fixture(vec![]);
    Arc::get_mut(&mut model.inner).unwrap().config.on_payload = Some(Arc::new(|body| {
        body["tools"] = serde_json::json!([{"type":"sandbox"}])
    }));
    assert!(matches!(
        model
            .invoke(&(), ModelRequest::new(vec![Message::user("q")]))
            .await,
        Err(Error::Unsupported(_))
    ));
    assert!(scripted.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn invoke_preserves_unknown_fields_but_redacts_client_credentials() {
    let mut value = complete();
    value["future_metadata"] = serde_json::json!({"credential":"test-key","ordinary":"retained"});
    let (model, _) = fixture(vec![json_reply(200, value)]);
    let response = model
        .invoke(&(), ModelRequest::new(vec![Message::user("q")]))
        .await
        .unwrap();
    assert_eq!(
        response.raw.as_ref().unwrap()["future_metadata"]["ordinary"],
        "retained"
    );
    assert_eq!(
        response.raw.as_ref().unwrap()["future_metadata"]["credential"],
        "[REDACTED]"
    );
    assert!(!format!("{model:?}").contains("test-key"));
}

#[tokio::test]
async fn public_function_replay_keeps_history_arguments_signature_and_call_id() {
    use crate::{
        ToolSchema,
        message::{ContentBlock, ToolMessage},
    };
    let first = serde_json::json!({"id":"resp_1","model":"google/test","status":"completed","output":[
        {"type":"function_call","id":"fc_1","call_id":"call_1","name":"lookup","arguments":"{ \"id\": 7 }","thought_signature":"opaque-signature"}
    ]});
    let (model, scripted) = fixture(vec![json_reply(200, first), json_reply(200, complete())]);
    let tool = ToolSchema::new(
        "lookup",
        "look up a record",
        serde_json::json!({"type":"object"}),
    );
    let mut request = ModelRequest::new(vec![
        Message::system("Use the supplied record."),
        Message::user("lookup"),
    ])
    .with_tools(vec![tool]);
    let first = model.invoke(&(), request.clone()).await.unwrap();
    let result = ToolMessage::for_call(
        &first.message.tool_calls[0],
        vec![ContentBlock::Text("found".into())],
    );
    request.messages.push(Message::Assistant(first.message));
    request.messages.push(Message::Tool(result));
    request.model = Some(first.execution.unwrap().model);
    request.continuation_id = None;
    assert_eq!(model.invoke(&(), request).await.unwrap().text(), "answer");
    let requests = scripted.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    let second: serde_json::Value =
        serde_json::from_slice(requests[1].body().unwrap().as_bytes().unwrap()).unwrap();
    assert!(second.get("previous_response_id").is_none());
    assert_eq!(second["model"], "google/test");
    assert_eq!(second["instructions"], "Use the supplied record.");
    assert_eq!(
        second["input"],
        serde_json::json!([
            {"type":"message","role":"user","content":[{"type":"input_text","text":"lookup"}]},
            {"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{ \"id\": 7 }","thought_signature":"opaque-signature"},
            {"type":"function_call_output","call_id":"call_1","name":"lookup","output":[{"type":"input_text","text":"found"}],"thought_signature":"opaque-signature"}
        ])
    );
    assert_eq!(second["tools"][0]["name"], "lookup");
}
