use super::*;

use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::Response;
use axum::routing::{get, post};
use axum::{Json, Router};
use futures::StreamExt;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Clone, Default)]
struct Fixture {
    requests: Arc<std::sync::Mutex<Vec<(HeaderMap, Value)>>>,
    response: Arc<std::sync::Mutex<FixtureResponse>>,
    rotate_token_before_response: Arc<std::sync::Mutex<Option<Arc<std::sync::Mutex<String>>>>>,
}

#[derive(Clone)]
struct FixtureResponse {
    status: StatusCode,
    content_type: &'static str,
    body: String,
}

impl Default for FixtureResponse {
    fn default() -> Self {
        Self {
            status: StatusCode::OK,
            content_type: "application/json",
            body: completion_body(),
        }
    }
}

fn completion_body() -> String {
    json!({
        "id": "chatcmpl-test",
        "choices": [{ "index": 0, "message": { "role": "assistant", "content": "ok" }, "finish_reason": "stop" }],
        "usage": { "prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22 },
        "openhuman": {
            "usage": { "cached_input_tokens": 8, "context_window": 128000 },
            "billing": { "charged_amount_usd": 0.0042 }
        }
    })
    .to_string()
}

async fn chat(State(fixture): State<Fixture>, headers: HeaderMap, body: String) -> Response {
    let body: Value = serde_json::from_str(&body).expect("request JSON");
    fixture.requests.lock().unwrap().push((headers, body));
    if let Some(token) = fixture.rotate_token_before_response.lock().unwrap().take() {
        *token.lock().unwrap() = "new-session-token".to_owned();
    }
    let response = fixture.response.lock().unwrap().clone();
    Response::builder()
        .status(response.status)
        .header("content-type", response.content_type)
        .body(Body::from(response.body))
        .unwrap()
}

async fn listing(State(fixture): State<Fixture>, headers: HeaderMap) -> Json<Value> {
    fixture
        .requests
        .lock()
        .unwrap()
        .push((headers, json!({ "listing": true })));
    Json(json!({ "data": [{ "id": "model-a" }] }))
}

async fn never_respond(
    State(entered): State<Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>>,
) -> Response {
    if let Some(sender) = entered.lock().unwrap().take() {
        let _ = sender.send(());
    }
    std::future::pending().await
}

async fn serve(fixture: Fixture) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let app = Router::new()
        .route("/openai/v1/chat/completions", post(chat))
        .route("/openai/v1/models", get(listing))
        .with_state(fixture);
    tokio::spawn(async move { axum::serve(listener, app).await.expect("serve fixture") });
    format!("http://{address}")
}

fn fixture_model(base: &str) -> TinyHumansModel {
    TinyHumansModel::new(base, "", "default-model", || {
        Box::pin(async { Ok("test-token".into()) })
    })
    .expect("model construction")
}

#[tokio::test]
async fn dynamic_bearer_is_resolved_again_for_invoke_and_stream() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    let reads = Arc::new(AtomicUsize::new(0));
    let source = reads.clone();
    let model = TinyHumansModel::new(&base, "", "default-model", move || {
        let source = Arc::clone(&source);
        Box::pin(async move {
            Ok(format!(
                "token-{}",
                source.fetch_add(1, Ordering::SeqCst) + 1
            ))
        })
    })
    .unwrap();
    model
        .invoke(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("invoke")]),
        )
        .await
        .unwrap();
    model
        .stream(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("stream")]),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].0["authorization"], "Bearer token-1");
    assert_eq!(requests[1].0["authorization"], "Bearer token-2");
}

#[tokio::test]
async fn constructor_model_is_default_and_nonblank_request_model_overrides_it() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    let model = TinyHumansModel::new(&base, " constructor-model ", "fallback-model", || {
        Box::pin(async { Ok("token".into()) })
    })
    .unwrap();
    for requested_model in [None, Some("request-model"), Some("  ")] {
        let mut request = ModelRequest::new(vec![crate::message::Message::user("hi")]);
        request.model = requested_model.map(str::to_owned);
        model.invoke(&(), request).await.unwrap();
    }
    {
        let requests = fixture.requests.lock().unwrap();
        assert_eq!(requests[0].1["model"], "constructor-model");
        assert_eq!(requests[1].1["model"], "request-model");
        assert_eq!(requests[2].1["model"], "constructor-model");
    }

    let fallback_model = TinyHumansModel::new(&base, "  ", "fallback-model", || {
        Box::pin(async { Ok("token".into()) })
    })
    .unwrap();
    fallback_model
        .invoke(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("hi")]),
        )
        .await
        .unwrap();
    assert_eq!(
        fixture.requests.lock().unwrap()[3].1["model"],
        "fallback-model"
    );
}

#[tokio::test]
async fn resolver_failure_stops_before_transport() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    let model = TinyHumansModel::new(&base, "m", "default-model", || {
        Box::pin(async { Err(Error::Model("credential source failed".into())) })
    })
    .unwrap();
    let error = model
        .invoke(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("hi")]),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("credential source failed"));
    assert!(fixture.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn auth_failure_callback_receives_dispatch_operation() {
    let fixture = Fixture::default();
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::UNAUTHORIZED,
        content_type: "application/json",
        body: json!({ "error": { "message": "Unauthorized" } }).to_string(),
    };
    let base = serve(fixture).await;
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let callback_seen = seen.clone();
    let model = fixture_model(&base).with_auth_failure_handler(move |error, operation| {
        callback_seen
            .lock()
            .unwrap()
            .push((error.status, operation));
    });
    let _ = model
        .invoke(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("hi")]),
        )
        .await;
    assert_eq!(
        *seen.lock().unwrap(),
        vec![(Some(401), TinyHumansOperation::Invoke)]
    );
}

#[tokio::test]
async fn auth_failure_callback_receives_in_band_stream_failure() {
    let fixture = Fixture::default();
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::OK,
        content_type: "text/event-stream",
        body: "data: {\"error\":{\"message\":\"Unauthorized\",\"code\":\"unauthorized\"}}\n\n"
            .into(),
    };
    let base = serve(fixture).await;
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let callback_seen = seen.clone();
    let model = fixture_model(&base).with_auth_failure_handler(move |error, operation| {
        callback_seen
            .lock()
            .unwrap()
            .push((error.status, operation));
    });
    let items = model
        .stream(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("hi")]),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(_))
    ));
    assert_eq!(
        *seen.lock().unwrap(),
        vec![(None, TinyHumansOperation::Stream)]
    );
}

#[tokio::test]
async fn managed_auth_callback_reports_typed_provenance_for_each_operation() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    let typed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let typed_seen = typed.clone();
    let legacy = Arc::new(std::sync::Mutex::new(Vec::new()));
    let legacy_seen = legacy.clone();
    let model = fixture_model(&base)
        .with_managed_auth_failure_handler(move |error, failure| {
            typed_seen.lock().unwrap().push((
                error.provider.clone(),
                error.status,
                failure.operation(),
                failure.kind(),
            ));
        })
        .with_auth_failure_handler(move |error, operation| {
            legacy_seen.lock().unwrap().push((error.status, operation));
        });

    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::UNAUTHORIZED,
        content_type: "application/json",
        body: json!({ "error": { "message": "Unauthorized" } }).to_string(),
    };
    assert!(
        model
            .invoke(
                &(),
                ModelRequest::new(vec![crate::message::Message::user("invoke")]),
            )
            .await
            .is_err()
    );

    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::FORBIDDEN,
        content_type: "application/json",
        body: json!({ "error": { "message": "Forbidden" } }).to_string(),
    };
    assert!(
        model
            .stream(
                &(),
                ModelRequest::new(vec![crate::message::Message::user("stream open")]),
            )
            .await
            .is_err()
    );

    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::OK,
        content_type: "text/event-stream",
        body: "data: {\"error\":{\"message\":\"Unauthorized: session expired\",\"code\":\"unauthorized\"}}\n\n"
            .into(),
    };
    let items = model
        .stream(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("stream")]),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(_))
    ));

    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::OK,
        content_type: "text/event-stream",
        body: "data: {\"error\":{\"message\":\"Unauthorized\",\"code\":401}}\n\n".into(),
    };
    let items = model
        .stream(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("stream status")]),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(_))
    ));

    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::FORBIDDEN,
        content_type: "application/json",
        body: json!({ "error": { "message": "Forbidden" } }).to_string(),
    };
    model.probe_readiness().await.unwrap();

    assert_eq!(
        *typed.lock().unwrap(),
        vec![
            (
                "tinyhumans".to_owned(),
                Some(401),
                TinyHumansOperation::Invoke,
                TinyHumansAuthFailureKind::Unauthorized,
            ),
            (
                "tinyhumans".to_owned(),
                Some(403),
                TinyHumansOperation::Stream,
                TinyHumansAuthFailureKind::Forbidden,
            ),
            (
                "tinyhumans".to_owned(),
                None,
                TinyHumansOperation::Stream,
                TinyHumansAuthFailureKind::Other,
            ),
            (
                "tinyhumans".to_owned(),
                Some(401),
                TinyHumansOperation::Stream,
                TinyHumansAuthFailureKind::Unauthorized,
            ),
            (
                "tinyhumans".to_owned(),
                Some(403),
                TinyHumansOperation::Readiness,
                TinyHumansAuthFailureKind::Forbidden,
            ),
        ]
    );
    assert_eq!(
        *legacy.lock().unwrap(),
        vec![
            (Some(401), TinyHumansOperation::Invoke),
            (Some(403), TinyHumansOperation::Stream),
            (None, TinyHumansOperation::Stream),
            (Some(401), TinyHumansOperation::Stream),
            (Some(403), TinyHumansOperation::Readiness),
        ]
    );

    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::BAD_REQUEST,
        content_type: "application/json",
        body: json!({ "error": { "message": "Bad request" } }).to_string(),
    };
    assert!(
        model
            .invoke(
                &(),
                ModelRequest::new(vec![crate::message::Message::user("not auth")]),
            )
            .await
            .is_err()
    );
    assert_eq!(typed.lock().unwrap().len(), 5);
    assert_eq!(legacy.lock().unwrap().len(), 5);
}

#[tokio::test]
async fn stream_completed_projects_managed_usage() {
    let fixture = Fixture::default();
    let base = serve(fixture).await;
    let mut items = fixture_model(&base)
        .stream(
            &(),
            ModelRequest::new(vec![crate::message::Message::user("hi")]),
        )
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let Some(ModelStreamItem::Completed(response)) = items.pop() else {
        panic!("stream should end in a completed response: {items:?}");
    };
    assert_eq!(response.usage.unwrap().cache_read_tokens, 8);
    assert_eq!(
        response.raw.unwrap()["openhuman_usage_meta"]["charged_amount_usd"],
        0.0042
    );
}

#[tokio::test]
async fn readiness_blocks_only_confirmed_missing_provider_configuration() {
    let missing = Fixture::default();
    *missing.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::BAD_REQUEST,
        content_type: "application/json",
        body: json!({ "error": "API key not configured for provider", "errorCode": "BAD_REQUEST" })
            .to_string(),
    };
    let error = fixture_model(&serve(missing).await)
        .probe_readiness()
        .await
        .expect_err("definitive configuration error");
    assert!(error.contains("API key not configured for provider"));

    let transient = Fixture::default();
    *transient.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::SERVICE_UNAVAILABLE,
        content_type: "application/json",
        body: json!({ "error": "temporary outage" }).to_string(),
    };
    fixture_model(&serve(transient).await)
        .probe_readiness()
        .await
        .expect("transient errors fail open");

    let unrelated = Fixture::default();
    *unrelated.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::BAD_REQUEST,
        content_type: "application/json",
        body: json!({ "error": "invalid request body" }).to_string(),
    };
    fixture_model(&serve(unrelated).await)
        .probe_readiness()
        .await
        .expect("unrelated 4xx errors fail open");
}

#[tokio::test]
async fn readiness_fails_open_when_bearer_resolution_fails() {
    let model = TinyHumansModel::new("http://127.0.0.1:1", "m", "default-model", || {
        Box::pin(async { Err(Error::Model("credential lookup unavailable".into())) })
    })
    .unwrap();
    model
        .probe_readiness()
        .await
        .expect("credential lookup errors are not provider configuration errors");
}

#[tokio::test]
async fn readiness_fails_open_on_transport_failure() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind unused port");
    let address = listener.local_addr().expect("unused address");
    drop(listener);
    fixture_model(&format!("http://{address}"))
        .probe_readiness()
        .await
        .expect("transport errors fail open");
}

#[tokio::test]
async fn readiness_fails_open_after_its_timeout() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind fixture");
    let address = listener.local_addr().expect("fixture address");
    let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
    let entered = Arc::new(std::sync::Mutex::new(Some(entered_tx)));
    let app = Router::new()
        .route("/openai/v1/chat/completions", post(never_respond))
        .with_state(entered);
    tokio::spawn(async move { axum::serve(listener, app).await.expect("serve fixture") });

    let model = fixture_model(&format!("http://{address}"));
    let probe = tokio::spawn(async move { model.probe_readiness().await });
    entered_rx.await.expect("probe request reached fixture");
    tokio::time::pause();
    tokio::time::advance(Duration::from_secs(5)).await;
    probe
        .await
        .expect("probe task")
        .expect("probe timeout fails open");
}

async fn spawn_counting_relay(upstream: String) -> (String, Arc<AtomicUsize>) {
    let accepted = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind relay");
    let address = listener.local_addr().expect("relay address");
    let count = accepted.clone();
    tokio::spawn(async move {
        while let Ok((mut inbound, _)) = listener.accept().await {
            count.fetch_add(1, Ordering::SeqCst);
            let upstream = upstream.clone();
            tokio::spawn(async move {
                if let Ok(mut outbound) = tokio::net::TcpStream::connect(&upstream).await {
                    let _ = tokio::io::copy_bidirectional(&mut inbound, &mut outbound).await;
                }
            });
        }
    });
    (format!("http://{address}"), accepted)
}

#[tokio::test]
async fn consecutive_managed_calls_reuse_one_connection() {
    let fixture = Fixture::default();
    let upstream = serve(fixture.clone()).await;
    let upstream = upstream
        .strip_prefix("http://")
        .expect("fixture uses http")
        .to_owned();
    let (relay, accepted) = spawn_counting_relay(upstream).await;
    let model = fixture_model(&relay);

    for _ in 0..3 {
        model
            .probe_readiness()
            .await
            .expect("managed readiness call");
    }

    assert_eq!(
        fixture.requests.lock().unwrap().len(),
        3,
        "all three probes reach the upstream fixture"
    );
    assert_eq!(
        accepted.load(Ordering::SeqCst),
        1,
        "three consecutive managed calls reuse one pooled connection"
    );
}

#[tokio::test]
async fn reasoning_thread_and_caller_headers_are_preserved() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    let headers = vec![("x-attribution".to_owned(), "host".to_owned())];
    let model = fixture_model(&base)
        .with_thread_id(Some(" thread-9 "))
        .with_extra_headers(&headers);
    let mut request = ModelRequest::new(vec![crate::message::Message::user("hi")]);
    request.metadata = json!({ REASONING_OFF_METADATA_KEY: true });
    request.reasoning = Some(crate::model::ReasoningConfig::default());
    request.provider_options = json!({ "caller_option": "preserved" });
    model.invoke(&(), request).await.unwrap();
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(requests[0].1["reasoning"], json!({ "enabled": false }));
    assert_eq!(requests[0].1["thread_id"], "thread-9");
    assert_eq!(requests[0].0["x-attribution"], "host");
    assert_eq!(requests[0].1["model"], "default-model");
    assert_eq!(requests[0].1["caller_option"], "preserved");
}

#[tokio::test]
async fn catalog_discovery_keeps_auth_and_host_headers() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    let request = discovery_request(
        &base,
        "model-a",
        Some("session-token"),
        &[("x-attribution".into(), "host".into())],
    );
    let models = list_models(&reqwest::Client::new(), request)
        .await
        .expect("catalog");
    assert_eq!(models[0].id, "model-a");
    let requests = fixture.requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].0["authorization"], "Bearer session-token");
    assert_eq!(requests[0].0["x-attribution"], "host");
}

#[test]
fn empty_model_uses_caller_supplied_default_and_endpoint_is_normalized() {
    assert_eq!(resolve_model("  ", "chosen-default"), "chosen-default");
    assert_eq!(resolve_model(" model-x ", "chosen-default"), "model-x");
    assert_eq!(
        discovery_request("https://example.test", "m", None, &[]).endpoint,
        "https://example.test/openai/v1"
    );
    assert_eq!(
        discovery_request("https://example.test/openai/v1/", "m", None, &[]).endpoint,
        "https://example.test/openai/v1"
    );
}

#[test]
fn explicit_reasoning_option_precedes_off_hint_and_neutral_config() {
    let mut request = ModelRequest::new(vec![crate::message::Message::user("hi")]);
    request.metadata = json!({ REASONING_OFF_METADATA_KEY: true });
    request.reasoning = Some(crate::model::ReasoningConfig {
        effort: Some(crate::model::ReasoningEffort::High),
        budget_tokens: Some(900),
        summary: None,
    });
    request.provider_options = json!({ "reasoning": { "effort": "low" } });
    let request = apply_reasoning(request);
    assert_eq!(
        request.provider_options["reasoning"],
        json!({ "effort": "low" })
    );
    assert!(request.reasoning.is_none());

    let mut request = ModelRequest::new(vec![crate::message::Message::user("hi")]);
    request.reasoning = Some(crate::model::ReasoningConfig {
        effort: Some(crate::model::ReasoningEffort::High),
        budget_tokens: Some(900),
        summary: None,
    });
    assert_eq!(
        apply_reasoning(request).provider_options["reasoning"],
        json!({ "max_tokens": 900 })
    );
}

#[test]
fn model_keeps_managed_tool_vision_and_cache_behavior() {
    use crate::model::{ChatModel, InputModality, InputSource};

    let model = fixture_model("https://api.example.test").with_native_tool_calling(false);
    let profile = model.profile().unwrap();
    assert!(!profile.tool_calling);
    assert!(!profile.parallel_tool_calls);
    assert!(model.supports_input(InputModality::Image, "image/png", InputSource::Base64));
    assert!(model.supports_input(InputModality::Image, "image/png", InputSource::Url));
    assert!(!model.supports_input(InputModality::Image, "image/png", InputSource::Path));
    assert!(model.supports_input(InputModality::Audio, "audio/wav", InputSource::Base64));
    assert!(!model.supports_input(InputModality::Audio, "audio/wav", InputSource::Url));
    assert_eq!(
        model.cache_identity().as_deref(),
        Some("tinyhumans:https://api.example.test/openai/v1:default-model")
    );
}

#[test]
fn model_advertises_native_tool_and_physical_attempt_capabilities() {
    use crate::model::ChatModel;

    let model = fixture_model("https://api.example.test").with_native_tool_calling(true);
    let profile = model.profile().unwrap();
    assert!(profile.tool_calling);
    assert!(profile.parallel_tool_calls);
    assert!(model.tracks_physical_attempts());
}

#[test]
fn model_advertises_only_serializable_image_and_audio_inputs() {
    use crate::model::{ChatModel, InputModality as M, InputSource as S};

    let model = fixture_model("https://api.example.test");
    for mime in ["image/jpeg", "image/webp", "image/gif"] {
        assert!(model.supports_input(M::Image, mime, S::Base64), "{mime}");
    }
    for mime in ["audio/x-wav", "audio/mpeg", "audio/mp3"] {
        assert!(model.supports_input(M::Audio, mime, S::Base64), "{mime}");
    }
    for (modality, mime) in [
        (M::Image, "application/pdf"),
        (M::Image, "video/mp4"),
        (
            M::Image,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (M::Image, "image/tiff"),
        (M::Audio, "audio/ogg"),
    ] {
        assert!(!model.supports_input(modality, mime, S::Base64), "{mime}");
    }
    assert!(!model.supports_input(M::Audio, "audio/wav", S::Url));
}

#[path = "cache_identity_tests.rs"]
mod cache_identity_tests;

#[path = "observer_tests.rs"]
mod observer_tests;

/// Each failure carries the credential kind its own request was sent with,
/// even when the resolver switches kinds between requests.
#[tokio::test]
async fn managed_auth_failure_reports_the_credential_its_request_carried() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::UNAUTHORIZED,
        content_type: "application/json",
        body: json!({ "error": { "message": "Unauthorized" } }).to_string(),
    };
    let use_api_key = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let resolver_use_api_key = use_api_key.clone();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_by_handler = seen.clone();
    let model = TinyHumansModel::new_with_credential(base, "", "default-model", move || {
        let resolver_use_api_key = Arc::clone(&resolver_use_api_key);
        Box::pin(async move {
            let api_key = resolver_use_api_key.load(Ordering::SeqCst);
            Ok(TinyHumansCredential {
                bearer: if api_key { "key".into() } else { "jwt".into() },
                kind: if api_key {
                    TinyHumansCredentialKind::ApiKey
                } else {
                    TinyHumansCredentialKind::Session
                },
            })
        })
    })
    .expect("model construction")
    .with_managed_auth_failure_handler(move |_, failure| {
        seen_by_handler
            .lock()
            .unwrap()
            .push((failure.operation(), failure.credential()));
    });

    let request = || ModelRequest::new(vec![crate::message::Message::user("hi")]);
    assert!(model.invoke(&(), request()).await.is_err());
    use_api_key.store(false, Ordering::SeqCst);
    assert!(model.invoke(&(), request()).await.is_err());
    assert!(model.stream(&(), request()).await.is_err());

    assert_eq!(
        *seen.lock().unwrap(),
        vec![
            (
                TinyHumansOperation::Invoke,
                Some(TinyHumansCredentialKind::ApiKey)
            ),
            (
                TinyHumansOperation::Invoke,
                Some(TinyHumansCredentialKind::Session)
            ),
            (
                TinyHumansOperation::Stream,
                Some(TinyHumansCredentialKind::Session)
            ),
        ]
    );
}

#[tokio::test]
async fn managed_auth_failure_carries_a_non_secret_request_credential_fingerprint() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::UNAUTHORIZED,
        content_type: "application/json",
        body: json!({ "error": { "message": "Unauthorized" } }).to_string(),
    };
    let token = Arc::new(std::sync::Mutex::new("old-request-token".to_owned()));
    *fixture.rotate_token_before_response.lock().unwrap() = Some(token.clone());
    let token_for_resolver = token.clone();
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_by_handler = seen.clone();
    let model = TinyHumansModel::new_with_credential(base, "", "default-model", move || {
        let token_for_resolver = Arc::clone(&token_for_resolver);
        Box::pin(async move {
            Ok(TinyHumansCredential {
                bearer: token_for_resolver.lock().unwrap().clone(),
                kind: TinyHumansCredentialKind::Session,
            })
        })
    })
    .unwrap()
    .with_managed_auth_failure_handler(move |_, failure| {
        seen_by_handler
            .lock()
            .unwrap()
            .push(failure.credential_fingerprint());
    });

    let request = ModelRequest::new(vec![crate::message::Message::user("hi")]);
    assert!(model.invoke(&(), request).await.is_err());
    assert_eq!(*token.lock().unwrap(), "new-session-token");

    let fingerprints = seen.lock().unwrap();
    let fingerprint = fingerprints[0].expect("managed credential fingerprint");
    assert!(fingerprint.matches_bearer("old-request-token"));
    assert!(!fingerprint.matches_bearer("new-session-token"));
    assert!(!format!("{fingerprint:?}").contains("old-request-token"));
}

#[test]
fn credential_fingerprint_has_a_stable_opaque_hex_round_trip() {
    let fingerprint = TinyHumansCredentialFingerprint::from_bearer("session-token");
    let encoded = fingerprint.to_hex();

    assert_eq!(encoded.len(), 64);
    assert!(
        encoded
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
    assert_eq!(
        TinyHumansCredentialFingerprint::from_hex(&encoded),
        Some(fingerprint)
    );
    assert_eq!(
        TinyHumansCredentialFingerprint::from_hex(&encoded.to_ascii_uppercase()),
        None,
        "the serialized form is canonical lower-case hex"
    );
    assert_eq!(TinyHumansCredentialFingerprint::from_hex("xyz"), None);
}

#[tokio::test]
async fn a_plain_bearer_resolver_reports_no_credential_kind() {
    let fixture = Fixture::default();
    let base = serve(fixture.clone()).await;
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::UNAUTHORIZED,
        content_type: "application/json",
        body: json!({ "error": { "message": "Unauthorized" } }).to_string(),
    };
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_by_handler = seen.clone();
    let model = fixture_model(&base).with_managed_auth_failure_handler(move |_, failure| {
        seen_by_handler.lock().unwrap().push(failure.credential());
    });
    let request = ModelRequest::new(vec![crate::message::Message::user("hi")]);
    assert!(model.invoke(&(), request).await.is_err());
    assert_eq!(*seen.lock().unwrap(), vec![None]);
}
