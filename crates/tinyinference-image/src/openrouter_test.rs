//! Offline tests for the OpenRouter image generator and media transport.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use axum::Router;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use serde_json::{Value, json};

use crate::mock::TINY_PNG;
use crate::{
    Error, ImageGenerator, ImageRequest, MediaAuth, MediaReference, MediaTransport,
    OpenRouterImageGenerator,
};

const KEY: &str = "sk-or-test-secret-key-0123456789";

#[derive(Clone, Default)]
struct Captured {
    bodies: Arc<Mutex<Vec<Value>>>,
    headers: Arc<Mutex<Vec<HeaderMap>>>,
    image_calls: Arc<AtomicUsize>,
}

struct Fixture {
    base_url: String,
    captured: Captured,
}

async fn serve(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{address}")
}

/// Serves `images` and `images/models` under `prefix` with canned handlers.
async fn fixture(
    prefix: &str,
    image_reply: fn(usize) -> Response,
    listing: Option<Value>,
) -> Fixture {
    let captured = Captured::default();
    let images_state = captured.clone();
    let images = post(
        move |State(state): State<Captured>, request: Request| async move {
            let headers = request.headers().clone();
            let body = axum::body::to_bytes(request.into_body(), usize::MAX)
                .await
                .unwrap();
            state.headers.lock().unwrap().push(headers);
            state
                .bodies
                .lock()
                .unwrap()
                .push(serde_json::from_slice(&body).unwrap());
            let call = state.image_calls.fetch_add(1, Ordering::SeqCst);
            image_reply(call)
        },
    );
    let models = get(move || async move {
        match listing {
            Some(listing) => axum::Json(listing).into_response(),
            None => StatusCode::NOT_FOUND.into_response(),
        }
    });
    let router = Router::new()
        .route(&format!("{prefix}/images"), images)
        .route(&format!("{prefix}/images/models"), models)
        .with_state(images_state);
    Fixture {
        base_url: format!("{}{prefix}", serve(router).await),
        captured,
    }
}

fn png_reply(_call: usize) -> Response {
    axum::Json(json!({
        "created": 1_748_372_400,
        "data": [{ "b64_json": BASE64.encode(TINY_PNG), "media_type": "image/png" }],
        "usage": { "cost": 0.035 }
    }))
    .into_response()
}

fn generator(base_url: &str) -> OpenRouterImageGenerator {
    OpenRouterImageGenerator::with_transport(
        MediaTransport::new(MediaAuth::ApiKey(KEY.into()))
            .with_base_url(base_url)
            .with_header("x-title", "tinyinference-tests")
            .with_max_retries(2),
    )
}

#[tokio::test]
async fn generates_decodes_and_reports_cost() {
    let fixture = fixture("/api/v1", png_reply, None).await;
    let response = generator(&fixture.base_url)
        .generate(
            ImageRequest::new("a red panda astronaut")
                .with_model("openrouter/bytedance-seed/seedream-5-0-lite")
                .with_aspect_ratio("landscape")
                .with_resolution("2k")
                .with_seed(7)
                .with_reference(MediaReference::Url("https://example.com/ref.png".into()))
                .with_reference(MediaReference::Bytes {
                    media_type: "image/png".into(),
                    data: Bytes::from_static(TINY_PNG),
                }),
        )
        .await
        .unwrap();

    assert_eq!(response.model, "bytedance-seed/seedream-5-0-lite");
    assert_eq!(response.images.len(), 1);
    assert_eq!(response.images[0].media_type, "image/png");
    assert_eq!(response.images[0].data.as_ref(), TINY_PNG);
    assert_eq!(response.cost_usd, Some(0.035));

    let body = fixture.captured.bodies.lock().unwrap()[0].clone();
    assert_eq!(
        body["model"], "bytedance-seed/seedream-5-0-lite",
        "openrouter/ prefix stripped"
    );
    assert_eq!(body["aspect_ratio"], "16:9");
    assert_eq!(body["resolution"], "2K");
    assert_eq!(body["seed"], 7);
    assert_eq!(body["input_references"][0]["type"], "image_url");
    assert_eq!(
        body["input_references"][0]["image_url"]["url"],
        "https://example.com/ref.png"
    );
    assert!(
        body["input_references"][1]["image_url"]["url"]
            .as_str()
            .unwrap()
            .starts_with("data:image/png;base64,")
    );

    let headers = fixture.captured.headers.lock().unwrap()[0].clone();
    assert_eq!(headers["authorization"], format!("Bearer {KEY}"));
    assert_eq!(headers["x-title"], "tinyinference-tests");
}

/// Regression (R1): an accepted, billed request that returns no images must be
/// an error that tells the caller not to retry — never an empty success.
#[tokio::test]
async fn accepted_request_without_images_is_no_media_error() {
    fn empty(_call: usize) -> Response {
        axum::Json(json!({ "created": 1, "data": [], "usage": { "cost": 0.035 } })).into_response()
    }
    let fixture = fixture("/api/v1", empty, None).await;
    let error = generator(&fixture.base_url)
        .generate(ImageRequest::new("anything"))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NoMedia { .. }), "{error:?}");
    assert!(error.to_string().contains("do not retry"), "{error}");
    assert!(!error.is_retryable());
    assert_eq!(fixture.captured.image_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn entries_without_payload_are_not_counted_as_images() {
    fn blank(_call: usize) -> Response {
        axum::Json(json!({ "created": 1, "data": [{ "b64_json": "" }] })).into_response()
    }
    let fixture = fixture("/api/v1", blank, None).await;
    let error = generator(&fixture.base_url)
        .generate(ImageRequest::new("anything"))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NoMedia { .. }), "{error:?}");
}

#[tokio::test]
async fn unsupported_aspect_ratio_fails_before_the_paid_call() {
    let listing = json!({ "data": [{
        "id": "bytedance-seed/seedream-5-0-lite",
        "supported_parameters": {
            "aspect_ratio": { "type": "enum", "values": ["1:1", "3:4"] },
            "n": { "type": "range", "min": 1, "max": 4 }
        }
    }]});
    let fixture = fixture("/api/v1", png_reply, Some(listing)).await;
    let error = generator(&fixture.base_url)
        .generate(ImageRequest::new("x").with_aspect_ratio("16:9"))
        .await
        .unwrap_err();
    match error {
        Error::Unsupported { field, allowed, .. } => {
            assert_eq!(field, "aspect_ratio");
            assert_eq!(allowed, vec!["1:1", "3:4"]);
        }
        other => panic!("expected Unsupported, got {other:?}"),
    }
    assert_eq!(fixture.captured.image_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn unsupported_seed_is_rejected_when_the_listing_omits_it() {
    let listing = json!({ "data": [{
        "id": "google/gemini-3.1-flash-lite-image",
        "supported_parameters": { "aspect_ratio": { "type": "enum", "values": ["1:1"] } }
    }]});
    let fixture = fixture("/api/v1", png_reply, Some(listing)).await;
    let error = generator(&fixture.base_url)
        .generate(
            ImageRequest::new("x")
                .with_model("google/gemini-3.1-flash-lite-image")
                .with_seed(1),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Unsupported { ref field, .. } if field == "seed"),
        "{error:?}"
    );
}

#[tokio::test]
async fn listing_without_capabilities_does_not_block_generation() {
    // A proxying backend may list ids only; that must never reject a request.
    let listing = json!({ "object": "list", "data": [{
        "id": "bytedance-seed/seedream-5-0-lite", "display_name": "Seedream 5.0 Lite"
    }]});
    let fixture = fixture("/api/v1", png_reply, Some(listing)).await;
    generator(&fixture.base_url)
        .generate(
            ImageRequest::new("x")
                .with_aspect_ratio("21:9")
                .with_seed(3),
        )
        .await
        .unwrap();
}

/// A proxying backend wraps OpenRouter's body in `{success, data}`; the
/// transport unwraps it transparently.
fn enveloped_png_reply(_call: usize) -> Response {
    axum::Json(json!({ "success": true, "data": {
        "created": 1,
        "data": [{ "b64_json": BASE64.encode(TINY_PNG), "media_type": "image/png" }],
        "usage": { "cost": 0.035 }
    }}))
    .into_response()
}

#[tokio::test]
async fn proxied_backend_base_url_and_bearer_resolver() {
    let fixture = fixture("/agent-integrations/openrouter", enveloped_png_reply, None).await;
    let resolver: crate::BearerResolver = Arc::new(|| Ok("session-jwt-abcdefgh".to_owned()));
    let generator = OpenRouterImageGenerator::with_transport(
        MediaTransport::new(MediaAuth::Bearer(resolver)).with_base_url(&fixture.base_url),
    );
    let response = generator.generate(ImageRequest::new("x")).await.unwrap();
    assert_eq!(response.images.len(), 1);
    assert_eq!(response.cost_usd, Some(0.035));
    let headers = fixture.captured.headers.lock().unwrap()[0].clone();
    assert_eq!(headers["authorization"], "Bearer session-jwt-abcdefgh");
}

#[tokio::test]
async fn failed_envelope_is_an_error_even_with_a_2xx_status() {
    fn failed(_call: usize) -> Response {
        axum::Json(json!({ "success": false, "error": "Insufficient balance" })).into_response()
    }
    let fixture = fixture("/agent-integrations/openrouter", failed, None).await;
    let error = generator(&fixture.base_url)
        .with_capability_check(false)
        .generate(ImageRequest::new("x"))
        .await
        .unwrap_err();
    assert!(
        error.to_string().contains("Insufficient balance"),
        "{error}"
    );
}

#[tokio::test]
async fn blank_bearer_fails_without_a_request() {
    let fixture = fixture("/api/v1", png_reply, None).await;
    let resolver: crate::BearerResolver = Arc::new(|| Ok("  ".to_owned()));
    let generator = OpenRouterImageGenerator::with_transport(
        MediaTransport::new(MediaAuth::Bearer(resolver)).with_base_url(&fixture.base_url),
    )
    .with_capability_check(false);
    let error = generator
        .generate(ImageRequest::new("x"))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Auth(_)), "{error:?}");
    assert_eq!(fixture.captured.image_calls.load(Ordering::SeqCst), 0);
}

/// A 5xx after a submit may already have started (and billed) a generation, so
/// the billable POST must not be retried.
#[tokio::test]
async fn billable_post_is_not_retried_on_server_error() {
    fn boom(_call: usize) -> Response {
        (
            StatusCode::BAD_GATEWAY,
            axum::Json(json!({"error": {"code": 502, "message": "upstream"}})),
        )
            .into_response()
    }
    let fixture = fixture("/api/v1", boom, None).await;
    let error = generator(&fixture.base_url)
        .with_capability_check(false)
        .generate(ImageRequest::new("x"))
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Http { status: 502, .. }),
        "{error:?}"
    );
    assert_eq!(fixture.captured.image_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn billable_post_is_retried_on_rate_limit() {
    fn limited_once(call: usize) -> Response {
        if call == 0 {
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("retry-after", "0")],
                "slow down",
            )
                .into_response()
        } else {
            png_reply(call)
        }
    }
    let fixture = fixture("/api/v1", limited_once, None).await;
    generator(&fixture.base_url)
        .with_capability_check(false)
        .generate(ImageRequest::new("x"))
        .await
        .unwrap();
    assert_eq!(fixture.captured.image_calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn provider_errors_and_debug_never_leak_the_key() {
    fn echo_key(_call: usize) -> Response {
        (
            StatusCode::UNAUTHORIZED,
            axum::Json(json!({"error": {"code": 401, "message": format!("bad key {KEY}")}})),
        )
            .into_response()
    }
    let fixture = fixture("/api/v1", echo_key, None).await;
    let generator = generator(&fixture.base_url).with_capability_check(false);
    let error = generator
        .generate(ImageRequest::new("x"))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Auth(_)), "{error:?}");
    assert!(!error.to_string().contains(KEY), "{error}");
    assert!(!format!("{generator:?}").contains(KEY));
}

#[tokio::test]
async fn validation_errors_are_local() {
    let fixture = fixture("/api/v1", png_reply, None).await;
    let generator = generator(&fixture.base_url);
    assert!(matches!(
        generator.generate(ImageRequest::new("  ")).await,
        Err(Error::Validation(_))
    ));
    assert!(matches!(
        generator.generate(ImageRequest::new("x").with_n(11)).await,
        Err(Error::Validation(_))
    ));
    assert_eq!(fixture.captured.image_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn lists_models_from_both_listing_shapes() {
    let listing = json!({ "object": "list", "data": [
        { "id": "a/one", "display_name": "One" },
        { "id": "b/two", "name": "Two", "supported_parameters": { "seed": { "type": "boolean" } } }
    ]});
    let fixture = fixture("/api/v1", png_reply, Some(listing)).await;
    let models = generator(&fixture.base_url).list_models().await.unwrap();
    assert_eq!(models.len(), 2);
    assert_eq!(models[0].name.as_deref(), Some("One"));
    assert_eq!(models[0].capabilities.seed, None);
    assert_eq!(models[1].capabilities.seed, Some(true));
}

/// A billed response whose image payloads cannot be decoded is a billed
/// non-delivery (`NoMedia`, do not retry), not a retryable decode error.
#[tokio::test]
async fn undecodable_images_are_a_billed_non_delivery() {
    fn garbage(_call: usize) -> Response {
        axum::Json(json!({ "created": 1, "data": [{ "b64_json": "!!!not base64!!!" }] }))
            .into_response()
    }
    let fixture = fixture("/api/v1", garbage, None).await;
    let error = generator(&fixture.base_url)
        .with_capability_check(false)
        .generate(ImageRequest::new("x"))
        .await
        .unwrap_err();
    assert!(matches!(error, Error::NoMedia { .. }), "{error:?}");
    assert!(!error.is_retryable());
}

#[test]
fn transport_debug_redacts_base_url_userinfo() {
    let transport = MediaTransport::new(MediaAuth::ApiKey(KEY.into()))
        .with_base_url("https://user:hunter2secret@proxy.example/agent-integrations/openrouter");
    let debug = format!("{transport:?}");
    assert!(!debug.contains("hunter2secret"), "{debug}");
    assert!(!transport.redacted_base_url().contains("hunter2secret"));
}
