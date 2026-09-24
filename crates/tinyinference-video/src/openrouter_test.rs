//! Offline tests for the OpenRouter video generator.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::extract::{Path, Query, Request, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use serde_json::{Value, json};
use tinyinference_image::{MediaAuth, MediaReference, MediaTransport};

use crate::{Error, OpenRouterVideoGenerator, VideoGenerator, VideoRequest, WaitPolicy};

#[derive(Clone)]
struct Server {
    submits: Arc<Mutex<Vec<Value>>>,
    polls: Arc<AtomicUsize>,
    /// Poll replies, in order; the last one repeats.
    script: Arc<Vec<(StatusCode, Value)>>,
    listing: Option<Value>,
    content_queries: Arc<Mutex<Vec<String>>>,
    /// Wrap JSON replies in the proxying backend's `{success, data}` envelope.
    envelope: bool,
}

fn wrap(envelope: bool, body: Value) -> Value {
    if envelope {
        json!({ "success": true, "data": body })
    } else {
        body
    }
}

async fn start(
    prefix: &str,
    script: Vec<(StatusCode, Value)>,
    listing: Option<Value>,
) -> (String, Server) {
    let server = Server {
        submits: Arc::default(),
        polls: Arc::default(),
        script: Arc::new(script),
        listing,
        content_queries: Arc::default(),
        envelope: prefix.starts_with("/agent-integrations"),
    };
    let router = Router::new()
        .route(
            &format!("{prefix}/videos"),
            post(|State(s): State<Server>, request: Request| async move {
                let body = axum::body::to_bytes(request.into_body(), usize::MAX).await.unwrap();
                s.submits.lock().unwrap().push(serde_json::from_slice(&body).unwrap());
                axum::Json(wrap(
                    s.envelope,
                    json!({
                        "id": "gen-vid-1-abc", "polling_url": "/api/v1/videos/gen-vid-1-abc", "status": "pending"
                    }),
                ))
            }),
        )
        .route(
            &format!("{prefix}/videos/models"),
            get(|State(s): State<Server>| async move {
                match s.listing {
                    Some(listing) => axum::Json(listing).into_response(),
                    None => StatusCode::NOT_FOUND.into_response(),
                }
            }),
        )
        .route(
            &format!("{prefix}/videos/{{id}}"),
            get(|State(s): State<Server>, Path(id): Path<String>| async move {
                let call = s.polls.fetch_add(1, Ordering::SeqCst);
                let (status, mut body) = s.script[call.min(s.script.len() - 1)].clone();
                body["id"] = json!(id);
                let body = if status.is_success() { wrap(s.envelope, body) } else { body };
                (status, [("retry-after", "0")], axum::Json(body)).into_response()
            }),
        )
        .route(
            &format!("{prefix}/videos/{{id}}/content"),
            get(
                |State(s): State<Server>, Query(q): Query<std::collections::HashMap<String, String>>| async move {
                    s.content_queries
                        .lock()
                        .unwrap()
                        .push(q.get("index").cloned().unwrap_or_default());
                    ([("content-type", "video/mp4")], &b"\x00\x00\x00\x18ftypmp42"[..]).into_response()
                },
            ),
        )
        .with_state(server.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
    (format!("http://{address}{prefix}"), server)
}

fn generator(base_url: &str) -> OpenRouterVideoGenerator {
    OpenRouterVideoGenerator::with_transport(
        MediaTransport::new(MediaAuth::ApiKey("sk-or-test-key-123456".into()))
            .with_base_url(base_url)
            .with_max_retries(2),
    )
}

fn fast() -> WaitPolicy {
    WaitPolicy::new(Duration::from_millis(1), Duration::from_secs(5))
}

fn poll(status: &str, urls: &[&str]) -> (StatusCode, Value) {
    (
        StatusCode::OK,
        json!({ "status": status, "unsigned_urls": urls, "usage": { "cost": 0.42 } }),
    )
}

/// Regression (R1) end to end over HTTP: `completed` with an empty
/// `unsigned_urls` is polled through, then both outputs are downloaded.
#[tokio::test]
async fn full_lifecycle_polls_through_completed_without_urls() {
    let (base, server) = start(
        "/api/v1",
        vec![
            poll("pending", &[]),
            poll("in_progress", &[]),
            poll("completed", &[]),
            poll(
                "completed",
                &["https://cdn.test/0.mp4", "https://cdn.test/1.mp4"],
            ),
        ],
        None,
    )
    .await;
    let response = generator(&base)
        .generate(
            VideoRequest::new("a lighthouse in a storm")
                .with_model("openrouter/bytedance/seedance-2.0-mini")
                .with_duration(5)
                .with_resolution("720")
                .with_aspect_ratio("landscape")
                .with_audio(true)
                .with_first_frame(MediaReference::Url("https://x.test/first.png".into()))
                .with_last_frame(MediaReference::DataUrl(
                    "data:image/png;base64,iVBORw0KGgo=".into(),
                ))
                .with_reference(MediaReference::Url("https://x.test/style.mp4".into())),
            &fast(),
        )
        .await
        .unwrap();

    assert_eq!(response.job_id, "gen-vid-1-abc");
    assert_eq!(response.model, "bytedance/seedance-2.0-mini");
    assert_eq!(response.videos.len(), 2);
    assert_eq!(response.cost_usd, Some(0.42));
    assert_eq!(server.polls.load(Ordering::SeqCst), 4);
    assert_eq!(*server.content_queries.lock().unwrap(), vec!["0", "1"]);

    let body = server.submits.lock().unwrap()[0].clone();
    assert_eq!(body["model"], "bytedance/seedance-2.0-mini");
    assert_eq!(body["duration"], 5);
    assert_eq!(body["resolution"], "720p");
    assert_eq!(body["aspect_ratio"], "16:9");
    assert_eq!(body["generate_audio"], true);
    assert_eq!(body["frame_images"][0]["frame_type"], "first_frame");
    assert_eq!(
        body["frame_images"][0]["image_url"]["url"],
        "https://x.test/first.png"
    );
    assert_eq!(body["frame_images"][1]["frame_type"], "last_frame");
    assert_eq!(body["input_references"][0]["type"], "video_url");
    assert_eq!(
        body["input_references"][0]["video_url"]["url"],
        "https://x.test/style.mp4"
    );
}

#[tokio::test]
async fn proxied_base_url_unwraps_the_backend_envelope() {
    let (base, server) = start(
        "/agent-integrations/openrouter",
        vec![poll("pending", &[]), poll("completed", &["u"])],
        None,
    )
    .await;
    let response = generator(&base)
        .generate(VideoRequest::new("x"), &fast())
        .await
        .unwrap();
    assert_eq!(response.job_id, "gen-vid-1-abc");
    assert_eq!(response.videos.len(), 1);
    assert_eq!(response.cost_usd, Some(0.42));
    assert_eq!(server.submits.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn failed_job_surfaces_the_provider_error() {
    let (base, _server) = start(
        "/api/v1",
        vec![(
            StatusCode::OK,
            json!({ "status": "failed", "error": "prompt rejected by safety filter" }),
        )],
        None,
    )
    .await;
    let error = generator(&base)
        .generate(VideoRequest::new("x"), &fast())
        .await
        .unwrap_err();
    match error {
        Error::JobFailed { message, .. } => assert_eq!(message, "prompt rejected by safety filter"),
        other => panic!("expected JobFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn transient_poll_server_errors_are_retried() {
    let (base, server) = start(
        "/api/v1",
        vec![
            (
                StatusCode::BAD_GATEWAY,
                json!({"error": {"message": "busy"}}),
            ),
            poll("completed", &["u"]),
        ],
        None,
    )
    .await;
    generator(&base)
        .generate(VideoRequest::new("x"), &fast())
        .await
        .unwrap();
    assert_eq!(server.polls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn unsupported_duration_fails_before_submit() {
    let listing = json!({ "data": [{
        "id": "bytedance/seedance-2.0-mini",
        "supported_durations": [4, 5, 6],
        "supported_resolutions": ["480p", "720p"],
        "supported_frame_images": ["first_frame", "last_frame"],
        "generate_audio": true,
        "seed": true
    }]});
    let (base, server) = start("/api/v1", vec![poll("completed", &["u"])], Some(listing)).await;
    let error = generator(&base)
        .submit(VideoRequest::new("x").with_duration(20))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::Media(tinyinference_image::Error::Unsupported { field, .. }) if field == "duration"),
        "{error:?}"
    );
    let error = generator(&base)
        .submit(VideoRequest::new("x").with_resolution("1080p"))
        .await
        .unwrap_err();
    assert!(
        matches!(&error, Error::Media(tinyinference_image::Error::Unsupported { field, .. }) if field == "resolution"),
        "{error:?}"
    );
    assert!(server.submits.lock().unwrap().is_empty());
}

#[tokio::test]
async fn hostile_job_ids_are_rejected_locally() {
    let generator = generator("http://127.0.0.1:9");
    for id in ["../admin", "a/b", "", "x?y=1"] {
        assert!(matches!(
            generator.poll(id).await,
            Err(Error::Media(tinyinference_image::Error::Validation(_)))
        ));
    }
}

/// Sparse `unsigned_urls` (a blank slot before a populated one) download the
/// populated slot's index, not `0`.
#[tokio::test]
async fn sparse_output_slots_download_their_own_index() {
    let (base, server) = start(
        "/api/v1",
        vec![poll("completed", &["", "https://cdn.test/1.mp4"])],
        None,
    )
    .await;
    let response = generator(&base)
        .generate(VideoRequest::new("x"), &fast())
        .await
        .unwrap();
    assert_eq!(response.videos.len(), 1);
    assert_eq!(*server.content_queries.lock().unwrap(), vec!["1"]);
}
