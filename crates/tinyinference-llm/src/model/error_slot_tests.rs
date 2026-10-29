//! `ErrorSlotModel` keeps the slot filled with the provider failure that ended
//! a model call, including one reported inside a stream.
use super::*;
use crate::message::{AssistantMessage, ContentBlock};
use async_trait::async_trait;
use futures::StreamExt;

const REJECTION: &str = "Message at index 2 has role 'tool' but is not preceded by an \
                         assistant message with a matching tool_call";

/// Streams one terminal item per call, in order; `invoke` fails with the
/// same provider error.
struct ScriptedStreamModel(Mutex<Vec<ModelStreamItem>>);

fn completed() -> ModelResponse {
    ModelResponse {
        output: Vec::new(),
        execution: None,
        message: AssistantMessage {
            id: None,
            content: vec![ContentBlock::Text("ok".to_string())],
            tool_calls: Vec::new(),
            usage: None,
            origin: None,
        },
        usage: None,
        finish_reason: Some("stop".to_string()),
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

fn rejection() -> ProviderError {
    ProviderError {
        provider: "OpenHuman".to_string(),
        status: Some(400),
        message: REJECTION.to_string(),
        ..ProviderError::default()
    }
}

#[async_trait]
impl ChatModel<()> for ScriptedStreamModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> crate::Result<ModelResponse> {
        Err(crate::Error::Provider(Box::new(rejection())))
    }

    async fn stream(&self, _state: &(), _request: ModelRequest) -> crate::Result<ModelStream> {
        let item = self.0.lock().unwrap().remove(0);
        Ok(ModelStream::new(Box::pin(futures::stream::iter(vec![
            item,
        ]))))
    }
}

async fn drain(model: &ErrorSlotModel<()>) {
    let mut stream = model.stream(&(), ModelRequest::default()).await.unwrap();
    while stream.next().await.is_some() {}
}

fn slot_text(slot: &ModelErrorSlot) -> Option<String> {
    slot.lock().unwrap().as_ref().map(|error| error.to_string())
}

#[tokio::test]
async fn an_in_stream_provider_failure_fills_the_slot_and_a_later_success_clears_it() {
    let slot: ModelErrorSlot = Arc::new(Mutex::new(None));
    let model = ErrorSlotModel::new(
        Arc::new(ScriptedStreamModel(Mutex::new(vec![
            ModelStreamItem::ProviderFailed(rejection()),
            ModelStreamItem::Completed(completed()),
        ]))),
        slot.clone(),
    );

    drain(&model).await;
    let recorded = slot_text(&slot).expect("an in-stream failure is recorded");
    assert!(recorded.contains(REJECTION), "{recorded}");
    assert!(recorded.contains("HTTP 400"), "{recorded}");

    // A retry or fallback that succeeds must not leave a stale failure behind
    // for an unrelated later error to re-surface.
    drain(&model).await;
    assert_eq!(slot_text(&slot), None);
}

#[tokio::test]
async fn an_invoke_error_fills_the_slot() {
    let slot: ModelErrorSlot = Arc::new(Mutex::new(None));
    let model = ErrorSlotModel::new(
        Arc::new(ScriptedStreamModel(Mutex::new(Vec::new()))),
        slot.clone(),
    );
    model
        .invoke(&(), ModelRequest::default())
        .await
        .expect_err("scripted invoke fails");
    assert!(slot_text(&slot).is_some_and(|text| text.contains(REJECTION)));
}

#[tokio::test]
async fn an_attempt_that_is_dropped_after_a_failed_one_leaves_no_stale_error() {
    // In-band 503, then the retry hangs and the harness drops it on a call
    // timeout: the 503 must not be re-surfaced as the cause.
    let slot: ModelErrorSlot = Arc::new(Mutex::new(None));
    let model = ErrorSlotModel::new(
        Arc::new(ScriptedStreamModel(Mutex::new(vec![
            ModelStreamItem::ProviderFailed(ProviderError {
                status: Some(503),
                ..rejection()
            }),
            ModelStreamItem::Completed(completed()),
        ]))),
        slot.clone(),
    );
    drain(&model).await;
    assert!(slot_text(&slot).is_some());

    let retry = model.stream(&(), ModelRequest::default()).await.unwrap();
    drop(retry);
    assert_eq!(slot_text(&slot), None, "a new attempt clears the old error");
}

#[tokio::test]
async fn the_recorded_error_is_secret_scrubbed() {
    // The slot's error reaches logs and Sentry via the run failure.
    let slot: ModelErrorSlot = Arc::new(Mutex::new(None));
    let model = ErrorSlotModel::new(
        Arc::new(ScriptedStreamModel(Mutex::new(vec![
            ModelStreamItem::ProviderFailed(ProviderError {
                message: "bad key sk-live0123456789abcdefghij in request".to_string(),
                ..rejection()
            }),
        ]))),
        slot.clone(),
    );
    drain(&model).await;
    let recorded = slot_text(&slot).expect("recorded");
    assert!(!recorded.contains("sk-live0123456789"), "{recorded}");
    assert!(recorded.contains("[REDACTED]"), "{recorded}");
}
