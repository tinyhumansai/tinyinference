//! Migration-contract tests for provider-neutral model boundary metadata.

use super::*;
use crate::providers::MockModel;
use crate::usage::ChargedAmount;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;

struct RecordingModel {
    requests: Mutex<Vec<ModelRequest>>,
    response: ModelResponse,
}

impl RecordingModel {
    fn new(response: ModelResponse) -> Self {
        Self {
            requests: Mutex::new(Vec::new()),
            response,
        }
    }
}

#[async_trait]
impl ChatModel<()> for RecordingModel {
    async fn invoke(&self, _state: &(), request: ModelRequest) -> crate::Result<ModelResponse> {
        self.requests.lock().unwrap().push(request);
        Ok(self.response.clone())
    }
}

struct FailingModel;

#[async_trait]
impl ChatModel<()> for FailingModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> crate::Result<ModelResponse> {
        Err(crate::Error::Model("synthetic failure".to_string()))
    }
}

struct ScriptedStreamModel {
    items: Vec<ModelStreamItem>,
    metadata: ModelStreamMetadata,
}

impl ScriptedStreamModel {
    fn new(items: Vec<ModelStreamItem>, metadata: ModelStreamMetadata) -> Self {
        Self { items, metadata }
    }
}

#[async_trait]
impl ChatModel<()> for ScriptedStreamModel {
    async fn invoke(&self, _state: &(), _request: ModelRequest) -> crate::Result<ModelResponse> {
        Ok(ModelResponse::assistant("not used by stream tests"))
    }

    async fn stream(&self, _state: &(), _request: ModelRequest) -> crate::Result<ModelStream> {
        Ok(
            ModelStream::new(Box::pin(futures::stream::iter(self.items.clone())))
                .with_metadata(self.metadata.clone()),
        )
    }
}

#[derive(Default)]
struct RecordingObserver(Mutex<Vec<ModelCallObservation>>);

impl ModelObserver for RecordingObserver {
    fn observe(&self, observation: ModelCallObservation) {
        self.0.lock().unwrap().push(observation);
    }
}

#[test]
fn usage_round_trips_billing_and_context_metadata_without_raw_json() {
    let usage = Usage {
        cache_read_tokens: 3,
        cache_creation_tokens: 5,
        reasoning_tokens: 7,
        charged_amount: Some(ChargedAmount::usd_micros(42)),
        context_window_tokens: Some(128_000),
        ..Usage::new(11, 13)
    };

    let encoded = serde_json::to_value(usage).expect("usage serializes");
    assert!(encoded.get("charged_amount").is_some());
    assert!(encoded.get("context_window_tokens").is_some());
    assert!(!encoded.to_string().contains("raw"));
    assert_eq!(serde_json::from_value::<Usage>(encoded).unwrap(), usage);
}

#[test]
fn request_correlation_and_resolved_route_survive_response_serialization() {
    let correlation = ModelCallCorrelation::new("run-7", "call-9");
    let route = ResolvedModelRoute::new("openai", "gpt-5", "chat-v1");
    let response = ModelResponse::assistant("ok")
        .with_correlation(correlation.clone())
        .with_resolved_route(route.clone());

    let decoded: ModelResponse =
        serde_json::from_value(serde_json::to_value(response).unwrap()).unwrap();
    assert_eq!(decoded.correlation.as_ref(), Some(&correlation));
    assert_eq!(decoded.resolved_route.as_ref(), Some(&route));
}

#[tokio::test]
async fn direct_provider_propagates_request_correlation_to_sync_and_stream_results() {
    let model = MockModel::constant("ok");
    let correlation = ModelCallCorrelation::new("run-direct", "call-direct");
    let request = ModelRequest::default().with_correlation(correlation.clone());

    let response = model.invoke(&(), request.clone()).await.unwrap();
    assert_eq!(response.correlation.as_ref(), Some(&correlation));

    let stream = model.stream(&(), request).await.unwrap();
    assert_eq!(stream.metadata().correlation.as_ref(), Some(&correlation));
    let items = stream.collect::<Vec<_>>().await;
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("mock stream must complete");
    };
    assert_eq!(response.correlation.as_ref(), Some(&correlation));
}

#[tokio::test]
async fn default_chat_model_stream_propagates_request_correlation_to_terminal_response() {
    let model = RecordingModel::new(ModelResponse::assistant("ok"));
    let correlation = ModelCallCorrelation::new("run-default", "call-default");
    let stream = model
        .stream(
            &(),
            ModelRequest::default().with_correlation(correlation.clone()),
        )
        .await
        .unwrap();
    assert_eq!(stream.metadata().correlation.as_ref(), Some(&correlation));
    let items = stream.collect::<Vec<_>>().await;
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("default stream must complete");
    };
    assert_eq!(response.correlation.as_ref(), Some(&correlation));

    let provider_correlation = ModelCallCorrelation::new("provider-run", "provider-call");
    let model = RecordingModel::new(
        ModelResponse::assistant("provider correlation")
            .with_correlation(provider_correlation.clone()),
    );
    let stream = model.stream(&(), ModelRequest::default()).await.unwrap();
    assert_eq!(stream.metadata(), &ModelStreamMetadata::default());
    let items = stream.collect::<Vec<_>>().await;
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("default stream must complete");
    };
    assert_eq!(response.correlation.as_ref(), Some(&provider_correlation));
}

#[tokio::test]
async fn decorators_apply_defaults_clamp_tokens_and_stamp_sync_and_stream_results() {
    let inner = Arc::new(RecordingModel::new(ModelResponse::assistant("ok")));
    let profile = ModelProfile {
        max_input_tokens: Some(128_000),
        ..ModelProfile::default()
    };
    let decorated: Arc<dyn ChatModel<()>> = Arc::new(MaxTokensModel::new(
        Arc::new(
            ProfileOverrideModel::new(inner.clone(), profile)
                .with_request_model("chat-v1")
                .with_request_temperature(0.3),
        ),
        64,
    ));
    let route = ResolvedModelRoute::new("openai", "gpt-5", "chat-v1");
    let model = RouteRecordingModel::new(decorated, route.clone());
    let request = ModelRequest::default()
        .with_max_tokens(100)
        .with_correlation(ModelCallCorrelation::new("run", "call"));

    let response = model.invoke(&(), request.clone()).await.unwrap();
    assert_eq!(response.correlation.as_ref().unwrap().call_id, "call");
    assert_eq!(response.resolved_route.as_ref(), Some(&route));

    let stream_items = model
        .stream(&(), request)
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let ModelStreamItem::Completed(stream_response) = stream_items.last().unwrap() else {
        panic!("default stream must complete");
    };
    assert_eq!(stream_response.resolved_route.as_ref(), Some(&route));

    let requests = inner.requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.model.as_deref() == Some("chat-v1"))
    );
    assert!(
        requests
            .iter()
            .all(|request| request.temperature == Some(0.3))
    );
    assert!(
        requests
            .iter()
            .all(|request| request.max_tokens == Some(64))
    );
}

#[tokio::test]
async fn stream_metadata_and_abort_guard_follow_the_consumer_lifetime() {
    let producer = tokio::spawn(std::future::pending::<()>());
    let stream = ModelStream::new(Box::pin(futures::stream::pending()))
        .with_correlation(ModelCallCorrelation::new("run", "call"))
        .with_resolved_route(ResolvedModelRoute::new("mock", "m", "route"))
        .abort_on_drop(AbortOnDrop::from_join_handle(&producer));
    assert_eq!(
        stream.metadata().correlation.as_ref().unwrap().run_id,
        "run"
    );
    assert_eq!(
        stream.metadata().resolved_route.as_ref().unwrap().provider,
        "mock"
    );
    drop(stream);
    assert!(producer.await.unwrap_err().is_cancelled());
}

#[test]
fn stream_metadata_serializes_with_defaults_and_stamps_terminal_responses() {
    assert_eq!(
        serde_json::to_value(ModelStreamMetadata::default()).unwrap(),
        serde_json::json!({})
    );

    let correlation = ModelCallCorrelation::new("run", "call");
    let route = ResolvedModelRoute::new("mock", "model", "route");
    let metadata = ModelStreamMetadata {
        correlation: Some(correlation.clone()),
        resolved_route: Some(route.clone()),
    };
    assert_eq!(
        serde_json::from_value::<ModelStreamMetadata>(serde_json::to_value(&metadata).unwrap())
            .unwrap(),
        metadata
    );

    let stream = ModelStream::new(Box::pin(futures::stream::iter(vec![
        ModelStreamItem::Completed(ModelResponse::assistant("done")),
    ])))
    .with_correlation(correlation.clone())
    .with_resolved_route(route.clone());
    assert_eq!(stream.metadata(), &metadata);

    let items = futures::executor::block_on(stream.collect::<Vec<_>>());
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("stream must complete");
    };
    assert_eq!(response.correlation.as_ref(), Some(&correlation));
    assert_eq!(response.resolved_route.as_ref(), Some(&route));
}

#[test]
fn repeated_stream_metadata_setters_replace_defaults_without_overwriting_provider_values() {
    let first_correlation = ModelCallCorrelation::new("first-run", "first-call");
    let second_correlation = ModelCallCorrelation::new("second-run", "second-call");
    let first_route = ResolvedModelRoute::new("first", "first-model", "first-route");
    let second_route = ResolvedModelRoute::new("second", "second-model", "second-route");
    let stream = ModelStream::new(Box::pin(futures::stream::iter(vec![
        ModelStreamItem::Completed(ModelResponse::assistant("done")),
    ])))
    .with_correlation(first_correlation)
    .with_resolved_route(first_route)
    .with_correlation(second_correlation.clone())
    .with_resolved_route(second_route.clone());
    assert_eq!(
        stream.metadata().correlation.as_ref(),
        Some(&second_correlation)
    );
    assert_eq!(
        stream.metadata().resolved_route.as_ref(),
        Some(&second_route)
    );
    let items = futures::executor::block_on(stream.collect::<Vec<_>>());
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("stream must complete");
    };
    assert_eq!(response.correlation.as_ref(), Some(&second_correlation));
    assert_eq!(response.resolved_route.as_ref(), Some(&second_route));

    let provider_correlation = ModelCallCorrelation::new("provider-run", "provider-call");
    let provider_route = ResolvedModelRoute::new("provider", "model", "provider-route");
    let stream = ModelStream::new(Box::pin(futures::stream::iter(vec![
        ModelStreamItem::Completed(
            ModelResponse::assistant("provider values")
                .with_correlation(provider_correlation.clone())
                .with_resolved_route(provider_route.clone()),
        ),
    ])))
    .with_correlation(second_correlation)
    .with_resolved_route(second_route);
    let items = futures::executor::block_on(stream.collect::<Vec<_>>());
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("stream must complete");
    };
    assert_eq!(response.correlation.as_ref(), Some(&provider_correlation));
    assert_eq!(response.resolved_route.as_ref(), Some(&provider_route));
}

#[tokio::test]
async fn every_terminal_stream_item_disarms_its_abort_guard() {
    let terminal_items = vec![
        ModelStreamItem::Completed(ModelResponse::assistant("done")),
        ModelStreamItem::Failed("failed".to_string()),
        ModelStreamItem::ProviderFailed(ProviderError {
            provider: "mock".to_string(),
            message: "provider failed".to_string(),
            ..ProviderError::default()
        }),
    ];

    for item in terminal_items {
        let producer = tokio::spawn(std::future::pending::<()>());
        let stream = ModelStream::new(Box::pin(futures::stream::iter(vec![item])))
            .abort_on_drop(AbortOnDrop::from_join_handle(&producer));
        let _ = stream.collect::<Vec<_>>().await;
        tokio::task::yield_now().await;
        assert!(
            !producer.is_finished(),
            "terminal streams must not abort producers"
        );
        producer.abort();
        assert!(producer.await.unwrap_err().is_cancelled());
    }
}

#[tokio::test]
async fn observer_reports_each_terminal_outcome_once() {
    let observer = Arc::new(RecordingObserver::default());
    let success = ObservingModel::new(
        Arc::new(RecordingModel::new(ModelResponse::assistant("ok"))),
        observer.clone(),
    );
    success.invoke(&(), ModelRequest::default()).await.unwrap();

    let cached = ObservingModel::new(
        Arc::new(RecordingModel::new(ModelResponse {
            served_from_cache: true,
            ..ModelResponse::assistant("cached")
        })),
        observer.clone(),
    );
    cached.invoke(&(), ModelRequest::default()).await.unwrap();

    let fallback = ObservingModel::new(
        Arc::new(RouteRecordingModel::new(
            Arc::new(RecordingModel::new(ModelResponse::assistant("fallback"))),
            ResolvedModelRoute::new("mock", "fallback", "fallback-route"),
        )),
        observer.clone(),
    );
    fallback
        .invoke(
            &(),
            ModelRequest::default()
                .with_model("provider-model-id")
                .with_requested_route("primary-route"),
        )
        .await
        .unwrap();

    let same_route = ObservingModel::new(
        Arc::new(RouteRecordingModel::new(
            Arc::new(RecordingModel::new(ModelResponse::assistant("same route"))),
            ResolvedModelRoute::new("mock", "other-provider-model", "same-route"),
        )),
        observer.clone(),
    );
    same_route
        .invoke(
            &(),
            ModelRequest::default()
                .with_model("different-provider-model-id")
                .with_requested_route("same-route"),
        )
        .await
        .unwrap();

    let failure = ObservingModel::new(Arc::new(FailingModel), observer.clone());
    assert!(failure.invoke(&(), ModelRequest::default()).await.is_err());

    let observations = observer.0.lock().unwrap();
    assert!(matches!(
        observations[0],
        ModelCallObservation::Succeeded { .. }
    ));
    assert!(matches!(
        observations[1],
        ModelCallObservation::CacheHit { .. }
    ));
    assert!(matches!(
        observations[2],
        ModelCallObservation::Fallback { .. }
    ));
    assert!(matches!(
        observations[3],
        ModelCallObservation::Succeeded { .. }
    ));
    assert!(matches!(
        observations[4],
        ModelCallObservation::Failed { .. }
    ));
    assert_eq!(observations.len(), 5);
}

#[tokio::test]
async fn observer_streams_report_one_terminal_outcome_with_stream_metadata() {
    let correlation = ModelCallCorrelation::new("stream-run", "stream-call");
    let route = ResolvedModelRoute::new("mock", "model", "route");
    let metadata = ModelStreamMetadata {
        correlation: Some(correlation.clone()),
        resolved_route: Some(route.clone()),
    };
    let observer = Arc::new(RecordingObserver::default());
    let success = ObservingModel::new(
        Arc::new(ScriptedStreamModel::new(
            vec![
                ModelStreamItem::Completed(ModelResponse::assistant("ok")),
                ModelStreamItem::Failed("ignored after completion".to_string()),
            ],
            metadata.clone(),
        )),
        observer.clone(),
    );
    let cached = ObservingModel::new(
        Arc::new(ScriptedStreamModel::new(
            vec![ModelStreamItem::Completed(ModelResponse {
                served_from_cache: true,
                ..ModelResponse::assistant("cached")
            })],
            metadata.clone(),
        )),
        observer.clone(),
    );
    let failed = ObservingModel::new(
        Arc::new(ScriptedStreamModel::new(
            vec![ModelStreamItem::Failed("stream failed".to_string())],
            metadata.clone(),
        )),
        observer.clone(),
    );
    let provider_failed = ObservingModel::new(
        Arc::new(ScriptedStreamModel::new(
            vec![ModelStreamItem::ProviderFailed(ProviderError {
                provider: "mock".to_string(),
                message: "provider failed".to_string(),
                ..ProviderError::default()
            })],
            metadata,
        )),
        observer.clone(),
    );

    for model in [&success, &cached, &failed, &provider_failed] {
        model
            .stream(&(), ModelRequest::default())
            .await
            .unwrap()
            .collect::<Vec<_>>()
            .await;
    }

    let observations = observer.0.lock().unwrap();
    assert_eq!(observations.len(), 4);
    assert!(matches!(
        &observations[0],
        ModelCallObservation::Succeeded { correlation: observed, route: observed_route, .. }
            if observed.as_ref() == Some(&correlation) && observed_route.as_ref() == Some(&route)
    ));
    assert!(matches!(
        &observations[1],
        ModelCallObservation::CacheHit { correlation: observed, route: observed_route, .. }
            if observed.as_ref() == Some(&correlation) && observed_route.as_ref() == Some(&route)
    ));
    assert!(matches!(
        &observations[2],
        ModelCallObservation::Failed { correlation: observed, message }
            if observed.as_ref() == Some(&correlation) && message == "stream failed"
    ));
    assert!(matches!(
        &observations[3],
        ModelCallObservation::Failed { correlation: observed, message }
            if observed.as_ref() == Some(&correlation) && message == "mock returned: provider failed"
    ));
}
