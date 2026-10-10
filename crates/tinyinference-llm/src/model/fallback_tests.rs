use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Reply {
    fails: bool,
    calls: AtomicUsize,
}
#[async_trait::async_trait]
impl ChatModel<()> for Reply {
    async fn invoke(&self, _: &(), request: ModelRequest) -> crate::Result<ModelResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.fails {
            Err(crate::Error::Model("unavailable".into()))
        } else {
            Ok(ModelResponse::assistant(
                request.model.unwrap_or_else(|| "answer".into()),
            ))
        }
    }
}
fn reply(fails: bool) -> Arc<Reply> {
    Arc::new(Reply {
        fails,
        calls: AtomicUsize::new(0),
    })
}

#[tokio::test]
async fn invocation_preserves_the_request_and_stops_at_the_first_success() {
    let primary = reply(true);
    let secondary = reply(false);
    let spare = reply(false);
    let chain = FallbackModel::new(primary.clone(), vec![secondary.clone(), spare.clone()]);
    let mut request = ModelRequest::new(Vec::new());
    request.model = Some("requested-model".into());
    assert_eq!(
        chain.invoke(&(), request).await.unwrap().text(),
        "requested-model"
    );
    assert_eq!(primary.calls.load(Ordering::SeqCst), 1);
    assert_eq!(secondary.calls.load(Ordering::SeqCst), 1);
    assert_eq!(spare.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn streaming_falls_back_before_any_content_and_retains_the_terminal_response() {
    let primary = reply(true);
    let secondary = reply(false);
    let chain = FallbackModel::new(primary, vec![secondary]);
    let mut stream = chain
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();
    let mut completed = None;
    while let Some(item) = stream.next().await {
        if let ModelStreamItem::Completed(response) = item {
            completed = Some(response);
        }
    }
    assert_eq!(completed.unwrap().text(), "answer");
}

#[tokio::test]
async fn exhausting_the_chain_returns_the_last_failure() {
    let chain = FallbackModel::new(reply(true), vec![reply(true)]);
    assert!(
        chain
            .invoke(&(), ModelRequest::new(Vec::new()))
            .await
            .is_err()
    );
    assert!(
        chain
            .stream(&(), ModelRequest::new(Vec::new()))
            .await
            .is_err()
    );
}

struct Scripted {
    items: Vec<ModelStreamItem>,
}
#[async_trait::async_trait]
impl ChatModel<()> for Scripted {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }
    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        Ok(ModelStream::new(Box::pin(futures::stream::iter(
            self.items.clone(),
        ))))
    }
}

#[tokio::test]
async fn started_then_failed_falls_back_without_leaking_failed_attempt_items() {
    let spare = reply(false);
    let chain = FallbackModel::new(
        Arc::new(Scripted {
            items: vec![
                ModelStreamItem::Started,
                ModelStreamItem::Failed("down".into()),
            ],
        }),
        vec![spare.clone()],
    );
    let items: Vec<_> = chain
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap()
        .collect()
        .await;
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, ModelStreamItem::Started))
            .count(),
        1
    );
    assert!(
        !items
            .iter()
            .any(|item| matches!(item, ModelStreamItem::Failed(_)))
    );
    assert_eq!(spare.calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn failure_after_visible_content_never_calls_a_fallback() {
    let spare = reply(false);
    let chain = FallbackModel::new(
        Arc::new(Scripted {
            items: vec![
                ModelStreamItem::Started,
                ModelStreamItem::MessageDelta(crate::message::MessageDelta::text("partial")),
                ModelStreamItem::Failed("down".into()),
            ],
        }),
        vec![spare.clone()],
    );
    let items: Vec<_> = chain
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap()
        .collect()
        .await;
    assert!(matches!(items.last(), Some(ModelStreamItem::Failed(message)) if message == "down"));
    assert_eq!(spare.calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn provider_failure_after_started_falls_back_but_bad_input_does_not() {
    for status in [503, 400] {
        let spare = reply(false);
        let failure = ProviderError {
            status: Some(status),
            message: "provider refused".into(),
            ..Default::default()
        };
        let chain = FallbackModel::new(
            Arc::new(Scripted {
                items: vec![
                    ModelStreamItem::Started,
                    ModelStreamItem::ProviderFailed(failure),
                ],
            }),
            vec![spare.clone()],
        );
        let result = chain.stream(&(), ModelRequest::new(Vec::new())).await;
        if status == 503 {
            assert_eq!(
                collect_model_stream(result.unwrap()).await.unwrap().text(),
                "answer"
            );
            assert_eq!(spare.calls.load(Ordering::SeqCst), 1);
        } else {
            assert!(
                matches!(result, Err(crate::Error::Provider(error)) if error.status == Some(400))
            );
            assert_eq!(spare.calls.load(Ordering::SeqCst), 0);
        }
    }
}

#[tokio::test]
async fn tool_and_reasoning_events_commit_to_the_current_attempt() {
    let events = vec![
        ModelStreamItem::ToolCallDelta(crate::tool::ToolDelta {
            call_id: "call-1".into(),
            content: "{}".into(),
            tool_name: Some("lookup".into()),
            content_index: None,
        }),
        ModelStreamItem::MessageDelta(crate::message::MessageDelta::reasoning("thinking")),
        ModelStreamItem::BlockStart {
            index: 0,
            kind: BlockKind::Text,
        },
        ModelStreamItem::BlockDelta {
            index: 0,
            delta: BlockDelta::Text("fragment".into()),
        },
    ];
    for event in events {
        let spare = reply(false);
        let chain = FallbackModel::new(
            Arc::new(Scripted {
                items: vec![
                    event,
                    ModelStreamItem::ProviderFailed(ProviderError {
                        status: Some(503),
                        message: "failed after output".into(),
                        ..Default::default()
                    }),
                ],
            }),
            vec![spare.clone()],
        );
        let items: Vec<_> = chain
            .stream(&(), ModelRequest::new(Vec::new()))
            .await
            .unwrap()
            .collect()
            .await;
        assert!(
            matches!(items.last(), Some(ModelStreamItem::ProviderFailed(error)) if error.status == Some(503))
        );
        assert_eq!(spare.calls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn missing_terminal_before_content_falls_back_and_after_content_is_a_failure() {
    for content in [false, true] {
        let spare = reply(false);
        let mut items = vec![ModelStreamItem::Started];
        if content {
            items.push(ModelStreamItem::MessageDelta(
                crate::message::MessageDelta::text("partial"),
            ));
        }
        let chain = FallbackModel::new(Arc::new(Scripted { items }), vec![spare.clone()]);
        let stream = chain
            .stream(&(), ModelRequest::new(Vec::new()))
            .await
            .unwrap();
        let result = collect_model_stream(stream).await;
        if content {
            assert!(
                matches!(result, Err(crate::Error::Model(message)) if message.contains("without a terminal"))
            );
            assert_eq!(spare.calls.load(Ordering::SeqCst), 0);
        } else {
            assert_eq!(result.unwrap().text(), "answer");
            assert_eq!(spare.calls.load(Ordering::SeqCst), 1);
        }
    }
    let chain = FallbackModel::new(Arc::new(Scripted { items: Vec::new() }), Vec::new());
    assert!(
        matches!(chain.stream(&(), ModelRequest::new(Vec::new())).await, Err(crate::Error::Model(message)) if message.contains("without a terminal"))
    );
}

struct Invalid {
    unsupported: bool,
}
#[async_trait::async_trait]
impl ChatModel<()> for Invalid {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        Err(if self.unsupported {
            crate::Error::Unsupported("modality".into())
        } else {
            crate::Error::Validation("bad input".into())
        })
    }
}
#[tokio::test]
async fn validation_and_unsupported_errors_do_not_try_another_model() {
    for unsupported in [false, true] {
        let spare = reply(false);
        let chain = FallbackModel::new(Arc::new(Invalid { unsupported }), vec![spare.clone()]);
        assert!(
            chain
                .invoke(&(), ModelRequest::new(Vec::new()))
                .await
                .is_err()
        );
        assert!(
            chain
                .stream(&(), ModelRequest::new(Vec::new()))
                .await
                .is_err()
        );
        assert_eq!(spare.calls.load(Ordering::SeqCst), 0);
    }
}

struct Recording {
    fails: bool,
    requests: Arc<std::sync::Mutex<Vec<serde_json::Value>>>,
    response: ModelResponse,
}
#[async_trait::async_trait]
impl ChatModel<()> for Recording {
    async fn invoke(&self, _: &(), request: ModelRequest) -> crate::Result<ModelResponse> {
        self.requests
            .lock()
            .unwrap()
            .push(serde_json::to_value(request).unwrap());
        if self.fails {
            Err(crate::Error::Model("primary offline".into()))
        } else {
            Ok(self.response.clone())
        }
    }
    async fn stream(&self, state: &(), request: ModelRequest) -> crate::Result<ModelStream> {
        let response = self.invoke(state, request).await?;
        let metadata = ModelStreamMetadata {
            correlation: response.correlation.clone(),
            resolved_route: response.resolved_route.clone(),
        };
        Ok(ModelStream::new(Box::pin(futures::stream::iter(vec![
            ModelStreamItem::Started,
            ModelStreamItem::Completed(response),
        ])))
        .with_metadata(metadata))
    }
}

#[tokio::test]
async fn options_requests_response_fields_and_selected_stream_metadata_are_preserved() {
    let requests = Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut response = ModelResponse::assistant("secondary answer");
    response.raw = Some(serde_json::json!({"marker": "provider envelope"}));
    response.correlation = Some(ModelCallCorrelation::new("run-1", "call-1"));
    response.resolved_route = Some(ResolvedModelRoute::new("secondary", "model-b", "route-b"));
    response.continue_turn = Some("continue-b".into());
    let primary = Arc::new(Recording {
        fails: true,
        requests: requests.clone(),
        response: response.clone(),
    });
    let secondary = Arc::new(Recording {
        fails: false,
        requests: requests.clone(),
        response: response.clone(),
    });
    let chain = FallbackModel::new(primary, vec![secondary]);
    let mut request = ModelRequest::new(vec![crate::message::Message::user("hello")]);
    request.provider_options = serde_json::json!({"keep": [1, 2, 3]});
    request.metadata = serde_json::json!({"session": "a"});
    request.requested_route = Some("primary-route".into());
    request.temperature = Some(0.42);
    request.correlation = response.correlation.clone();
    request.max_tokens = Some(17);
    request.timeout_ms = Some(200);
    let expected = serde_json::to_value(&request).unwrap();
    assert_eq!(
        serde_json::to_value(chain.invoke(&(), request.clone()).await.unwrap()).unwrap(),
        serde_json::to_value(&response).unwrap()
    );
    let stream = chain.stream(&(), request).await.unwrap();
    assert_eq!(stream.metadata().correlation, response.correlation);
    assert_eq!(stream.metadata().resolved_route, response.resolved_route);
    assert_eq!(
        serde_json::to_value(collect_model_stream(stream).await.unwrap()).unwrap(),
        serde_json::to_value(response).unwrap()
    );
    assert_eq!(*requests.lock().unwrap(), vec![expected.clone(); 4]);
}

struct Profiled {
    profile: ModelProfile,
    input: bool,
}
#[async_trait::async_trait]
impl ChatModel<()> for Profiled {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }
    fn supports_input(&self, _: InputModality, _: &str, _: InputSource) -> bool {
        self.input
    }
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }
}
#[test]
fn mixed_profiles_never_claim_one_providers_tool_or_schema_dialect() {
    let mut first = ModelProfile {
        provider: Some("first".into()),
        tool_calling: true,
        json_schema: true,
        max_input_tokens: Some(100),
        ..Default::default()
    };
    first.modalities.image_in = true;
    let second = ModelProfile {
        provider: Some("second".into()),
        tool_calling: true,
        json_schema: true,
        max_input_tokens: Some(50),
        ..Default::default()
    };
    let primary = Arc::new(Profiled {
        profile: first.clone(),
        input: true,
    });
    let same = FallbackModel::new(primary.clone(), vec![primary.clone()]);
    assert_eq!(same.profile(), Some(&first));
    let mixed = FallbackModel::new(
        primary,
        vec![Arc::new(Profiled {
            profile: second,
            input: false,
        })],
    );
    let profile = mixed.profile().unwrap();
    assert!(!profile.tool_calling);
    assert!(!profile.json_schema);
    assert!(!profile.modalities.image_in);
    assert_eq!(profile.max_input_tokens, Some(50));
    assert!(profile.provider.is_none());
    assert!(!mixed.supports_input(InputModality::Image, "image/png", InputSource::Url));
    assert!(mixed.cache_identity().is_none());
}

struct Cancellable {
    dropped: Arc<std::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
}
struct SignalDrop(Option<tokio::sync::oneshot::Sender<()>>);
impl Drop for SignalDrop {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}
#[async_trait::async_trait]
impl ChatModel<()> for Cancellable {
    async fn invoke(&self, _: &(), _: ModelRequest) -> crate::Result<ModelResponse> {
        unreachable!()
    }
    async fn stream(&self, _: &(), _: ModelRequest) -> crate::Result<ModelStream> {
        let sender = self.dropped.lock().unwrap().take();
        let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(async move {
            let _guard = SignalDrop(sender);
            ready_tx.send(()).unwrap();
            futures::future::pending::<()>().await;
        });
        ready_rx.await.unwrap();
        let items = futures::stream::iter(vec![ModelStreamItem::MessageDelta(
            crate::message::MessageDelta::text("first"),
        )])
        .chain(futures::stream::pending());
        Ok(ModelStream::new(Box::pin(items)).abort_on_drop(AbortOnDrop::from_join_handle(&task)))
    }
}
#[tokio::test]
async fn dropping_selected_stream_cancels_its_original_producer() {
    let (sender, receiver) = tokio::sync::oneshot::channel();
    let model = Arc::new(Cancellable {
        dropped: Arc::new(std::sync::Mutex::new(Some(sender))),
    });
    let chain = FallbackModel::new(model, vec![reply(false)]);
    let stream = chain
        .stream(&(), ModelRequest::new(Vec::new()))
        .await
        .unwrap();
    drop(stream);
    receiver.await.unwrap();
}

#[tokio::test]
async fn empty_success_and_deferral_stop_the_chain_without_more_attempts() {
    for terminal in [
        ModelStreamItem::Completed(ModelResponse::assistant("")),
        ModelStreamItem::Deferred(DeferredHandle::new("provider", "job-1")),
    ] {
        let spare = reply(false);
        let expected = serde_json::to_value(&terminal).unwrap();
        let chain = FallbackModel::new(
            Arc::new(Scripted {
                items: vec![ModelStreamItem::Started, terminal],
            }),
            vec![spare.clone()],
        );
        let items: Vec<_> = chain
            .stream(&(), ModelRequest::new(Vec::new()))
            .await
            .unwrap()
            .collect()
            .await;
        assert_eq!(
            serde_json::to_value(items.last().unwrap()).unwrap(),
            expected
        );
        assert_eq!(spare.calls.load(Ordering::SeqCst), 0);
    }
}

#[tokio::test]
async fn exhausting_stream_attempts_preserves_the_last_structured_failure() {
    let last = ProviderError {
        provider: "backup".into(),
        status: Some(503),
        code: Some("outage".into()),
        message: "final attempt".into(),
        retry_after_ms: Some(20),
        raw: Some(serde_json::json!({"detail":"last"})),
        ..Default::default()
    };
    let chain = FallbackModel::new(
        Arc::new(Scripted {
            items: vec![ModelStreamItem::Failed("first failed".into())],
        }),
        vec![Arc::new(Scripted {
            items: vec![ModelStreamItem::ProviderFailed(last.clone())],
        })],
    );
    assert!(
        matches!(chain.stream(&(), ModelRequest::new(Vec::new())).await, Err(crate::Error::Provider(error)) if *error == last)
    );
}
