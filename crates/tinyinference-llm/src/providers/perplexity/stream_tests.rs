use super::*;

use super::tests::{complete, fixture};
use crate::message::Message;
use crate::model::{ModelStreamItem, collect_model_stream};
use bytes::Bytes;
use futures::{FutureExt, StreamExt};
use serde_json::{Value, json};

fn event(value: Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

fn created() -> Value {
    json!({"type":"response.created","sequence_number":0,
        "response":{"id":"resp_1","model":"openai/test","status":"in_progress","output":[]}})
}

#[tokio::test]
async fn live_provisional_model_and_null_message_content_complete_normally() {
    let items = run_events(vec![
        json!({"type":"response.in_progress","sequence_number":1,"response":{"id":"resp_1","model":"","status":"in_progress","output":[]}}),
        json!({"type":"response.output_item.added","sequence_number":4,"output_index":0,"item":{"content":null,"id":"msg_1","role":"assistant","status":"completed","type":"message"}}),
        json!({"type":"response.output_text.delta","sequence_number":5,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"answer"}),
        json!({"type":"response.completed","sequence_number":9,"response":complete()}),
    ], true).await;
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("live event sequence must complete: {items:?}");
    };
    assert_eq!(response.text(), "answer");
    assert_eq!(response.execution.as_ref().unwrap().model, "openai/test");
}

async fn run_events(events: Vec<Value>, split_bytes: bool) -> Vec<ModelStreamItem> {
    let body = events
        .into_iter()
        .flat_map(|v| event(v).to_vec())
        .collect::<Vec<_>>();
    let chunks = if split_bytes {
        body.into_iter()
            .map(|b| Bytes::from(vec![b]))
            .collect::<Vec<_>>()
    } else {
        vec![Bytes::from(body)]
    };
    let incoming = http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(reqwest::Body::wrap_stream(futures::stream::iter(
            chunks.into_iter().map(Ok::<_, std::io::Error>),
        )))
        .unwrap()
        .into();
    let (model, _) = fixture(vec![Ok(incoming)]);
    model
        .stream(&(), ModelRequest::new(vec![Message::user("q")]))
        .await
        .unwrap()
        .collect()
        .await
}

#[tokio::test]
async fn split_utf8_and_final_snapshots_do_not_repeat_text_or_charges() {
    let mut final_response = complete();
    final_response["output"][0]["content"][0]["text"] = json!("Café");
    final_response["usage"] = json!({"input_tokens":4,"output_tokens":2,"total_tokens":6});
    let item = final_response["output"][0].clone();
    let items=run_events(vec![created(),
        json!({"type":"response.output_item.added","sequence_number":1,"output_index":0,
            "item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}),
        json!({"type":"response.output_text.delta","sequence_number":2,"output_index":0,"content_index":0,"delta":"Caf"}),
        json!({"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"delta":"é"}),
        json!({"type":"response.output_text.done","sequence_number":4,"output_index":0,"content_index":0,"text":"Café"}),
        json!({"type":"response.output_item.done","sequence_number":5,"output_index":0,"item":item}),
        json!({"type":"response.completed","sequence_number":6,"response":final_response}),
        json!({"type":"response.completed","sequence_number":6,"response":final_response}),
    ],true).await;
    let text = items
        .iter()
        .filter_map(|item| {
            if let ModelStreamItem::MessageDelta(delta) = item {
                Some(delta.text.as_str())
            } else {
                None
            }
        })
        .collect::<String>();
    assert_eq!(text, "Café");
    assert_eq!(
        items
            .iter()
            .filter(|i| matches!(i, ModelStreamItem::Completed(_)))
            .count(),
        1
    );
    let Some(ModelStreamItem::Completed(result)) = items.last() else {
        panic!("missing completion: {items:?}");
    };
    assert_eq!(result.usage.unwrap().total_tokens, 6);
    assert_eq!(
        result.output,
        super::response::parse(final_response).unwrap().output
    );
}

#[tokio::test]
async fn function_argument_fragments_preserve_call_ids_and_signatures() {
    let call = json!({"type":"function_call","id":"fc1","call_id":"call1","name":"lookup","arguments":"{\"id\":7}","thought_signature":"signature"});
    let items=run_events(vec![created(),
        json!({"type":"response.output_item.added","sequence_number":1,"output_index":0,
            "item":{"type":"function_call","id":"fc1","call_id":"call1","name":"lookup","arguments":""}}),
        json!({"type":"response.function_call_arguments.delta","sequence_number":2,"item_id":"fc1","delta":"{\"id\":"}),
        json!({"type":"response.function_call_arguments.delta","sequence_number":3,"item_id":"fc1","delta":"7}"}),
        json!({"type":"response.output_item.done","sequence_number":4,"output_index":0,"item":call}),
        json!({"type":"response.completed","sequence_number":5,"response":{"id":"resp_1","model":"openai/test","status":"completed","output":[call]}}),
    ],true).await;
    let args = items
        .iter()
        .filter_map(|item| {
            if let ModelStreamItem::ToolCallDelta(delta) = item {
                assert_eq!(delta.call_id, "call1");
                Some(delta.content.as_str())
            } else {
                None
            }
        })
        .collect::<String>();
    assert_eq!(args, "{\"id\":7}");
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("missing completion: {items:?}");
    };
    assert_eq!(
        response.message.tool_calls[0]
            .replay
            .as_ref()
            .unwrap()
            .thought_signature
            .as_deref(),
        Some("signature")
    );
}

#[tokio::test]
async fn truncated_stream_keeps_received_text_and_recovery_identity() {
    let items=run_events(vec![created(),
        json!({"type":"response.output_text.delta","sequence_number":1,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"partial"}),
    ],false).await;
    let Some(ModelStreamItem::ProviderFailed(error)) = items.last() else {
        panic!("truncation must fail");
    };
    assert_eq!(error.code.as_deref(), Some("truncated_stream"));
    let partial = error.partial_response.as_ref().unwrap();
    assert_eq!(partial.text(), "partial");
    assert_eq!(partial.execution.as_ref().unwrap().id, "resp_1");
    assert_eq!(partial.execution.as_ref().unwrap().sequence_number, Some(1));
    assert_eq!(
        items
            .iter()
            .filter(|i| matches!(i, ModelStreamItem::ProviderFailed(_)))
            .count(),
        1
    );
}

#[tokio::test]
async fn malformed_events_and_changed_identities_fail_without_panicking() {
    for bad in [
        json!(7),
        json!({"type":"response.output_text.delta","output_index":0,"content_index":5000,"delta":"x"}),
        json!({"type":"response.completed","response":{"id":"other","model":"openai/test","status":"completed","output":[]}}),
        json!({"type":"response.output_item.done","output_index":0,"item":{"type":"message","content":42}}),
    ] {
        let items = run_events(vec![created(), bad], false).await;
        assert!(matches!(
            items.last(),
            Some(ModelStreamItem::ProviderFailed(_))
        ));
    }
}

#[tokio::test]
async fn unknown_events_and_output_items_are_preserved() {
    let unknown = json!({"type":"future_item","opaque":{"a":1}});
    let items=run_events(vec![created(),json!({"type":"future_event","sequence_number":1,"opaque":"value","response":"future representation"}),
        json!({"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","model":"openai/test","status":"completed","output":[unknown]}})
    ],false).await;
    assert!(items.iter().any(|item|matches!(item,ModelStreamItem::OutputEvent(crate::ModelOutputEvent::Progress(crate::ModelProgress::Extension {event_type,..})) if event_type=="future_event")));
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("missing completion");
    };
    assert!(matches!(
        response.output[0].kind,
        crate::ModelOutputKind::Extension { .. }
    ));
    assert!(
        matches!(&response.execution.as_ref().unwrap().progress[0],crate::ModelProgress::Extension {data,..} if data["opaque"]=="value")
    );
}

#[tokio::test]
async fn crlf_and_multiline_sse_data_are_decoded() {
    let body = format!(
        "event: response.created\r\ndata: {{\r\ndata: \"response\":{{\"id\":\"resp_1\",\"model\":\"openai/test\",\"status\":\"in_progress\",\"output\":[]}}\r\ndata: }}\r\n\r\ndata: {}\r\n\r\ndata: [DONE]\r\n\r\n",
        json!({"type":"response.completed","response":complete()})
    );
    let incoming = http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(body)
        .unwrap()
        .into();
    let (model, _) = fixture(vec![Ok(incoming)]);
    let response = collect_model_stream(
        model
            .stream(&(), ModelRequest::new(vec![Message::user("q")]))
            .await
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(response.text(), "answer");
}

#[tokio::test(start_paused = true)]
async fn stream_idle_timeout_and_event_limit_are_terminal_failures() {
    let incoming = http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(reqwest::Body::wrap_stream(futures::stream::pending::<
            std::result::Result<Bytes, std::io::Error>,
        >()))
        .unwrap()
        .into();
    let (mut model, _) = fixture(vec![Ok(incoming)]);
    Arc::get_mut(&mut model.inner).unwrap().config.idle_timeout = std::time::Duration::from_secs(1);
    let items = model
        .stream(&(), ModelRequest::new(vec![Message::user("q")]))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(items.last(),Some(ModelStreamItem::ProviderFailed(error)) if error.code.as_deref()==Some("timeout"))
    );
    let incoming = http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(event(created()))
        .unwrap()
        .into();
    let (mut model, _) = fixture(vec![Ok(incoming)]);
    Arc::get_mut(&mut model.inner)
        .unwrap()
        .config
        .max_event_bytes = 16;
    let items = model
        .stream(&(), ModelRequest::new(vec![Message::user("q")]))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(_))
    ));
}

#[tokio::test]
async fn contradictory_final_text_keeps_the_text_already_delivered_in_partial_failure() {
    for final_event in [
        json!({"type":"response.completed","sequence_number":2,"response":complete()}),
        json!({"type":"response.output_text.done","sequence_number":2,"output_index":0,"content_index":0,"text":"different"}),
        json!({"type":"response.content_part.done","sequence_number":2,"output_index":0,"content_index":0,"part":{"type":"output_text","text":"different"}}),
    ] {
        let items=run_events(vec![created(),
        json!({"type":"response.output_text.delta","sequence_number":1,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"delivered"}),
        final_event
    ],false).await;
        let Some(ModelStreamItem::ProviderFailed(error)) = items.last() else {
            panic!("contradiction must fail");
        };
        assert_eq!(error.partial_response.as_ref().unwrap().text(), "delivered");
    }
}

#[tokio::test]
async fn dropping_a_stream_releases_its_body_without_remote_cancellation() {
    let (sender, receiver) = futures::channel::mpsc::unbounded::<Bytes>();
    let incoming = http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(reqwest::Body::wrap_stream(
            receiver.map(Ok::<_, std::io::Error>),
        ))
        .unwrap()
        .into();
    let (model, scripted) = fixture(vec![Ok(incoming)]);
    let stream = model
        .stream(&(), ModelRequest::new(vec![Message::user("q")]))
        .await
        .unwrap();
    assert!(!sender.is_closed());
    drop(stream);
    assert!(sender.is_closed());
    assert_eq!(scripted.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn terminal_failure_preserves_previously_streamed_work() {
    let items=run_events(vec![created(),
        json!({"type":"response.output_text.delta","sequence_number":1,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"partial"}),
        json!({"type":"response.failed","sequence_number":2,"response":{"id":"resp_1","model":"openai/test","status":"failed","output":[],"error":{"message":"failed"}}})
    ],false).await;
    let Some(ModelStreamItem::ProviderFailed(error)) = items.last() else {
        panic!("missing failure");
    };
    assert_eq!(error.partial_response.as_ref().unwrap().text(), "partial");
    assert_eq!(
        error
            .partial_response
            .as_ref()
            .unwrap()
            .execution
            .as_ref()
            .unwrap()
            .status,
        crate::ExecutionStatus::Failed
    );
}

#[tokio::test]
async fn reasoning_summary_fragments_stay_separate_from_visible_text() {
    let reasoning = json!({"id":"reason_1","type":"reasoning","summary":[{"type":"summary_text","text":"considering"}]});
    let mut final_response = complete();
    final_response["output"]
        .as_array_mut()
        .unwrap()
        .insert(0, reasoning.clone());
    let items=run_events(vec![created(),
        json!({"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"reason_1","type":"reasoning","summary":[]}}),
        json!({"type":"response.reasoning_summary_text.delta","sequence_number":2,"output_index":0,"summary_index":0,"delta":"considering"}),
        json!({"type":"response.output_item.done","sequence_number":3,"output_index":0,"item":reasoning}),
        json!({"type":"response.completed","sequence_number":4,"response":final_response})
    ],true).await;
    let reasoning = items
        .iter()
        .filter_map(|i| {
            if let ModelStreamItem::MessageDelta(d) = i {
                Some(d.reasoning.as_str())
            } else {
                None
            }
        })
        .collect::<String>();
    assert_eq!(reasoning, "considering");
    let Some(ModelStreamItem::Completed(response)) = items.last() else {
        panic!("missing completion: {items:?}");
    };
    assert_eq!(response.text(), "answer");
}

#[tokio::test(start_paused = true)]
async fn text_arrives_before_completion_and_collecting_preserves_the_full_response() {
    let (sender, receiver) = futures::channel::mpsc::unbounded::<Bytes>();
    let incoming = http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(reqwest::Body::wrap_stream(
            receiver.map(Ok::<_, std::io::Error>),
        ))
        .unwrap()
        .into();
    let (model, _) = fixture(vec![Ok(incoming)]);
    sender
        .unbounded_send(event(json!({"type":"response.created","sequence_number":0,
        "response":{"id":"resp_1","model":"openai/test","status":"in_progress","output":[]}})))
        .unwrap();
    sender
        .unbounded_send(event(
            json!({"type":"response.output_item.added","sequence_number":1,"output_index":0,
        "item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}),
        ))
        .unwrap();
    sender
        .unbounded_send(event(
            json!({"type":"response.output_text.delta","sequence_number":2,"output_index":0,
        "content_index":0,"item_id":"msg_1","delta":"answer"}),
        ))
        .unwrap();
    let mut stream = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        model.stream(&(), ModelRequest::new(vec![Message::user("question")])),
    )
    .await
    .expect("opening the stream must not wait for its complete body")
    .unwrap();
    assert!(matches!(
        stream.next().await,
        Some(ModelStreamItem::Started)
    ));
    loop {
        match stream.next().await.unwrap() {
            ModelStreamItem::MessageDelta(delta) => {
                assert_eq!(delta.text, "answer");
                break;
            }
            ModelStreamItem::Completed(_) | ModelStreamItem::ProviderFailed(_) => {
                panic!("stream ended before text was delivered")
            }
            _ => {}
        }
    }
    while let Some(item) = stream.next().now_or_never() {
        assert!(matches!(item, Some(ModelStreamItem::OutputEvent(_))));
    }
    sender
        .unbounded_send(event(
            json!({"type":"response.completed","sequence_number":3,"response":complete()}),
        ))
        .unwrap();
    drop(sender);
    let result = collect_model_stream(stream).await.unwrap();
    assert_eq!(result.text(), "answer");
    assert_eq!(result.execution.unwrap().sequence_number, Some(3));
}
