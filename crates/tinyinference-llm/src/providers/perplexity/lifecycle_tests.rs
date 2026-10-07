use super::*;

use super::tests::{complete, fixture, json_reply};
use crate::{
    message::Message,
    model::{DeferredStatus, ExecutionStatus},
};
use serde_json::json;

#[tokio::test]
async fn responses_alias_submits_once_and_uses_canonical_agent_recovery_routes() {
    let mut queued = complete();
    queued["status"] = json!("queued");
    queued["output"] = json!([]);
    let mut cancelled = queued.clone();
    cancelled["status"] = json!("cancelled");
    let (mut model, scripted) = fixture(vec![
        json_reply(200, queued.clone()),
        json_reply(200, queued),
        json_reply(200, json!({"response_id":"resp_1","status":"cancelling"})),
        json_reply(200, cancelled),
    ]);
    Arc::get_mut(&mut model.inner)
        .unwrap()
        .config
        .responses_alias = true;
    let handle = model
        .submit_background(ModelRequest::new(vec![Message::user("question")]))
        .await
        .unwrap();
    assert_eq!(handle.id, "resp_1");
    assert!(matches!(
        <PerplexityModel as ChatModel<()>>::fetch_deferred(&model, &handle)
            .await
            .unwrap(),
        DeferredStatus::Pending
    ));
    assert_eq!(
        model.cancel_background(&handle).await.unwrap(),
        ExecutionStatus::Cancelling
    );
    assert!(matches!(
        <PerplexityModel as ChatModel<()>>::fetch_deferred(&model, &handle)
            .await
            .unwrap(),
        DeferredStatus::Cancelled { .. }
    ));
    let requests = scripted.requests.lock().unwrap();
    assert_eq!(
        requests
            .iter()
            .filter(|r| r.method() == reqwest::Method::POST && r.url().path() == "/v1/responses")
            .count(),
        1
    );
    assert_eq!(requests[0].url().path(), "/v1/responses");
    assert_eq!(requests[1].url().path(), "/v1/agent/resp_1");
    assert_eq!(requests[2].url().path(), "/v1/agent/resp_1/cancel");
}

fn handle(model: &PerplexityModel) -> crate::model::DeferredHandle {
    let response = super::response::parse(complete()).unwrap();
    let mut handle = model.response_handle(&response).unwrap();
    handle.kind = Some("background".into());
    handle
}

#[tokio::test]
async fn file_listing_and_download_use_file_ids_without_treating_filenames_as_paths() {
    let bytes = http::Response::builder()
        .header("content-type", "text/csv")
        .body("a,b\n1,2\n")
        .unwrap()
        .into();
    let (model, scripted) = fixture(vec![
        json_reply(
            200,
            json!({"data":[{"id":"file/1","filename":"../../report.csv","bytes":8,"created_at":7}]}),
        ),
        Ok(bytes),
    ]);
    let handle = handle(&model);
    let files = model.list_response_files(&handle).await.unwrap();
    assert_eq!(files[0].filename, "../../report.csv");
    assert_eq!(files[0].response_id, "resp_1");
    let content = model
        .download_response_file(&handle, &files[0].id)
        .await
        .unwrap();
    assert_eq!(content.bytes, b"a,b\n1,2\n");
    assert_eq!(content.media_type.as_deref(), Some("text/csv"));
    let requests = scripted.requests.lock().unwrap();
    assert_eq!(
        requests[1].url().path(),
        "/v1/agent/resp_1/files/file%2F1/content"
    );
    assert!(requests.iter().all(|r| r.method() == reqwest::Method::GET));
}

#[tokio::test]
async fn foreign_handles_invalid_ids_and_duplicate_files_fail_clearly() {
    let (model, scripted) = fixture(vec![]);
    let mut foreign = handle(&model);
    foreign
        .metadata
        .insert("base_url".into(), json!("https://attacker.example"));
    assert!(matches!(
        model.retrieve_response(&foreign).await,
        Err(Error::Validation(_))
    ));
    let mut invalid = handle(&model);
    invalid.id = "..".into();
    assert!(model.list_response_files(&invalid).await.is_err());
    assert!(scripted.requests.lock().unwrap().is_empty());
    let (model, _) = fixture(vec![json_reply(
        200,
        json!({"data":[
            {"id":"f","filename":"one","bytes":1,"created_at":1},
            {"id":"f","filename":"two","bytes":1,"created_at":1}
        ]}),
    )]);
    assert!(model.list_response_files(&handle(&model)).await.is_err());
}

#[tokio::test]
async fn oversized_file_and_failed_download_never_regenerate_the_run() {
    let (mut model, scripted) = fixture(vec![
        Ok(http::Response::new("123456").into()),
        json_reply(404, json!({"error":{"message":"missing file"}})),
    ]);
    Arc::get_mut(&mut model.inner)
        .unwrap()
        .config
        .max_file_bytes = 3;
    let handle = handle(&model);
    let first = model
        .download_response_file(&handle, "f")
        .await
        .unwrap_err();
    assert!(
        matches!(first,Error::Provider(error) if error.code.as_deref()==Some("response_too_large"))
    );
    assert!(
        model
            .download_response_file(&handle, "missing")
            .await
            .is_err()
    );
    assert!(
        scripted
            .requests
            .lock()
            .unwrap()
            .iter()
            .all(|r| r.method() == reqwest::Method::GET)
    );
}

#[tokio::test]
async fn reconnect_uses_saved_cursor_and_keeps_the_authoritative_full_response() {
    use futures::StreamExt;
    let mut progress = complete();
    progress["status"] = json!("in_progress");
    progress["output"] = json!([]);
    let body=[
        json!({"type":"response.output_text.delta","sequence_number":3,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"old"}),
        json!({"type":"response.output_text.delta","sequence_number":4,"output_index":0,"content_index":0,"item_id":"msg_1","delta":"swer"}),
        json!({"type":"response.completed","sequence_number":5,"response":complete()}),
    ].into_iter().map(|event|format!("data: {event}\n\n")).collect::<String>();
    let incoming = http::Response::builder()
        .header("content-type", "text/event-stream")
        .body(body)
        .unwrap()
        .into();
    let (model, scripted) = fixture(vec![json_reply(200, progress), Ok(incoming)]);
    let items = model
        .resume_background(&handle(&model), Some(3))
        .await
        .unwrap()
        .collect::<Vec<_>>()
        .await;
    let deltas = items
        .iter()
        .filter_map(|i| {
            if let crate::model::ModelStreamItem::MessageDelta(d) = i {
                Some(d.text.as_str())
            } else {
                None
            }
        })
        .collect::<String>();
    assert_eq!(deltas, "swer");
    let Some(crate::model::ModelStreamItem::Completed(response)) = items.last() else {
        panic!("missing completion {items:?}");
    };
    assert_eq!(response.text(), "answer");
    assert_eq!(
        scripted.requests.lock().unwrap()[1].url().query(),
        Some("stream=true&starting_after=3")
    );
}

#[tokio::test]
async fn expired_reconnect_and_cancel_errors_are_not_replayed() {
    let mut progress = complete();
    progress["status"] = json!("in_progress");
    progress["output"] = json!([]);
    let (model, scripted) = fixture(vec![
        json_reply(200, progress),
        json_reply(400, json!({"error":{"message":"expired window"}})),
        json_reply(503, json!({})),
    ]);
    let handle = handle(&model);
    assert!(
        matches!(model.resume_background(&handle,Some(3)).await,Err(Error::Provider(error)) if error.code.as_deref()==Some("reconnect_unavailable"))
    );
    assert!(model.cancel_background(&handle).await.is_err());
    let requests = scripted.requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests
            .iter()
            .filter(|r| r.method() == reqwest::Method::POST)
            .all(|r| r.url().path().ends_with("/cancel"))
    );
}

#[tokio::test]
async fn failed_and_incomplete_background_runs_keep_their_terminal_state() {
    let mut incomplete = complete();
    incomplete["status"] = json!("incomplete");
    incomplete["incomplete_details"] = json!({"reason":"max_output_tokens"});
    let mut failed = complete();
    failed["status"] = json!("failed");
    failed["error"] = json!({"message":"failed","code":"provider_error"});
    let (model, _) = fixture(vec![json_reply(200, incomplete), json_reply(200, failed)]);
    let handle = handle(&model);
    let DeferredStatus::Completed(response) =
        <PerplexityModel as ChatModel<()>>::fetch_deferred(&model, &handle)
            .await
            .unwrap()
    else {
        panic!("missing incomplete snapshot");
    };
    assert_eq!(response.finish_reason.as_deref(), Some("max_output_tokens"));
    assert!(matches!(
        <PerplexityModel as ChatModel<()>>::fetch_deferred(&model, &handle)
            .await
            .unwrap(),
        DeferredStatus::ProviderFailed(_)
    ));
}
