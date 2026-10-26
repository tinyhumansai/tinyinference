use super::*;
use futures::StreamExt;
use serde_json::json;

async fn collect(chunks: Vec<&str>) -> Vec<ModelStreamItem> {
    stream_from_bytes(
        chunks.into_iter().map(|c| c.as_bytes().to_vec()).collect(),
        "m",
    )
    .collect()
    .await
}

fn completed(items: &[ModelStreamItem]) -> &ModelResponse {
    match items.last() {
        Some(ModelStreamItem::Completed(response)) => response,
        other => panic!("expected Completed, got {other:?}"),
    }
}

#[tokio::test]
async fn generic_text_stream_without_done_sentinel() {
    let items = collect(vec![
        "data: {\"index\":0,\"message\":{\"role\":\"ASSISTANT\",\"content\":[{\"type\":\"TEXT\",\"text\":\"Hel\"}]}}\n\n",
        "data: {\"index\":0,\"message\":{\"role\":\"ASSISTANT\",\"content\":[{\"type\":\"TEXT\",\"text\":\"lo\"}]}}\n\n",
        "data: {\"finishReason\":\"stop\"}\n\n",
        "data: {\"usage\":{\"promptTokens\":5,\"completionTokens\":2,\"totalTokens\":7}}\n",
    ])
    .await;
    assert!(matches!(items[0], ModelStreamItem::Started));
    let response = completed(&items);
    assert_eq!(
        response.message.content,
        vec![ContentBlock::Text("Hello".into())]
    );
    assert_eq!(response.finish_reason.as_deref(), Some("stop"));
    assert_eq!(response.usage.unwrap().total_tokens, 7);
    assert_eq!(response.message.origin.as_ref().unwrap().provider, "oci");
    let deltas = items
        .iter()
        .filter(|item| matches!(item, ModelStreamItem::MessageDelta(_)))
        .count();
    assert_eq!(deltas, 2);
}

#[tokio::test]
async fn done_sentinel_and_split_chunks_are_handled() {
    let line =
        "data: {\"index\":0,\"message\":{\"content\":[{\"type\":\"TEXT\",\"text\":\"é\"}]}}\n";
    let bytes = line.as_bytes();
    let (a, b) = bytes.split_at(bytes.len() / 2 + 3);
    let items = stream_from_bytes(
        vec![
            a.to_vec(),
            b.to_vec(),
            b"data: {\"finishReason\":\"stop\"}\r\ndata: [DONE]\n".to_vec(),
        ],
        "m",
    )
    .collect::<Vec<_>>()
    .await;
    assert_eq!(
        completed(&items).message.content,
        vec![ContentBlock::Text("é".into())]
    );
}

#[tokio::test]
async fn generic_tool_call_fragments_accumulate() {
    let items = collect(vec![
        "data: {\"message\":{\"toolCalls\":[{\"id\":\"c1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"ci\"}]}}\n",
        "data: {\"message\":{\"toolCalls\":[{\"arguments\":\"ty\\\":\\\"Rome\\\"}\"}]}}\n",
        "data: {\"finishReason\":\"tool_calls\"}\n",
    ])
    .await;
    let response = completed(&items);
    assert_eq!(response.message.tool_calls.len(), 1);
    assert_eq!(response.message.tool_calls[0].id, "c1");
    assert_eq!(
        response.message.tool_calls[0].arguments,
        json!({"city": "Rome"})
    );
    let deltas: Vec<_> = items
        .iter()
        .filter_map(|item| match item {
            ModelStreamItem::ToolCallDelta(delta) => Some(delta.call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(deltas, vec!["c1", "c1"]);
}

#[tokio::test]
async fn cohere_final_chunk_does_not_duplicate_text() {
    let items = collect(vec![
        "data: {\"apiFormat\":\"COHERE\",\"text\":\"Hi \"}\n",
        "data: {\"apiFormat\":\"COHERE\",\"text\":\"there\"}\n",
        "data: {\"apiFormat\":\"COHERE\",\"text\":\"Hi there\",\"finishReason\":\"COMPLETE\"}\n",
    ])
    .await;
    let response = completed(&items);
    assert_eq!(
        response.message.content,
        vec![ContentBlock::Text("Hi there".into())]
    );
    assert_eq!(response.finish_reason.as_deref(), Some("COMPLETE"));
}

#[tokio::test]
async fn eof_without_finish_reason_is_a_provider_failure_with_partial() {
    let items = collect(vec![
        "data: {\"message\":{\"content\":[{\"type\":\"TEXT\",\"text\":\"partial\"}]}}\n",
    ])
    .await;
    match items.last() {
        Some(ModelStreamItem::ProviderFailed(error)) => {
            assert_eq!(error.provider, "oci");
            let partial = error.partial_message.as_ref().unwrap();
            assert_eq!(partial.content, vec![ContentBlock::Text("partial".into())]);
        }
        other => panic!("expected failure, got {other:?}"),
    }
}

#[tokio::test]
async fn error_event_and_bad_json_fail_the_stream() {
    let items = collect(vec![
        "data: {\"code\":\"TooManyRequests\",\"message\":\"slow down\"}\n",
    ])
    .await;
    match items.last() {
        Some(ModelStreamItem::ProviderFailed(error)) => {
            assert_eq!(error.code.as_deref(), Some("TooManyRequests"));
            assert_eq!(error.message, "slow down");
        }
        other => panic!("expected failure, got {other:?}"),
    }
    let items = collect(vec!["data: {not json}\n"]).await;
    assert!(matches!(
        items.last(),
        Some(ModelStreamItem::ProviderFailed(_))
    ));
}
