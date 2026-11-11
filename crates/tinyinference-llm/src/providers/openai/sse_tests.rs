use super::*;
use serde_json::json;

/// Runs a synthetic SSE byte stream through the parser and returns every item.
async fn stream_items(raw: Vec<&'static [u8]>) -> Vec<ModelStreamItem> {
    use futures::StreamExt;

    let bytes = futures::stream::iter(
        raw.into_iter()
            .map(|v| Ok::<bytes::Bytes, crate::Error>(bytes::Bytes::from_static(v))),
    );
    let state = SseState {
        bytes: Box::pin(bytes),
        buf: Vec::new(),
        pending: VecDeque::new(),
        acc: OpenAiStreamAcc::default(),
        provider: "openai".to_string(),
        model: "gpt-4.1-mini".to_string(),
        started: false,
        finished: false,
        completion_seen: false,
        terminal_emitted: false,
    };
    futures::stream::unfold(state, sse_next).collect().await
}

async fn completed(raw: Vec<&'static [u8]>) -> ModelResponse {
    let items = stream_items(raw).await;
    items
        .into_iter()
        .find_map(|item| match item {
            ModelStreamItem::Completed(response) => Some(response),
            _ => None,
        })
        .expect("stream ends with Completed")
}

/// Gemini behind an OpenAI-compatible gateway reports the output cap as
/// `MAX_TOKENS`; the streamed response must carry the canonical `length`.
#[tokio::test]
async fn sse_stream_reports_an_output_limit_stop_as_length() {
    let response = completed(vec![
        b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"cut\"},\"finish_reason\":\"MAX_TOKENS\"}]}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;
    assert_eq!(response.text(), "cut");
    assert_eq!(response.finish_reason.as_deref(), Some("length"));
}

/// A gateway that bills the call sends its charge on its own SSE frame after
/// the usage chunk (`event: openhuman-metadata`). The streamed response must
/// carry it on `raw`, as the non-streaming response does, or a streaming
/// caller can never see what it was charged.
#[tokio::test]
async fn sse_stream_keeps_gateway_extension_frame_on_raw() {
    let response = completed(vec![
        b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"}]}\n\n",
        b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,\"total_tokens\":105}}\n\n",
        b"event: openhuman-metadata\ndata: {\"openhuman\":{\"usage\":{\"input_tokens\":100,\"output_tokens\":5,\"total_tokens\":105,\"cached_input_tokens\":80},\"billing\":{\"charged_amount_usd\":0.0000183}}}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;

    assert_eq!(response.text(), "hi");
    assert_eq!(response.usage.unwrap().total_tokens, 105);
    let raw = response.raw.as_ref().expect("extension frame kept on raw");
    assert_eq!(
        raw["openhuman"]["billing"]["charged_amount_usd"],
        json!(0.0000183)
    );
    assert_eq!(raw["openhuman"]["usage"]["cached_input_tokens"], json!(80));
    // Chunk-schema fields belong to the normalized response, not `raw`.
    assert!(raw.get("choices").is_none());
    assert!(raw.get("usage").is_none());
}

/// OpenRouter reports what a call cost inside `usage` (`cost`, `cost_details`,
/// `is_byok`). The non-streaming response keeps them on `raw`, under `usage`,
/// as part of the full wire body; a streamed response must keep them at the
/// same path, or a caller reading `/usage/cost` sees the charge only when it
/// did not stream. The token counts the normalized usage already reads stay
/// off `raw`.
#[tokio::test]
async fn sse_stream_keeps_usage_extension_fields_on_raw() {
    let items = stream_items(vec![
        b"data: {\"id\":\"c1\",\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}],\"usage\":null}\n\n",
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5,\"total_tokens\":105,\"prompt_tokens_details\":{\"cached_tokens\":80},\"cost\":0.000183,\"is_byok\":false,\"cost_details\":{\"upstream_inference_cost\":null}}}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;

    let usage_delta = items
        .iter()
        .find_map(|item| match item {
            ModelStreamItem::UsageDelta(usage) => Some(usage),
            _ => None,
        })
        .expect("usage frame emits a usage delta");
    assert_eq!(usage_delta.charged_amount_nanos, Some(183_000));

    let response = items
        .iter()
        .find_map(|item| match item {
            ModelStreamItem::Completed(response) => Some(response),
            _ => None,
        })
        .expect("stream ends with completed response");

    let usage = response.usage.as_ref().unwrap();
    assert_eq!(usage.total_tokens, 105);
    assert_eq!(usage.cache_read_tokens, 80);
    assert_eq!(usage.charged_amount_nanos, Some(183_000));
    let raw = response.raw.as_ref().expect("usage extensions kept on raw");
    assert_eq!(raw.pointer("/usage/cost"), Some(&json!(0.000183)));
    assert_eq!(raw["usage"]["is_byok"], json!(false));
    assert!(raw["usage"]["cost_details"].is_object());
    // Normalized fields are not duplicated onto `raw`.
    for key in [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
    ] {
        assert!(raw["usage"].get(key).is_none(), "{key} on raw");
    }
}

/// A later usage frame's extension fields replace an earlier one's, as
/// top-level extensions do (last value per key).
#[tokio::test]
async fn sse_stream_usage_extensions_last_value_wins() {
    let response = completed(vec![
        b"data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}],\"usage\":{\"prompt_tokens\":1,\"cost\":0.1}}\n\n",
        b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"cost\":0.2}}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;
    let raw = response.raw.as_ref().expect("usage extensions kept");
    assert_eq!(raw["usage"]["cost"], json!(0.2));
}

/// Extension fields riding on ordinary chunks are kept too (last value per
/// key).
#[tokio::test]
async fn sse_stream_keeps_chunk_extension_fields_last_value_wins() {
    let response = completed(vec![
        b"data: {\"provider\":\"BaseTen\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n",
        b"data: {\"provider\":\"Modal\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;
    let raw = response.raw.as_ref().expect("extension field kept");
    assert_eq!(raw["provider"], json!("Modal"));
}

/// A plain OpenAI stream carries only chunk-schema fields, so its response
/// keeps `raw: None` exactly as before.
#[tokio::test]
async fn sse_stream_without_extensions_has_no_raw() {
    let response = completed(vec![
        b"data: {\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"system_fingerprint\":\"fp\",\"choices\":[{\"delta\":{\"content\":\"a\"},\"finish_reason\":\"stop\"}]}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;
    assert!(response.raw.is_none());
}

/// A hostile `tool_calls[].index` must not size the slot vector. 2_000_000
/// (about 144 MB of slots) rather than `u32::MAX`, which would abort on OOM.
#[tokio::test]
async fn huge_tool_call_index_is_rejected() {
    let items = stream_items(vec![
        b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":2000000,\"id\":\"call-1\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;
    assert!(
        items
            .iter()
            .any(|item| matches!(item, ModelStreamItem::ProviderFailed(_))),
        "an out-of-range tool-call index was accepted"
    );
}

/// Concatenates the visible-text and reasoning fragments across every streamed
/// `MessageDelta`, as `(text, reasoning)`.
fn streamed_channels(items: &[ModelStreamItem]) -> (String, String) {
    let mut text = String::new();
    let mut reasoning = String::new();
    for item in items {
        if let ModelStreamItem::MessageDelta(delta) = item {
            text.push_str(&delta.text);
            reasoning.push_str(&delta.reasoning);
        }
    }
    (text, reasoning)
}

/// The text of every `Thinking` block on a response, concatenated.
fn thinking_text(response: &ModelResponse) -> String {
    response
        .message
        .content
        .iter()
        .filter_map(|block| block.as_thinking().map(|(text, _)| text.to_string()))
        .collect()
}

/// A thinking model behind an OpenAI-compatible gateway (Surplus serving
/// `glm-5.3-flash`) streams `delta.content` as a list of typed parts while it
/// thinks, and the chunk that closes the thinking also carries the first
/// visible words. These are the captured frames. That transition chunk used
/// to fail to deserialize and was silently dropped, so the answer lost its
/// opening ("ALPHA BRAVO CHARLIE" arrived as " CHARLIE").
#[tokio::test]
async fn sse_stream_keeps_the_answer_opening_carried_with_list_content_thinking() {
    let items = stream_items(vec![
        br#"data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":" user wants"}],"closed":false}]}}]}

"#,
        br#"data: {"id":"c1","choices":[{"index":0,"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"."}],"closed":true},{"type":"text","text":"ALPHA BRAVO"}]}}]}

"#,
        br#"data: {"id":"c1","choices":[{"index":0,"delta":{"content":" CHARLIE"},"finish_reason":"stop"}]}

"#,
        b"data: [DONE]\n\n",
    ])
    .await;

    let (text, reasoning) = streamed_channels(&items);
    assert_eq!(text, "ALPHA BRAVO CHARLIE");
    assert_eq!(reasoning, " user wants.");

    // Within the transition chunk the closing thought streams before the
    // first visible words, so the block order stays thinking -> text.
    let position = |wanted: &dyn Fn(&MessageDelta) -> bool| {
        items
            .iter()
            .position(|item| matches!(item, ModelStreamItem::MessageDelta(d) if wanted(d)))
            .expect("delta present")
    };
    assert!(position(&|d| d.reasoning == ".") < position(&|d| d.text == "ALPHA BRAVO"));

    let response = items
        .into_iter()
        .find_map(|item| match item {
            ModelStreamItem::Completed(response) => Some(response),
            _ => None,
        })
        .expect("stream ends with Completed");
    assert_eq!(response.text(), "ALPHA BRAVO CHARLIE");
    assert_eq!(thinking_text(&response), " user wants.");
    assert!(
        matches!(
            response.message.content.as_slice(),
            [ContentBlock::Thinking { .. }, ContentBlock::Text(_)]
        ),
        "{:?}",
        response.message.content
    );
}

/// A chunk whose list content holds only thinking parts feeds the reasoning
/// channel and never the visible text. The `thinking` field may be a plain
/// string as well as a list of text parts.
#[tokio::test]
async fn sse_stream_routes_a_thinking_only_list_content_to_reasoning() {
    let items = stream_items(vec![
        br#"data: {"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"weigh "}],"closed":false}]}}]}

"#,
        br#"data: {"choices":[{"delta":{"content":[{"type":"thinking","thinking":"it up","closed":true}]},"finish_reason":"stop"}]}

"#,
        b"data: [DONE]\n\n",
    ])
    .await;

    let (text, reasoning) = streamed_channels(&items);
    assert_eq!(text, "");
    assert_eq!(reasoning, "weigh it up");
    let response = items
        .into_iter()
        .find_map(|item| match item {
            ModelStreamItem::Completed(response) => Some(response),
            _ => None,
        })
        .expect("stream ends with Completed");
    assert_eq!(response.text(), "");
    assert_eq!(thinking_text(&response), "weigh it up");
}

/// Plain string content, the shape every non-thinking reply uses, streams
/// exactly as before: all visible text, no reasoning.
#[tokio::test]
async fn sse_stream_plain_string_content_is_all_visible_text() {
    let items = stream_items(vec![
        br#"data: {"choices":[{"delta":{"content":"ALPHA"}}]}

"#,
        br#"data: {"choices":[{"delta":{"content":" BRAVO"},"finish_reason":"stop"}]}

"#,
        b"data: [DONE]\n\n",
    ])
    .await;

    assert_eq!(
        streamed_channels(&items),
        ("ALPHA BRAVO".to_string(), String::new())
    );
}

/// A list mixing text, `reasoning` parts (carrying `text`), and a part type
/// this parser does not know: the text parts concatenate in order, the
/// reasoning part goes to the reasoning channel, and the unknown part is
/// skipped without losing the rest of the chunk.
#[tokio::test]
async fn sse_stream_list_content_skips_unknown_parts_and_keeps_the_rest() {
    let response = completed(vec![
        br#"data: {"choices":[{"delta":{"content":[{"type":"text","text":"one "},{"type":"image_url","image_url":{"url":"https://example.invalid/x.png"}},{"type":"reasoning","text":"aside"},{"type":"text","text":"two"}]},"finish_reason":"stop"}]}

"#,
        b"data: [DONE]\n\n",
    ])
    .await;

    assert_eq!(response.text(), "one two");
    assert_eq!(thinking_text(&response), "aside");
}

/// Formatted log output, captured for assertions on what was warned.
#[derive(Clone, Default)]
struct LogBuffer(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl LogBuffer {
    fn contents(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().unwrap()).into_owned()
    }
}

impl std::io::Write for LogBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for LogBuffer {
    type Writer = LogBuffer;

    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

/// A chunk that still does not fit the chunk schema is skipped, as before,
/// but no longer silently: it is warned about with a truncated copy of the
/// payload, and the rest of the stream still arrives.
#[tokio::test]
async fn sse_stream_warns_about_a_chunk_it_cannot_read() {
    let logs = LogBuffer::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(logs.clone())
        .with_ansi(false)
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);

    let unreadable: &'static [u8] = format!(
        "data: {{\"choices\":[{{\"index\":\"zero\",\"delta\":{{\"content\":\"{}\"}}}}]}}\n\n",
        "x".repeat(4_000)
    )
    .leak()
    .as_bytes();
    let response = completed(vec![
        b"data: {\"choices\":[{\"delta\":{\"content\":\"kept\"}}]}\n\n",
        unreadable,
        b"data: {\"choices\":[{\"delta\":{\"content\":\" too\"},\"finish_reason\":\"stop\"}]}\n\n",
        b"data: [DONE]\n\n",
    ])
    .await;

    assert_eq!(response.text(), "kept too");
    let logs = logs.contents();
    assert!(logs.contains("WARN"), "{logs}");
    assert!(logs.contains(r#"\"index\":\"zero\""#) || logs.contains(r#""index":"zero""#), "{logs}");
    assert!(!logs.contains(&"x".repeat(4_000)), "payload not truncated: {logs}");
}
