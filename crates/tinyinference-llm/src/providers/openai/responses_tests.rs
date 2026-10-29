use super::*;
use crate::message::Message;
use serde_json::json;

#[test]
fn build_input_folds_system_into_instructions_and_keys_roles() {
    let messages = vec![
        Message::system("be terse"),
        Message::system("and correct"),
        Message::user("hi"),
        Message::assistant("hello"),
        Message::user("  "), // empty → skipped
    ];
    let (instructions, input) = build_responses_input(&messages, false).unwrap();
    assert_eq!(instructions.as_deref(), Some("be terse\n\nand correct"));
    assert_eq!(input.len(), 2);
    assert_eq!(input[0].role, "user");
    assert_eq!(input[0].content[0].kind, "input_text");
    assert_eq!(input[0].content[0].text, "hi");
    // Assistant items must use `output_text`, not `input_text`.
    assert_eq!(input[1].role, "assistant");
    assert_eq!(input[1].content[0].kind, "output_text");
    assert_eq!(input[1].content[0].text, "hello");
}

#[test]
fn build_input_skips_custom_messages() {
    let messages = vec![
        Message::user("hi"),
        Message::Custom(crate::message::CustomMessage {
            kind: "compaction".into(),
            payload: json!({"summary": "..."}),
            display: Some("Compacted".into()),
        }),
        Message::assistant("hello"),
    ];
    let (_, input) = build_responses_input(&messages, false).unwrap();
    assert_eq!(input.len(), 2);
    assert_eq!(input[0].content[0].text, "hi");
    assert_eq!(input[1].content[0].text, "hello");
}

#[test]
fn extract_text_prefers_output_text_then_scans_content() {
    let with_convenience = ResponsesResponse {
        status: None,
        incomplete_details: None,
        output: Vec::new(),
        output_text: Some("  final  ".to_string()),
        usage: None,
    };
    assert_eq!(
        extract_responses_text(&with_convenience).as_deref(),
        Some("final")
    );

    let via_content = ResponsesResponse {
        status: None,
        incomplete_details: None,
        output: vec![ResponsesOutput {
            content: vec![
                ResponsesContent {
                    kind: Some("reasoning".into()),
                    text: Some("...".into()),
                },
                ResponsesContent {
                    kind: Some("output_text".into()),
                    text: Some("answer".into()),
                },
            ],
            ..ResponsesOutput::default()
        }],
        output_text: None,
        usage: None,
    };
    assert_eq!(
        extract_responses_text(&via_content).as_deref(),
        Some("answer")
    );

    let empty = ResponsesResponse {
        status: None,
        incomplete_details: None,
        output: Vec::new(),
        output_text: None,
        usage: None,
    };
    assert_eq!(extract_responses_text(&empty), None);
}

#[test]
fn parse_maps_text_and_usage_onto_model_response() {
    let body = json!({
        "output_text": "the answer",
        "usage": { "input_tokens": 12, "output_tokens": 5 }
    });
    let resp = parse_responses_response(body);
    assert_eq!(resp.text(), "the answer");
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    let usage = resp.usage.expect("usage mapped");
    assert_eq!(usage.input_tokens, 12);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 17);
}

#[test]
fn parse_tolerates_a_body_without_output() {
    let resp = parse_responses_response(json!({ "id": "resp_1" }));
    assert_eq!(resp.text(), "");
}

#[test]
fn an_output_limit_stop_is_reported_as_length() {
    // The Responses API spells the output cap `max_output_tokens`; consumers
    // compare against the canonical `length`. The wire value stays on `raw`.
    let resp = parse_responses_response(json!({
        "status": "incomplete",
        "incomplete_details": { "reason": "max_output_tokens" },
        "output_text": "partial"
    }));
    assert_eq!(resp.text(), "partial");
    assert_eq!(resp.finish_reason.as_deref(), Some("length"));
    assert_eq!(
        resp.raw.as_ref().unwrap()["incomplete_details"]["reason"],
        json!("max_output_tokens")
    );
}

#[test]
fn other_incomplete_reasons_pass_through() {
    let resp = parse_responses_response(json!({
        "status": "incomplete",
        "incomplete_details": { "reason": "content_filter" },
        "output_text": ""
    }));
    assert_eq!(resp.finish_reason.as_deref(), Some("content_filter"));
}

#[test]
fn parse_responses_wire_reads_sse_completed_event() {
    let sse = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output_text\":\"hi\"}}\n\n";
    let value = parse_responses_wire(sse).expect("sse");
    assert_eq!(value["output_text"], "hi");
}

#[test]
fn parse_responses_wire_uses_output_text_done_when_completed_output_is_empty() {
    let sse = "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\nevent: response.output_text.done\ndata: {\"type\":\"response.output_text.done\",\"text\":\"ok\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output\":[]}}\n\n";
    let value = parse_responses_wire(sse).expect("sse");
    assert_eq!(value["output_text"], "ok");
}

#[tokio::test]
async fn responses_invoke_avoids_codex_stream_required_400() {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use crate::model::{ChatModel, ModelRequest};
    use crate::providers::openai::OpenAiModel;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (mut sock, _) = listener.accept().unwrap();
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = sock.read(&mut tmp).unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(split) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let headers = std::str::from_utf8(&buf[..split]).unwrap_or("");
                let content_len = headers
                    .lines()
                    .find_map(|line| {
                        line.split_once(':').and_then(|(k, v)| {
                            k.eq_ignore_ascii_case("content-length")
                                .then(|| v.trim().parse::<usize>().ok())
                                .flatten()
                        })
                    })
                    .unwrap_or(0);
                let start = split + 4;
                while buf.len() < start + content_len {
                    let n = sock.read(&mut tmp).unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                break;
            }
        }
        let req = String::from_utf8_lossy(&buf);
        let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
        let stream_true = body.contains("\"stream\":true") || body.contains("\"stream\": true");
        let (status, content_type, payload) = if stream_true {
            let sse = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"output_text\":\"ok\"}}\n\n";
            ("200 OK", "text/event-stream", sse.to_string())
        } else {
            (
                "400 Bad Request",
                "application/json",
                r#"{"detail":"Stream must be set to true"}"#.to_string(),
            )
        };
        let resp = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
            payload.len()
        );
        sock.write_all(resp.as_bytes()).unwrap();
    });

    let model = OpenAiModel::compatible("k", format!("http://{addr}/v1"), "gpt-5.6-luna")
        .with_responses_api_primary();
    let response = model
        .invoke(&(), ModelRequest::new(vec![Message::user("hi")]))
        .await
        .expect("codex-shaped mock must succeed when stream is true");
    assert_eq!(response.text(), "ok");
    server.join().unwrap();
}

#[test]
fn native_inputs_preserve_order_and_emit_responses_wire_shapes() {
    use crate::message::{ContentBlock, ImageRef, MediaRef, UserMessage};
    let messages = vec![Message::User(UserMessage {
        content: vec![
            ContentBlock::Text("before".into()),
            ContentBlock::Image(ImageRef {
                url: "data:image/png;base64,QQ==".into(),
                mime_type: Some("image/png".into()),
            }),
            ContentBlock::Text("after".into()),
            ContentBlock::Document(MediaRef::base64("QQ==", "application/pdf")),
        ],
    })];
    let (_, input) = build_responses_input(&messages, true).unwrap();
    assert_eq!(
        serde_json::to_value(&input).unwrap(),
        json!([{"role":"user","content":[
            {"type":"input_text","text":"before"},
            {"type":"input_image","image_url":"data:image/png;base64,QQ=="},
            {"type":"input_text","text":"after"},
            {"type":"input_file","filename":"document.pdf","file_data":"data:application/pdf;base64,QQ=="}
        ]}])
    );
}

#[test]
fn responses_native_inputs_reject_unsupported_sources_and_modalities() {
    use crate::message::{ContentBlock, ImageRef, MediaRef, UserMessage};
    for block in [
        ContentBlock::Audio(MediaRef::base64("QQ==", "audio/wav")),
        ContentBlock::Video(MediaRef::base64("QQ==", "video/mp4")),
        ContentBlock::Document(MediaRef::path("/secret.pdf")),
        ContentBlock::Document(MediaRef::base64("QQ==", "application/msword")),
        ContentBlock::Image(ImageRef {
            url: "file:///secret.png".into(),
            mime_type: Some("image/png".into()),
        }),
    ] {
        let messages = [Message::User(UserMessage {
            content: vec![block],
        })];
        assert!(build_responses_input(&messages, true).is_err());
    }
    let pdf = [Message::User(UserMessage {
        content: vec![ContentBlock::Document(MediaRef::base64(
            "QQ==",
            "application/pdf",
        ))],
    })];
    assert!(
        build_responses_input(&pdf, false).is_err(),
        "Codex endpoint must decline files"
    );
}

#[test]
fn responses_pdf_url_uses_file_url() {
    use crate::message::{ContentBlock, MediaRef, UserMessage};
    let messages = [Message::User(UserMessage {
        content: vec![ContentBlock::Document(MediaRef::Url {
            url: "https://example.com/a.pdf".into(),
            media_type: Some("application/pdf".into()),
        })],
    })];
    let (_, input) = build_responses_input(&messages, true).unwrap();
    assert_eq!(
        serde_json::to_value(input).unwrap()[0]["content"][0],
        json!({"type":"input_file","file_url":"https://example.com/a.pdf"})
    );
}

#[test]
fn parse_preserves_ordered_output_and_execution_metadata() {
    let resp = parse_responses_response(json!({
        "id": "resp_42",
        "model": "gpt-5",
        "status": "completed",
        "output": [
            {
                "id": "rs_1",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "think"}],
                "encrypted_content": "sig"
            },
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": "answer",
                    "annotations": [{
                        "type": "url_citation",
                        "url": "https://example.com",
                        "title": "Example",
                        "start_index": 0,
                        "end_index": 6
                    }]
                }]
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": "{\"q\":1}"
            },
            {"id": "ws_1", "type": "web_search_call", "status": "completed"}
        ]
    }));

    let execution = resp.execution.expect("execution metadata");
    assert_eq!(execution.id, "resp_42");
    assert_eq!(execution.model, "gpt-5");
    assert_eq!(execution.status, ExecutionStatus::Completed);

    let ids: Vec<_> = resp.output.iter().map(|item| item.id.as_deref()).collect();
    assert_eq!(
        ids,
        [Some("rs_1"), Some("msg_1"), Some("fc_1"), Some("ws_1")]
    );
    assert!(matches!(
        &resp.output[0].kind,
        ModelOutputKind::Reasoning { content }
            if content == &[ContentBlock::Thinking { text: "think".into(), signature: Some("sig".into()) }]
    ));
    let ModelOutputKind::Message { role, content } = &resp.output[1].kind else {
        panic!("expected message item");
    };
    assert_eq!(role, "assistant");
    assert_eq!(content[0].block, ContentBlock::Text("answer".into()));
    assert!(matches!(
        &content[0].annotations[0],
        OutputAnnotation::UrlCitation { url, start: Some(0), end: Some(6), .. } if url == "https://example.com"
    ));
    let ModelOutputKind::FunctionCall { call } = &resp.output[2].kind else {
        panic!("expected function call item");
    };
    assert_eq!(call.id, "call_1");
    assert_eq!(call.arguments, json!({"q": 1}));
    assert!(matches!(
        &resp.output[3].kind,
        ModelOutputKind::Extension { item_type, .. } if item_type == "web_search_call"
    ));
}

#[test]
fn parse_keeps_malformed_function_arguments_as_an_invalid_call() {
    let resp = parse_responses_response(json!({
        "output": [{"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": "{bad"}]
    }));
    let ModelOutputKind::FunctionCall { call } = &resp.output[0].kind else {
        panic!("expected function call item");
    };
    assert!(call.invalid.is_some());
    assert_eq!(call.arguments, json!("{bad"));
}

#[test]
fn parse_omits_execution_without_a_response_id() {
    let resp = parse_responses_response(json!({ "output_text": "hi" }));
    assert!(resp.execution.is_none());
    assert!(resp.output.is_empty());
}
