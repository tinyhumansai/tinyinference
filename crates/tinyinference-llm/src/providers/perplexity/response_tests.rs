use super::*;
use serde_json::json;

#[test]
fn response_preserves_sources_annotations_cost_and_distinct_ids() {
    let response = parse(json!({
        "id":"resp_1","model":"openai/model","status":"completed",
        "output":[
            {"type":"search_results","queries":["question"],"results":[
                {"id":1,"url":"https://example.com","title":"Source","snippet":"evidence"}
            ]},
            {"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[
                {"type":"output_text","text":"Café [1]","annotations":[{"type":"url_citation","url":"https://example.com","start_index":0,"end_index":4}]}
            ]}
        ],
        "usage":{"input_tokens":20,"output_tokens":5,"total_tokens":25,
            "cost":{"currency":"USD","total_cost":0.0000007,"tool_calls_cost":0.0025},
            "tool_calls_details":{"web_search":{"invocation":1}}
        }
    })).unwrap();
    assert_eq!(response.text(), "Café [1]");
    assert_eq!(response.message.id.as_deref(), Some("msg_1"));
    assert_eq!(response.execution.as_ref().unwrap().id, "resp_1");
    assert_eq!(response.output.len(), 2);
    assert!(response.message.tool_calls.is_empty());
    assert_eq!(response.usage.unwrap().charged_amount.unwrap().micros, 1);
    match &response.output[1].kind {
        ModelOutputKind::Message { content, .. } => assert_eq!(content[0].annotations.len(), 1),
        _ => panic!("expected normalized message"),
    }
}

#[test]
fn all_documented_hosted_outputs_remain_typed_and_are_never_local_calls() {
    let values = vec![
        json!({"type":"search_results","queries":["q"],"results":[{"id":1,"url":"u","title":"t","snippet":"s"}]}),
        json!({"type":"people_search_results","results":[{"id":1,"url":"u","title":"t","snippet":"s","source":"people"}]}),
        json!({"type":"image_search_results","queries":["q"],"results":[{"image_url":"i","origin_url":"o","width":10,"height":20}],"error":"partial"}),
        json!({"type":"fetch_url_results","contents":[{"url":"u","title":"t","snippet":"s"}]}),
        json!({"type":"finance_results","tickers":["ABC"],"categories":["quote"],"results":[{"category":"quote","content":"price table","sources":["u"]}]}),
        json!({"type":"sandbox_results","status":"failed","code":"exit(1)","stdout":"out","stderr":"err","exit_code":1,"duration_ms":7}),
        json!({"type":"mcp_list_tools","id":"m1","server_label":"source","connector_id":"connector1","tools":[{"name":"lookup","input_schema":{"type":"object"}}]}),
        json!({"type":"mcp_call","id":"m2","server_label":"source","name":"lookup","arguments":"{}","output":"found","error":null}),
        json!({"type":"tool_search_output","id":"t1","status":"future_state","execution":"server","call_id":null,"arguments":"{}","tools":[{"type":"namespace","name":"source","description":"tools","tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}]}),
        json!({"type":"share_file","filename":"report.csv","new_provider_field":7}),
        json!({"type":"future_output","payload":{"answer":42}}),
    ];
    let response = completed(
        parse(json!({"id":"r","model":"openai/m","status":"completed","output":values})).unwrap(),
    )
    .unwrap();
    assert_eq!(response.output.len(), 11);
    let round_trip: ModelResponse =
        serde_json::from_slice(&serde_json::to_vec(&response).unwrap()).unwrap();
    assert_eq!(round_trip, response);
    assert!(response.message.tool_calls.is_empty());
    assert!(response.text().is_empty());
    assert!(
        matches!(&response.output[0].kind,ModelOutputKind::Search {tool,..} if tool=="web_search")
    );
    assert!(
        matches!(&response.output[1].kind,ModelOutputKind::Search {tool,..} if tool=="people_search")
    );
    assert!(
        matches!(&response.output[2].kind,ModelOutputKind::ImageSearch {error:Some(error),..} if error=="partial")
    );
    assert!(
        matches!(&response.output[3].kind,ModelOutputKind::Fetch {contents} if contents[0].snippet=="s")
    );
    assert!(
        matches!(&response.output[4].kind,ModelOutputKind::Finance {results,..} if results[0].content=="price table")
    );
    assert!(matches!(
        &response.output[5].kind,
        ModelOutputKind::Sandbox {
            exit_code: Some(1),
            ..
        }
    ));
    assert!(
        matches!(&response.output[6].kind,ModelOutputKind::ToolDiscovery {tools,..} if tools[0].name=="lookup")
    );
    assert!(
        matches!(&response.output[7].kind,ModelOutputKind::RemoteCall {output:Some(output),..} if output=="found")
    );
    assert!(
        matches!(&response.output[8].kind,ModelOutputKind::ToolSearch {tools,..} if tools[0].tools[0].name=="lookup")
    );
    assert_eq!(
        response.output[8].status,
        Some(ExecutionStatus::Other("future_state".into()))
    );
    assert!(
        matches!(&response.output[9].kind,ModelOutputKind::GeneratedFiles {metadata,..} if metadata["new_provider_field"]==7)
    );
    assert!(
        matches!(&response.output[10].kind,ModelOutputKind::Extension {data,..} if data["payload"]["answer"]==42)
    );
}

#[test]
fn function_calls_preserve_raw_arguments_and_signatures_even_when_invalid() {
    let response=parse(json!({"id":"r","model":"google/m","status":"completed","output":[
        {"type":"function_call","id":"item1","call_id":"call1","name":"lookup","arguments":"{ broken","thought_signature":"opaque"}
    ]})).unwrap();
    assert_eq!(response.finish_reason.as_deref(), Some("tool_calls"));
    let call = &response.message.tool_calls[0];
    assert!(call.is_invalid());
    assert_eq!(call.id, "call1");
    assert_eq!(
        call.replay.as_ref().unwrap().item_id.as_deref(),
        Some("item1")
    );
    assert_eq!(
        call.replay.as_ref().unwrap().thought_signature.as_deref(),
        Some("opaque")
    );
    assert_eq!(call.replay.as_ref().unwrap().arguments, "{ broken");
}

#[test]
fn malformed_known_items_fail_and_retain_preceding_rich_output() {
    for bad in [
        json!({"type":"search_results","results":"wrong"}),
        json!({"type":"message","role":"assistant","content":[{"type":"output_text"}]}),
        json!({"type":"function_call","arguments":"{}","name":"x"}),
        json!({"type":"sandbox_results","status":"completed","duration_ms":-1}),
    ] {
        let result = parse(
            json!({"id":"r","model":"openai/m","status":"completed","output":[
                {"type":"share_file","file_id":"f"},bad
            ]}),
        );
        let Err(Error::Provider(error)) = result else {
            panic!("malformed known output must fail");
        };
        let partial = error.partial_response.unwrap();
        assert_eq!(partial.output.len(), 1);
        assert!(matches!(
            partial.output[0].kind,
            ModelOutputKind::GeneratedFiles { .. }
        ));
    }
}

#[test]
fn incomplete_failed_cancelled_and_unknown_states_are_not_reported_as_normal_success() {
    let incomplete=completed(parse(json!({"id":"r","model":"openai/m","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"output":[]})).unwrap()).unwrap();
    assert_eq!(
        incomplete.finish_reason.as_deref(),
        Some("max_output_tokens")
    );
    for status in [
        "failed",
        "cancelled",
        "in_progress",
        "queued",
        "future_state",
    ] {
        let result = completed(
            parse(json!({"id":"r","model":"openai/m","status":status,"output":[]})).unwrap(),
        );
        let Err(Error::Provider(error)) = result else {
            panic!("state {status} is not completed");
        };
        assert_eq!(error.partial_response.unwrap().execution.unwrap().id, "r");
    }
}

#[test]
fn missing_cost_is_unknown_and_invalid_usage_is_not_fabricated() {
    let response =
        parse(json!({"id":"r","model":"openai/m","status":"completed","output":[]})).unwrap();
    assert!(response.usage.is_none());
    for usage in [
        json!({"input_tokens":-1,"output_tokens":0,"total_tokens":0}),
        json!({"input_tokens":1,"output_tokens":2,"total_tokens":3,"cost":{"currency":"USD","total_cost":-0.1}}),
        json!({"input_tokens":1,"output_tokens":2,"total_tokens":3,"cost":{"currency":"USD","total_cost":1e30}}),
    ] {
        assert!(
            parse(
                json!({"id":"r","model":"openai/m","status":"completed","output":[],"usage":usage})
            )
            .is_err()
        );
    }
}

#[test]
fn reported_decimal_cost_keeps_precision_at_micro_unit_rounding_boundaries() {
    let value = decode(
        br#"{"id":"r","model":"openai/m","status":"completed","output":[],
        "usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2,
            "cost":{"currency":"USD","total_cost":0.000000499999999999999999}}}"#,
    )
    .unwrap();
    let response = parse(value).unwrap();
    assert_eq!(response.usage.unwrap().charged_amount.unwrap().micros, 0);
    assert_eq!(
        response.execution.unwrap().cost.unwrap().components["total"],
        "0.000000499999999999999999"
    );
}
