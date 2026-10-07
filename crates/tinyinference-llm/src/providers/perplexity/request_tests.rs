use super::*;

use super::super::{PerplexityConfig, PerplexitySelection};
use crate::message::Message;
use crate::model::ModelRequest;
use serde_json::json;

use super::super::{
    PerplexityOptions, PerplexityRemoteCredentials, PerplexitySearchFilters, PerplexityTool,
    PerplexityWebSearch,
};
use crate::message::{ContentBlock, MediaRef, ToolMessage};
use crate::model::{ReasoningConfig, ReasoningEffort, ResponseFormat};
use crate::tool::{ToolCall, ToolCallReplay, ToolSchema};

#[test]
fn preset_request_preserves_server_defaults() {
    let config = PerplexityConfig::new(PerplexitySelection::preset("low"));
    let request = ModelRequest::new(vec![Message::user("hello")]);
    assert_eq!(
        build(&config, &request, false, false).unwrap(),
        json!({
            "preset":"low", "stream":false,
            "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}]
        })
    );
}

#[test]
fn invalid_hosted_options_are_rejected_without_mutating_the_request() {
    for tool in [
        PerplexityTool::ImageSearch {
            max_results: Some(0),
            filters: None,
        },
        PerplexityTool::ImageSearch {
            max_results: Some(31),
            filters: None,
        },
        PerplexityTool::FetchUrl { max_urls: Some(0) },
        PerplexityTool::FetchUrl { max_urls: Some(11) },
    ] {
        let mut request = ModelRequest::new(vec![Message::user("question")]);
        request.provider_options = json!({"perplexity":{"max_steps":2}});
        let original = request.provider_options.clone();
        let options = PerplexityOptions {
            tools: Some(vec![tool]),
            ..Default::default()
        };
        assert!(options.apply_to(&mut request).is_err());
        assert_eq!(request.provider_options, original);
    }
}

#[test]
fn forced_hosted_tools_require_a_declaration_unless_a_preset_can_supply_it() {
    let tool = PerplexityTool::web_search(PerplexityWebSearch::default());
    for selection in [
        PerplexitySelection::Model("openai/test".into()),
        PerplexitySelection::Models(vec!["openai/test".into()]),
        PerplexitySelection::preset("low"),
    ] {
        for declared in [false, true] {
            let mut config = PerplexityConfig::new(selection.clone());
            config.options.tool_choice = Some(super::super::PerplexityToolChoice::Hosted(
                "web_search".into(),
            ));
            if declared {
                config.options.tools = Some(vec![tool.clone()]);
            }
            let request = ModelRequest::new(vec![Message::user("question")]);
            let result = build(&config, &request, false, false);
            if declared || matches!(selection, PerplexitySelection::Preset { .. }) {
                assert_eq!(result.unwrap()["tool_choice"], json!({"type":"web_search"}));
            } else {
                assert!(matches!(result, Err(Error::Validation(_))));
            }
        }
    }
}

#[test]
fn each_preset_remains_dynamic_and_explicit_model_overrides_are_preserved() {
    for preset in ["fast", "low", "medium", "high", "xhigh", "future-preset"] {
        let config = PerplexityConfig::new(PerplexitySelection::preset(preset));
        let mut request = ModelRequest::new(vec![
            Message::system("my instructions"),
            Message::user("question"),
        ]);
        request.model = Some("google/model".into());
        request.max_tokens = Some(128);
        let body = build(&config, &request, false, false).unwrap();
        assert_eq!(body["preset"], preset);
        assert_eq!(body["model"], "google/model");
        assert_eq!(body["instructions"], "my instructions");
        assert_eq!(body["max_output_tokens"], 128);
        assert!(body.get("tools").is_none());
    }
}

#[test]
fn hosted_tool_options_survive_typed_serialization_and_preserve_omissions() {
    let config = PerplexityConfig::new(PerplexitySelection::preset("low"));
    let mut request = ModelRequest::new(vec![Message::user("question")]);
    PerplexityOptions {
        tools: Some(vec![PerplexityTool::WebSearch {
            options: Box::new(PerplexityWebSearch {
                max_results: Some(3),
                filters: Some(PerplexitySearchFilters {
                    search_domain_filter: Some(vec!["example.com".into()]),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        }]),
        ..Default::default()
    }
    .apply_to(&mut request)
    .unwrap();
    let body = build(&config, &request, false, false).unwrap();
    assert_eq!(
        body["tools"],
        json!([{"type":"web_search","max_results":3,"filters":{"search_domain_filter":["example.com"]}}])
    );
    assert!(body.get("max_steps").is_none());
    assert!(body.get("tool_choice").is_none());
}

#[test]
fn invalid_provider_options_are_rejected_without_silent_dropping() {
    let config = PerplexityConfig::new(PerplexitySelection::Model("openai/model".into()));
    for options in [
        json!({"profile":"id"}),
        json!({"skills":[]}),
        json!({"unexpected":true}),
        json!({"max_steps":0}),
        json!({"tools":[{"type":"mcp","server_label":"x","server_url":"https://example.com","require_approval":"always"}]}),
        json!({"tools":[{"type":"web_search","max_results":51}]}),
        json!({"tools":[{"type":"web_search","filters":{"search_domain_filter":["a","-b"]}}]}),
        json!({"tools":[{"type":"web_search","filters":{"search_after_date_filter":"02/30/2026"}}]}),
    ] {
        let mut request = ModelRequest::new(vec![Message::user("question")]);
        request.provider_options = json!({"perplexity":options});
        assert!(build(&config, &request, false, false).is_err(), "{options}");
    }
    let mut request = ModelRequest::new(vec![Message::user("question")]);
    request.seed = Some(1);
    assert!(matches!(
        build(&config, &request, false, false),
        Err(Error::Unsupported(_))
    ));
    request.seed = None;
    request.reasoning = Some(ReasoningConfig::effort(ReasoningEffort::None));
    assert!(matches!(
        build(&config, &request, false, false),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn anthropic_and_fallback_selections_require_an_explicit_output_cap() {
    let mut request = ModelRequest::new(vec![Message::user("question")]);
    for selection in [
        PerplexitySelection::Model("anthropic/model".into()),
        PerplexitySelection::Models(vec!["openai/model".into(), "anthropic/model".into()]),
    ] {
        let config = PerplexityConfig::new(selection);
        assert!(build(&config, &request, false, false).is_err());
        request.max_tokens = Some(32);
        assert_eq!(
            build(&config, &request, false, false).unwrap()["max_output_tokens"],
            32
        );
        request.max_tokens = None;
    }
    let config = PerplexityConfig::new(PerplexitySelection::Models(vec!["openai/model".into()]));
    request.model = Some("google/model".into());
    assert!(build(&config, &request, false, false).is_err());
}

#[test]
fn structured_output_and_document_inputs_use_the_native_agent_shape() {
    let config = PerplexityConfig::new(PerplexitySelection::Model("openai/model".into()));
    let mut request = ModelRequest::new(vec![Message::User(crate::message::UserMessage {
        content: vec![
            ContentBlock::Text("summarize".into()),
            ContentBlock::Document(MediaRef::base64("aGVsbG8=", "text/plain")),
        ],
    })]);
    request.response_format = Some(ResponseFormat::json_schema(
        "answer",
        json!({"type":"object"}),
    ));
    let body = build(&config, &request, false, false).unwrap();
    assert_eq!(
        body["input"][0]["content"][1],
        json!({"type":"input_file","file_data":"aGVsbG8=","filename":"document.txt"})
    );
    assert_eq!(body["response_format"]["type"], "json_schema");
    assert!(body.get("text").is_none());
    request.messages = vec![Message::User(crate::message::UserMessage {
        content: vec![ContentBlock::Document(MediaRef::path("/tmp/private"))],
    })];
    assert!(matches!(
        build(&config, &request, false, false),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn remote_credentials_are_bound_to_their_server_and_never_serialized_into_requests() {
    let mut config = PerplexityConfig::new(PerplexitySelection::Model("openai/model".into()));
    config.remote_credentials.insert(
        "source".into(),
        PerplexityRemoteCredentials {
            server_url: "https://trusted.example/mcp".into(),
            authorization: Some("remote-secret".into()),
            headers: Default::default(),
        },
    );
    let options = PerplexityOptions {
        tools: Some(vec![PerplexityTool::Mcp {
            server_label: "source".into(),
            server_url: "https://trusted.example/mcp".into(),
            allowed_tools: vec!["read".into()],
            defer_loading: Some(true),
        }]),
        ..Default::default()
    };
    let mut request = ModelRequest::new(vec![Message::user("question")]);
    options.apply_to(&mut request).unwrap();
    assert!(
        !serde_json::to_string(&request)
            .unwrap()
            .contains("remote-secret")
    );
    assert_eq!(
        build(&config, &request, false, false).unwrap()["tools"][0]["authorization"],
        "remote-secret"
    );
    request.provider_options["perplexity"]["tools"][0]["server_url"] =
        json!("https://other.example/mcp");
    assert!(build(&config, &request, false, false).is_err());
    assert!(!format!("{config:?}").contains("remote-secret"));
}

#[test]
fn signed_function_calls_and_results_replay_natively_with_original_arguments() {
    let config = PerplexityConfig::new(PerplexitySelection::Model("google/model".into()));
    let mut call = ToolCall::new("call_1", "lookup", json!({"id":7}));
    call.replay = Some(ToolCallReplay {
        origin: crate::message::MessageOrigin {
            provider: "perplexity".into(),
            api: "agent".into(),
            model: "google/model".into(),
        },
        item_id: Some("fc_1".into()),
        arguments: "{ \"id\": 7 }".into(),
        thought_signature: Some("signed".into()),
    });
    let result = ToolMessage::for_call(&call, vec![ContentBlock::Json(json!({"ok":true}))]);
    let mut assistant = crate::model::ModelResponse::assistant("").message;
    assistant.content.clear();
    assistant.tool_calls = vec![call.clone()];
    let mut request = ModelRequest::new(vec![
        Message::user("lookup"),
        Message::Assistant(assistant),
        Message::Tool(result.clone()),
    ]);
    request.tools = vec![ToolSchema::new(
        "lookup",
        "lookup",
        json!({"type":"object"}),
    )];
    let body = build(&config, &request, false, false).unwrap();
    assert_eq!(body["input"][1]["type"], "function_call");
    assert_eq!(body["input"][1]["arguments"], "{ \"id\": 7 }");
    assert_eq!(body["input"][2]["call_id"], "call_1");
    assert_eq!(body["input"][2]["thought_signature"], "signed");
    request.continuation_id = Some("resp_1".into());
    request.messages = vec![Message::Tool(result)];
    assert_eq!(
        build(&config, &request, false, false).unwrap()["previous_response_id"],
        "resp_1"
    );
    request.model = Some("openai/model".into());
    assert!(matches!(
        build(&config, &request, false, false),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn unmatched_or_duplicate_function_results_are_rejected() {
    let config = PerplexityConfig::new(PerplexitySelection::Model("openai/model".into()));
    let call = ToolCall::new("call_1", "lookup", json!({}));
    let result = Message::Tool(ToolMessage::for_call(
        &call,
        vec![ContentBlock::Text("result".into())],
    ));
    let request = ModelRequest::new(vec![Message::user("question"), result.clone()]);
    assert!(build(&config, &request, false, false).is_err());
    let mut request = ModelRequest::new(vec![result.clone(), result]);
    request.continuation_id = Some("resp_1".into());
    assert!(build(&config, &request, false, false).is_err());
}
