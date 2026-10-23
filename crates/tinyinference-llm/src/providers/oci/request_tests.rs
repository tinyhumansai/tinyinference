use super::*;
use crate::message::{AssistantMessage, ImageRef};
use crate::tool::{ToolCall, ToolSchema};

fn weather_tool() -> ToolSchema {
    ToolSchema::new(
        "get_weather",
        "Look up weather",
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" },
                "days": { "type": "integer" }
            },
            "required": ["city"]
        }),
    )
}

fn assistant_with_call() -> Message {
    Message::Assistant(AssistantMessage {
        id: None,
        content: vec![],
        tool_calls: vec![ToolCall::new(
            "call_1",
            "get_weather",
            json!({"city": "Rome"}),
        )],
        usage: None,
        origin: None,
    })
}

fn details(
    request: &ModelRequest,
    serving: &OciServing,
    format: OciApiFormat,
    stream: bool,
) -> Value {
    chat_details(
        request,
        "meta.llama-3.3-70b-instruct",
        serving,
        "ocid1.compartment.x",
        format,
        stream,
    )
}

#[test]
fn generic_on_demand_envelope_and_messages() {
    let mut request = ModelRequest::new(vec![
        Message::system("be brief"),
        Message::user("hi"),
        assistant_with_call(),
        Message::tool("call_1", "sunny"),
    ]);
    request.max_tokens = Some(64);
    request.temperature = Some(0.3);
    request.top_p = Some(0.9);
    request.stop_sequences = vec!["END".into()];
    request.tools = vec![weather_tool()];
    let body = details(
        &request,
        &OciServing::OnDemand,
        OciApiFormat::Generic,
        false,
    );

    assert_eq!(body["compartmentId"], "ocid1.compartment.x");
    assert_eq!(
        body["servingMode"],
        json!({"servingType": "ON_DEMAND", "modelId": "meta.llama-3.3-70b-instruct"})
    );
    let chat = &body["chatRequest"];
    assert_eq!(chat["apiFormat"], "GENERIC");
    assert_eq!(chat["maxTokens"], 64);
    assert_eq!(chat["temperature"], 0.3);
    assert_eq!(chat["topP"], 0.9);
    assert_eq!(chat["stop"], json!(["END"]));
    assert!(chat.get("isStream").is_none());
    assert_eq!(
        chat["messages"],
        json!([
            {"role": "SYSTEM", "content": [{"type": "TEXT", "text": "be brief"}]},
            {"role": "USER", "content": [{"type": "TEXT", "text": "hi"}]},
            {"role": "ASSISTANT", "toolCalls": [
                {"type": "FUNCTION", "id": "call_1", "name": "get_weather", "arguments": "{\"city\":\"Rome\"}"}
            ]},
            {"role": "TOOL", "toolCallId": "call_1", "content": [{"type": "TEXT", "text": "sunny"}]}
        ])
    );
    assert_eq!(chat["tools"][0]["type"], "FUNCTION");
    assert_eq!(chat["tools"][0]["name"], "get_weather");
    assert_eq!(chat["tools"][0]["parameters"]["required"], json!(["city"]));
    assert_eq!(chat["toolChoice"], json!({"type": "AUTO"}));
}

#[test]
fn model_override_and_tool_choice_variants() {
    let mut request = ModelRequest::new(vec![Message::user("x")]);
    request.model = Some("xai.grok-4".into());
    request.tools = vec![weather_tool()];
    request.tool_choice = ToolChoice::Tool("get_weather".into());
    let body = details(
        &request,
        &OciServing::OnDemand,
        OciApiFormat::Generic,
        false,
    );
    assert_eq!(body["servingMode"]["modelId"], "xai.grok-4");
    assert_eq!(
        body["chatRequest"]["toolChoice"],
        json!({"type": "FUNCTION", "name": "get_weather"})
    );
    request.tool_choice = ToolChoice::Required;
    let body = details(
        &request,
        &OciServing::OnDemand,
        OciApiFormat::Generic,
        false,
    );
    assert_eq!(
        body["chatRequest"]["toolChoice"],
        json!({"type": "REQUIRED"})
    );
}

#[test]
fn dedicated_serving_uses_the_endpoint_id() {
    let request = ModelRequest::new(vec![Message::user("x")]);
    let serving = OciServing::Dedicated {
        endpoint_id: "ocid1.generativeaiendpoint.y".into(),
    };
    let body = details(&request, &serving, OciApiFormat::Generic, false);
    assert_eq!(
        body["servingMode"],
        json!({"servingType": "DEDICATED", "endpointId": "ocid1.generativeaiendpoint.y"})
    );
}

#[test]
fn streaming_sets_is_stream_and_include_usage() {
    let request = ModelRequest::new(vec![Message::user("x")]);
    for format in [OciApiFormat::Generic, OciApiFormat::Cohere] {
        let body = details(&request, &OciServing::OnDemand, format, true);
        assert_eq!(body["chatRequest"]["isStream"], true);
        assert_eq!(
            body["chatRequest"]["streamOptions"],
            json!({"isIncludeUsage": true})
        );
    }
}

#[test]
fn images_become_image_parts_and_reasoning_effort_maps() {
    let mut request = ModelRequest::new(vec![Message::User(crate::message::UserMessage {
        content: vec![
            ContentBlock::Text("look".into()),
            ContentBlock::Image(ImageRef {
                url: "https://img.example/a.png".into(),
                mime_type: None,
            }),
        ],
    })]);
    request.reasoning = Some(crate::model::ReasoningConfig {
        effort: Some(ReasoningEffort::XHigh),
        budget_tokens: None,
        summary: None,
    });
    let body = details(
        &request,
        &OciServing::OnDemand,
        OciApiFormat::Generic,
        false,
    );
    let content = &body["chatRequest"]["messages"][0]["content"];
    assert_eq!(
        content[1],
        json!({"type": "IMAGE", "imageUrl": {"url": "https://img.example/a.png"}})
    );
    assert_eq!(body["chatRequest"]["reasoningEffort"], "HIGH");
}

#[test]
fn provider_options_merge_but_reserved_keys_are_dropped() {
    let mut request = ModelRequest::new(vec![Message::user("x")]);
    request.provider_options = json!({"seed": 7, "servingMode": "evil", "messages": []});
    let body = details(
        &request,
        &OciServing::OnDemand,
        OciApiFormat::Generic,
        false,
    );
    assert_eq!(body["chatRequest"]["seed"], 7);
    assert_eq!(body["servingMode"]["servingType"], "ON_DEMAND");
    assert_eq!(body["chatRequest"]["messages"].as_array().unwrap().len(), 1);
}

#[test]
fn cohere_maps_message_history_preamble_and_tools() {
    let mut request = ModelRequest::new(vec![
        Message::system("You are terse."),
        Message::user("first"),
        Message::assistant("reply"),
        Message::user("second"),
    ]);
    request.max_tokens = Some(10);
    request.stop_sequences = vec!["X".into()];
    request.tools = vec![weather_tool()];
    let body = details(&request, &OciServing::OnDemand, OciApiFormat::Cohere, false);
    let chat = &body["chatRequest"];
    assert_eq!(chat["apiFormat"], "COHERE");
    assert_eq!(chat["message"], "second");
    assert_eq!(chat["preambleOverride"], "You are terse.");
    assert_eq!(
        chat["chatHistory"],
        json!([
            {"role": "USER", "message": "first"},
            {"role": "CHATBOT", "message": "reply"}
        ])
    );
    assert_eq!(chat["maxTokens"], 10);
    assert_eq!(chat["stopSequences"], json!(["X"]));
    assert_eq!(
        chat["tools"][0]["parameterDefinitions"],
        json!({
            "city": {"type": "string", "isRequired": true, "description": "City name"},
            "days": {"type": "integer", "isRequired": false}
        })
    );
}

#[test]
fn cohere_trailing_tool_messages_become_tool_results() {
    let request = ModelRequest::new(vec![
        Message::user("weather?"),
        assistant_with_call(),
        Message::tool("call_1", "sunny"),
    ]);
    let body = details(&request, &OciServing::OnDemand, OciApiFormat::Cohere, false);
    let chat = &body["chatRequest"];
    assert_eq!(chat["message"], "");
    assert_eq!(
        chat["toolResults"],
        json!([{
            "call": {"name": "get_weather", "parameters": {"city": "Rome"}},
            "outputs": [{"output": "sunny"}]
        }])
    );
    assert_eq!(
        chat["chatHistory"][1]["toolCalls"][0]["name"],
        "get_weather"
    );
}

#[test]
fn generic_openai_models_send_max_completion_tokens() {
    // OCI forwards `openai.*` to OpenAI, whose reasoning models reject
    // `max_tokens` (`unsupported_parameter`); seen live with openai.gpt-5.5.
    let mut request = ModelRequest::new(vec![Message::user("hi")]);
    request.max_tokens = Some(64);
    let body = chat_details(
        &request,
        "openai.gpt-5.5",
        &OciServing::OnDemand,
        "ocid1.compartment.x",
        OciApiFormat::Generic,
        false,
    );
    let chat = &body["chatRequest"];
    assert_eq!(chat["maxCompletionTokens"], 64);
    assert!(chat.get("maxTokens").is_none());
}

#[test]
fn generic_request_model_override_selects_the_token_field() {
    let mut request = ModelRequest::new(vec![Message::user("hi")]);
    request.max_tokens = Some(8);
    request.model = Some("openai.o4-mini".into());
    let chat = &details(
        &request,
        &OciServing::OnDemand,
        OciApiFormat::Generic,
        false,
    )["chatRequest"];
    assert_eq!(chat["maxCompletionTokens"], 8);

    request.model = Some("meta.llama-4-scout-17b-16e-instruct".into());
    let chat = &details(
        &request,
        &OciServing::OnDemand,
        OciApiFormat::Generic,
        false,
    )["chatRequest"];
    assert_eq!(chat["maxTokens"], 8);
    assert!(chat.get("maxCompletionTokens").is_none());
}

#[test]
fn generic_reasoning_models_omit_sampling_parameters() {
    // Live: openai.gpt-5.5 rejects temperature 0.7 ("only the default (1)").
    let mut request = ModelRequest::new(vec![Message::user("hi")]);
    request.temperature = Some(0.7);
    request.top_p = Some(0.9);
    for model in ["openai.gpt-5.5", "openai.gpt-6-sol", "openai.o4-mini"] {
        request.model = Some(model.into());
        let chat = &details(
            &request,
            &OciServing::OnDemand,
            OciApiFormat::Generic,
            false,
        )["chatRequest"];
        assert!(chat.get("temperature").is_none(), "{model}");
        assert!(chat.get("topP").is_none(), "{model}");
    }
    for model in [
        "openai.gpt-4.1",
        "meta.llama-3.3-70b-instruct",
        "xai.grok-4.7",
    ] {
        request.model = Some(model.into());
        let chat = &details(
            &request,
            &OciServing::OnDemand,
            OciApiFormat::Generic,
            false,
        )["chatRequest"];
        assert_eq!(chat["temperature"], 0.7, "{model}");
        assert_eq!(chat["topP"], 0.9, "{model}");
    }
}

#[test]
fn native_drops_openai_wire_provider_options() {
    // Live: the OpenHuman harness sets `prompt_cache_key`; the native endpoint
    // answered 400 "Please pass in correct format of request" until it was dropped.
    let mut request = ModelRequest::new(vec![Message::user("hi")]);
    request.provider_options = json!({
        "prompt_cache_key": "tap-1",
        "reasoningEffort": "LOW",
        "seed": 7
    });
    let chat = &details(
        &request,
        &OciServing::OnDemand,
        OciApiFormat::Generic,
        false,
    )["chatRequest"];
    assert!(chat.get("prompt_cache_key").is_none());
    assert_eq!(chat["reasoningEffort"], "LOW");
    assert_eq!(chat["seed"], 7);
}
