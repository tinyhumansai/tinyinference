use super::AuthStyle;
use super::config::*;

#[test]
fn builds_a_chat_model_with_the_configured_profile() {
    let model = build_openai_model(OpenAiConfig {
        provider_name: "deepseek",
        endpoint: "https://api.deepseek.com/v1",
        api_key: "secret",
        auth_style: AuthStyle::Bearer,
        model: "deepseek-chat",
        temperature_unsupported_models: &[],
        temperature_override: None,
        merge_system_into_user: false,
        extra_headers: &[],
        native_tool_calling: None,
        vision: None,
        default_provider_options: None,
        responses_api_primary: false,
        responses_omit_max_output_tokens: false,
        extra_query_params: &[],
        user_agent: None,
        explicit_cache_control: false,
    });
    let profile = model.profile().expect("openai models expose a profile");
    assert_eq!(profile.provider.as_deref(), Some("deepseek"));
    assert_eq!(profile.model.as_deref(), Some("deepseek-chat"));
    assert!(profile.tool_calling);
}

#[test]
fn local_runtime_builder_disables_native_tools_and_vision() {
    let model = build_local_runtime_chat_model(
        "ollama",
        "http://localhost:11434/v1",
        "",
        AuthStyle::None,
        "qwen2.5",
        &[],
        None,
        Some(8192),
    );
    let profile = model.profile().expect("openai models expose a profile");
    assert!(!profile.tool_calling);
    assert!(!profile.modalities.image_in);
    // One leading system message only, so the harness must not place a
    // system message mid-transcript (tinyhumansai/openhuman#7440).
    assert!(!profile.mid_conversation_system_messages);
    assert!(profile.hoists_system_messages);
}

#[test]
fn local_runtime_builder_can_opt_into_native_tools() {
    // An OpenAI-compatible endpoint behind `local-openai` may serve a model
    // with working native tool calls (GLM on a hosted gateway). Prompting it
    // with the text protocol instead lets the gateway's own tool parser read
    // the JSON-in-tag call as a tool *name*, so the host must be able to
    // turn native calls on for it.
    let model = build_local_runtime_chat_model_with_native_tools(
        "local-openai",
        "https://gateway.example/v1",
        "secret",
        AuthStyle::Bearer,
        "glm-5.3-flash",
        &[],
        None,
        None,
        true,
    );
    let profile = model.profile().expect("openai models expose a profile");
    assert!(profile.tool_calling);
    // Everything else stays the local-runtime profile.
    assert!(!profile.modalities.image_in);
    assert!(!profile.mid_conversation_system_messages);
    assert!(profile.hoists_system_messages);

    let off = build_local_runtime_chat_model_with_native_tools(
        "local-openai",
        "https://gateway.example/v1",
        "secret",
        AuthStyle::Bearer,
        "glm-5.3-flash",
        &[],
        None,
        None,
        false,
    );
    assert!(!off.profile().expect("profile").tool_calling);
}

#[test]
fn openrouter_endpoints_are_recognized() {
    assert!(endpoint_is_openrouter("https://openrouter.ai/api/v1"));
    assert!(endpoint_is_openrouter("HTTPS://OpenRouter.ai:443/api/v1/"));
    assert!(!endpoint_is_openrouter("https://notopenrouter.ai/api/v1"));
}

#[test]
fn debug_redacts_credentials_and_header_values() {
    let headers = vec![(
        "Authorization".to_string(),
        "Bearer header-secret".to_string(),
    )];
    let query = vec![("api_key".to_string(), "query-secret".to_string())];
    let config = OpenAiConfig {
        provider_name: "test",
        endpoint: "https://endpoint-user:endpoint-pass@example.com/v1?token=endpoint-secret#fragment-secret",
        api_key: "api-secret",
        auth_style: AuthStyle::Bearer,
        model: "model",
        temperature_unsupported_models: &[],
        temperature_override: None,
        merge_system_into_user: false,
        extra_headers: &headers,
        native_tool_calling: None,
        vision: None,
        default_provider_options: None,
        responses_api_primary: false,
        responses_omit_max_output_tokens: false,
        extra_query_params: &query,
        user_agent: None,
        explicit_cache_control: false,
    };
    let debug = format!("{config:?}");
    assert!(!debug.contains("api-secret"));
    assert!(!debug.contains("header-secret"));
    assert!(!debug.contains("query-secret"));
    assert!(!debug.contains("endpoint-user"));
    assert!(!debug.contains("endpoint-pass"));
    assert!(!debug.contains("endpoint-secret"));
    assert!(!debug.contains("fragment-secret"));
    assert!(debug.contains("[REDACTED]"));
    assert!(debug.contains("Authorization"));
    assert!(debug.contains("api_key"));
}
