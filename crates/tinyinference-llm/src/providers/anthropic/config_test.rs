use super::config::*;

#[test]
fn only_first_party_anthropic_endpoint_selects_messages_api() {
    assert!(endpoint_is_anthropic_messages(
        "https://api.anthropic.com/v1"
    ));
    assert!(!endpoint_is_anthropic_messages(
        "https://anthropic-proxy.example/v1"
    ));
}

#[test]
fn builds_a_native_anthropic_model_with_the_configured_profile() {
    let model = build_anthropic_model(AnthropicConfig {
        endpoint: "https://api.anthropic.com/v1",
        api_key: "sk-ant-secret",
        model: "claude-sonnet-4-6",
        temperature_override: Some(0.2),
        temperature_unsupported_models: &[],
        extra_headers: &[],
    });
    let profile = model.profile().expect("anthropic models expose a profile");
    assert_eq!(profile.provider.as_deref(), Some("anthropic"));
    assert_eq!(profile.model.as_deref(), Some("claude-sonnet-4-6"));
    assert!(profile.tool_calling);
    assert!(profile.streaming);
    let identity = model.cache_identity().expect("model identity");
    assert!(identity.contains("api.anthropic.com"));
    assert!(!identity.contains("sk-ant-secret"));
}

#[test]
fn debug_redacts_api_key() {
    let headers = vec![("anthropic-beta".to_string(), "secret-beta".to_string())];
    let config = AnthropicConfig {
        endpoint: "https://endpoint-user:endpoint-pass@api.anthropic.com/v1?token=query-secret#fragment-secret",
        api_key: "sk-ant-secret",
        model: "claude-sonnet-4-6",
        temperature_override: None,
        temperature_unsupported_models: &[],
        extra_headers: &headers,
    };
    let debug = format!("{config:?}");
    assert!(!debug.contains("sk-ant-secret"));
    assert!(!debug.contains("endpoint-user"));
    assert!(!debug.contains("endpoint-pass"));
    assert!(!debug.contains("query-secret"));
    assert!(!debug.contains("fragment-secret"));
    assert!(!debug.contains("secret-beta"));
    assert!(debug.contains("anthropic-beta"));
    assert!(debug.contains("token"));
    assert!(debug.contains("[REDACTED]"));
}
