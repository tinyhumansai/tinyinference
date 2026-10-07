use super::*;
use crate::model::ProviderError;

#[test]
fn structured_status_ignores_unanchored_numbers() {
    assert_eq!(
        structured_http_status("API error (403 Forbidden): nope"),
        Some(403)
    );
    assert_eq!(structured_http_status("HTTP 404 Not Found"), Some(404));
    assert_eq!(structured_http_status("status: 401"), Some(401));
    assert_eq!(structured_http_status("408 Request Timeout"), Some(408));
    assert_eq!(structured_http_status("upstream took 450ms"), None);
    assert_eq!(structured_http_status("gpt-4-0409 returned nothing"), None);
}

#[test]
fn provider_failure_taxonomy_is_complete() {
    assert_eq!(
        classify_provider_failure(Some(401), None, "invalid api key"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(Some(429), None, "too many requests"),
        ProviderFailureClass::RateLimited
    );
    assert_eq!(
        classify_provider_failure(Some(429), None, "insufficient_balance"),
        ProviderFailureClass::NonRetryableRateLimit
    );
    assert_eq!(
        classify_provider_failure(None, None, "503 Service Unavailable"),
        ProviderFailureClass::UpstreamUnhealthy
    );
    assert_eq!(
        classify_provider_failure(None, None, "api key not set"),
        ProviderFailureClass::NonRetryable
    );
}

#[test]
fn structured_status_takes_precedence_over_message_heuristics() {
    assert_eq!(
        classify_provider_failure(Some(400), None, "proxy said bad gateway"),
        ProviderFailureClass::NonRetryable
    );
    assert_eq!(
        classify_provider_failure(None, Some("invalid_request"), "502 bad gateway"),
        ProviderFailureClass::NonRetryable
    );
}

#[test]
fn structured_provider_error_uses_the_same_classifier() {
    let error = ProviderError {
        partial_response: None,
        status: Some(429),
        code: Some("insufficient_quota".into()),
        message: "quota exhausted".into(),
        ..ProviderError::default()
    };
    assert_eq!(
        classify_provider_error(&error),
        ProviderFailureClass::NonRetryableRateLimit
    );
    assert!(!provider_error_is_retryable(&error));
}

#[test]
fn normalized_provider_retryability_is_authoritative() {
    let error = ProviderError {
        partial_response: None,
        message: "malformed streaming payload".into(),
        retryable: false,
        ..ProviderError::default()
    };
    assert_eq!(
        classify_provider_error(&error),
        ProviderFailureClass::NonRetryable
    );
    assert!(!provider_error_is_retryable(&error));
}

#[test]
fn retry_after_accepts_integer_and_fractional_seconds() {
    assert_eq!(parse_retry_after_ms("Retry-After: 5"), Some(5_000));
    assert_eq!(
        parse_retry_after_ms("retry_after: 2.5 seconds"),
        Some(2_500)
    );
    assert_eq!(parse_retry_after_ms("Retry-After 7"), Some(7_000));
    assert_eq!(parse_retry_after_ms("no retry hint"), None);
    assert_eq!(
        parse_retry_after_ms("Retry-After: 1000000000000000000000000000000"),
        None
    );
    assert_eq!(parse_retry_after_ms("Retry-After: inf"), None);
}

#[test]
fn business_limit_text_is_recognised_without_a_status() {
    for message in [
        "your plan does not include this model",
        "insufficient_balance",
        "out of credits",
        "error code 1311 returned",
        "1113",
    ] {
        assert!(contains_business_limit(message), "{message}");
    }
    for message in [
        "too many requests",
        "rate limit exceeded",
        "code 13110 upstream",
        "code 21113",
    ] {
        assert!(!contains_business_limit(message), "{message}");
    }
}

// ── Body phrase matchers ─────────────────────────────────────────────────────

/// Verbatim TAURI-RUST-C9A provider body: the Kiro IDE proxy wraps its own
/// 402 monthly-quota refusal inside a 500 envelope.
const C9A_BODY: &str = "kiro API error (500 Internal Server Error): \
    {\"error\":{\"message\":\"HTTP 402 from Kiro IDE: {\\\"message\\\":\\\"You have \
    reached the limit.\\\",\\\"reason\\\":\\\"MONTHLY_REQUEST_COUNT\\\"}\",\
    \"type\":\"server_error\"}}";

/// Verbatim TAURI-RUST-AFE Responses-API plan-cap body.
const AFE_BODY: &str = "openai Responses API error: {\"error\":{\"type\":\
    \"usage_limit_reached\",\"message\":\"The usage limit has been reached\",\
    \"plan_type\":\"plus\",\"resets_at\":1750000000}}";

#[test]
fn context_window_matches_wrapped_500_body() {
    assert!(is_context_window_exceeded_message(
        "{\"error\":{\"code\":500,\"message\":\"Context size has been exceeded.\",\"type\":\"server_error\"}}"
    ));
}

#[test]
fn context_window_matches_established_phrasings() {
    for body in [
        "This model's maximum context length is 8192 tokens",
        "request exceeds the context window of this model",
        "context length exceeded",
        "too many tokens in the prompt",
        "token limit exceeded",
        "prompt is too long for the selected model",
        "input is too long",
    ] {
        assert!(
            is_context_window_exceeded_message(body),
            "should match context-overflow body: {body}"
        );
    }
}

#[test]
fn context_window_matches_lmstudio_n_keep_body() {
    let body = "lmstudio API error (400 Bad Request): {\"error\":\"The number of tokens to keep from the initial prompt is greater than the context length (n_keep: 10978 >= n_ctx: 8192). Try to load the model with a larger context length, or provide a shorter input.\"}";
    assert!(is_context_window_exceeded_message(body));
    assert!(is_context_window_exceeded_message(
        "request rejected: prompt is greater than the context length of the loaded model"
    ));
    assert!(is_context_window_exceeded_message(
        "n_keep: 9000 >= n_ctx: 4096"
    ));
}

#[test]
fn context_window_ignores_unrelated_bodies() {
    for body in [
        "rate limit exceeded, retry after 30s",
        "Invalid request: model not found",
        "Insufficient budget",
        "tool call exceeded the allowed budget",
        // Only one of the paired n_keep/n_ctx tokens.
        "loaded model with n_ctx: 8192 and 32 layers",
    ] {
        assert!(
            !is_context_window_exceeded_message(body),
            "must NOT match unrelated body: {body}"
        );
    }
}

#[test]
fn context_window_token_rate_limits_are_not_overflow() {
    for body in [
        "Rate limit reached: too many tokens per minute (TPM) for this org",
        "rate_limit_exceeded: token limit exceeded, retry after 12s",
        "You have hit too many tokens per min; try again in 30s",
    ] {
        assert!(
            !is_context_window_exceeded_message(body),
            "TPM rate-limit must NOT match as context overflow: {body}"
        );
    }
    assert!(is_context_window_exceeded_message(
        "Request rejected: too many tokens in the input for this model"
    ));
}

#[test]
fn quota_exhausted_matches_verbatim_bodies() {
    assert!(body_indicates_quota_exhausted(C9A_BODY));
    assert!(body_indicates_quota_exhausted(AFE_BODY));
    assert!(body_indicates_quota_exhausted("usage_limit_reached"));
    assert!(body_indicates_quota_exhausted(
        "The usage limit has been reached"
    ));
}

#[test]
fn quota_exhausted_matches_common_phrasings() {
    for body in [
        "{\"reason\":\"MONTHLY_REQUEST_COUNT\"}",
        "You have reached the limit on your monthly requests",
        "monthly request quota reached",
        "monthly limit reached",
        "plan quota exceeded",
        "usage limit exceeded for this period",
    ] {
        assert!(
            body_indicates_quota_exhausted(body),
            "should match: {body:?}"
        );
    }
}

#[test]
fn quota_exhausted_ignores_unrelated_500_and_rate_limit() {
    for body in [
        "kiro API error (500 Internal Server Error): {\"error\":\
         {\"message\":\"upstream connection reset\",\"type\":\"server_error\"}}",
        "rate_limit_exceeded: too many requests, retry after 12s",
        "429 Too Many Requests",
        "context length exceeded: reduce the number of tokens",
    ] {
        assert!(
            !body_indicates_quota_exhausted(body),
            "should NOT match: {body:?}"
        );
    }
}

#[test]
fn rate_cap_matches_hxf_body_but_not_transient_or_context() {
    assert!(is_provider_rate_cap_exceeded_message(
        "groq API error (413 Payload Too Large): {\"error\":{\"message\":\"Request too large \
         for model `openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` on \
         tokens per minute (TPM): Limit 8000, Requested 42084.\",\"code\":\"rate_limit_exceeded\"}}"
    ));
    assert!(!is_provider_rate_cap_exceeded_message(
        "groq API error (429 Too Many Requests): Rate limit reached. Please try again in 2.5s."
    ));
    assert!(!is_provider_rate_cap_exceeded_message(
        "openai API error (400): This model's maximum context length is 8192 tokens"
    ));
    assert!(!is_provider_rate_cap_exceeded_message(
        "openai API error (413 Payload Too Large): request entity too large"
    ));
}

#[test]
fn insufficient_credits_matches_phrasings_and_ignores_unrelated() {
    for body in [
        "This request requires more credits, or fewer max_tokens. You requested up to 65536 tokens, but can only afford 4096",
        "Insufficient credits",
        "insufficient balance",
        "insufficient funds",
        "Payment Required",
    ] {
        assert!(
            body_indicates_insufficient_credits(body),
            "should match: {body:?}"
        );
    }
    assert!(!body_indicates_insufficient_credits(
        "{\"error\":{\"message\":\"some unrelated condition\"}}"
    ));
    // Quota and credits are distinct buckets: the 500-wrapped C9A body is quota.
    assert!(!body_indicates_insufficient_credits(C9A_BODY));
}

#[test]
fn local_and_hosted_ollama_body_matchers() {
    assert!(body_indicates_no_model_loaded(
        "{\"error\":\"No models loaded. Please load a model in the developer page\"}"
    ));
    assert!(!body_indicates_no_model_loaded("model not found"));

    assert!(body_indicates_ollama_cloud_internal_error(
        "{\"error\":\"Internal Server Error (ref: 3f2b1c7e-1111-2222-3333-444455556666)\"}"
    ));
    // A local daemon 500 has no `ref:` UUID.
    assert!(!body_indicates_ollama_cloud_internal_error(
        "{\"error\":\"Internal Server Error\"}"
    ));
}

#[test]
fn policy_moderation_and_upstream_body_matchers() {
    assert!(body_indicates_provider_access_policy_denied(
        "{\"error\":{\"type\":\"access_terminated_error\"}}"
    ));
    assert!(body_indicates_provider_access_policy_denied(
        "This endpoint is currently only available for Coding Agents"
    ));
    assert!(!body_indicates_provider_access_policy_denied("forbidden"));

    assert!(body_indicates_moderation_rejection(
        "{\"error\":\"Message rejected by Ombudsman\",\"score\":80}"
    ));
    assert!(body_indicates_moderation_rejection("{\"score\": 3}"));
    assert!(!body_indicates_moderation_rejection(
        "invalid request: score must be positive"
    ));

    assert!(body_indicates_custom_openai_upstream_bad_request(
        "{\"error\":{\"message\":\"Bad request to upstream provider\",\"type\":\"upstream_error\",\"status\":400}}"
    ));
    assert!(!body_indicates_custom_openai_upstream_bad_request(
        "Bad request to upstream provider"
    ));
}

#[test]
fn auth_key_error_body_matcher() {
    for body in [
        "{\"type\":\"authentication_error\"}",
        "{\"code\":\"invalid_api_key\"}",
        "Incorrect API key provided",
        "no api key supplied",
        "Invalid or missing API key",
    ] {
        assert!(
            body_indicates_auth_key_error(body),
            "should match: {body:?}"
        );
    }
    assert!(!body_indicates_auth_key_error("quota exceeded"));
    // Provider-specific clauses (e.g. OpenRouter "user not found") stay in the host.
    assert!(!body_indicates_auth_key_error("User not found."));
}

// ── String-level predicates and extractors (failure::text) ──────────────

#[test]
fn extract_provider_error_detail_pulls_openai_message() {
    let raw = r#"custom_openai API error (404 Not Found): {"error":{"message":"Project `proj_X` does not have access to model `gpt-5.5`","type":"invalid_request_error","param":null,"code":"model_not_found"}}"#;
    let detail = extract_provider_error_detail(raw).expect("expected JSON message");
    assert!(
        detail.contains("does not have access to model"),
        "got: {detail}"
    );
    assert!(detail.contains("gpt-5.5"));
}

#[test]
fn extract_provider_error_detail_returns_none_for_transport_errors() {
    // Plain transport failure — no provider JSON body to quote. Surfacing
    // raw transport text would leak internal infra URLs.
    let raw = "error sending request for url (https://internal-api.example.invalid/openai/v1/chat/completions)";
    assert!(extract_provider_error_detail(raw).is_none());
}

#[test]
fn extract_provider_error_detail_decodes_standard_json_escapes() {
    // The escaped solidus matters most in practice: provider bodies routinely
    // carry URLs as `https:\/\/…`. `\r`, `\b` and `\f` complete the JSON
    // standard set. Every one of them must decode — none may survive as a
    // literal backslash.
    let raw = r#"provider API error (400): {"error":{"message":"GET https:\/\/api.example.com\/v1\/models failed\r\nretry\tlater\b\f done"}}"#;
    let detail = extract_provider_error_detail(raw).expect("expected JSON message");
    assert!(
        !detail.contains('\\'),
        "no escape should survive decoding, got: {detail:?}"
    );
    assert!(
        detail.contains("https://api.example.com/v1/models"),
        "escaped solidus must decode, got: {detail:?}"
    );
    assert!(
        detail.contains('\r'),
        "carriage return must decode: {detail:?}"
    );
    assert!(detail.contains('\n'), "newline must decode: {detail:?}");
    assert!(detail.contains('\t'), "tab must decode: {detail:?}");
    assert!(
        detail.contains('\u{8}'),
        "backspace must decode: {detail:?}"
    );
    assert!(
        detail.contains('\u{c}'),
        "form feed must decode: {detail:?}"
    );
}

#[test]
fn extract_provider_error_detail_preserves_unknown_escapes() {
    // Genuinely unsupported sequences keep both characters — an unhandled
    // `\uXXXX` is better shown to the user as visible literal text than
    // silently mangled into a character nobody asked for. `\"` and `\\`
    // keep their existing meaning.
    let raw = r#"provider API error: {"error":{"message":"quote \" hi and slash \\ then unicode \u263A and \q \s \&"}}"#;
    let detail = extract_provider_error_detail(raw).expect("expected JSON message");
    assert!(detail.contains("quote \" hi"), "got: {detail:?}");
    assert!(detail.contains("slash \\ then"), "got: {detail:?}");
    assert!(detail.contains(r"\u263A"), "got: {detail:?}");
    assert!(detail.contains(r"\q"), "got: {detail:?}");
    assert!(detail.contains(r"\s"), "got: {detail:?}");
    assert!(detail.contains(r"\&"), "got: {detail:?}");
}

#[test]
fn retry_after_secs_rounds_up_and_reads_camel_case_and_quoted_keys() {
    assert_eq!(parse_retry_after_secs("Retry-After: 30"), Some(30));
    assert_eq!(parse_retry_after_secs("retry_after 1.2"), Some(2));
    assert_eq!(parse_retry_after_secs(r#"{"retry_after": 30}"#), Some(30));
    assert_eq!(parse_retry_after_secs(r#"{"retryAfter": 7}"#), Some(7));
    assert_eq!(parse_retry_after_secs("no hint here"), None);
}

#[test]
fn text_predicates_match_their_anchors() {
    assert!(is_empty_provider_response_text(
        "model returned an empty response"
    ));
    assert!(!is_empty_provider_response_text(
        "summarizer returned empty response"
    ));
    assert!(is_malformed_tool_history_text(
        "role 'tool' must match a tool_call"
    ));
    assert!(is_connection_dropped_text("error sending request for url"));
    assert!(!is_connection_dropped_text("request timed out"));
    assert!(is_provider_request_rejected_text(
        "openrouter api error (422 unprocessable)"
    ));
    assert!(!is_provider_request_rejected_text(
        "offset 400 of 404 bytes"
    ));
    assert!(is_transient_unavailability_text("model is overloaded"));
    assert!(!is_transient_unavailability_text(
        "model unavailable on this endpoint"
    ));
    assert!(is_fallback_chain_exhausted(
        "All providers/models failed. Attempts:"
    ));
}

#[test]
fn provider_name_is_extracted_from_api_error_prefix() {
    assert_eq!(
        extract_provider_name("OpenRouter API error (429 Too Many Requests): x"),
        Some("openrouter".to_string())
    );
    assert_eq!(extract_provider_name("https://x API error (1)"), None);
    assert_eq!(extract_provider_name("plain failure"), None);
}

#[test]
fn provider_detail_is_quoted_below_the_summary() {
    let raw = r#"p API error (400): {"error":{"message":"bad param"}}"#;
    assert_eq!(
        with_provider_detail("Summary.", raw),
        "Summary.\n\n> bad param"
    );
    assert_eq!(with_provider_detail("Summary.", "boom"), "Summary.");
}

#[test]
fn provider_detail_is_bounded_and_ellipsized() {
    let long = "x".repeat(400);
    let raw = format!(r#"{{"message":"{long}"}}"#);
    let detail = extract_provider_error_detail(&raw).unwrap();
    assert!(detail.chars().count() <= 303);
    assert!(detail.ends_with("..."));
}

#[test]
fn status_word_predicates_match_the_phrases_the_chat_ladder_branches_on() {
    assert!(is_rate_limit_text("rate limit reached"));
    assert!(is_rate_limit_text("http 429"));
    assert!(!is_rate_limit_text("rate cap"));
    assert!(is_timeout_text("request timed out"));
    assert!(is_timeout_text("timeout"));
    assert!(!is_timeout_text("time out"));
    assert!(is_auth_error_text("401"));
    assert!(is_auth_error_text("invalid api key"));
    assert!(is_auth_error_text("unauthorized"));
    assert!(!is_auth_error_text("forbidden"));
    assert!(is_payment_required_text("402"));
    assert!(is_payment_required_text("payment required"));
    assert!(is_payment_required_text("insufficient balance"));
    assert!(!is_payment_required_text("insufficient budget"));
    assert!(is_server_error_text("internal server error"));
    assert!(is_server_error_text("503"));
    assert!(!is_server_error_text("502 bad gateway"));
    assert!(is_context_length_text("context length exceeded"));
    assert!(is_context_length_text("context token cap"));
    assert!(!is_context_length_text("length only"));
    assert!(is_model_unavailable_text("model foo not found"));
    assert!(is_model_unavailable_text("the model does not have access"));
    assert!(!is_model_unavailable_text("not found"));
    assert!(is_vision_unsupported_text("capability=vision"));
    assert!(is_vision_unsupported_text("does not support vision input"));
    assert!(is_codex_token_expired_text(
        "codex authentication token is expired"
    ));
    assert!(!is_codex_token_expired_text(
        "authentication token is expired"
    ));
}

#[test]
fn recoverable_failure_text_matches_transient_markers_only() {
    assert!(is_recoverable_failure_text("Request TIMED OUT"));
    assert!(is_recoverable_failure_text(
        "dns error: failed to lookup address"
    ));
    assert!(is_recoverable_failure_text("HTTP 503 Service Unavailable"));
    assert!(is_recoverable_failure_text("Too Many Requests"));
    assert!(!is_recoverable_failure_text("permission denied"));
    assert!(!is_recoverable_failure_text("exit code 1"));
}

#[test]
fn context_window_matches_dashscope_input_length_range() {
    let body = "Provider returned error: {\"error\":{\"code\":\"invalid_parameter_error\",\
                \"message\":\"Range of input length should be [1, 98304]\"}}";
    assert!(is_context_window_exceeded_message(body));
}
