//! Classifier tests. The branch-order corpus is ported from OpenCompany's
//! `probe_tests_classify.rs`; the vendor-body fixtures are the shapes named in
//! the plan (all keys and ids are obviously fake).

use std::time::Duration;

use proptest::prelude::*;

use super::*;
use crate::error::HubError;

fn raw(text: &str) -> ReasonCode {
    classify_text(None, None, text, false).reason
}

fn body(status: u16, text: &str) -> ProviderFailure {
    classify(status, &[], text)
}

// ---- the branch-order cases ----------------------------------------------

#[test]
fn a_407_proxy_challenge_is_unknown_not_auth() {
    assert_eq!(
        raw("HTTP 407 Proxy Authentication Required"),
        ReasonCode::Unknown
    );
    assert!(!ReasonCode::Unknown.destroys_credential());
}

#[test]
fn a_bare_waf_403_is_unknown_not_auth() {
    assert_eq!(raw("error from cloudflare: 403"), ReasonCode::Unknown);
    assert_eq!(raw("502 Bad Gateway"), ReasonCode::Unknown);
    assert_eq!(raw("504 Gateway Timeout"), ReasonCode::Unknown);
}

#[test]
fn a_403_that_names_no_credential_refusal_keeps_the_key() {
    for text in [
        "403: Input token count + max_tokens must be less than the context length of the model being queried",
        "403: Forbidden (insufficient permissions, guardrail block, or moderation flag)",
        "403: Country, region, or territory not supported",
        "403: Your API key does not have permission to use the specified resource.",
        "403: PERMISSION_DENIED",
        "403: not allowed due to permission restrictions",
        "403: Ask your team admin for permission.",
        "403: PermissionDeniedError",
        "403: FireRouter is not available for Fireworks accounts with data residency enabled",
    ] {
        assert!(
            !raw(text).destroys_credential(),
            "this 403 must not delete the key: {text:?}"
        );
        assert!(!body(403, text).reason.destroys_credential(), "{text:?}");
    }
}

#[test]
fn a_403_that_does_name_a_credential_refusal_is_still_auth() {
    assert_eq!(
        raw("403: The API key you provided is invalid"),
        ReasonCode::Auth
    );
    assert_eq!(raw("403: You must provide an API key"), ReasonCode::Auth);
    assert_eq!(body(403, "invalid credential").reason, ReasonCode::Auth);
    assert_eq!(
        body(
            403,
            "Google: API key not valid. Please pass a valid API key."
        )
        .reason,
        ReasonCode::Auth
    );
}

#[test]
fn the_reason_phrase_is_not_part_of_what_is_classified() {
    // A 403 with a body that says nothing about the credential must not become
    // auth just because the status is 403; and a 401 with an empty body is
    // auth because the status alone is the signal.
    let failure = body(403, r#"{"error":"context length exceeded"}"#);
    assert!(!failure.reason.destroys_credential());
    let failure = body(401, "");
    assert_eq!(failure.reason, ReasonCode::Auth);
    assert_eq!(failure.retry, Retry::Never);
    assert_eq!(failure.status, Some(401));
}

#[test]
fn a_400_about_our_request_shape_does_not_delete_the_key() {
    assert!(
        !raw("400: Bearer authentication is not supported, use x-api-key").destroys_credential()
    );
    assert!(
        !raw("424: dependent request failed (Remote MCP authentication)").destroys_credential()
    );
    assert_eq!(raw("401: authentication_error"), ReasonCode::Auth);
    assert_eq!(
        raw("400: Authentication Fails (no such user)"),
        ReasonCode::Auth
    );
}

#[test]
fn a_status_code_inside_an_id_does_not_match() {
    assert_eq!(raw("request id req_1403 failed"), ReasonCode::Unknown);
    assert_eq!(raw("trace 4032 aborted"), ReasonCode::Unknown);
    assert_eq!(raw("model gpt-4010 is odd"), ReasonCode::Unknown);
    assert_eq!(raw("request id req_4291 failed"), ReasonCode::Unknown);
}

// ---- every class ------------------------------------------------------------

#[test]
fn every_class_has_a_real_error_string_that_reaches_it() {
    let cases: &[(&str, ReasonCode)] = &[
        ("401 Unauthorized", ReasonCode::Auth),
        ("Incorrect API key provided", ReasonCode::Auth),
        ("invalid_api_key", ReasonCode::Auth),
        (
            "The model `gpt-5.6-sol-pro` does not exist",
            ReasonCode::Model,
        ),
        ("model_not_found", ReasonCode::Model),
        (
            "Model 'anthropic/claude-sonnet-5' is not available",
            ReasonCode::Model,
        ),
        ("You exceeded your current quota", ReasonCode::Quota),
        ("insufficient credits", ReasonCode::Quota),
        ("429 Too Many Requests", ReasonCode::RateLimited),
        ("404 Not Found", ReasonCode::Endpoint),
        ("dns error: not found", ReasonCode::Endpoint),
        ("operation timed out", ReasonCode::Timeout),
        ("request timeout after 10s", ReasonCode::Timeout),
        ("something nobody has seen before", ReasonCode::Unknown),
        ("", ReasonCode::Unknown),
    ];
    for (text, expected) in cases {
        assert_eq!(raw(text), *expected, "classifying {text:?}");
    }
}

#[test]
fn a_missing_model_is_not_read_as_a_missing_endpoint() {
    assert_eq!(raw("The model `acme-1` was not found"), ReasonCode::Model);
    assert_eq!(raw("404 page not found"), ReasonCode::Endpoint);
}

#[test]
fn dns_and_refusal_are_endpoint_facts_not_unknowns() {
    for text in [
        "connection refused",
        "no such host",
        "could not resolve host",
        "temporary failure in name resolution",
        "dns error",
        "network is unreachable",
        "connection reset by peer",
    ] {
        assert_eq!(raw(text), ReasonCode::Endpoint, "{text}");
    }
}

#[test]
fn classification_is_case_insensitive_and_ignores_surrounding_noise() {
    assert_eq!(raw("  \n401 UNAUTHORIZED\n "), ReasonCode::Auth);
}

// ---- D12: quota vs rate limit ----------------------------------------------------

#[test]
fn an_anthropic_spend_cap_is_quota_and_never_retried() {
    let text = r#"{"type":"error","error":{"type":"rate_limit_error","message":"You have reached your specified API usage limits. You will regain access on 2026-10-01 at 00:00 UTC."}}"#;
    let failure = body(429, text);
    assert_eq!(failure.reason, ReasonCode::Quota);
    assert_eq!(failure.retry, Retry::Never);
    assert_eq!(failure.status, Some(429));
    // The same phrase as a 400 (Anthropic's other spelling of the same cap).
    assert_eq!(
        body(400, "You have reached your specified API usage limits.").reason,
        ReasonCode::Quota
    );
}

#[test]
fn an_enforced_spend_limit_code_is_quota() {
    let failure = body(
        429,
        r#"{"error":{"code":"enforced_spend_limit_reached","message":"Monthly spend limit hit"}}"#,
    );
    assert_eq!(failure.reason, ReasonCode::Quota);
    assert_eq!(failure.retry, Retry::Never);
    assert_eq!(
        failure.provider_code.as_deref(),
        Some("enforced_spend_limit_reached")
    );
}

#[test]
fn a_402_is_quota_whatever_the_body_says() {
    let failure = body(402, "");
    assert_eq!(failure.reason, ReasonCode::Quota);
    assert_eq!(failure.retry, Retry::Never);
    let failure = body(
        402,
        "This request requires more credits, or fewer max_tokens. You requested up to 4096 tokens, but can only afford 1000",
    );
    assert_eq!(failure.reason, ReasonCode::Quota);
}

#[test]
fn an_openai_insufficient_quota_429_is_quota_not_a_cooldown() {
    let failure = body(
        429,
        r#"{"error":{"message":"You exceeded your current quota, please check your plan and billing details.","type":"insufficient_quota","code":"insufficient_quota"}}"#,
    );
    assert_eq!(failure.reason, ReasonCode::Quota);
    assert_eq!(failure.retry, Retry::Never);
    assert_eq!(failure.provider_code.as_deref(), Some("insufficient_quota"));
}

#[test]
fn an_openai_slow_down_429_is_rate_limited_and_retryable() {
    let failure = body(
        429,
        r#"{"error":{"message":"Slow down","type":"requests","code":"slow_down"}}"#,
    );
    assert_eq!(failure.reason, ReasonCode::RateLimited);
    assert_eq!(failure.retry, Retry::Later(None));
    assert_eq!(failure.provider_code.as_deref(), Some("slow_down"));
}

#[test]
fn an_openai_tokens_per_minute_429_carries_the_go_style_delay() {
    let text = "Rate limit reached for gpt-4 in organization org-fake on tokens per min (TPM): Limit 10000, Used 9000. Please try again in 6m0s.";
    let failure = body(429, text);
    assert_eq!(failure.reason, ReasonCode::RateLimited);
    assert_eq!(failure.retry, Retry::Later(Some(Duration::from_secs(360))));
}

#[test]
fn a_google_per_minute_quota_message_is_a_rate_limit_not_a_spend_cap() {
    let failure = body(
        429,
        "Quota exceeded for quota metric 'Generate Content API requests per minute' and limit 'GenerateContent request limit per minute'",
    );
    assert_eq!(failure.reason, ReasonCode::RateLimited);
    assert_eq!(failure.retry, Retry::Later(None));
}

#[test]
fn a_plain_429_honours_retry_after_headers() {
    let failure = classify(429, &[("Retry-After", "30")], "");
    assert_eq!(failure.reason, ReasonCode::RateLimited);
    assert_eq!(failure.retry, Retry::Later(Some(Duration::from_secs(30))));
    let failure = classify(429, &[("retry-after-ms", "1500")], "");
    assert_eq!(
        failure.retry,
        Retry::Later(Some(Duration::from_millis(1500)))
    );
    // ms header wins over the seconds header.
    let failure = classify(429, &[("retry-after", "30"), ("retry-after-ms", "250")], "");
    assert_eq!(
        failure.retry,
        Retry::Later(Some(Duration::from_millis(250)))
    );
    // A body-only hint is found too.
    let failure = classify(429, &[], "Retry-After: 7");
    assert_eq!(failure.retry, Retry::Later(Some(Duration::from_secs(7))));
    let failure = classify(429, &[], "please try again in 250ms");
    assert_eq!(
        failure.retry,
        Retry::Later(Some(Duration::from_millis(250)))
    );
}

#[test]
fn a_retry_after_is_capped_so_a_hostile_header_cannot_park_a_caller() {
    let failure = classify(429, &[("retry-after", "99999999999")], "");
    assert_eq!(failure.retry, Retry::Later(Some(MAX_RETRY_AFTER)));
    let failure = classify(429, &[("retry-after-ms", "-5")], "");
    assert_eq!(failure.retry, Retry::Later(None));
    let failure = classify(429, &[("retry-after-ms", "NaN")], "");
    assert_eq!(failure.retry, Retry::Later(None));
}

#[test]
fn a_soft_quota_word_is_trusted_only_when_nothing_says_pace() {
    // Generic "billing" wording with no rate marker: out of credit.
    assert_eq!(
        body(403, "Please check your billing details").reason,
        ReasonCode::Quota
    );
    // The same wording with a retry-after is a pace problem.
    let failure = classify(429, &[("retry-after", "10")], "quota window resets soon");
    assert_eq!(failure.reason, ReasonCode::RateLimited);
    assert_eq!(failure.retry, Retry::Later(Some(Duration::from_secs(10))));
    // "insufficient permissions" is an access problem, never a credit one.
    assert_eq!(
        body(403, "insufficient permissions for this route").reason,
        ReasonCode::Unknown
    );
    assert_eq!(body(403, "insufficient scope").reason, ReasonCode::Unknown);
}

#[test]
fn the_openhuman_credit_and_quota_predicates_are_covered() {
    for text in [
        "requires more credits",
        "You can only afford 10",
        "insufficient balance",
        "insufficient funds",
        "payment required",
        "monthly_request_count exceeded",
        "monthly limit reached",
        "monthly quota exhausted",
        "usage_limit_reached",
        "usage limit has been reached",
        "Your credit balance is too low to access the API",
        "billing_hard_limit_reached",
    ] {
        let failure = body(400, text);
        assert_eq!(failure.reason, ReasonCode::Quota, "{text}");
        assert_eq!(failure.retry, Retry::Never, "{text}");
    }
}

// ---- OpenHuman predicates ---------------------------------------------------------

#[test]
fn a_context_window_overflow_is_a_model_failure_not_an_auth_failure() {
    for text in [
        "This model's maximum context length is 8192 tokens",
        "context_length_exceeded",
        "prompt is too long: 210000 tokens",
        "Input is too long for requested model",
        "the request exceeds the context window of this model",
        "n_keep (5000) >= n_ctx (4096)",
        "too many tokens in the prompt",
    ] {
        let failure = body(400, text);
        assert_eq!(failure.reason, ReasonCode::Model, "{text}");
        assert_eq!(failure.retry, Retry::Never);
        assert_eq!(
            failure.provider_code.as_deref(),
            Some("context_length_exceeded"),
            "{text}"
        );
    }
    // "too many tokens per minute" is pace, not size.
    assert_eq!(
        body(429, "too many tokens per minute").reason,
        ReasonCode::RateLimited
    );
}

#[test]
fn a_local_runtime_with_no_model_loaded_says_so() {
    let failure = body(
        400,
        "No models loaded. Please load a model in the developer page.",
    );
    assert_eq!(failure.reason, ReasonCode::Model);
    assert_eq!(failure.provider_code.as_deref(), Some("no_model_loaded"));
    // Only a 400 counts; the phrase elsewhere is just a missing model.
    assert_ne!(
        body(500, "no models loaded").provider_code.as_deref(),
        Some("no_model_loaded")
    );
}

#[test]
fn ollama_clouds_internal_500_is_retryable_and_keeps_the_key() {
    let failure = body(500, "Internal Server Error (ref: 0b8a9f2c-1234)");
    assert_eq!(failure.reason, ReasonCode::Unknown);
    assert_eq!(failure.retry, Retry::Later(None));
    assert_eq!(
        failure.provider_code.as_deref(),
        Some("provider_internal_error")
    );
}

#[test]
fn moderation_and_access_policy_rejections_keep_the_key() {
    let failure = body(
        400,
        r#"{"error":"Message rejected by moderation","score":0.98}"#,
    );
    assert_eq!(failure.reason, ReasonCode::Unknown);
    assert_eq!(failure.retry, Retry::Never);
    assert_eq!(failure.provider_code.as_deref(), Some("content_moderation"));
    let failure = body(
        403,
        r#"{"error":{"type":"access_terminated_error","message":"nope"}}"#,
    );
    assert_eq!(failure.reason, ReasonCode::Unknown);
    assert_eq!(failure.provider_code.as_deref(), Some("access_policy"));
    let failure = body(
        403,
        "This model is currently only available for coding agents",
    );
    assert_eq!(failure.provider_code.as_deref(), Some("access_policy"));
}

#[test]
fn openrouters_user_not_found_is_a_bad_key_only_on_openrouter() {
    let text = r#"{"error":{"message":"User not found.","code":401}}"#;
    assert_eq!(
        classify_for(Some("openrouter"), 403, &[], text).reason,
        ReasonCode::Auth
    );
    assert_ne!(
        classify_for(Some("groq"), 403, &[], "User not found.").reason,
        ReasonCode::Auth
    );
    assert_ne!(
        classify(403, &[], "User not found.").reason,
        ReasonCode::Auth
    );
}

#[test]
fn the_openhuman_auth_markers_are_recognised() {
    for text in [
        "Invalid or missing API key",
        "No API key supplied",
        "invalid authentication",
    ] {
        assert_eq!(body(403, text).reason, ReasonCode::Auth, "{text}");
    }
}

// ---- 5xx and unknowns -------------------------------------------------------------

#[test]
fn a_gateway_5xx_is_unknown_but_retryable_and_never_destroys_the_key() {
    for status in [500, 502, 503, 504, 529, 408] {
        let failure = body(status, "upstream connect error");
        assert!(!failure.reason.destroys_credential(), "{status}");
        assert!(
            matches!(failure.retry, Retry::Later(_)),
            "{status}: {:?}",
            failure.retry
        );
    }
    // A 4xx nobody recognises is not retried.
    let failure = body(418, "teapot");
    assert_eq!(failure.reason, ReasonCode::Unknown);
    assert_eq!(failure.retry, Retry::Never);
}

#[test]
fn transport_conditions_classify_without_reading_the_error_text() {
    let cases = [
        (
            TransportCondition::Timeout,
            ReasonCode::Timeout,
            Retry::Later(None),
        ),
        (
            TransportCondition::ConnectFailed,
            ReasonCode::Endpoint,
            Retry::Later(None),
        ),
        (
            TransportCondition::RedirectRefused,
            ReasonCode::Endpoint,
            Retry::Never,
        ),
        (TransportCondition::Other, ReasonCode::Unknown, Retry::Never),
    ];
    for (condition, reason, retry) in cases {
        // The detail text mentions /models; it must not steer the class.
        let failure = classify_transport(
            condition,
            "error sending request for url (https://x.test/v1/models)",
        );
        assert_eq!(failure.reason, reason);
        assert_eq!(failure.retry, retry);
        assert!(
            failure.raw.expose().contains("/models"),
            "the URL is still worth having in a log"
        );
        assert_eq!(failure.status, None);
    }
}

#[test]
fn the_request_url_never_reaches_the_classifier() {
    // Without URL stripping, "models" + "not found" reads as a missing model.
    let text = "404 not found for https://api.acme.test/v1/models";
    assert_eq!(body(404, text).reason, ReasonCode::Endpoint);
    assert_eq!(
        body(
            200,
            "error sending request for url (https://x.test/v1/models)"
        )
        .reason,
        ReasonCode::Unknown
    );
    // A URL that contains a status-looking or phrase-looking path is inert.
    assert_eq!(
        body(400, "see https://docs.test/401/invalid-api-key for help").reason,
        ReasonCode::Unknown
    );
}

#[test]
fn strip_urls_replaces_every_url_token() {
    assert_eq!(
        strip_urls("see https://a.test/x?y=1, then http://b.test."),
        "see <url>, then <url>."
    );
    assert_eq!(strip_urls("no urls here"), "no urls here");
    assert_eq!(strip_urls("HTTPS://UPPER.test/x end"), "<url> end");
    assert_eq!(strip_urls("(http://a.test/x)"), "(<url>)");
    assert_eq!(strip_urls(""), "");
    assert_eq!(strip_urls("http://only.test"), "<url>");
}

// ---- metadata extraction --------------------------------------------------------------

#[test]
fn provider_code_prefers_error_code_then_type_then_top_level() {
    let f = body(400, r#"{"error":{"code":"invalid_thing","type":"t"}}"#);
    assert_eq!(f.provider_code.as_deref(), Some("invalid_thing"));
    let f = body(400, r#"{"error":{"type":"overloaded_error"}}"#);
    assert_eq!(f.provider_code.as_deref(), Some("overloaded_error"));
    let f = body(400, r#"{"code":"top_level","type":"other"}"#);
    assert_eq!(f.provider_code.as_deref(), Some("top_level"));
    let f = body(400, r#"{"type":"error_only"}"#);
    assert_eq!(f.provider_code.as_deref(), Some("error_only"));
    let f = body(400, r#"{"error":{"code":429}}"#);
    assert_eq!(f.provider_code.as_deref(), Some("429"));
    // Unsafe or oversize codes are dropped, not displayed.
    assert_eq!(
        body(400, r#"{"error":{"code":"has space <script>"}}"#).provider_code,
        None
    );
    assert_eq!(
        body(
            400,
            &format!(r#"{{"error":{{"code":"{}"}}}}"#, "x".repeat(200))
        )
        .provider_code,
        None
    );
    assert_eq!(
        body(400, r#"{"error":{"code":{"nested":1}}}"#).provider_code,
        None
    );
    assert_eq!(body(400, "not json").provider_code, None);
    assert_eq!(body(400, r#"{"error":{"code":""}}"#).provider_code, None);
}

#[test]
fn request_ids_come_from_headers_first_then_the_body() {
    let f = classify(500, &[("X-Request-Id", "req_abc123")], "boom");
    assert_eq!(f.request_id.as_deref(), Some("req_abc123"));
    let f = classify(500, &[("request-id", "req_anthropic")], "boom");
    assert_eq!(f.request_id.as_deref(), Some("req_anthropic"));
    let f = classify(500, &[("openai-request-id", "req_openai")], "boom");
    assert_eq!(f.request_id.as_deref(), Some("req_openai"));
    let f = classify(500, &[], r#"{"request_id":"req_body"}"#);
    assert_eq!(f.request_id.as_deref(), Some("req_body"));
    let f = classify(500, &[], r#"{"error":{"request_id":"req_nested"}}"#);
    assert_eq!(f.request_id.as_deref(), Some("req_nested"));
    // A hostile id is dropped and the next source is tried.
    let f = classify(
        500,
        &[("x-request-id", "bad id\r\nSet-Cookie: x")],
        r#"{"request_id":"ok_1"}"#,
    );
    assert_eq!(f.request_id.as_deref(), Some("ok_1"));
    assert_eq!(classify(500, &[], "boom").request_id, None);
}

#[test]
fn the_raw_body_is_kept_log_only_and_never_displayed() {
    let secretish = "Authorization: Bearer sk-not-a-real-key rejected";
    let failure = body(401, secretish);
    // The raw text is scrubbed on the way in: the echoed credential is gone.
    assert_eq!(failure.raw.expose(), "Authorization: <redacted>");
    assert!(!format!("{failure}").contains("sk-not"));
    assert!(!format!("{failure:?}").contains("sk-not"));
    let error = HubError::Provider(failure);
    assert!(!error.to_string().contains("sk-not"));
    assert!(!format!("{error:?}").contains("sk-not"));
}

// ---- properties ---------------------------------------------------------------------

proptest! {
    #[test]
    fn classify_is_total_and_never_leaks_its_input(
        status in 0u16..1000,
        text in "\\PC{0,200}",
        header in "\\PC{0,20}",
    ) {
        let failure = classify(status, &[("retry-after", &header), ("x-request-id", &header)], &text);
        // Display and Debug never contain the raw text (unless it is trivially
        // a substring of our own fixed words).
        let shown = format!("{failure}{failure:?}");
        if text.chars().count() >= 12 && !shown.contains("provider_code") {
            prop_assert!(!shown.contains(&text), "{shown}");
        }
        prop_assert_eq!(failure.status, Some(status));
    }

    #[test]
    fn quota_auth_and_model_are_never_retried(status in 100u16..600, text in "\\PC{0,120}") {
        let failure = classify(status, &[], &text);
        if matches!(failure.reason, ReasonCode::Quota | ReasonCode::Auth | ReasonCode::Model) {
            prop_assert_eq!(failure.retry, Retry::Never);
        }
    }

    #[test]
    fn only_a_rejected_credential_ever_rolls_back_a_cloud_add(status in 100u16..600, text in "\\PC{0,120}") {
        let failure = classify(status, &[], &text);
        prop_assert_eq!(
            failure.rolls_back(crate::taxonomy::ProviderGroup::Cloud),
            failure.reason == ReasonCode::Auth
        );
    }

    #[test]
    fn a_status_code_embedded_in_a_longer_number_never_reads_as_auth(
        prefix in "[1-9][0-9]{0,3}", suffix in "[0-9]{1,3}",
    ) {
        let text = format!("request id {prefix}401{suffix} failed");
        prop_assert_ne!(raw(&text), ReasonCode::Auth);
    }

    #[test]
    fn appending_a_url_never_changes_the_class(status in 100u16..600, text in "[ -~]{0,100}") {
        prop_assume!(!text.to_ascii_lowercase().contains("http"));
        let plain = classify(status, &[], &text);
        let with_url = classify(
            status,
            &[],
            &format!("{text} see https://docs.test/models/404/invalid-api-key/insufficient_quota"),
        );
        prop_assert_eq!(plain.reason, with_url.reason);
        prop_assert_eq!(plain.retry, with_url.retry);
    }

    #[test]
    fn strip_urls_never_panics_and_removes_every_scheme(text in "[ -~]{0,120}") {
        let out = strip_urls(&text);
        prop_assert!(!out.to_ascii_lowercase().contains("http://"));
        prop_assert!(!out.to_ascii_lowercase().contains("https://"));
    }
}

// ---- regressions found while writing the suite --------------------------------------

#[test]
fn a_provider_cooldown_longer_than_thirty_seconds_is_reported_whole() {
    // core's `parse_retry_after_ms` is a backoff bound (30 s); using it for the
    // provider's own delay misreported a six-minute cooldown as thirty seconds.
    let failure = classify(429, &[("retry-after", "360")], "");
    assert_eq!(failure.retry, Retry::Later(Some(Duration::from_secs(360))));
    let failure = classify(429, &[("retry-after", "1.5")], "");
    assert_eq!(
        failure.retry,
        Retry::Later(Some(Duration::from_millis(1500)))
    );
    // The capped date form still goes through core.
    let failure = classify(429, &[("retry-after", "Wed, 21 Oct 2015 07:28:00 GMT")], "");
    assert_eq!(failure.retry, Retry::Later(Some(Duration::ZERO)));
}

#[test]
fn a_sentence_ending_period_after_a_duration_is_not_a_number() {
    // `parse_try_again_in` once collected the trailing "." as a digit run and
    // gave up on the whole delay.
    let f = classify(429, &[], "Rate limit hit. Please try again in 2s.");
    assert_eq!(f.retry, Retry::Later(Some(Duration::from_secs(2))));
    let f = classify(429, &[], "try again in 1h2m3s");
    assert_eq!(f.retry, Retry::Later(Some(Duration::from_secs(3723))));
    let f = classify(429, &[], "try again in 1.5s...");
    assert_eq!(f.retry, Retry::Later(Some(Duration::from_millis(1500))));
    // No unit, or no number, is no delay.
    assert_eq!(
        classify(429, &[], "try again in 5 parsecs").retry,
        Retry::Later(None)
    );
    assert_eq!(
        classify(429, &[], "try again in a while").retry,
        Retry::Later(None)
    );
    assert_eq!(
        classify(429, &[], "try again in ...").retry,
        Retry::Later(None)
    );
}

#[test]
fn trailing_punctuation_is_not_part_of_a_stripped_url() {
    assert_eq!(strip_urls("failed: https://a.test/x."), "failed: <url>.");
    assert_eq!(strip_urls("see https://a.test/x; then"), "see <url>; then");
}

#[test]
fn a_malformed_number_in_a_try_again_hint_is_no_hint() {
    // "1.2.3" collects as one digit run but does not parse.
    assert_eq!(
        classify(429, &[], "try again in 1.2.3s").retry,
        Retry::Later(None)
    );
}

// ---- regressions from the fresh-eyes review (round 1) -----------------------------------

#[test]
fn a_known_status_decides_the_status_rules_not_a_number_in_the_body() {
    // A 400 whose body merely contains `401` is not a rejected credential.
    let f = body(400, "max_tokens must be less than 401");
    assert_ne!(f.reason, ReasonCode::Auth);
    assert!(!f.reason.destroys_credential());
    // Neither is a 400/500 whose body mentions 404, 429 or 407.
    assert_ne!(
        body(400, "field 404 is invalid").reason,
        ReasonCode::Endpoint
    );
    assert_ne!(body(400, "max 429 items").reason, ReasonCode::RateLimited);
    assert!(matches!(
        body(500, "code 407 in trace").retry,
        Retry::Later(_)
    ));
    // The real statuses still classify.
    assert_eq!(body(401, "anything").reason, ReasonCode::Auth);
    assert_eq!(body(404, "anything").reason, ReasonCode::Endpoint);
    assert_eq!(body(429, "anything").reason, ReasonCode::RateLimited);
    assert_eq!(body(407, "anything").reason, ReasonCode::Unknown);
    // With no status (text-only) or a 2xx envelope the whole-token rule stays.
    assert_eq!(raw("error 401 from upstream"), ReasonCode::Auth);
    assert_eq!(
        body(200, r#"{"code":401,"msg":"bad"}"#).reason,
        ReasonCode::Auth
    );
}

#[test]
fn the_bare_word_unauthorized_is_auth_only_without_a_status() {
    // 403 entitlement wording must keep the key.
    for text in [
        "Unauthorized: model X is not enabled for your region",
        "unauthorized region",
    ] {
        let f = body(403, text);
        assert!(!f.reason.destroys_credential(), "{text}");
        assert!(!body(400, text).reason.destroys_credential(), "{text}");
    }
    // With no status the word is still a body signal (text-only errors), and a
    // real 401 needs no word at all.
    assert_eq!(raw("Unauthorized"), ReasonCode::Auth);
    assert_eq!(body(401, "Unauthorized").reason, ReasonCode::Auth);
}

#[test]
fn insufficient_access_is_never_out_of_credit_in_any_spelling() {
    for text in [
        r#"{"error":"insufficient_scope"}"#,
        r#"{"error":"insufficient_permissions"}"#,
        "insufficient permission to call this route",
        "Insufficient access for this resource",
        "insufficient_privileges",
    ] {
        let f = body(403, text);
        assert_ne!(f.reason, ReasonCode::Quota, "{text}");
        assert!(!f.reason.destroys_credential(), "{text}");
    }
    // A genuine soft credit wording is still quota.
    assert_eq!(
        body(403, "insufficient account balance for billing").reason,
        ReasonCode::Quota
    );
}

#[test]
fn an_id_or_word_containing_404_or_dns_is_not_an_endpoint_failure() {
    // Regression: bare substring matches read `req_9a404c...` as a 404 and any
    // word containing "dns" as a DNS failure, which rolls back a good local add.
    for text in [
        "unrecognised failure req_9a404cbeef",
        "trace id 1404 aborted",
        "the wordsdnsish thing",
        "ddnsx failure",
    ] {
        let f = body(400, text);
        assert_ne!(f.reason, ReasonCode::Endpoint, "{text}");
        assert!(
            !f.rolls_back(crate::taxonomy::ProviderGroup::Local),
            "{text}"
        );
    }
    // The whole tokens still classify.
    assert_eq!(raw("dns error: lookup failed"), ReasonCode::Endpoint);
    assert_eq!(raw("dns: no answer"), ReasonCode::Endpoint);
    assert_eq!(raw("HTTP 404 page"), ReasonCode::Endpoint);
}

#[test]
fn an_http_date_retry_after_is_resolved_against_the_supplied_instant_and_not_capped_at_thirty_seconds()
 {
    use std::time::{Duration, SystemTime};
    // Regression: the date form went through core's 30 s backoff cap.
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_445_412_475); // 2015-10-21 07:27:55 GMT
    let f = classify_at(
        now,
        None,
        429,
        &[("Retry-After", "Wed, 21 Oct 2015 07:33:55 GMT")],
        "",
    );
    assert_eq!(f.retry, Retry::Later(Some(Duration::from_secs(360))));
    // A date in the past is an immediate retry, not a panic.
    let f = classify_at(
        now,
        None,
        429,
        &[("retry-after", "Wed, 21 Oct 2015 07:00:00 GMT")],
        "",
    );
    assert_eq!(f.retry, Retry::Later(Some(Duration::ZERO)));
    // A far-future date is capped at the maximum.
    let f = classify_at(
        now,
        None,
        429,
        &[("retry-after", "Fri, 31 Dec 9999 23:59:59 GMT")],
        "",
    );
    assert_eq!(f.retry, Retry::Later(Some(MAX_RETRY_AFTER)));
    // Garbage is no hint at all.
    let f = classify_at(now, None, 429, &[("retry-after", "next tuesday")], "");
    assert_eq!(f.retry, Retry::Later(None));
    // The default entry points use the real clock and agree on the numeric form.
    assert_eq!(
        classify(429, &[("retry-after", "5")], "").retry,
        Retry::Later(Some(Duration::from_secs(5)))
    );
    assert_eq!(
        classify_for(Some("openai"), 429, &[("retry-after", "5")], "").retry,
        Retry::Later(Some(Duration::from_secs(5)))
    );
}

// ---- regressions from the fresh-eyes review (round 2) -----------------------------------

#[test]
fn raw_text_never_keeps_a_url_credential_or_an_echoed_authorization_value() {
    // Regression: `raw` was documented log-safe but held the body verbatim, so a
    // Gemini `?key=` URL or an echoed header reached whatever logged it.
    let raw = |text: &str| classify(400, &[], text).raw.expose().clone();
    let gemini = raw(
        "bad request for https://generativelanguage.test/v1/models?key=AIzaFAKE123&alt=json#frag",
    );
    assert_eq!(
        gemini,
        "bad request for https://generativelanguage.test/v1/models?<redacted>"
    );
    let userinfo = raw("failed at https://alice:hunter2@host.test/v1");
    assert!(
        !userinfo.contains("hunter2") && userinfo.contains("host.test"),
        "{userinfo}"
    );
    let fragment_only = raw("see https://a.test/x#secretfrag now");
    assert_eq!(fragment_only, "see https://a.test/x now");
    for (text, needle) in [
        ("x-api-key: sk-not-a-real-key sent", "sk-not-a-real-key"),
        ("X-Api-Key:sk-not-a-real-key sent", "sk-not-a-real-key"),
        (
            r#"{"headers":{"api-key":"sk-not-a-real-key"}}"#,
            "sk-not-a-real-key",
        ),
        ("Authorization: Basic dXNlcjpwYXNz", "dXNlcjpwYXNz"),
        ("bad Bearer sk-not-a-real-key, retry", "sk-not-a-real-key"),
    ] {
        let scrubbed = raw(text);
        assert!(!scrubbed.contains(needle), "{text} -> {scrubbed}");
        assert!(scrubbed.contains("<redacted>"), "{scrubbed}");
    }
    // Ordinary text and an already-redacted value are left alone.
    assert_eq!(scrub_log_text("plain failure text"), "plain failure text");
    assert_eq!(scrub_log_text("Bearer <redacted>"), "Bearer <redacted>");
    assert_eq!(scrub_log_text("Bearer"), "Bearer");
    // Transport detail goes through the same scrub.
    let t = classify_transport(
        TransportCondition::Other,
        "error for url (https://x.test/v1?api_key=sk-not-a-real-key)",
    );
    assert!(!t.raw.expose().contains("sk-not"), "{}", t.raw.expose());
}

#[test]
fn many_urls_in_a_hostile_body_are_handled_in_linear_time() {
    // Regression: `strip_urls` re-lowercased the remainder per URL (quadratic).
    let body = "http://a ".repeat(30_000);
    let start = std::time::Instant::now();
    let stripped = strip_urls(&body);
    assert_eq!(stripped.matches("<url>").count(), 30_000);
    let _ = scrub_log_text(&body);
    // Generous bound: quadratic behaviour took many seconds at this size.
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
}

proptest! {
    #[test]
    fn scrubbing_never_panics_and_never_grows_a_credential(text in "[ -~]{0,200}") {
        let scrubbed = scrub_log_text(&text);
        prop_assert!(!scrubbed.contains("?key="));
        let _ = strip_urls(&text);
    }
}

// ---- round 3 review regressions --------------------------------------------------------

#[test]
fn a_url_password_containing_url_punctuation_is_still_redacted() {
    for (text, secret) in [
        ("call https://user:pa,ss@host.test/v1 failed", "pa,ss"),
        ("call https://user:pa)ss@host.test/v1 failed", "pa)ss"),
        ("call https://user:pa'ss@host.test/v1 failed", "pa'ss"),
    ] {
        let scrubbed = scrub_log_text(text);
        assert!(!scrubbed.contains(secret), "{scrubbed}");
        assert!(scrubbed.contains("host.test"), "{scrubbed}");
    }
    // Trailing punctuation is still trimmed back off the URL.
    assert_eq!(
        scrub_log_text("see (https://a.test/x), then"),
        "see (https://a.test/x), then"
    );
    assert_eq!(
        strip_urls("see (https://a.test/x), then"),
        "see (<url>), then"
    );
}

#[test]
fn a_non_bearer_authorization_scheme_and_escaped_quotes_are_redacted_whole() {
    let a = scrub_log_text("Authorization: Token abc123 was rejected");
    assert!(!a.contains("abc123"), "{a}");
    let b = scrub_log_text(r#"{"password":"ab\"cd-secret","other":"ok"}"#);
    assert!(
        !b.contains("cd-secret") && b.contains(r#""other":"ok""#),
        "{b}"
    );
    let c = scrub_log_text("x-api-key: sk one two\nnext line");
    assert!(!c.contains("one two") && c.contains("next line"), "{c}");
}

#[test]
fn a_server_error_is_never_read_as_a_missing_model_or_an_empty_account() {
    // Regression (review round 3): body phrases ran under an authoritative 5xx.
    for (status, text) in [
        (503, "Service is not available, please retry"),
        (500, "internal error: billing service unavailable"),
        (502, "the model gateway does not exist right now"),
        (504, "upstream said: maximum context length"),
    ] {
        let f = body(status, text);
        assert!(
            !matches!(f.reason, ReasonCode::Model | ReasonCode::Quota),
            "{status} {text}: {:?}",
            f.reason
        );
        assert!(
            matches!(f.retry, Retry::Later(_)),
            "{status} {text}: {:?}",
            f.retry
        );
    }
    // A 404 about the endpoint is an endpoint failure; one about a model is a model failure.
    assert_eq!(
        body(404, "The requested endpoint does not exist").reason,
        ReasonCode::Endpoint
    );
    assert_eq!(
        body(404, "The model `x` does not exist").reason,
        ReasonCode::Model
    );
    assert_eq!(
        body(404, "The API deployment for this resource does not exist").reason,
        ReasonCode::Model
    );
    // Explicit spend codes still win on any status.
    assert_eq!(body(500, "insufficient_quota").reason, ReasonCode::Quota);
}

#[test]
fn a_429_is_a_rate_limit_unless_it_says_the_spend_is_gone() {
    // Regression: a bare "quota" on a 429 (Google RESOURCE_EXHAUSTED) was a
    // permanent stop.
    let f = body(429, "Resource has been exhausted (e.g. check quota).");
    assert_eq!(f.reason, ReasonCode::RateLimited);
    assert_eq!(f.retry, Retry::Later(None));
    assert_eq!(
        body(429, r#"{"error":{"status":"RESOURCE_EXHAUSTED"}}"#).reason,
        ReasonCode::RateLimited
    );
    assert_eq!(
        body(429, "please check your quota").reason,
        ReasonCode::RateLimited
    );
    // Hard phrases still win, and the same soft wording off a 429 is still credit.
    assert_eq!(
        body(429, "You exceeded your current quota").reason,
        ReasonCode::Quota
    );
    assert_eq!(
        body(403, "check your quota and billing").reason,
        ReasonCode::Quota
    );
    // A proxy-page 429 (Cloudflare 1015) is retryable, not a permanent Unknown.
    let f = body(429, "cloudflare error 1015");
    assert_eq!(f.reason, ReasonCode::Unknown);
    assert!(matches!(f.retry, Retry::Later(_)));
}

#[test]
fn any_credential_named_json_member_is_redacted_from_raw() {
    // Regression (review round 4): only an 8-name list of exact members was.
    let scrubbed = scrub_log_text(
        r#"{"client_secret":"s3","refresh_token":"rt","token":"t","key":"k","subscription-key":"sk","note":"keep","n":5}"#,
    );
    for leaked in ["s3", "\"rt\"", "\"t\"", "\"k\"", "\"sk\""] {
        assert!(!scrubbed.contains(leaked), "{leaked} in {scrubbed}");
    }
    assert!(
        scrubbed.contains(r#""note":"keep""#) && scrubbed.contains(r#""n":5"#),
        "{scrubbed}"
    );
    let cookies = scrub_log_text(
        "Set-Cookie: sid=abc123; HttpOnly\nCookie: a=b\nProxy-Authorization: Basic zzz\nok line",
    );
    assert!(
        !cookies.contains("abc123") && !cookies.contains("a=b") && !cookies.contains("zzz"),
        "{cookies}"
    );
    assert!(cookies.contains("ok line"), "{cookies}");
    // A member name is only matched as a name (before a colon), not in prose.
    assert_eq!(
        scrub_log_text(r#"the "token" was fine"#),
        r#"the "token" was fine"#
    );
    // An unterminated string does not panic or loop.
    let _ = scrub_log_text(r#"{"token":"abc"#);
    let _ = scrub_log_text(r#"{"token"#);
}

#[test]
fn stray_quotes_in_the_surrounding_text_do_not_hide_a_json_credential() {
    // Regression (review round 5): pairing quotes from the start of the text
    // let one odd quote shift every later member.
    for text in [
        r#"upstream said "bad {"api_key":"sk-LEAK1"}"#,
        r#"the "model" x said "oops" then {"client_secret":"sk-LEAK2","ok":1}"#,
        r#"unterminated " quote then {"token":"sk-LEAK3"}"#,
        r#"{"a":"b","password":"sk-LEAK4"}"#,
    ] {
        let scrubbed = scrub_log_text(text);
        assert!(!scrubbed.contains("sk-LEAK"), "{text} -> {scrubbed}");
        assert!(scrubbed.contains("<redacted>"), "{scrubbed}");
    }
    let kept = scrub_log_text(r#"upstream said "bad {"note":"keep","n":5}"#);
    assert!(kept.contains(r#""note":"keep""#), "{kept}");
    // Two adjacent credential members are both redacted (the first
    // implementation re-read the closing quote as an opening one).
    let two = scrub_log_text(r#"{"client_secret":"a1","refresh_token":"b2","x":"y"}"#);
    assert!(
        !two.contains("a1") && !two.contains("b2") && two.contains(r#""x":"y""#),
        "{two}"
    );
}

proptest! {
    #[test]
    fn a_credential_member_is_always_redacted_whatever_surrounds_it(
        // A prefix with no quote or backslash cannot open a string, so the
        // member is well formed; the suffix is unconstrained.
        prefix in "[ !#-\\[\\]-~]{0,40}", suffix in "[ -~]{0,40}", value in "sk-[A-Za-z0-9]{8,16}",
    ) {
        let text = format!("{prefix}{{\"api_key\":\"{value}\"}}{suffix}");
        let scrubbed = scrub_log_text(&text);
        prop_assert!(!scrubbed.contains(&value), "{scrubbed}");
    }
}

#[test]
fn a_credential_inside_a_json_document_carried_as_a_string_is_redacted() {
    // Regression (review round 6): gateways stringify a nested provider body,
    // so its quotes arrive escaped.
    let scrubbed = scrub_log_text(r#"{"error":"{\"api_key\":\"sk-LEAK1\",\"ok\":1}"}"#);
    assert!(!scrubbed.contains("sk-LEAK1"), "{scrubbed}");
    assert!(scrubbed.contains("<redacted>"), "{scrubbed}");
    // The unescaped and escaped forms in one text are both handled.
    let both = scrub_log_text(r#"{"token":"sk-LEAK2","echo":"{\"password\":\"sk-LEAK3\"}"}"#);
    assert!(!both.contains("sk-LEAK"), "{both}");
}

#[test]
fn a_secret_containing_an_escaped_quote_inside_an_escaped_json_string_is_redacted_whole() {
    // Regression (review round 7): the escaped-quote pass ended the value at
    // the first `\"`, leaving the tail of the secret in the log text.
    let text = r#"{"error":"{\"token\":\"a\\\"b\"}"}"#;
    let scrubbed = scrub_log_text(text);
    assert!(
        !scrubbed.contains("a\\\"b") && !scrubbed.contains("\"b\\\""),
        "{scrubbed}"
    );
    assert_eq!(scrubbed, r#"{"error":"{\"token\":\"<redacted>\"}"}"#);
}

#[test]
fn a_secret_ending_in_a_backslash_inside_an_escaped_json_string_keeps_the_structure() {
    // Regression (review round 8): a secret ending in `\\` closes with five
    // backslashes before the quote, which the "exactly one" rule skipped,
    // over-redacting into the next member.
    let text = r#"{"error":"{\"token\":\"ab\\\\\",\"x\":\"keep\"}"}"#;
    let scrubbed = scrub_log_text(text);
    assert!(!scrubbed.contains("ab\\"), "{scrubbed}");
    assert!(scrubbed.contains(r#"\"x\":\"keep\""#), "{scrubbed}");
    assert_eq!(
        scrubbed,
        r#"{"error":"{\"token\":\"<redacted>\",\"x\":\"keep\"}"}"#
    );
}
