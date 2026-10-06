//! Provider-neutral failure classification shared by transport adapters.

/// Provider failure class used for retry and telemetry decisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderFailureClass {
    /// A transient failure without a more specific classification.
    Retryable,
    /// A permanent caller, account, or model error.
    NonRetryable,
    /// A generic rate limit where retrying after backoff may succeed.
    RateLimited,
    /// A rate limit caused by exhausted quota, balance, or plan access.
    NonRetryableRateLimit,
    /// A provider outage, timeout, or capacity failure.
    UpstreamUnhealthy,
}

impl ProviderFailureClass {
    /// Returns whether retrying the same request may succeed.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::Retryable | Self::RateLimited | Self::UpstreamUnhealthy
        )
    }

    /// Returns a stable telemetry label.
    pub fn reason(self) -> &'static str {
        match self {
            Self::Retryable => "retryable",
            Self::NonRetryable => "non_retryable",
            Self::RateLimited => "rate_limited",
            Self::NonRetryableRateLimit => "rate_limited_non_retryable",
            Self::UpstreamUnhealthy => "upstream_unhealthy",
        }
    }
}

fn parse_status_at(text: &str, start: usize) -> Option<u16> {
    let digits: String = text
        .get(start..)?
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    (digits.len() == 3).then(|| digits.parse().ok()).flatten()
}

// ── Body phrase matchers ─────────────────────────────────────────────────────
//
// Pure, status-agnostic matchers over a provider error body. They carry no
// host policy: a host adds its own status / provider gates and decides what to
// do with a match (retry, demote a log, suppress telemetry).

/// Whether a provider error body indicates the request exceeded the model's
/// context window (the prompt or conversation is too long for the model).
///
/// This is a deterministic usage condition, not a transient fault: retrying
/// the same oversized request cannot help, and the remediation (trim the
/// conversation or pick a larger-context model) lies with the user.
///
/// Status-agnostic on purpose: providers disagree on the HTTP code for this
/// condition (most emit `400 context_length_exceeded`, some self-hosted
/// gateways mis-report it as `500`), so matching the body keeps them in one
/// bucket.
///
/// Anchoring is two-tier so an over-broad match cannot mark a retryable error
/// as permanent:
///
/// - **Length/context phrases** are unambiguous ("context window", "context
///   length", "prompt is too long" only describe request-size overflow) and
///   match alone.
/// - **Token-count phrases** collide with per-minute token *rate* limits
///   ("rate limit reached ... too many tokens per min"), which are transient
///   and must stay retryable. They only count as context overflow when no
///   rate-limit marker is present.
pub fn is_context_window_exceeded_message(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();

    // Unambiguous request-size / context phrases — match on their own.
    const CONTEXT_HINTS: &[&str] = &[
        "exceeds the context window",
        "context window of this model",
        "maximum context length",
        "context length exceeded",
        "context size has been exceeded",
        "prompt is too long",
        "input is too long",
        // LM Studio / llama.cpp un-evictable-prefix overflow (TAURI-RUST-6V0):
        // `"The number of tokens to keep from the initial prompt is greater
        //   than the context length (n_keep: 10978 >= n_ctx: 8192). Try to
        //   load the model with a larger context length, …"`. The user's local
        // model was loaded with an `n_ctx` smaller than the system/un-evictable
        // prefix; the remediation lives in the user's local server (reload with
        // a larger context), so this is expected user-state, not a product bug.
        "greater than the context length",
        // Alibaba / DashScope (Qwen): `"Range of input length should be
        // [1, 98304]"` — the window is the range's upper bound.
        "range of input length should be",
    ];
    if CONTEXT_HINTS.iter().any(|hint| lower.contains(hint)) {
        return true;
    }

    // LM Studio / llama.cpp emit the overflow as a paired `n_keep … n_ctx`
    // diagnostic. Require BOTH tokens so the arm stays anchored to that exact
    // shape (TAURI-RUST-6V0) and never broadens to unrelated `n_ctx` logging.
    if lower.contains("n_keep") && lower.contains("n_ctx") {
        return true;
    }

    // Token-count phrases are ambiguous with token-per-minute RATE limits.
    // Treat them as context-overflow only when the body carries no
    // rate-limit marker — otherwise a transient TPM 429 would be silenced
    // from Sentry and (via `reliable`) wrongly classified as non-retryable.
    const TOKEN_HINTS: &[&str] = &["too many tokens", "token limit exceeded"];
    if TOKEN_HINTS.iter().any(|hint| lower.contains(hint)) {
        const RATE_LIMIT_MARKERS: &[&str] = &[
            "per minute",
            "per min",
            "rate limit",
            "rate_limit",
            "tpm",
            "requests per",
            "retry after",
            "try again in",
        ];
        return !RATE_LIMIT_MARKERS
            .iter()
            .any(|marker| lower.contains(marker));
    }

    false
}

/// Phrase-level matcher for an insufficient-credits / out-of-balance provider
/// error body (the caller's own provider account lacks the balance to satisfy
/// the request). Status-agnostic: callers add their own status gate (for
/// example `402`) when they need one.
pub fn body_indicates_insufficient_credits(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("requires more credits")
        || lower.contains("more credits")
        || lower.contains("can only afford")
        || lower.contains("insufficient credit")
        || lower.contains("insufficient balance")
        || lower.contains("insufficient funds")
        || lower.contains("payment required")
}

/// Phrase-level matcher for a provider monthly-quota / usage-limit exhausted
/// body: the plan has spent its allotment for the period and no request
/// succeeds until it resets.
///
/// Status-agnostic, because an upstream proxy may wrap its own `402` inside a
/// `500` envelope. Keyed on quota-specific wording only, so a generic outage
/// or a transient `429` rate limit is not matched.
pub fn body_indicates_quota_exhausted(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("monthly_request_count")
        || lower.contains("monthly request")
        || lower.contains("monthly limit")
        || lower.contains("monthly quota")
        || lower.contains("quota exceeded")
        || lower.contains("usage limit exceeded")
        // Codex/ChatGPT OAuth `/responses` plan-cap body (TAURI-RUST-AFE):
        // `usage_limit_reached` / "The usage limit has been reached" — a plan
        // quota with no "monthly"/"quota" co-marker, so the phrases above miss
        // it. Both are quota-specific enough to match on their own (the loop
        // retries until `resets_at`, flooding from a single capped Plus user).
        || lower.contains("usage_limit_reached")
        || lower.contains("usage limit has been reached")
        // "reached the limit" alone is ambiguous (rate-limit, token-limit), so
        // require a quota/plan/request/monthly co-marker to keep the blast
        // radius on plan-quota exhaustion only.
        || (lower.contains("reached the limit")
            && (lower.contains("request")
                || lower.contains("quota")
                || lower.contains("monthly")
                || lower.contains("plan")))
}

/// Whether a provider error body is a **permanent per-request rate-cap
/// rejection**: a *single* request's token count exceeds the account's
/// tokens-per-minute budget, so no amount of retrying or spacing lets it
/// through on the current tier.
///
/// Distinct from a transient TPM `429` ("rate limit reached ... try again in
/// 2s"), from a monthly-plan quota ([`body_indicates_quota_exhausted`]), and
/// from context-window overflow ([`is_context_window_exceeded_message`]).
///
/// Anchored on BOTH the permanence marker `"request too large"` AND a
/// per-minute-tokens marker (`"tokens per minute"` / `"(tpm)"`), so a
/// transient burst limit, which lacks "request too large", is not matched.
pub fn is_provider_rate_cap_exceeded_message(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("request too large")
        && (lower.contains("tokens per minute") || lower.contains("(tpm)"))
}

/// Whether a provider error body says a local inference server is running but
/// has **no model loaded** (for example an idle LM Studio server answering
/// `No models loaded. Please load a model ...`). Status-agnostic; callers add
/// their own status gate.
pub fn body_indicates_no_model_loaded(body: &str) -> bool {
    body.to_ascii_lowercase().contains("no models loaded")
}

/// Whether a provider error body is Ollama Cloud's opaque hosted-inference
/// internal error envelope: `{"error":"Internal Server Error (ref: <uuid>)"}`.
///
/// Anchored on the `internal server error (ref:` shape, which is specific to
/// the hosted envelope; a local Ollama daemon failure does not carry a `ref:`
/// UUID. Status- and provider-agnostic; callers add both gates.
pub fn body_indicates_ollama_cloud_internal_error(body: &str) -> bool {
    body.to_ascii_lowercase()
        .contains("internal server error (ref:")
}

/// Whether a provider error body is a provider access-policy denial, for
/// example Kimi's coding endpoint rejecting non-agent clients with
/// `access_terminated_error` / "currently only available for Coding Agents".
/// Status-agnostic; callers add their own status gate (`403`).
pub fn body_indicates_provider_access_policy_denied(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("access_terminated_error")
        || lower.contains("currently only available for coding agents")
}

/// Whether a provider error body is an external content-moderation rejection
/// such as `{"error":"Message rejected by Ombudsman","score":80}`.
///
/// Anchored on the moderation-verdict shape: the rejection wording
/// (`message rejected` / `ombudsman`) or the quoted `"score"` verdict key
/// (not a bare `score`, so prose in an unrelated 400 is not matched).
/// Status-agnostic; callers add their own status gate (`400`).
pub fn body_indicates_moderation_rejection(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("message rejected") || lower.contains("ombudsman") || lower.contains("\"score\"")
}

/// Whether a provider error body is the known generic upstream envelope a
/// custom OpenAI-compatible proxy returns:
/// `{"error":{"message":"Bad request to upstream provider","type":"upstream_error","status":400}}`.
/// Status- and provider-agnostic; callers add both gates.
pub fn body_indicates_custom_openai_upstream_bad_request(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    lower.contains("bad request to upstream provider") && lower.contains("upstream_error")
}

/// Whether a provider error body looks like an OpenAI-style authentication
/// envelope (a missing or invalid API key): `authentication_error`,
/// `invalid_api_key`, "incorrect api key", and the bare-message variants
/// gateways emit. Status- and provider-agnostic; callers add both gates.
pub fn body_indicates_auth_key_error(body: &str) -> bool {
    let lower = body.to_ascii_lowercase();
    const AUTH_ERROR_MARKERS: &[&str] = &[
        "authentication_error",
        "invalid_api_key",
        "invalid api key",
        "invalid or missing api key",
        "missing api key",
        "no api key supplied",
        "incorrect api key",
        "invalid authentication",
    ];
    AUTH_ERROR_MARKERS
        .iter()
        .any(|marker| lower.contains(marker))
}

/// Extracts an HTTP status from normalized provider error text.
pub fn structured_http_status(message: &str) -> Option<u16> {
    let trimmed = message.trim_start();
    if let Some(status) = parse_status_at(trimmed, 0) {
        return Some(status);
    }
    for (index, _) in message.match_indices('(') {
        if let Some(status) = parse_status_at(message, index + 1) {
            return Some(status);
        }
    }
    let lower = message.to_ascii_lowercase();
    for marker in ["http ", "status:", "status "] {
        if let Some(index) = lower.find(marker)
            && let Some(status) = parse_status_at(message, index + marker.len())
        {
            return Some(status);
        }
    }
    None
}

/// Whether an already-lowercased rate-limit message is a **business** limit
/// (plan, balance, quota, package) rather than a transient throttle.
///
/// Retrying a business limit is futile, so callers that hold only the flattened
/// error text (a host that has already stringified the provider error) use this
/// to decide whether to offer a retry. The provider-code scan matches the Z.AI
/// business codes 1113 and 1311 as standalone integer tokens.
pub fn contains_business_limit(lower: &str) -> bool {
    [
        "plan does not include",
        "doesn't include",
        "not include",
        "insufficient balance",
        "insufficient_balance",
        "insufficient quota",
        "insufficient_quota",
        "quota exhausted",
        "out of credits",
        "no available package",
        "package not active",
        "purchase package",
        "model not available for your plan",
    ]
    .iter()
    .any(|hint| lower.contains(hint))
        || lower.split(|ch: char| !ch.is_ascii_digit()).any(|token| {
            token
                .parse::<u16>()
                .is_ok_and(|code| matches!(code, 1113 | 1311))
        })
}

fn indicates_upstream_failure(lower: &str) -> bool {
    [
        "no healthy upstream",
        "upstream unavailable",
        "service unavailable",
        "408 request timeout",
        "409 conflict",
        "500 internal server error",
        "502 bad gateway",
        "503 service unavailable",
        "504 gateway timeout",
        "bad gateway",
        "gateway timeout",
    ]
    .iter()
    .any(|hint| lower.contains(hint))
}

fn indicates_terminal_request(lower: &str) -> bool {
    [
        "invalid api key",
        "incorrect api key",
        "missing api key",
        "api key not set",
        "authentication failed",
        "authentication_error",
        "auth failed",
        "unauthorized",
        "forbidden",
        "permission denied",
        "access denied",
        "invalid token",
        "invalid_request",
    ]
    .iter()
    .any(|hint| lower.contains(hint))
        || (lower.contains("model")
            && [
                "not found",
                "unknown",
                "unsupported",
                "does not exist",
                "invalid",
            ]
            .iter()
            .any(|hint| lower.contains(hint)))
}

/// Whether the provider itself said the condition is transient and named a
/// retry.
///
/// Provider-neutral on purpose: it keys on the retry instruction, not on who
/// sent it or why. A 4xx is normally a permanent caller error, which is why the
/// blanket `400..500` arm below classifies one as [`ProviderFailureClass::NonRetryable`],
/// but a provider that returns a 4xx *and* tells the caller to retry is
/// describing a momentary condition on its own side — a concurrency or
/// reservation window — and the body is the only place that distinction is
/// carried.
///
/// Observed as a `402` whose body read "This request would exceed your
/// available credits given your current in-flight requests. Retry after
/// in-flight requests settle, or add credits." It is a credit *reservation*
/// held against requests still outstanding, not an empty account; the next call
/// succeeds. Classified non-retryable it discarded a 34-minute, 96-call agent
/// run on its 97th call with the account at $13.76 of $40.
///
/// Gated on the absence of a terminal indicator, so "invalid api key — fix it
/// and try again" stays permanent: an instruction to retry *after changing
/// something* is not an instruction to retry the same request.
fn indicates_provider_requested_retry(lower: &str) -> bool {
    [
        "retry after",
        "please retry",
        "retry the request",
        "retry in",
        "try again in",
    ]
    .iter()
    .any(|hint| lower.contains(hint))
        && !indicates_terminal_request(lower)
}

/// Classifies a provider failure from status, code, and message detail.
pub fn classify_provider_failure(
    status: Option<u16>,
    code: Option<&str>,
    message: &str,
) -> ProviderFailureClass {
    let has_structured_status = status.is_some();
    let has_structured_code = code.is_some_and(|value| !value.trim().is_empty());
    let structured_code = code.unwrap_or_default().to_ascii_lowercase();
    let status = status.or_else(|| {
        (!has_structured_code)
            .then(|| structured_http_status(message))
            .flatten()
    });
    let lower = match code {
        Some(code) if !code.trim().is_empty() => format!("{message} {code}").to_ascii_lowercase(),
        _ => message.to_ascii_lowercase(),
    };

    let rate_limited = status == Some(429)
        || (lower.contains("429")
            && (lower.contains("too many") || lower.contains("rate") || lower.contains("limit")));
    if rate_limited {
        return if contains_business_limit(&lower) {
            ProviderFailureClass::NonRetryableRateLimit
        } else {
            ProviderFailureClass::RateLimited
        };
    }

    if status.is_some_and(|value| matches!(value, 408 | 409) || value >= 500)
        || (has_structured_code && indicates_upstream_failure(&structured_code))
        || (!has_structured_status && !has_structured_code && indicates_upstream_failure(&lower))
    {
        return ProviderFailureClass::UpstreamUnhealthy;
    }

    // Before the blanket 4xx arm: a provider that asks for a retry is
    // describing its own momentary state, which the status code cannot express.
    if indicates_provider_requested_retry(&lower) {
        return ProviderFailureClass::RateLimited;
    }

    if status.is_some_and(|value| (400..500).contains(&value))
        || (has_structured_code && indicates_terminal_request(&structured_code))
        || (!has_structured_status && !has_structured_code && indicates_terminal_request(&lower))
    {
        return ProviderFailureClass::NonRetryable;
    }

    ProviderFailureClass::Retryable
}

/// Classifies a normalized structured provider error.
pub fn classify_provider_error(error: &crate::model::ProviderError) -> ProviderFailureClass {
    let class = classify_provider_failure(error.status, error.code.as_deref(), &error.message);
    if !error.retryable && class.is_retryable() {
        ProviderFailureClass::NonRetryable
    } else {
        class
    }
}

/// Returns whether a normalized provider error is safe to retry.
pub fn provider_error_is_retryable(error: &crate::model::ProviderError) -> bool {
    error.retryable
}

/// Parses a `Retry-After` / `retry_after` value from provider error text.
///
/// Integer and fractional seconds are accepted and returned as milliseconds.
pub fn parse_retry_after_ms(message: &str) -> Option<u64> {
    let lower = message.to_ascii_lowercase();
    for prefix in &[
        "retry-after:",
        "retry_after:",
        "retry-after ",
        "retry_after ",
    ] {
        if let Some(position) = lower.find(prefix) {
            let number: String = message[position + prefix.len()..]
                .trim_start()
                .chars()
                .take_while(|character| character.is_ascii_digit() || *character == '.')
                .collect();
            if let Ok(seconds) = number.parse::<f64>()
                && seconds.is_finite()
                && seconds >= 0.0
            {
                let milliseconds = seconds * 1_000.0;
                if milliseconds <= u64::MAX as f64 {
                    return Some(milliseconds as u64);
                }
            }
        }
    }
    None
}

mod context_limit;
mod text;

pub use context_limit::parse_context_limit_from_error;
pub use text::{
    extract_provider_error_detail, extract_provider_name, is_auth_error_text,
    is_codex_token_expired_text, is_connection_dropped_text, is_context_length_text,
    is_empty_provider_response_text, is_fallback_chain_exhausted, is_malformed_tool_history_text,
    is_model_unavailable_text, is_payment_required_text, is_provider_request_rejected_text,
    is_rate_limit_text, is_recoverable_failure_text, is_server_error_text, is_timeout_text,
    is_transient_unavailability_text, is_vision_unsupported_text, parse_retry_after_secs,
    with_provider_detail,
};

#[cfg(test)]
#[path = "failure_tests.rs"]
mod tests;
