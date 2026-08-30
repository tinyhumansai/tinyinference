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

fn contains_business_limit(lower: &str) -> bool {
    [
        "plan does not include",
        "doesn't include",
        "insufficient balance",
        "insufficient quota",
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

/// Classifies a provider failure from status, code, and message detail.
pub fn classify_provider_failure(
    status: Option<u16>,
    code: Option<&str>,
    message: &str,
) -> ProviderFailureClass {
    let status = status.or_else(|| structured_http_status(message));
    let lower = match code {
        Some(code) if !code.trim().is_empty() => format!("{message} {code}").to_ascii_lowercase(),
        _ => message.to_ascii_lowercase(),
    };

    let rate_limited = status == Some(429)
        || (lower.contains("429")
            && (lower.contains("too many")
                || lower.contains("rate")
                || lower.contains("limit")));
    if rate_limited {
        return if contains_business_limit(&lower) {
            ProviderFailureClass::NonRetryableRateLimit
        } else {
            ProviderFailureClass::RateLimited
        };
    }

    if status.is_some_and(|value| matches!(value, 408 | 409) || value >= 500)
        || [
            "no healthy upstream",
            "upstream unavailable",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        ]
        .iter()
        .any(|hint| lower.contains(hint))
    {
        return ProviderFailureClass::UpstreamUnhealthy;
    }

    if status.is_some_and(|value| (400..500).contains(&value))
        || [
            "invalid api key",
            "incorrect api key",
            "missing api key",
            "authentication failed",
            "unauthorized",
            "forbidden",
            "permission denied",
        ]
        .iter()
        .any(|hint| lower.contains(hint))
        || (lower.contains("model")
            && ["not found", "unknown", "unsupported", "does not exist", "invalid"]
                .iter()
                .any(|hint| lower.contains(hint)))
    {
        return ProviderFailureClass::NonRetryable;
    }

    ProviderFailureClass::Retryable
}

/// Classifies a normalized structured provider error.
pub fn classify_provider_error(
    error: &crate::model::ProviderError,
) -> ProviderFailureClass {
    classify_provider_failure(error.status, error.code.as_deref(), &error.message)
}
