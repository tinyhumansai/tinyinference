//! Retry-After parsing and bounded exponential backoff for embedding providers.

use std::time::SystemTime;

/// Maximum number of provider retries.
pub const MAX_RETRIES: u32 = 3;
/// Initial exponential backoff in milliseconds.
pub const BASE_BACKOFF_MS: u64 = 1_000;
/// Maximum provider backoff in milliseconds.
pub const MAX_BACKOFF_MS: u64 = 30_000;

/// Parses delta seconds or an HTTP date into a bounded millisecond delay.
pub fn parse_retry_after_ms(value: Option<&str>) -> Option<u64> {
    parse_retry_after_ms_at(value, SystemTime::now())
}

fn parse_retry_after_ms_at(value: Option<&str>, now: SystemTime) -> Option<u64> {
    let value = value?.trim();
    if value.is_empty() {
        return None;
    }
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(seconds.saturating_mul(1_000).min(MAX_BACKOFF_MS));
    }
    let retry_at = httpdate::parse_http_date(value).ok()?;
    let delay = retry_at.duration_since(now).unwrap_or_default();
    Some(
        u64::try_from(delay.as_millis())
            .unwrap_or(u64::MAX)
            .min(MAX_BACKOFF_MS),
    )
}

/// Returns a Retry-After delay or bounded exponential fallback for `attempt`.
pub fn backoff_ms_for_attempt(attempt: u32, retry_after: Option<&str>) -> u64 {
    parse_retry_after_ms(retry_after).unwrap_or_else(|| {
        BASE_BACKOFF_MS
            .saturating_mul(2u64.saturating_pow(attempt))
            .min(MAX_BACKOFF_MS)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_delta_seconds_and_caps() {
        assert_eq!(parse_retry_after_ms(Some(" 5 ")), Some(5_000));
        assert_eq!(parse_retry_after_ms(Some("999")), Some(MAX_BACKOFF_MS));
        assert_eq!(parse_retry_after_ms(Some("-1")), None);
    }

    #[test]
    fn parses_http_dates() {
        let now = httpdate::parse_http_date("Wed, 21 Oct 2015 07:27:55 GMT").unwrap();
        assert_eq!(
            parse_retry_after_ms_at(Some("Wed, 21 Oct 2015 07:28:00 GMT"), now),
            Some(5_000)
        );
    }

    #[test]
    fn falls_back_to_bounded_exponential_backoff() {
        assert_eq!(backoff_ms_for_attempt(0, None), 1_000);
        assert_eq!(backoff_ms_for_attempt(2, None), 4_000);
        assert_eq!(backoff_ms_for_attempt(20, None), MAX_BACKOFF_MS);
    }
}
