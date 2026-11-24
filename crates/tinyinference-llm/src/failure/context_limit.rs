//! Extracting the context limit a provider states in an overflow error.
//!
//! When a request overflows, most providers say how big the window really is:
//!
//! - OpenAI, DeepSeek, OpenRouter and vLLM: `"This model's maximum context
//!   length is 131072 tokens. However, you requested 140000 tokens …"` (OpenRouter
//!   says `"This endpoint's …"`).
//! - Anthropic: `"prompt is too long: 208000 tokens > 200000 maximum"`.
//! - Newer vLLM: `"… is longer than the maximum model length of 32768"`.
//! - Mistral: `"… too large for model with 32768 maximum context length"`.
//! - Gemini: `"… exceeds the maximum number of tokens allowed (1048576)"`.
//! - llama.cpp / LM Studio: `"(n_keep: 10978 >= n_ctx: 8192)"` and
//!   `"exceeds the available context size (8192 tokens)"`.
//! - Alibaba / DashScope (Qwen): `"Range of input length should be [1, 98304]"`;
//!   the window is the upper bound of the range. DashScope endpoints can admit
//!   less than their advertised window (Qwen3-8B lists 131072, accepts 98304).
//!
//! That number is the authoritative window for the endpoint that answered, so a
//! host can record it as a correction (see
//! [`crate::model::discover::record_overflow_error`]) instead of retrying
//! against a guessed one.

/// The smallest value accepted as a context window. Anything below this is
/// almost certainly a different number in the message (a status, a count of
/// messages), not a window.
const MIN_PLAUSIBLE_WINDOW: u64 = 512;
/// The largest value accepted as a context window (100M tokens).
const MAX_PLAUSIBLE_WINDOW: u64 = 100_000_000;

/// Phrases after which the stated limit is the first number that follows.
///
/// Matched against the lowercased message. Each anchor is specific enough that
/// the next number is the window, not the requested size.
const LIMIT_FOLLOWS: &[&str] = &[
    "maximum context length is",
    "maximum context length of",
    "max context length is",
    "context length is only",
    "maximum model length of",
    "maximum model length is",
    "max_model_len",
    "maximum number of tokens allowed",
    "available context size",
    "n_ctx:",
    "n_ctx =",
    "context window of",
    "context window is",
    "context_length of",
];

/// Phrases before which the stated limit is the last number that precedes.
const LIMIT_PRECEDES: &[&str] = &["maximum context length"];

/// Extracts the context window a provider states in an overflow error body.
///
/// Returns `None` when the text states no recognisable limit, including when it
/// is not a context-overflow message at all. The caller does not need to check
/// [`crate::failure::is_context_window_exceeded_message`] first, but recording
/// the result is only meaningful for overflow errors.
#[must_use]
pub fn parse_context_limit_from_error(message: &str) -> Option<u64> {
    let lower = message.to_ascii_lowercase();

    // Anthropic: "prompt is too long: 208000 tokens > 200000 maximum". The
    // window is the number after `>`.
    if let Some(start) = lower.find("prompt is too long") {
        let tail = &lower[start..];
        if let Some(limit) = tail
            .find('>')
            .and_then(|gt| first_number(&tail[gt + 1..], 24))
        {
            return plausible(limit);
        }
    }

    // DashScope: "range of input length should be [1, 98304]". The first
    // number is the range's lower bound; the window is the one after the comma.
    if let Some(start) = lower.find("range of input length should be") {
        let tail = &lower[start..];
        if let Some(limit) = tail
            .find('[')
            .zip(tail.find(']'))
            .and_then(|(open, close)| tail.get(open + 1..close))
            .and_then(|range| range.rsplit(',').next())
            .and_then(|upper| first_number(upper, 8))
        {
            return plausible(limit);
        }
    }

    let following = LIMIT_FOLLOWS.iter().find_map(|anchor| {
        let start = lower.find(anchor)?;
        first_number(&lower[start + anchor.len()..], 24).and_then(plausible)
    });
    if following.is_some() {
        return following;
    }

    // Mistral: "too large for model with 32768 maximum context length". The
    // window is the number right before the phrase.
    LIMIT_PRECEDES.iter().find_map(|anchor| {
        let end = lower.find(anchor)?;
        trailing_number(&lower[..end]).and_then(plausible)
    })
}

fn plausible(value: u64) -> Option<u64> {
    (MIN_PLAUSIBLE_WINDOW..=MAX_PLAUSIBLE_WINDOW)
        .contains(&value)
        .then_some(value)
}

/// The integer `text` starts with, after skipping at most `window` bytes of
/// whitespace and punctuation. A letter before the first digit means the
/// anchor was not followed by a number (`"context window of this model"`), so
/// a figure later in the sentence is never mistaken for the window.
fn first_number(text: &str, window: usize) -> Option<u64> {
    for (index, character) in text.char_indices() {
        if index > window || character.is_alphabetic() {
            return None;
        }
        if character.is_ascii_digit() {
            return parse_grouped(&text[index..]);
        }
    }
    None
}

/// The integer that `text` ends with, ignoring trailing whitespace. Returns
/// `None` when anything other than whitespace follows the last digit, so a
/// requested-size figure earlier in the sentence is never mistaken for the
/// window.
fn trailing_number(text: &str) -> Option<u64> {
    let trimmed = text.trim_end();
    // Step past the whole character that precedes the digits: it can be
    // multi-byte (a no-break space, a CJK character), and `index + 1` would
    // land inside it.
    let begin = trimmed
        .char_indices()
        .rev()
        .find(|&(_, character)| !(character.is_ascii_digit() || character == ','))
        .map_or(0, |(index, character)| index + character.len_utf8());
    let digits = trimmed[begin..].trim_start_matches(',');
    if digits.is_empty() {
        return None;
    }
    parse_grouped(digits)
}

/// Parses a leading integer, accepting `,` / `_` digit grouping.
fn parse_grouped(text: &str) -> Option<u64> {
    let mut digits = String::new();
    let mut chars = text.chars().peekable();
    while let Some(character) = chars.next() {
        if character.is_ascii_digit() {
            digits.push(character);
        } else if (character == ',' || character == '_')
            && chars.peek().is_some_and(char::is_ascii_digit)
        {
            continue;
        } else {
            break;
        }
    }
    if digits.is_empty() || digits.len() > 12 {
        return None;
    }
    digits.parse().ok()
}

#[cfg(test)]
#[path = "context_limit_tests.rs"]
mod tests;
