//! Secret-bearing wrappers: [`Secret`] for credentials and [`LogOnly`] for
//! upstream text that may echo request material.
//!
//! Both wrappers exist so that the *type* carries the redaction rule. A
//! `String` in a struct is one `{:?}` away from a log line; a [`Secret`] is not,
//! because its `Debug` and `Display` never print the value and it has no
//! `Serialize` implementation, so it cannot reach a config file or a wire
//! payload by accident (invariant 1: a credential is never on a record).
//!
//! ```compile_fail,E0277
//! use tinyinference_hub::Secret;
//! let secret = Secret::new("sk-not-a-real-key");
//! // `Secret` deliberately does not implement `serde::Serialize`.
//! let _ = serde_json::to_string(&secret);
//! ```

use std::fmt;

/// The text every redacting `Debug`/`Display` prints in place of a value.
pub(crate) const REDACTED: &str = "<redacted>";

/// A credential. Redacts itself in `Debug` and `Display`; never serialisable.
///
/// The only way to read the value is [`Secret::expose`], which makes each use
/// greppable in review.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// Wraps a credential value.
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    /// The credential itself. Call this only where the value is put on the wire.
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// Whether the wrapped value is empty (an empty key is "no key").
    pub fn is_empty(&self) -> bool {
        self.0.trim().is_empty()
    }

    /// Length of the wrapped value in bytes. Safe to log; the value is not.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl From<String> for Secret {
    fn from(value: String) -> Self {
        Self(value)
    }
}

impl From<&str> for Secret {
    fn from(value: &str) -> Self {
        Self(value.to_string())
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Secret({REDACTED})")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// A value that may be written to a log channel by the code that owns it but
/// must never appear in a user-facing sentence, `Display`, or `Debug` dump.
///
/// Used for raw upstream error text: it can echo request headers or key
/// fragments, and the sentence built from it is one someone screenshots into a
/// ticket (`probe.rs` design note, OpenCompany).
#[derive(Clone, PartialEq, Eq, Default)]
pub struct LogOnly<T>(T);

impl<T> LogOnly<T> {
    /// Wraps a log-only value.
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// The wrapped value, for a log or detail channel only.
    pub fn expose(&self) -> &T {
        &self.0
    }

    /// Consumes the wrapper. Same rule as [`LogOnly::expose`].
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> fmt::Debug for LogOnly<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LogOnly({REDACTED})")
    }
}

impl<T> fmt::Display for LogOnly<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(REDACTED)
    }
}

/// Whether a field, header or query-parameter name marks its value as a
/// credential.
///
/// camelCase is split (`accessToken` is `access_token`), and `-` and case are
/// ignored. Three rules, on the name's words:
///
/// * a word contains, or two adjacent words join into, a compound that is a
///   credential:
///   `apikey`, `accesskey`, `privatekey`, `masterkey`, `subscriptionkey`,
///   `authorization`, `bearer`, `password`, `passwd`, `passphrase`,
///   `credential`, `accesstoken`, `authtoken`, `refreshtoken`, `sessiontoken`,
///   `idtoken`, `apitoken`, `bearertoken`, or `secret` (except `secretary`);
/// * its last word is `signature`, `sig`, `cookie`, `pwd` or `auth`, or it is
///   `auth`;
/// * its last word is `key` or `token` and no word in it is plainly benign
///   (`public_key`, `cache_key`, `page_token`, `next_page_token`).
///
/// Ordinary names (`max_tokens`, `tokenizer`, `keywords`, `monkey`,
/// `secretary`, `signature_algorithm`) are not credentials.
pub(crate) fn is_credential_name(name: &str) -> bool {
    let mut words = String::with_capacity(name.len() + 4);
    let mut previous: Option<char> = None;
    for c in name.trim().chars() {
        if c.is_ascii_uppercase()
            && previous.is_some_and(|p| p.is_ascii_lowercase() || p.is_ascii_digit())
        {
            words.push('_');
        }
        words.push(if c == '-' {
            '_'
        } else {
            c.to_ascii_lowercase()
        });
        previous = Some(c);
    }
    const COMPOUNDS: &[&str] = &[
        "apikey",
        "accesskey",
        "privatekey",
        "masterkey",
        "subscriptionkey",
        "authorization",
        "bearer",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "accesstoken",
        "authtoken",
        "refreshtoken",
        "sessiontoken",
        "idtoken",
        "apitoken",
        "bearertoken",
    ];
    const LAST_WORDS: &[&str] = &["signature", "sig", "cookie", "pwd", "auth"];
    const BENIGN_WORDS: &[&str] = &[
        "public",
        "cache",
        "sort",
        "foreign",
        "partition",
        "row",
        "idempotency",
        "shard",
        "lookup",
        "page",
        "pagination",
        "continuation",
    ];
    let parts: Vec<&str> = words.split('_').collect();
    // A compound counts inside one word (`secretkey`, `accesstoken`) or as two
    // adjacent whole words joined (`access_token`, `api_key`); it does not count
    // when it merely spans a word boundary (`valid_tokens` is not `idtoken`).
    let in_word = parts.iter().any(|w| {
        COMPOUNDS.iter().any(|c| w.contains(c)) || (w.contains("secret") && !w.contains("secretar"))
    });
    // The second word may carry a plural or a number (`api_keys`, `api_key2`).
    let joined = parts.windows(2).any(|pair| {
        let second = pair[1].trim_end_matches(|c: char| c.is_ascii_digit());
        let singular = second.strip_suffix('s').unwrap_or(second);
        [second, singular]
            .iter()
            .any(|tail| COMPOUNDS.contains(&format!("{}{tail}", pair[0]).as_str()))
    });
    if in_word || joined {
        return true;
    }
    let last = words.rsplit('_').next().unwrap_or("");
    if LAST_WORDS.contains(&last) {
        return true;
    }
    matches!(last, "key" | "token") && !words.split('_').any(|word| BENIGN_WORDS.contains(&word))
}

#[cfg(test)]
#[path = "secret_test.rs"]
mod tests;
