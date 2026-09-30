//! The hub's typed error taxonomy (built first, per D12).
//!
//! * `types`: [`HubError`], [`ReasonCode`], [`Retry`], [`ProviderFailure`],
//!   and the small enums they carry.
//! * `classify`: vendor status + headers + body to a [`ProviderFailure`],
//!   ported from OpenCompany's classifier and OpenHuman's `http_error/*`.
//! * `copy`: user-facing sentences that never interpolate raw upstream text.

mod classify;
mod copy;
mod types;

use std::time::Duration;

use crate::taxonomy::ProviderGroup;

pub use classify::{
    MAX_RETRY_AFTER, TransportCondition, classify, classify_at, classify_for, classify_transport,
    scrub_log_text, strip_urls,
};
pub use copy::{describe, describe_refusal};
pub use types::{
    CopyContext, HubError, InputField, InvalidInput, NotFound, Operation, PolicyViolation,
    PortName, ProviderFailure, ReasonCode, Retry, Unresolved, UsedBy,
};

/// Result alias for hub operations.
pub type Result<T> = std::result::Result<T, HubError>;

impl ProviderFailure {
    /// Whether meeting this failure should undo the add that just saved a
    /// credential, given the kind of provider it was.
    ///
    /// Only a rejected credential is evidence about the credential, so only
    /// [`ReasonCode::Auth`] rolls back in general. The one category exception
    /// (from OpenCompany's design): **a local runtime also rolls back on an
    /// unreachable endpoint or a timeout.** A runtime that is not running is not
    /// a connection worth creating; the next move is to start it and retry. For
    /// a cloud provider the same class means the opposite: a proxy, a WAF or a
    /// slow gateway sits between a good key and a fine endpoint.
    pub fn rolls_back(&self, group: ProviderGroup) -> bool {
        if self.reason.destroys_credential() {
            return true;
        }
        group == ProviderGroup::Local
            && matches!(self.reason, ReasonCode::Endpoint | ReasonCode::Timeout)
    }
}

impl HubError {
    /// The stable reason code for this error.
    pub fn reason(&self) -> ReasonCode {
        match self {
            Self::Provider(failure) => failure.reason,
            Self::SignedOut { .. } => ReasonCode::SignedOut,
            Self::Unsupported { .. } => ReasonCode::Unsupported,
            Self::Policy(_) => ReasonCode::Policy,
            Self::Invalid(_) => ReasonCode::Invalid,
            Self::NotFound(_) => ReasonCode::NotFound,
            Self::AlreadyExists { .. } => ReasonCode::AlreadyExists,
            Self::InUse(_) => ReasonCode::InUse,
            Self::Conflict => ReasonCode::Conflict,
            Self::StoreUnreadable { .. } => ReasonCode::StoreUnreadable,
            Self::Unresolved(_) => ReasonCode::Unresolved,
        }
    }

    /// Whether and when trying again may help. Quota and auth failures are
    /// [`Retry::Never`]; a lost compare-and-swap is [`Retry::Now`].
    pub fn retry(&self) -> Retry {
        match self {
            Self::Provider(failure) => failure.retry,
            Self::Conflict => Retry::Now,
            Self::StoreUnreadable { .. } => Retry::Later(None),
            _ => Retry::Never,
        }
    }

    /// Whether this error, on an add, should undo the credential just written.
    /// See [`ProviderFailure::rolls_back`]; only provider failures roll back.
    pub fn rolls_back(&self, group: ProviderGroup) -> bool {
        matches!(self, Self::Provider(failure) if failure.rolls_back(group))
    }
}

impl From<ProviderFailure> for HubError {
    fn from(failure: ProviderFailure) -> Self {
        Self::Provider(failure)
    }
}

impl From<InvalidInput> for HubError {
    fn from(invalid: InvalidInput) -> Self {
        Self::Invalid(invalid)
    }
}

impl From<PolicyViolation> for HubError {
    fn from(violation: PolicyViolation) -> Self {
        Self::Policy(violation)
    }
}

impl From<tinyinference_llm::Error> for HubError {
    /// Maps llm's error into the hub's. Structured provider detail keeps its
    /// status, code and retry hint; free text is classified after its URLs are
    /// stripped; a validation error becomes [`HubError::Invalid`].
    fn from(error: tinyinference_llm::Error) -> Self {
        use tinyinference_llm::Error;
        match error {
            Error::Provider(provider) => {
                let has_retry_after = provider.retry_after_ms.is_some();
                // The structured code is the most reliable signal a provider
                // gives (`insufficient_quota`, `invalid_api_key`), so it is
                // classified together with the message.
                let text = strip_urls(&provider.message);
                let text = match provider.code.as_deref() {
                    Some(code) if !code.is_empty() => format!("{code} {text}"),
                    _ => text,
                };
                let mut failure = classify::classify_text(
                    Some(provider.provider.as_str()),
                    provider.status,
                    &text,
                    has_retry_after,
                );
                classify::set_delay(
                    &mut failure,
                    provider.retry_after_ms.map(Duration::from_millis),
                );
                if provider.retryable
                    && failure.reason == ReasonCode::Unknown
                    && failure.retry == Retry::Never
                {
                    failure.retry = Retry::Later(None);
                }
                failure.status = provider.status;
                if failure.provider_code.is_none() {
                    failure.provider_code = provider
                        .code
                        .as_deref()
                        .and_then(classify::sanitize_identifier);
                }
                Self::Provider(failure.with_raw(provider.message))
            }
            Error::Model(text) => Self::Provider(classify_llm_text(&text)),
            Error::Catalog(text) => Self::Provider(
                ProviderFailure::new(ReasonCode::Unknown, Retry::Never).with_raw(text),
            ),
            Error::Serialization(source) => Self::Provider(
                ProviderFailure::new(ReasonCode::Unknown, Retry::Never)
                    .with_raw(source.to_string()),
            ),
            Error::Unsupported(text) => Self::Provider(
                ProviderFailure::new(ReasonCode::Unsupported, Retry::Never).with_raw(text),
            ),
            // The message can echo the input (a key, a URL), so it is kept
            // log-only rather than dropped or displayed.
            Error::Validation(text) => {
                Self::Invalid(InvalidInput::Rejected(crate::secret::LogOnly::new(text)))
            }
        }
    }
}

/// Classifies llm's free-text model error: URLs stripped first (transport text
/// always contains the request URL).
///
/// The status llm's extractor guesses from free text is **reported** but not
/// treated as authoritative: it accepts any three digits after a `(`
/// (`"error (401 bytes dropped)"`), so the classifier is handed no status and
/// falls back to its whole-token rules, as it does for any text-only error.
fn classify_llm_text(text: &str) -> ProviderFailure {
    let has_retry_after = tinyinference_llm::parse_retry_after_ms(text).is_some();
    let mut failure = classify::classify_text(None, None, &strip_urls(text), has_retry_after);
    classify::set_delay(
        &mut failure,
        tinyinference_llm::parse_retry_after_ms(text).map(Duration::from_millis),
    );
    failure.status = tinyinference_llm::structured_http_status(&strip_urls(text));
    failure.with_raw(text)
}

#[cfg(test)]
#[path = "test.rs"]
mod tests;
