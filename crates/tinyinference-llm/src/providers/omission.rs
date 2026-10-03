//! Learned request-parameter omission: what an endpoint has told us, by name,
//! that a model does not accept.
//!
//! A provider that rejects a request parameter with an HTTP 400 almost always
//! names it (`Unsupported parameter: 'max_tokens'`, `Unsupported value:
//! 'temperature' does not support 0.2 with this model`). An adapter can then drop
//! that one parameter, retry once, and [`remember_omit`] it so every later
//! request to the same endpoint and model leaves it off without spending the
//! round-trip again. [`OpenAiModel`](crate::providers::openai::OpenAiModel) does
//! this on its Chat Completions path; the functions are public so a host that
//! builds its own wire body can share the same evidence rule and the same store.
//!
//! This is what keeps static capability tables (such as
//! `OpenAiModel::with_temperature_unsupported_models` or the o-series
//! `max_completion_tokens` rename) an optimisation rather than a dependency: a
//! vendor that changes a restriction silently between releases is corrected by
//! one wasted round-trip (a 400 is billed nothing), not by a library release.
//!
//! Ported from OpenCompany's `company/inference/dialect.rs` learning layer.
//!
//! # Scope of what is learned
//!
//! * **Only parameters we sent.** [`parameter_blamed_by`] looks for the names
//!   the caller says went out, never a vendor error format, so it cannot go
//!   stale and a 400 about a field we did not send never drops one we did.
//! * **Keyed on endpoint and model.** A model id is not unique across gateways;
//!   one gateway refusing `max_tokens` for `gpt-4o` says nothing about another.
//!   The endpoint key keeps the base path (a path-routed gateway fronts several
//!   upstreams) but drops the operation suffix, so `/chat/completions` and
//!   `/responses` on one base share what was learnt.
//! * **Process-wide and in-memory.** It is a cache, not a record. Losing it on
//!   restart costs one round-trip per model, and a vendor that lifts a
//!   restriction is not remembered as broken forever.

use std::collections::HashSet;
use std::sync::{OnceLock, RwLock};

/// Phrases that mark a provider error as a statement about the request's
/// *shape*. Without one, a body that merely mentions a parameter name (say,
/// model output echoed into an error) could talk us into dropping it.
const REJECTION_MARKERS: &[&str] = &[
    "unsupported",
    "not supported",
    "unsupported_value",
    "unsupported_parameter",
    "unknown parameter",
    "unrecognized",
    "not permitted",
    "invalid_request_error",
    "is deprecated",
    "does not support",
    "extra inputs",
    "unexpected",
];

/// Operation suffixes stripped from an endpoint before it keys the store:
/// they name what was asked of a service, not which service answered.
const OPERATIONS: &[&str] = &[
    "/chat/completions",
    "/completions",
    "/responses",
    "/messages",
    "/models",
];

/// One learned omission: `(endpoint scope, lowercased model id, parameter)`.
type OmissionKey = (String, String, String);

/// Which parameter, if any, a provider rejection blames — chosen only from the
/// parameters the request actually sent.
///
/// `body` is the provider's error text (message, code, or raw body); `sent` is
/// the wire names that went out. Returns `None` when the text carries no
/// rejection marker (it is not about the request's shape) or names none of
/// `sent`. When several sent names appear, the longest wins, so a name that
/// contains another is not mistaken for it.
///
/// Matching is against the *wire* names on purpose: after a rename
/// (`max_tokens` → `max_completion_tokens`) the field a vendor rejects is the
/// one that went out, not the one the caller asked with.
pub fn parameter_blamed_by<'a, S: AsRef<str>>(body: &str, sent: &'a [S]) -> Option<&'a str> {
    let body = body.to_ascii_lowercase();
    if !REJECTION_MARKERS.iter().any(|marker| body.contains(marker)) {
        return None;
    }
    let mut candidates: Vec<&str> = sent.iter().map(AsRef::as_ref).collect();
    candidates.sort_by_key(|name| std::cmp::Reverse(name.len()));
    candidates
        .into_iter()
        .find(|name| !name.is_empty() && body.contains(&name.to_ascii_lowercase()))
}

/// Records that `model` at `endpoint` rejects `parameter`, so the next request
/// there omits it without spending a round-trip.
///
/// Keyed on the endpoint as well as the model because that is who answered:
/// keyed on the model alone, one gateway's 400 would silently drop the
/// parameter for every gateway serving that id — and dropping an output cap is
/// a bill, not just a difference. The model id is compared case-insensitively.
pub fn remember_omit(endpoint: &str, model: &str, parameter: &str) {
    tracing::debug!(
        target: "tinyinference::omission",
        model,
        parameter,
        "[providers][omission] remembering a rejected request parameter"
    );
    learned_store()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .insert(key(endpoint, model, parameter));
}

/// Whether `model` at `endpoint` is known to reject `parameter` — i.e. whether
/// [`remember_omit`] recorded it for the same service in this process.
pub fn is_omitted(endpoint: &str, model: &str, parameter: &str) -> bool {
    learned_store()
        .read()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .contains(&key(endpoint, model, parameter))
}

fn key(endpoint: &str, model: &str, parameter: &str) -> OmissionKey {
    (
        endpoint_scope(endpoint),
        model.to_ascii_lowercase(),
        parameter.to_string(),
    )
}

/// The part of a URL that identifies the service: scheme, host, port and base
/// path, lowercased, with query, fragment and any [`OPERATIONS`] suffix removed.
///
/// Anything that does not parse as `scheme://rest` is used whole: a key that is
/// too specific costs one round-trip, one that is too broad is the bug the
/// endpoint scoping exists to prevent.
pub(super) fn endpoint_scope(endpoint: &str) -> String {
    let lowered = endpoint.trim().to_ascii_lowercase();
    let Some((scheme, rest)) = lowered.split_once("://") else {
        return lowered;
    };
    let authority_and_path = rest.split(['?', '#']).next().unwrap_or(rest);
    let trimmed = authority_and_path.trim_end_matches('/');
    if trimmed.is_empty() {
        return lowered;
    }
    let base = OPERATIONS
        .iter()
        .find_map(|operation| trimmed.strip_suffix(operation))
        .unwrap_or(trimmed);
    format!("{scheme}://{}", base.trim_end_matches('/'))
}

/// The single cell the reader and the writer share.
fn learned_store() -> &'static RwLock<HashSet<OmissionKey>> {
    static LEARNED: OnceLock<RwLock<HashSet<OmissionKey>>> = OnceLock::new();
    LEARNED.get_or_init(|| RwLock::new(HashSet::new()))
}

#[cfg(test)]
#[path = "omission_tests.rs"]
mod tests;
