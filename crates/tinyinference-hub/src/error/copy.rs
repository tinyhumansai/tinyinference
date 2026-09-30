//! User-facing copy for hub errors.
//!
//! **Never interpolates raw upstream text.** That text can carry request
//! material (headers, fragments of a key), and these sentences land in banners
//! that get screenshotted into tickets. The raw text lives in
//! [`ProviderFailure::raw`](super::ProviderFailure) and goes to a log channel.
//!
//! Every provider-facing sentence but the first begins with "Saved" in
//! [`describe`], because every provider class but `auth` kept the record and
//! the credential: the save is a fact and only reachability is in question. The
//! hub-internal classes (`signed_out`, `invalid`, `conflict`, ...) describe
//! themselves instead. When a class rolls the add back,
//! [`describe_refusal`] has its own sentences, each naming the next thing to
//! do, because telling the operator something was saved while no row appears is
//! worse than telling them nothing.

use super::types::{CopyContext, HubError, ReasonCode};

/// What to tell the operator after a change that was **saved**, given a class
/// and the provider's label.
pub fn describe(reason: ReasonCode, provider: &str) -> String {
    match reason {
        ReasonCode::Auth => {
            format!("Could not reach {provider}: the provider rejected the credential.")
        }
        ReasonCode::Endpoint => format!("Saved, but nothing answered at {provider}."),
        ReasonCode::Model => "Saved. The endpoint did not recognise that model id.".to_string(),
        ReasonCode::Quota => {
            "Saved. The account is out of credit or over its spend limit.".to_string()
        }
        ReasonCode::RateLimited => {
            format!("Saved. {provider} is rate limiting requests; try again shortly.")
        }
        ReasonCode::Timeout => format!("Saved, but {provider} did not answer in time."),
        ReasonCode::Unknown => "Saved, but the check did not complete.".to_string(),
        ReasonCode::SignedOut => format!("Sign in to use {provider}."),
        ReasonCode::Unsupported => format!("{provider} does not support that."),
        ReasonCode::Policy => "That endpoint is not allowed on this deployment.".to_string(),
        ReasonCode::Invalid => "That input is not valid.".to_string(),
        ReasonCode::NotFound => format!("{provider} was not found."),
        ReasonCode::AlreadyExists => format!("{provider} is already connected."),
        ReasonCode::InUse => format!("{provider} is still in use."),
        ReasonCode::Conflict => "The settings changed while saving. Try again.".to_string(),
        ReasonCode::StoreUnreadable => "The settings store could not be read.".to_string(),
        ReasonCode::Unresolved => "No provider is available for this turn.".to_string(),
    }
}

/// What to tell the operator when the add was **undone**. Never says "Saved".
pub fn describe_refusal(reason: ReasonCode, subject: &str) -> String {
    match reason {
        ReasonCode::Auth => {
            format!("Could not reach {subject}: the provider rejected the credential.")
        }
        ReasonCode::Endpoint => format!(
            "Nothing answered at {subject}, so it was not connected. Start it and try again."
        ),
        ReasonCode::Timeout => {
            format!("{subject} did not answer in time, so it was not connected.")
        }
        // Not reachable through `rolls_back` for a cloud kind. Answered rather
        // than panicked, because a future class joining the rollback set should
        // degrade to a true sentence.
        _ => format!("Could not verify {subject}, so it was not connected."),
    }
}

impl HubError {
    /// A sentence safe to show an operator. Never contains raw upstream text.
    pub fn user_message(&self, ctx: CopyContext) -> String {
        let reason = self.reason();
        let provider_facing = matches!(
            reason,
            ReasonCode::Auth
                | ReasonCode::Model
                | ReasonCode::Quota
                | ReasonCode::RateLimited
                | ReasonCode::Endpoint
                | ReasonCode::Timeout
                | ReasonCode::Unknown
        );
        // A validation or policy refusal is not a failed connectivity check, so
        // it keeps its own sentence even when the add was undone.
        if ctx.undone && provider_facing {
            describe_refusal(reason, &ctx.subject)
        } else {
            describe(reason, &ctx.subject)
        }
    }
}
