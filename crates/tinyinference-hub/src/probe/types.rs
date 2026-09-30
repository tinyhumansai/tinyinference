//! The result of a probe.

use std::time::Duration;

use crate::catalog::ModelEntry;
use crate::error::{HubError, PolicyViolation, ProviderFailure};
use crate::taxonomy::TestDepth;

/// Something a probe learned that a bare pass/fail does not say.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeNote {
    /// The catalog was readable but says nothing about the key: the kind's
    /// listing is public (`Quirk::CatalogUnauthenticated`), or the driver fell
    /// back from an account-scoped listing to the public one (OpenRouter's
    /// `/models/user` answering 404). Only a completion can fail a bad key.
    CatalogDoesNotProveKey,
    /// The listing is scoped to the account and came back empty: a fresh
    /// Fireworks key sees no public models, so an empty list is expected there,
    /// not a fault (`Quirk::CatalogAccountScoped`).
    AccountScopedCatalogIsEmpty,
    /// A **successful** catalog read had more pages than one read follows, so
    /// the models are a prefix. (A listing too large to read at all is a failure
    /// with `truncated` set, not this note.)
    CatalogTruncated,
}

/// What one probe found.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeReport {
    /// The depth that ran.
    pub depth: TestDepth,
    /// Why the provider refused, or `None` when the check passed.
    pub failure: Option<ProviderFailure>,
    /// Set when the failure was the endpoint policy refusing the request before
    /// (or between) sending. The failure is then an `endpoint` failure: the
    /// endpoint is not usable here, which says nothing about the key.
    pub refusal: Option<PolicyViolation>,
    /// How long the check took, by the hub's clock.
    pub latency: Duration,
    /// When the check started, in wall-clock milliseconds by the hub's clock.
    /// The health tracker uses it so a pass never supersedes a failure that was
    /// recorded while the probe was still running.
    pub started_ms: u64,
    /// The models read, for a passing `Catalog` probe.
    pub models: Vec<ModelEntry>,
    /// Whether a pass proves the key works (see [`ProbeNote`]).
    pub proves_key: bool,
    /// Things worth telling the operator.
    pub notes: Vec<ProbeNote>,
}

impl ProbeReport {
    /// Whether the check passed.
    pub fn ok(&self) -> bool {
        self.failure.is_none()
    }

    /// How many models a passing `Catalog` probe read.
    pub fn model_count(&self) -> Option<usize> {
        (self.depth == TestDepth::Catalog && self.ok()).then_some(self.models.len())
    }

    /// The report as a `Result`: a failed check becomes
    /// [`HubError::Provider`] (or [`HubError::Policy`] for a refusal).
    ///
    /// # Errors
    ///
    /// The failure the check produced.
    pub fn into_result(self) -> Result<Self, HubError> {
        match (&self.refusal, &self.failure) {
            (Some(violation), _) => Err(HubError::Policy(violation.clone())),
            (None, Some(failure)) => Err(HubError::Provider(failure.clone())),
            (None, None) => Ok(self),
        }
    }
}
