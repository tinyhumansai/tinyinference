//! The probe runner.

use crate::catalog::ModelEntry;
use crate::descriptor::Quirk;
use crate::error::{
    HubError, InputField, InvalidInput, Operation, PolicyViolation, ProviderFailure, ReasonCode,
    Retry,
};
use crate::ids::ModelId;
use crate::kinds::{DriverContext, KindDriver, Target};
use crate::policy::check_endpoint_with_credential;
use crate::taxonomy::{ProviderGroup, TestDepth};

use super::types::{ProbeNote, ProbeReport};

/// What a probe does, with what it needs already in hand.
enum Step<'a> {
    KeyOnly,
    Catalog,
    Completion(&'a ModelId),
}

/// A failure that stands for "the endpoint policy said no". It is an
/// `endpoint` failure so the add flow keeps the key (only `auth` rolls one
/// back), and never retried: retrying a refused address cannot help.
fn refusal_failure(violation: &PolicyViolation) -> ProviderFailure {
    ProviderFailure::new(ReasonCode::Endpoint, Retry::Never).with_raw(violation.to_string())
}

/// Runs one probe of `target` at `depth`.
///
/// Before anything is sent, the endpoint is checked against the policy with the
/// credential in hand (so a key is never sent over cleartext http off this
/// host); a refusal is reported as an `endpoint` failure with
/// [`ProbeReport::refusal`] set, and nothing is requested. The transport
/// applies the same policy again on every redirect hop.
///
/// # Errors
///
/// [`HubError::Unsupported`] when the kind does not support `depth`;
/// [`HubError::SignedOut`] when a managed provider has no credential;
/// [`HubError::Invalid`] when a kind that needs a key was given none, or a
/// completion was asked for without a model. A provider failing the check is
/// **not** an error: it is in the report.
pub async fn run_probe(
    cx: &DriverContext<'_>,
    driver: &dyn KindDriver,
    target: &Target<'_>,
    depth: TestDepth,
) -> Result<ProbeReport, HubError> {
    let descriptor = driver.descriptor();
    if !descriptor.supports_depth(depth) {
        return Err(HubError::Unsupported {
            op: Operation::Test(depth),
            kind: descriptor.kind.clone(),
        });
    }
    let has_key = target.key().is_some();
    // A key is only *presented* when the style sends one (a keyless local
    // runtime may be given a key it never receives).
    let credentialed = has_key && target.auth.needs_credential();
    if target.auth.needs_credential() && !has_key {
        if target.group == ProviderGroup::Managed {
            return Err(HubError::SignedOut {
                provider: target.slug.clone(),
            });
        }
        if descriptor.needs_key {
            return Err(HubError::Invalid(InvalidInput::Empty(InputField::Key)));
        }
    }
    let step = match depth {
        TestDepth::KeyOnly => Step::KeyOnly,
        TestDepth::Catalog => Step::Catalog,
        TestDepth::Completion => Step::Completion(target.model.ok_or(HubError::Invalid(
            InvalidInput::Malformed {
                field: InputField::ModelId,
                reason: "a model id is required for a completion test",
            },
        ))?),
    };

    let mut report = ProbeReport {
        depth,
        failure: None,
        refusal: None,
        latency: std::time::Duration::ZERO,
        started_ms: cx.clock.wall_ms(),
        models: Vec::new(),
        proves_key: false,
        notes: Vec::new(),
    };

    if let Err(refusal) = check_endpoint_with_credential(target.base_url, cx.policy, credentialed) {
        let violation = PolicyViolation::from(refusal);
        report.failure = Some(refusal_failure(&violation));
        report.refusal = Some(violation);
        return Ok(report);
    }

    let mut public_fallback = false;
    let started = cx.clock.now();
    let outcome: Result<Vec<ModelEntry>, HubError> = match step {
        Step::KeyOnly => driver.key_check(cx, target).await.map(|()| Vec::new()),
        Step::Catalog => driver.list_models(cx, target).await.map(|fetched| {
            if fetched.truncated {
                report.notes.push(ProbeNote::CatalogTruncated);
            }
            public_fallback = fetched.public_fallback;
            fetched.models
        }),
        Step::Completion(model) => driver
            .completion_ping(cx, target, model)
            .await
            .map(|()| Vec::new()),
    };
    report.latency = cx.clock.now().saturating_duration_since(started);

    match outcome {
        Ok(models) => {
            report.proves_key = credentialed
                && match depth {
                    TestDepth::KeyOnly | TestDepth::Completion => true,
                    TestDepth::Catalog => {
                        !descriptor.has_quirk(Quirk::CatalogUnauthenticated) && !public_fallback
                    }
                };
            if depth == TestDepth::Catalog {
                if credentialed
                    && (public_fallback || descriptor.has_quirk(Quirk::CatalogUnauthenticated))
                {
                    report.notes.push(ProbeNote::CatalogDoesNotProveKey);
                }
                if models.is_empty() && descriptor.has_quirk(Quirk::CatalogAccountScoped) {
                    report.notes.push(ProbeNote::AccountScopedCatalogIsEmpty);
                }
            }
            report.models = models;
            Ok(report)
        }
        Err(HubError::Provider(failure)) => {
            report.failure = Some(failure);
            Ok(report)
        }
        Err(HubError::Policy(violation)) => {
            report.failure = Some(refusal_failure(&violation));
            report.refusal = Some(violation);
            Ok(report)
        }
        Err(other) => Err(other),
    }
}
