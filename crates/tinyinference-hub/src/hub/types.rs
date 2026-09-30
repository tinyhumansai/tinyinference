//! The value types the [`Hub`](super::Hub) operations take and return.

use std::fmt;
use std::time::Duration;

use crate::config::DefaultChoice;
use crate::credential::CredentialOrigin;
use crate::descriptor::ProviderRecord;
use crate::error::{ProviderFailure, UsedBy};
use crate::health::{HealthSnapshot, ProviderHealth};
use crate::ids::{ModelId, Slug};
use crate::probe::ProbeReport;
use crate::secret::Secret;
use crate::taxonomy::{ProviderGroup, TestDepth};

/// Confirmation the caller gives for an operation that would orphan a
/// reference (the default choice, an agent pin, a workload route).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Confirm {
    /// The caller knows the provider is in use and wants to go ahead. The
    /// references are left in place and fail closed on the turn path.
    pub in_use: bool,
}

impl Confirm {
    /// No confirmation: an in-use provider refuses the operation.
    pub fn no() -> Self {
        Self::default()
    }

    /// Confirmed: go ahead even though the provider is in use.
    pub fn in_use() -> Self {
        Self { in_use: true }
    }
}

/// Options for [`Hub::connect`](super::Hub::connect).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConnectOptions {
    /// Make the new provider the default choice (its model is required).
    pub make_default: bool,
    /// Keep the provider even when the check fails (the operator wants to add
    /// it now and fix the endpoint later). Never overrides a policy refusal.
    pub add_anyway: bool,
    /// How deep the check goes.
    pub depth: TestDepth,
}

impl Default for ConnectOptions {
    fn default() -> Self {
        Self {
            make_default: false,
            add_anyway: false,
            depth: TestDepth::Catalog,
        }
    }
}

impl ConnectOptions {
    /// Sets `make_default`.
    #[must_use]
    pub fn make_default(mut self, on: bool) -> Self {
        self.make_default = on;
        self
    }

    /// Sets `add_anyway`.
    #[must_use]
    pub fn add_anyway(mut self, on: bool) -> Self {
        self.add_anyway = on;
        self
    }

    /// Sets the depth.
    #[must_use]
    pub fn depth(mut self, depth: TestDepth) -> Self {
        self.depth = depth;
        self
    }
}

/// A change to a saved provider. Fields left `None` are unchanged; the kind
/// never changes.
#[non_exhaustive]
#[derive(Clone, Default)]
pub struct ProviderPatch {
    /// A new display name. The slug never changes.
    pub label: Option<String>,
    /// A new endpoint (ignored for a catalogue preset, guard G2).
    pub base_url: Option<String>,
    /// A new model.
    pub model: Option<ModelId>,
    /// A new key (a rotation is never guarded).
    pub key: Option<Secret>,
}

impl ProviderPatch {
    /// A patch that changes nothing.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the label.
    #[must_use]
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the endpoint.
    #[must_use]
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the model.
    #[must_use]
    pub fn model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the key.
    #[must_use]
    pub fn key(mut self, key: Secret) -> Self {
        self.key = Some(key);
        self
    }
}

impl fmt::Debug for ProviderPatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProviderPatch")
            .field("label", &self.label)
            .field(
                "base_url",
                &self
                    .base_url
                    .as_deref()
                    .map(crate::endpoint::redact_endpoint),
            )
            .field("model", &self.model)
            .field("key", &self.key)
            .finish()
    }
}

/// What a mutation did.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutationStatus {
    /// Saved, and the check (if any) passed.
    Saved,
    /// Saved, but something needs attention; see the note. A probe that failed or
    /// could not run ([`Mutation::probe`]), or, for an `edit` that moved the
    /// endpoint, a provider that could not be switched back on (call
    /// `Hub::set_enabled`).
    SavedWithWarning,
    /// The request changed nothing.
    Unchanged,
}

/// Whether a key exists for a provider and where it comes from.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyState {
    /// A source answered.
    Configured(CredentialOrigin),
    /// Nothing answered.
    Missing,
    /// The credential store could not be read, so nothing is known.
    Unreadable,
}

impl KeyState {
    /// Whether a key exists.
    pub fn is_configured(&self) -> bool {
        matches!(self, Self::Configured(_))
    }

    /// The source that answered, if any.
    pub fn origin(&self) -> Option<&CredentialOrigin> {
        match self {
            Self::Configured(origin) => Some(origin),
            _ => None,
        }
    }
}

/// A provider as an operator sees it: the record and what surrounds it. Never
/// carries a credential.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderView {
    /// The saved record.
    pub record: ProviderRecord,
    /// The record's group.
    pub group: ProviderGroup,
    /// The kind's display label.
    pub kind_label: String,
    /// Whether a key exists and where it comes from.
    pub key: KeyState,
    /// Whether this provider is the default choice.
    pub is_default: bool,
}

/// What a mutating operation returns.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Mutation {
    /// What happened.
    pub status: MutationStatus,
    /// A sentence safe to show an operator. Never contains raw upstream text.
    pub note: String,
    /// The check that ran, if the operation ran one.
    pub probe: Option<ProbeReport>,
    /// What referenced the provider, for a guarded operation that went ahead
    /// on confirmation.
    pub used_by: Option<UsedBy>,
    /// The provider after the change (the removed record for a removal), or
    /// `None` for an operation that names no provider (clearing the default,
    /// removing a pin or a route).
    pub record: Option<ProviderView>,
}

/// One provider's health and key state.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderStatus {
    /// The provider.
    pub view: ProviderView,
    /// The health an operator sees: `Disabled` for a disabled or offline-excluded
    /// row, `SignedOut` for a managed row with no credential, otherwise what
    /// probes and turns have folded to.
    pub health: ProviderHealth,
    /// The signals behind the health.
    pub snapshot: HealthSnapshot,
}

/// The whole scope at a glance. Managed first (D5), then the operator's rows in
/// the order they were added.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct HubStatus {
    /// Every provider.
    pub providers: Vec<ProviderStatus>,
    /// The scope's default choice.
    pub default: DefaultChoice,
    /// The provider a UI should show as current: the default's provider when it
    /// is enabled, otherwise the first enabled row. Only for display: the turn
    /// path never falls back like this.
    pub primary: Option<Slug>,
}

/// One provider a scheduled re-test ran against.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Retested {
    /// The provider.
    pub slug: Slug,
    /// The check's outcome: `None` when it passed.
    pub failure: Option<ProviderFailure>,
    /// The health afterwards.
    pub health: ProviderHealth,
}

/// Which of the hub's policy knobs a host set.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HubPolicy {
    /// Restrict each catalogue kind to a **single** row (OpenCompany's rule); a
    /// second account of `openai` is then refused and has to be a `custom`
    /// endpoint. The default is `false`: a second account of a kind is allowed
    /// as another instance with its own slug.
    pub one_row_per_kind: bool,
    /// Words that are not model ids (a host's workload tiers). The hub reserves
    /// none itself (D7).
    pub reserved_model_words: Vec<String>,
    /// How long a provider that is `Down` on a rejected key or an exhausted
    /// account waits before [`Hub::retest_down`](super::Hub::retest_down) tries
    /// it again.
    pub retest_after: Duration,
}

impl Default for HubPolicy {
    fn default() -> Self {
        Self {
            one_row_per_kind: false,
            reserved_model_words: Vec::new(),
            retest_after: Duration::from_secs(300),
        }
    }
}

impl HubPolicy {
    /// The defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets `one_row_per_kind`.
    #[must_use]
    pub fn one_row_per_kind(mut self, on: bool) -> Self {
        self.one_row_per_kind = on;
        self
    }

    /// Sets the reserved model words.
    #[must_use]
    pub fn reserved_model_words<I, S>(mut self, words: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.reserved_model_words = words.into_iter().map(Into::into).collect();
        self
    }

    /// Sets the re-test interval.
    #[must_use]
    pub fn retest_after(mut self, after: Duration) -> Self {
        self.retest_after = after;
        self
    }
}
