//! The data types of the hub's error taxonomy.
//!
//! [`HubError`] is deliberately the hub's own `#[non_exhaustive]` enum rather
//! than a new variant on `tinyinference_llm::Error`: llm's error is matched
//! exhaustively by downstream crates, so growing it would break them
//! (`docs/spec` compat rule 2). [`ReasonCode`] is the stable wire vocabulary
//! shared by both hosts' classifiers, with `rate_limited` split from `quota`
//! so a spend cap is never retried as a cooldown.

use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::ids::{AgentKey, KindId, ModelId, Slug, WorkloadKey};
use crate::policy::EndpointRefusal;
use crate::secret::LogOnly;
use crate::taxonomy::TestDepth;

/// Why an operation failed, as a stable snake_case wire string.
///
/// The provider-facing codes (`auth` .. `unknown`) are the union of both
/// hosts' classifiers; the rest describe failures inside the hub itself.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    /// The provider rejected the credential. The only destructive class.
    Auth,
    /// The endpoint does not know that model id.
    Model,
    /// Spend or plan quota is exhausted. Never retried as a cooldown.
    Quota,
    /// A transient rate limit; retrying later may succeed.
    RateLimited,
    /// Nothing answered at that address.
    Endpoint,
    /// Something answered too slowly.
    Timeout,
    /// The managed provider has no credential source that answers.
    SignedOut,
    /// The kind cannot perform the operation.
    Unsupported,
    /// The check did not complete and the hub will not guess why.
    Unknown,
    /// The endpoint policy refused the request.
    Policy,
    /// Caller input failed validation.
    Invalid,
    /// A named provider, model, agent or workload does not exist.
    NotFound,
    /// The slug is already taken by a provider in this scope.
    AlreadyExists,
    /// The target is referenced and the caller did not confirm.
    InUse,
    /// The stored configuration changed concurrently.
    Conflict,
    /// A host store could not be read; never treated as "absent".
    StoreUnreadable,
    /// No provider resolves for the turn.
    Unresolved,
}

impl ReasonCode {
    /// Every code, in wire order. Used by tests and by hosts that render a
    /// legend.
    pub const ALL: [ReasonCode; 17] = [
        Self::Auth,
        Self::Model,
        Self::Quota,
        Self::RateLimited,
        Self::Endpoint,
        Self::Timeout,
        Self::SignedOut,
        Self::Unsupported,
        Self::Unknown,
        Self::Policy,
        Self::Invalid,
        Self::NotFound,
        Self::AlreadyExists,
        Self::InUse,
        Self::Conflict,
        Self::StoreUnreadable,
        Self::Unresolved,
    ];

    /// The stable wire spelling (identical to the serde form).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auth => "auth",
            Self::Model => "model",
            Self::Quota => "quota",
            Self::RateLimited => "rate_limited",
            Self::Endpoint => "endpoint",
            Self::Timeout => "timeout",
            Self::SignedOut => "signed_out",
            Self::Unsupported => "unsupported",
            Self::Unknown => "unknown",
            Self::Policy => "policy",
            Self::Invalid => "invalid",
            Self::NotFound => "not_found",
            Self::AlreadyExists => "already_exists",
            Self::InUse => "in_use",
            Self::Conflict => "conflict",
            Self::StoreUnreadable => "store_unreadable",
            Self::Unresolved => "unresolved",
        }
    }

    /// Whether meeting this code should roll back a credential that was just
    /// written. Exactly one code says yes; every other one is a connection
    /// fact, not a key fact.
    pub fn destroys_credential(self) -> bool {
        matches!(self, Self::Auth)
    }
}

impl fmt::Display for ReasonCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Whether and when trying the same request again may help.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Retry {
    /// Retrying cannot help (bad key, spend cap, unknown model, ...).
    Never,
    /// Retrying later may help, optionally after the provider's own delay.
    Later(Option<Duration>),
    /// Retrying immediately is reasonable (a lost compare-and-swap).
    Now,
}

/// The field a validation failure is about.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InputField {
    /// A provider slug.
    Slug,
    /// A provider display name.
    ProviderName,
    /// A model id.
    ModelId,
    /// An endpoint URL.
    Endpoint,
    /// A kind id.
    Kind,
    /// A scope, agent or workload key.
    Key,
}

impl fmt::Display for InputField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Slug => "a provider slug",
            Self::ProviderName => "a provider name",
            Self::ModelId => "a model id",
            Self::Endpoint => "an endpoint",
            Self::Kind => "a provider kind",
            Self::Key => "a key",
        })
    }
}

/// Why caller input was refused.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InvalidInput {
    /// Nothing was supplied, or it normalised to nothing.
    #[error("{0} cannot be empty")]
    Empty(InputField),
    /// Longer than the bound, counted in characters.
    #[error("{field} can be at most {max} characters")]
    TooLong {
        /// The field.
        field: InputField,
        /// The inclusive maximum, in characters.
        max: usize,
    },
    /// Contains a control character.
    #[error("{0} cannot contain control characters")]
    ControlCharacters(InputField),
    /// Contains whitespace where none is allowed.
    #[error("{0} cannot contain spaces")]
    Whitespace(InputField),
    /// A word the host or the catalogue reserves.
    #[error("`{value}` is reserved and cannot be used as {field}")]
    Reserved {
        /// The field.
        field: InputField,
        /// The refused value.
        value: String,
    },
    /// Contains characters outside the allowed alphabet.
    #[error("{0} has characters that are not allowed")]
    BadCharacters(InputField),
    /// Structurally wrong.
    #[error("{field} is malformed: {reason}")]
    Malformed {
        /// The field.
        field: InputField,
        /// A fixed, secret-free reason.
        reason: &'static str,
    },
    /// A record carries a field whose name marks it as a credential. A
    /// credential is never stored on a record (invariant 1); it belongs in the
    /// credential store.
    #[error("record field `{name}` looks like a credential and cannot be stored on a record")]
    CredentialField {
        /// The offending field name (a name, never a value).
        name: String,
    },
    /// A lower layer rejected the request and its own message may echo the
    /// input, so the detail is log-only.
    #[error("the request was rejected as invalid")]
    Rejected(LogOnly<String>),
}

/// Why the endpoint policy refused something.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PolicyViolation {
    /// The address or scheme is not allowed.
    #[error("{0}")]
    Endpoint(EndpointRefusal),
    /// The URL carries `user:password@`.
    #[error("the endpoint carries a username or password")]
    CredentialInEndpoint,
    /// A redirect chain ran past the limit.
    #[error("more than {max} redirects")]
    TooManyRedirects {
        /// The configured limit.
        max: usize,
    },
    /// A credentialed request was redirected to another origin.
    #[error("a credentialed request cannot follow a redirect to another origin")]
    CrossOriginRedirect,
}

impl From<EndpointRefusal> for PolicyViolation {
    /// A URL that carries a credential maps to
    /// [`PolicyViolation::CredentialInEndpoint`]; every other refusal is wrapped
    /// as [`PolicyViolation::Endpoint`].
    fn from(refusal: EndpointRefusal) -> Self {
        match refusal {
            EndpointRefusal::CredentialInUrl => Self::CredentialInEndpoint,
            other => Self::Endpoint(other),
        }
    }
}

/// What was looked up and not found.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum NotFound {
    /// No provider with that slug.
    #[error("provider `{0}`")]
    Provider(Slug),
    /// No such model on that provider.
    #[error("model `{model}` on `{provider}`")]
    Model {
        /// The provider.
        provider: Slug,
        /// The model.
        model: ModelId,
    },
    /// No such agent.
    #[error("agent `{0}`")]
    Agent(AgentKey),
    /// No such workload.
    #[error("workload `{0}`")]
    Workload(WorkloadKey),
    /// No such provider kind.
    #[error("provider kind `{0}`")]
    Kind(KindId),
}

/// Why no provider resolves for a turn.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Unresolved {
    /// Nothing is configured and no default exists.
    #[error("no provider is configured")]
    NoProvider,
    /// The default names only a provider, which fails closed on the turn path.
    #[error("the default names provider `{0}` but no model")]
    ProviderOnlyDefault(Slug),
    /// The chosen provider does not exist.
    #[error("provider `{0}` does not exist")]
    Missing(Slug),
    /// The chosen provider is disabled.
    #[error("provider `{0}` is disabled")]
    Disabled(Slug),
    /// The chosen provider needs a key and has none.
    #[error("provider `{0}` has no key")]
    NoKey(Slug),
}

/// What references a provider, for the in-use guard.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UsedBy {
    /// The provider is the default choice.
    pub default_choice: bool,
    /// Agents pinned to the provider.
    pub agents: Vec<AgentKey>,
    /// Workload routes that name the provider.
    pub workloads: Vec<WorkloadKey>,
    /// Other host-defined references, described in plain words.
    pub other: Vec<String>,
}

impl UsedBy {
    /// Whether nothing references the provider.
    pub fn is_empty(&self) -> bool {
        !self.default_choice
            && self.agents.is_empty()
            && self.workloads.is_empty()
            && self.other.is_empty()
    }

    /// How many references there are in total.
    pub fn count(&self) -> usize {
        usize::from(self.default_choice)
            + self.agents.len()
            + self.workloads.len()
            + self.other.len()
    }
}

impl fmt::Display for UsedBy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts: Vec<String> = Vec::new();
        if self.default_choice {
            parts.push("the default".to_string());
        }
        if !self.agents.is_empty() {
            parts.push(format!("{} agent pin(s)", self.agents.len()));
        }
        if !self.workloads.is_empty() {
            parts.push(format!("{} workload route(s)", self.workloads.len()));
        }
        if !self.other.is_empty() {
            parts.push(format!("{} other reference(s)", self.other.len()));
        }
        if parts.is_empty() {
            f.write_str("nothing")
        } else {
            f.write_str(&parts.join(", "))
        }
    }
}

/// A host port, named in errors so a host can tell which store failed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PortName {
    /// The credential store.
    Credentials,
    /// The configuration store.
    Config,
    /// The health store.
    Health,
    /// The HTTP transport.
    Http,
    /// A rotating token source.
    Token,
    /// The subprocess spawner.
    Process,
}

impl fmt::Display for PortName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Credentials => "credential store",
            Self::Config => "config store",
            Self::Health => "health store",
            Self::Http => "http transport",
            Self::Token => "token source",
            Self::Process => "process spawner",
        })
    }
}

/// An operation of the hub, named in `Unsupported` errors.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Add a provider and probe it.
    Connect,
    /// Add a provider without probing.
    Add,
    /// Edit a provider.
    Edit,
    /// Remove a provider.
    Remove,
    /// Enable or disable a provider.
    SetEnabled,
    /// Set a provider's key.
    SetKey,
    /// Clear a provider's key.
    ClearKey,
    /// Probe a draft that is not saved.
    ProbeDraft,
    /// Test a saved provider at a depth.
    Test(TestDepth),
    /// List a provider's models.
    ListModels,
    /// Read a provider's health.
    Health,
    /// Record a turn outcome.
    RecordOutcome,
    /// Set the default choice.
    SetDefault,
    /// Pin an agent to a model.
    PinAgent,
    /// Resolve a turn to a provider and model.
    ResolveForTurn,
    /// Build a chat model.
    ChatModel,
    /// Read a local runtime's status.
    LocalStatus,
    /// Check a CLI login.
    CliReadiness,
    /// Start or finish an OAuth flow.
    OAuth,
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Connect => f.write_str("connect"),
            Self::Add => f.write_str("add"),
            Self::Edit => f.write_str("edit"),
            Self::Remove => f.write_str("remove"),
            Self::SetEnabled => f.write_str("set_enabled"),
            Self::SetKey => f.write_str("set_key"),
            Self::ClearKey => f.write_str("clear_key"),
            Self::ProbeDraft => f.write_str("probe_draft"),
            Self::Test(depth) => write!(f, "test({depth})"),
            Self::ListModels => f.write_str("list_models"),
            Self::Health => f.write_str("health"),
            Self::RecordOutcome => f.write_str("record_outcome"),
            Self::SetDefault => f.write_str("set_default"),
            Self::PinAgent => f.write_str("pin_agent"),
            Self::ResolveForTurn => f.write_str("resolve_for_turn"),
            Self::ChatModel => f.write_str("chat_model"),
            Self::LocalStatus => f.write_str("local_status"),
            Self::CliReadiness => f.write_str("cli_readiness"),
            Self::OAuth => f.write_str("oauth"),
        }
    }
}

/// A classified failure reported by a provider (or by the transport in front of
/// one).
///
/// Carries the class *and* the raw upstream text, because they go to different
/// places: the class decides what happens to the credential and what the
/// operator is told, while the raw text is [`LogOnly`] and never reaches
/// `Display`, `Debug` or [`HubError::user_message`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderFailure {
    /// What the failure means.
    pub reason: ReasonCode,
    /// Whether retrying may help.
    pub retry: Retry,
    /// The HTTP status, when the failure came from HTTP.
    pub status: Option<u16>,
    /// The provider's own code or type, for example
    /// `enforced_spend_limit_reached`.
    pub provider_code: Option<String>,
    /// The provider's request id, for support tickets.
    pub request_id: Option<String>,
    /// The body ran past its cap and was cut.
    pub truncated: bool,
    /// The upstream text, for a log or detail channel only.
    pub raw: LogOnly<String>,
}

impl ProviderFailure {
    /// A failure with just a class and a retry hint.
    pub fn new(reason: ReasonCode, retry: Retry) -> Self {
        Self {
            reason,
            retry,
            status: None,
            provider_code: None,
            request_id: None,
            truncated: false,
            raw: LogOnly::default(),
        }
    }

    /// Sets the HTTP status.
    #[must_use]
    pub fn with_status(mut self, status: u16) -> Self {
        self.status = Some(status);
        self
    }

    /// Sets the provider's own code.
    #[must_use]
    pub fn with_provider_code(mut self, code: impl Into<String>) -> Self {
        self.provider_code = Some(code.into());
        self
    }

    /// Sets the provider's request id.
    #[must_use]
    pub fn with_request_id(mut self, id: impl Into<String>) -> Self {
        self.request_id = Some(id.into());
        self
    }

    /// Marks the body as cut at its cap.
    #[must_use]
    pub fn with_truncated(mut self, truncated: bool) -> Self {
        self.truncated = truncated;
        self
    }

    /// Attaches the raw upstream text (log-only). The text is passed through
    /// [`scrub_log_text`](super::scrub_log_text) first, so a URL's userinfo or
    /// query string and an echoed `Authorization` value never sit in the field
    /// a host will write to a log.
    #[must_use]
    pub fn with_raw(mut self, raw: impl AsRef<str>) -> Self {
        self.raw = LogOnly::new(super::scrub_log_text(raw.as_ref()));
        self
    }
}

impl fmt::Display for ProviderFailure {
    /// Never prints [`ProviderFailure::raw`].
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.reason)?;
        if let Some(status) = self.status {
            write!(f, " (status {status})")?;
        }
        if let Some(code) = &self.provider_code {
            write!(f, " [{code}]")?;
        }
        Ok(())
    }
}

/// The hub's typed error.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum HubError {
    /// A provider (or the transport in front of it) failed.
    #[error("provider refused: {0}")]
    Provider(ProviderFailure),
    /// The managed provider has no credential source that answers.
    #[error("signed out of {provider}")]
    SignedOut {
        /// The managed provider's slug.
        provider: Slug,
    },
    /// The kind cannot perform the operation.
    #[error("{op} is not supported by {kind}")]
    Unsupported {
        /// The operation.
        op: Operation,
        /// The kind.
        kind: KindId,
    },
    /// The endpoint policy refused.
    #[error("endpoint refused by policy: {0}")]
    Policy(PolicyViolation),
    /// Caller input failed validation.
    #[error("invalid input: {0}")]
    Invalid(InvalidInput),
    /// Something named does not exist.
    #[error("{0} not found")]
    NotFound(NotFound),
    /// The slug is taken.
    #[error("{slug} already connected")]
    AlreadyExists {
        /// The slug.
        slug: Slug,
    },
    /// The provider is referenced and the caller did not confirm.
    #[error("in use: {0}")]
    InUse(UsedBy),
    /// The stored configuration changed while the operation ran.
    #[error("configuration changed concurrently")]
    Conflict,
    /// A host store could not be read.
    #[error("{port} unreadable")]
    StoreUnreadable {
        /// Which store.
        port: PortName,
        /// The store's own message, for a log only.
        detail: LogOnly<String>,
    },
    /// No provider resolves for this turn.
    #[error("no provider resolves for this turn: {0}")]
    Unresolved(Unresolved),
}

/// The context [`HubError::user_message`] needs to write a sentence.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyContext {
    /// The provider's display label, or the subject of the sentence.
    pub subject: String,
    /// Whether the operation was undone (a rolled-back add) rather than saved.
    pub undone: bool,
}

impl CopyContext {
    /// Context for an operation that saved its change.
    pub fn saved(subject: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            undone: false,
        }
    }

    /// Context for an operation that was rolled back.
    pub fn undone(subject: impl Into<String>) -> Self {
        Self {
            subject: subject.into(),
            undone: true,
        }
    }
}
