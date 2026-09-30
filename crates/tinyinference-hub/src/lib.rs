//! One provider hub for every TinyInference host.
//!
//! OpenCompany and OpenHuman each implemented provider management separately:
//! two catalogues of the same hosted vendors, two error classifiers, two managed
//! paths, and safety invariants (SSRF policy, tenant-scoped model cache, "only a
//! rejected key rolls a new key back") that only one of them had. This crate is
//! the shared answer, built as a **leaf**: it depends on `tinyinference-llm`
//! (plus small utility crates), nothing depends on it, and no existing public
//! item of any other crate changed.
//!
//! # The shape of it
//!
//! A host implements four small ports and gets one [`Hub`]:
//!
//! | Port | What it is | Notes |
//! |---|---|---|
//! | [`CredentialStore`](ports::CredentialStore) | where keys live | `get`/`set`/`delete`; an error means *unreadable*, never *no key* |
//! | [`ConfigStore`](ports::ConfigStore) | the persisted [`HubConfig`] | compare-and-swap, so two processes cannot clobber each other |
//! | [`Http`](ports::Http) | the transport probes and catalogs use | must apply the endpoint policy on every redirect hop and pin the address it checked (`ports::ReqwestHttp`, feature `http-reqwest`, is a reference) |
//! | [`Clock`](ports::Clock) | the only source of time | so a simulation can run an hour of cache expiry without sleeping |
//!
//! Everything else has a default: health and events in memory, no detection, no
//! rotating token. The hub is runtime-agnostic and spawns nothing.
//!
//! ```
//! use std::sync::Arc;
//! use tinyinference_hub::ports::memory::{MemoryConfig, MemoryCredentials};
//! use tinyinference_hub::ports::{Http, HttpError, HubRequest, HubResponse, SystemClock};
//! use tinyinference_hub::{
//!     ConnectOptions, EndpointPolicy, Hub, ModelId, ProviderDraft, ScopeKey, Secret, TurnQuery,
//! };
//!
//! /// Your transport goes here. This one answers every request with a model list.
//! #[derive(Debug)]
//! struct Canned;
//!
//! #[async_trait::async_trait]
//! impl Http for Canned {
//!     async fn send(&self, request: HubRequest, _: &EndpointPolicy) -> Result<HubResponse, HttpError> {
//!         Ok(HubResponse::new(200, r#"{"data":[{"id":"gpt-x"}]}"#, request.url))
//!     }
//! }
//!
//! # futures::executor::block_on(async {
//! let hub = Hub::builder()
//!     .credentials(MemoryCredentials::new()) // wrap your keychain or vault
//!     .config(MemoryConfig::new())           // or a file, a database row
//!     .http(Canned)
//!     .clock(SystemClock)
//!     .policy(EndpointPolicy::desktop())     // `hosted()` in a multi-tenant server
//!     .build()?;
//!
//! let me = ScopeKey::new("user:local");
//! let draft = ProviderDraft::new("openai")
//!     .with_key(Secret::new("sk-not-a-real-key"))
//!     .with_model(ModelId::parse("gpt-x")?);
//! hub.connect(&me, draft, ConnectOptions::default()).await?; // checks the key, keeps the row
//!
//! let turn = hub.resolve_for_turn(&me, &TurnQuery::new()).await?; // no credential inside
//! let model = hub.chat_model(&me, &turn).await?;                  // Arc<dyn ChatModel<()>>
//! # let _: Arc<dyn tinyinference_hub::llm::ChatModel<()>> = model;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! # }).unwrap();
//! ```
//!
//! # What is here
//!
//! * [`Hub`]: `connect`, `add`, `edit`, `remove`, `set_enabled`, `set_key`,
//!   `clear_key`, `probe_draft`, `test`, `list_models`, `health`, `status`,
//!   `record_outcome`, `retest_down`, `set_default`, `pin_agent`,
//!   `set_workload_route`, `resolve_for_turn`, `chat_model`, `detect`,
//!   `local_status` (and `cli_readiness`, `oauth_*` behind features). Every
//!   change is *load, check the guards, save with the version loaded*, retried on
//!   a lost compare-and-swap; the state machine is spelled out on [`hub`].
//! * [`error`]: [`HubError`], the stable [`ReasonCode`] vocabulary, and
//!   [`classify`], which turns a vendor's status, headers and body into a
//!   [`ProviderFailure`] (a spend cap is `quota` and never retried, distinct from
//!   `rate_limited`).
//! * [`Secret`] and [`LogOnly`]: wrappers that redact themselves everywhere.
//! * [`ids`]: [`Slug`], [`ModelId`], [`KindId`], [`ScopeKey`] and the validators.
//! * [`taxonomy`], [`catalogue`], [`descriptor`]: every built-in kind (managed,
//!   26 cloud, 5 local, 2 CLI) as data, and [`LocalRuntime`], which reconciles
//!   the four local-runtime enums by conversion.
//! * [`policy`], [`endpoint`]: the SSRF policy and endpoint credential redaction.
//! * [`credential`]: the ordered credential chain (pasted key, account key,
//!   instance identity, session token, environment) that reports which source
//!   answered.
//! * [`catalog`], [`probe`], [`health`], [`kinds`]: tolerant listing parsers, the
//!   scope-partitioned cache, three-depth probing, folded health, and the kind
//!   drivers.
//! * [`route`], [`import`]: structured routes with both legacy string grammars,
//!   and pure readers that turn OpenCompany's and OpenHuman's stored shapes into
//!   a [`HubConfig`] plus a report of everything they could not carry over.
//! * [`client`]: the `ChatModel` the hub hands out, which resolves the credential
//!   on every call and feeds real turns back into health.
//! * [`detect`]: local-runtime and environment detection (never persisted).
//! * `testkit` (feature `testing`): `FakeClock`, `ScriptedHttp`, `MemoryPorts`,
//!   the contract suite, and the seeded `ScenarioRunner`.
//!
//! # Guarantees
//!
//! A credential is never on a record and never printed: [`Secret`] has no
//! `Serialize`, its `Debug` and `Display` redact, and [`ProviderRecord`] has no
//! credential field. Raw upstream error text is log-only and never reaches a
//! user-facing sentence. Only a rejected credential rolls back an add (a local
//! runtime also when nothing answers). A model list cached for one scope is never
//! served to another. The default choice is never silently changed by another
//! operation. A key change drops the provider's health and the scope's cached
//! catalogs.
//!
//! # Known limits
//!
//! * Turn traffic uses `tinyinference-llm`'s own transport, so the per-redirect
//!   policy and address pinning cover probes and catalogs, not chat turns. The
//!   resolved endpoint **is** re-checked against the policy at
//!   [`Hub::resolve_for_turn`].
//! * The managed provider's backend differs between OpenCompany and OpenHuman;
//!   the host supplies its endpoint and catalog shape ([`ManagedConfig`]).
//! * An ephemeral route ([`RouteTarget::Ephemeral`]) parses but is not resolved.
//! * Failover, budgets, cooldowns and rate-aware routing are deliberately absent:
//!   the hub exposes the signals ([`Hub::record_outcome`], health, [`Retry`]) a
//!   router consumes.
//!
//! # Features
//!
//! `default = []`.
//!
//! | Feature | Effect |
//! |---|---|
//! | `testing` | the no-socket simulation kit |
//! | `cli` | the `ProcessSpawner` port and `Hub::cli_readiness` |
//! | `oauth` | OAuth **types only**; every flow answers [`HubError::Unsupported`] |
//! | `http-reqwest` | the reference `Http`, `ports::ReqwestHttp` |
//! | `local-bridge` | `From`/`TryFrom` to `tinyinference-local`'s enums |

pub mod catalog;
pub mod catalogue;
#[cfg(feature = "cli")]
pub mod cli;
pub mod client;
pub mod config;
pub mod credential;
pub mod descriptor;
pub mod detect;
pub mod endpoint;
pub mod error;
pub mod health;
pub mod hub;
pub mod ids;
pub mod import;
pub mod kinds;
#[cfg(feature = "oauth")]
pub mod oauth;
mod ops;
pub mod policy;
pub mod ports;
pub mod probe;
pub mod route;
mod secret;
pub mod taxonomy;
#[cfg(any(test, feature = "testing"))]
pub mod testkit;

pub use config::{DefaultChoice, HubConfig, ModelChoice, ProviderDraft};
pub use descriptor::{Capabilities, ProviderDescriptor, ProviderRecord, Quirk};
pub use error::{
    CopyContext, HubError, InvalidInput, NotFound, Operation, PolicyViolation, PortName,
    ProviderFailure, ReasonCode, Result, Retry, Unresolved, UsedBy, classify, classify_for,
    classify_transport,
};
pub use hub::{
    Confirm, ConnectOptions, Hub, HubBuilder, HubPolicy, HubStatus, KeyState, ManagedConfig,
    Mutation, MutationStatus, ProviderPatch, ProviderStatus, ProviderView, Retested,
};
pub use ids::{AgentKey, KindId, ModelId, ScopeKey, Slug, WorkloadKey};
pub use import::{Imported, LossEntry, LossKind, LossReport};
pub use policy::{EndpointPolicy, EndpointRefusal, HeaderPolicy};
pub use route::{ProviderRoute, ResolvedTurn, ResolvedVia, RouteTarget, TurnQuery};
pub use secret::{LogOnly, Secret, SecretId};
pub use taxonomy::{
    AuthStyle, CatalogShape, CliKind, LocalRuntime, Protocol, ProviderGroup, TestDepth, Transport,
};

/// Re-exports of the llm types that appear in the hub's own signatures, so a
/// host needs one import. Deliberately not a wholesale re-export.
pub mod llm {
    pub use tinyinference_llm::ProviderKind;
    pub use tinyinference_llm::catalog::ModelInfo;
    pub use tinyinference_llm::model::ChatModel;
}
