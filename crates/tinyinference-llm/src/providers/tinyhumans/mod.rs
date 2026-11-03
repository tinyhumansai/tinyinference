//! TinyHumans' managed OpenAI-compatible inference gateway.
//!
//! The host supplies credential resolution, endpoint policy, and attribution
//! headers. This adapter owns only the generic wire details shared by callers:
//! dynamic bearer lookup, managed request extensions, usage projection, and
//! catalog discovery.

mod catalog;
mod reasoning;
mod usage;

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use sha2::{Digest, Sha256};
use tinyinference_core::sanitize;

use crate::model::{
    ChatModel, InputModality, InputSource, ModelProfile, ModelRequest, ModelResponse, ModelStream,
    ModelStreamItem, ProviderError,
};
use crate::providers::openai::{AuthStyle, OpenAiModel};
use crate::providers::{BearerSource, ProviderRequestOptions};
use crate::{Error, Result};

pub use catalog::{discovery_request, list_models};
pub use reasoning::{REASONING_OFF_METADATA_KEY, apply_reasoning, with_thread_id};
pub use usage::project_tinyhumans_envelope;

/// Resolves the current bearer credential for a managed inference call.
pub type BearerResolver =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<String>> + Send>> + Send + Sync>;

/// Which kind of credential a managed request authenticated with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyHumansCredentialKind {
    /// The host's signed-in app session token.
    Session,
    /// A long-lived API key.
    ApiKey,
}

/// A resolved bearer together with the kind of credential it is.
#[derive(Clone, Debug)]
pub struct TinyHumansCredential {
    /// The bearer sent with the request.
    pub bearer: String,
    /// What the bearer is, reported back on an authentication failure.
    pub kind: TinyHumansCredentialKind,
}

type KindedResolver = Arc<
    dyn Fn() -> Pin<
            Box<dyn Future<Output = Result<(String, Option<TinyHumansCredentialKind>)>> + Send>,
        > + Send
        + Sync,
>;

/// Opaque, non-secret identity for a bearer credential used by one request.
///
/// The fingerprint is a domain-separated SHA-256 digest. It supports comparing
/// the credential used by a failed request with a host's current bearer without
/// retaining or exposing the bearer itself.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct TinyHumansCredentialFingerprint([u8; 32]);

impl TinyHumansCredentialFingerprint {
    fn from_bearer(bearer: &str) -> Self {
        let mut digest = Sha256::new();
        digest.update(b"tinyinference/tinyhumans-credential-fingerprint/v1\0");
        digest.update(bearer.as_bytes());
        Self(digest.finalize().into())
    }

    /// Returns the canonical lower-case hexadecimal representation of this
    /// fingerprint. The bearer is not recoverable from this representation.
    #[must_use]
    pub fn to_hex(self) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut encoded = String::with_capacity(self.0.len() * 2);
        for byte in self.0 {
            encoded.push(char::from(HEX[usize::from(byte >> 4)]));
            encoded.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        encoded
    }

    /// Parses a canonical lower-case hexadecimal fingerprint representation.
    ///
    /// Returns `None` when `encoded` is not exactly 64 lower-case hex digits.
    #[must_use]
    pub fn from_hex(encoded: &str) -> Option<Self> {
        if encoded.len() != 64 {
            return None;
        }
        let mut bytes = [0_u8; 32];
        for (index, pair) in encoded.as_bytes().as_chunks::<2>().0.iter().enumerate() {
            let high = decode_lower_hex(pair[0])?;
            let low = decode_lower_hex(pair[1])?;
            bytes[index] = (high << 4) | low;
        }
        Some(Self(bytes))
    }

    /// Returns whether this fingerprint belongs to `bearer`.
    #[must_use]
    pub fn matches_bearer(self, bearer: &str) -> bool {
        let candidate = Self::from_bearer(bearer);
        // Compare the fixed-width digest without early exit.
        self.0
            .iter()
            .zip(candidate.0.iter())
            .fold(0_u8, |different, (left, right)| different | (left ^ right))
            == 0
    }
}

fn decode_lower_hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

impl std::fmt::Debug for TinyHumansCredentialFingerprint {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("TinyHumansCredentialFingerprint(<opaque>)")
    }
}

#[derive(Clone, Copy)]
struct SentCredential {
    kind: Option<TinyHumansCredentialKind>,
    fingerprint: TinyHumansCredentialFingerprint,
}

tokio::task_local! {
    /// Request-time credential identity so a failure is attributed to the
    /// credential actually sent, rather than whichever one resolves later.
    static SENT_CREDENTIAL: std::cell::Cell<Option<SentCredential>>;
}

/// Operation that received a provider authentication failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyHumansOperation {
    /// A unary model invocation.
    Invoke,
    /// A streaming model invocation, including in-band stream failures.
    Stream,
    /// The lightweight readiness probe.
    Readiness,
}

/// Stage at which a managed-model operation failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyHumansFailurePoint {
    /// A unary invocation failed before returning a response.
    Invoke,
    /// A stream could not be opened.
    StreamOpen,
    /// A failure was delivered inside an already-open stream.
    StreamItem,
    /// The lightweight readiness probe encountered an error.
    Readiness,
}

/// The HTTP-authentication classification for a managed gateway failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinyHumansAuthFailureKind {
    /// The gateway returned HTTP 401.
    Unauthorized,
    /// The gateway returned HTTP 403.
    Forbidden,
    /// Authentication failed without a matching HTTP 401 or 403 status.
    Other,
}

/// Typed provenance for an authentication failure observed by the managed model.
///
/// Values can only be created by [`TinyHumansModel`] after it recognizes an
/// authentication failure from the managed endpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TinyHumansAuthFailure {
    operation: TinyHumansOperation,
    kind: TinyHumansAuthFailureKind,
    credential: Option<TinyHumansCredentialKind>,
    credential_fingerprint: Option<TinyHumansCredentialFingerprint>,
}

impl TinyHumansAuthFailure {
    /// The managed inference operation that observed the failure.
    #[must_use]
    pub fn operation(self) -> TinyHumansOperation {
        self.operation
    }

    /// The HTTP-authentication category supplied by the managed endpoint.
    #[must_use]
    pub fn kind(self) -> TinyHumansAuthFailureKind {
        self.kind
    }

    /// The credential kind the failing request was sent with, when the model
    /// was built with [`TinyHumansModel::new_with_credential`]; `None` for a
    /// plain bearer resolver.
    #[must_use]
    pub fn credential(self) -> Option<TinyHumansCredentialKind> {
        self.credential
    }

    /// The opaque identity of the bearer sent with the failed request, when
    /// the request used a managed credential resolver.
    #[must_use]
    pub fn credential_fingerprint(self) -> Option<TinyHumansCredentialFingerprint> {
        self.credential_fingerprint
    }
}

/// Handles authentication failures from a managed endpoint.
pub type AuthFailureHandler = Arc<dyn Fn(&ProviderError, TinyHumansOperation) + Send + Sync>;

/// Handles authentication failures with explicit managed-endpoint provenance.
pub type ManagedAuthFailureHandler =
    Arc<dyn Fn(&ProviderError, TinyHumansAuthFailure) + Send + Sync>;

/// Observes model failures without changing their returned or streamed values.
pub type ErrorObserver = Arc<dyn Fn(&Error, TinyHumansFailurePoint) + Send + Sync>;

/// An OpenAI-compatible model for the TinyHumans managed gateway.
pub struct TinyHumansModel {
    model: OpenAiModel,
    base_url: String,
    default_model: String,
    cache_namespace: String,
    thread_id: Option<String>,
    auth_failure_handler: Option<AuthFailureHandler>,
    managed_auth_failure_handler: Option<ManagedAuthFailureHandler>,
    error_observer: Option<ErrorObserver>,
}

impl std::fmt::Debug for TinyHumansModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TinyHumansModel")
            .field("base_url", &sanitize::redact_url(&self.base_url))
            .field("default_model", &self.default_model)
            .field("thread_id", &self.thread_id)
            .field(
                "auth_failure_handler",
                &self.auth_failure_handler.as_ref().map(|_| "<fn>"),
            )
            .field(
                "managed_auth_failure_handler",
                &self.managed_auth_failure_handler.as_ref().map(|_| "<fn>"),
            )
            .field(
                "error_observer",
                &self.error_observer.as_ref().map(|_| "<fn>"),
            )
            .finish_non_exhaustive()
    }
}

impl TinyHumansModel {
    /// Creates a managed model using the supplied endpoint, model, and live
    /// bearer resolver.
    ///
    /// `base_url` is the backend root; `/openai/v1` is appended automatically.
    /// An empty constructor `model` uses `default_model`, which the host supplies
    /// so this crate does not own product defaults. A request with no model, or
    /// a blank model, uses the constructor's effective model; a nonblank request
    /// model remains an explicit override.
    ///
    /// # Errors
    /// Returns [`Error::Validation`] when the endpoint or default model is empty.
    pub fn new(
        base_url: impl AsRef<str>,
        model: impl AsRef<str>,
        default_model: impl AsRef<str>,
        resolver: impl Fn() -> Pin<Box<dyn Future<Output = Result<String>> + Send>>
        + Send
        + Sync
        + 'static,
    ) -> Result<Self> {
        Self::build(
            base_url,
            model,
            default_model,
            Arc::new(move || {
                let future = resolver();
                Box::pin(async move { future.await.map(|bearer| (bearer, None)) })
            }),
        )
    }

    /// Like [`Self::new`], but the resolver also reports which kind of
    /// credential each bearer is, and every [`TinyHumansAuthFailure`] carries
    /// the kind its own request was sent with.
    ///
    /// # Errors
    /// Returns [`Error::Validation`] when the endpoint or default model is empty.
    pub fn new_with_credential(
        base_url: impl AsRef<str>,
        model: impl AsRef<str>,
        default_model: impl AsRef<str>,
        resolver: impl Fn() -> Pin<Box<dyn Future<Output = Result<TinyHumansCredential>> + Send>>
        + Send
        + Sync
        + 'static,
    ) -> Result<Self> {
        Self::build(
            base_url,
            model,
            default_model,
            Arc::new(move || {
                let future = resolver();
                Box::pin(async move {
                    future
                        .await
                        .map(|credential| (credential.bearer, Some(credential.kind)))
                })
            }),
        )
    }

    fn build(
        base_url: impl AsRef<str>,
        model: impl AsRef<str>,
        default_model: impl AsRef<str>,
        resolver: KindedResolver,
    ) -> Result<Self> {
        let base = base_url.as_ref().trim().trim_end_matches('/');
        if base.is_empty() {
            return Err(Error::Validation(
                "TinyHumans base URL must not be empty".into(),
            ));
        }
        let fallback_model = default_model.as_ref().trim();
        if fallback_model.is_empty() {
            return Err(Error::Validation(
                "TinyHumans default model must not be empty".into(),
            ));
        }
        let endpoint = catalog::endpoint(base);
        let default_model = resolve_model(model.as_ref(), fallback_model);
        let source = Arc::new(ResolvedBearer(resolver));
        let wire = OpenAiModel::compatible_provider("tinyhumans", "", &endpoint, &default_model)
            .with_auth_style(AuthStyle::Bearer)
            .with_bearer_source(source)
            .with_native_tool_calling(true)
            .with_vision(true)
            .with_request_options(ProviderRequestOptions {
                http: Some(pooled_http_client()),
                ..ProviderRequestOptions::default()
            });
        Ok(Self {
            model: wire,
            base_url: endpoint,
            default_model,
            cache_namespace: "tinyhumans".to_owned(),
            thread_id: None,
            auth_failure_handler: None,
            managed_auth_failure_handler: None,
            error_observer: None,
        })
    }

    /// Replaces the shared default client, for host-specific TLS and proxy policy.
    #[must_use]
    pub fn with_http_client(mut self, client: reqwest::Client) -> Self {
        self.model = self.model.with_request_options(ProviderRequestOptions {
            http: Some(client),
            ..ProviderRequestOptions::default()
        });
        self
    }

    /// Adds headers such as host-selected attribution to every inference call.
    #[must_use]
    pub fn with_extra_headers(mut self, headers: &[(String, String)]) -> Self {
        for (name, value) in headers {
            self.model = self.model.with_header(name.clone(), value.clone());
        }
        self
    }

    /// Sets whether the endpoint should receive native tool declarations.
    #[must_use]
    pub fn with_native_tool_calling(mut self, enabled: bool) -> Self {
        self.model = self.model.with_native_tool_calling(enabled);
        self
    }

    /// Associates calls with this explicitly owned thread identifier.
    #[must_use]
    pub fn with_thread_id(mut self, thread_id: Option<impl AsRef<str>>) -> Self {
        self.thread_id = thread_id.and_then(|value| {
            let value = value.as_ref().trim();
            (!value.is_empty()).then(|| value.to_owned())
        });
        self
    }

    /// Sets the stable, credential-free host namespace used for response-cache
    /// identity. This does not change the wire provider label or auth provenance.
    /// Blank input leaves the current namespace unchanged.
    #[must_use]
    pub fn with_cache_namespace(mut self, namespace: impl AsRef<str>) -> Self {
        let namespace = namespace.as_ref().trim();
        if !namespace.is_empty() {
            self.cache_namespace = namespace.to_owned();
        }
        self
    }

    /// Installs a callback for structured authentication failures.
    #[must_use]
    pub fn with_auth_failure_handler(
        mut self,
        handler: impl Fn(&ProviderError, TinyHumansOperation) + Send + Sync + 'static,
    ) -> Self {
        self.auth_failure_handler = Some(Arc::new(handler));
        self
    }

    /// Installs a callback carrying typed provenance for managed auth failures.
    #[must_use]
    pub fn with_managed_auth_failure_handler(
        mut self,
        handler: impl Fn(&ProviderError, TinyHumansAuthFailure) + Send + Sync + 'static,
    ) -> Self {
        self.managed_auth_failure_handler = Some(Arc::new(handler));
        self
    }

    /// Installs an observer for failures at the model-call boundary.
    ///
    /// The observer receives the original normalized error and a typed failure
    /// point. It does not alter returned errors or stream items, and is called
    /// independently of the authentication-specific callbacks.
    #[must_use]
    pub fn with_error_observer(
        mut self,
        observer: impl Fn(&Error, TinyHumansFailurePoint) + Send + Sync + 'static,
    ) -> Self {
        self.error_observer = Some(Arc::new(observer));
        self
    }

    /// Performs a small completion to check whether an account has a provider
    /// configured. Only the definitive provider-not-configured response blocks;
    /// transport failures and unrelated provider errors fail open.
    pub async fn probe_readiness(&self) -> std::result::Result<(), String> {
        let request =
            ModelRequest::new(vec![crate::message::Message::user("ping")]).with_max_tokens(1);
        match tokio::time::timeout(
            Duration::from_secs(5),
            self.invoke_operation(request, TinyHumansOperation::Readiness),
        )
        .await
        {
            Err(_) => Ok(()),
            Ok(Ok(_)) => Ok(()),
            Ok(Err(Error::Provider(error))) if is_provider_not_configured_error(&error) => {
                Err(error.message.clone())
            }
            Ok(Err(_)) => Ok(()),
        }
    }

    /// The endpoint used for model calls and discovery.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn prepare_request(&self, mut request: ModelRequest) -> ModelRequest {
        request.model = Some(resolve_model(
            request.model.as_deref().unwrap_or(&self.default_model),
            &self.default_model,
        ));
        request = apply_reasoning(request);
        with_thread_id(request, self.thread_id.as_deref())
    }

    async fn invoke_operation(
        &self,
        request: ModelRequest,
        operation: TinyHumansOperation,
    ) -> Result<ModelResponse> {
        let (result, credential) = SENT_CREDENTIAL
            .scope(std::cell::Cell::new(None), async {
                let result = self.model.invoke(&(), self.prepare_request(request)).await;
                (result, SENT_CREDENTIAL.with(std::cell::Cell::get))
            })
            .await;
        match result {
            Ok(response) => Ok(project_tinyhumans_envelope(response)),
            Err(error) => {
                let point = match operation {
                    TinyHumansOperation::Readiness => TinyHumansFailurePoint::Readiness,
                    TinyHumansOperation::Invoke => TinyHumansFailurePoint::Invoke,
                    TinyHumansOperation::Stream => unreachable!("invoke operation is not stream"),
                };
                self.observe_error(&error, point);
                self.observe_auth_error(&error, operation, credential);
                Err(error)
            }
        }
    }

    fn observe_auth_error(
        &self,
        error: &Error,
        operation: TinyHumansOperation,
        credential: Option<SentCredential>,
    ) {
        if let Error::Provider(provider_error) = error
            && is_auth_failure(provider_error)
        {
            self.notify_auth_failure(provider_error, operation, credential);
        }
    }

    fn observe_error(&self, error: &Error, point: TinyHumansFailurePoint) {
        if let Some(observer) = &self.error_observer {
            observer(error, point);
        }
    }

    fn notify_auth_failure(
        &self,
        error: &ProviderError,
        operation: TinyHumansOperation,
        credential: Option<SentCredential>,
    ) {
        if let Some(handler) = &self.auth_failure_handler {
            handler(error, operation);
        }
        if let Some(handler) = &self.managed_auth_failure_handler {
            handler(
                error,
                TinyHumansAuthFailure {
                    operation,
                    kind: auth_failure_kind(error),
                    credential: credential.and_then(|sent| sent.kind),
                    credential_fingerprint: credential.map(|sent| sent.fingerprint),
                },
            );
        }
    }
}

#[async_trait]
impl ChatModel<()> for TinyHumansModel {
    fn profile(&self) -> Option<&ModelProfile> {
        <OpenAiModel as ChatModel<()>>::profile(&self.model)
    }

    fn tracks_physical_attempts(&self) -> bool {
        <OpenAiModel as ChatModel<()>>::tracks_physical_attempts(&self.model)
    }

    fn has_pre_invocation_admission_gate(&self) -> bool {
        <OpenAiModel as ChatModel<()>>::has_pre_invocation_admission_gate(&self.model)
    }

    fn supports_input(&self, modality: InputModality, mime: &str, source: InputSource) -> bool {
        <OpenAiModel as ChatModel<()>>::supports_input(&self.model, modality, mime, source)
    }

    fn cache_identity(&self) -> Option<String> {
        Some(format!(
            "{}:{}:{}",
            self.cache_namespace, self.base_url, self.default_model
        ))
    }

    async fn invoke(&self, _state: &(), request: ModelRequest) -> Result<ModelResponse> {
        self.invoke_operation(request, TinyHumansOperation::Invoke)
            .await
    }

    async fn stream(&self, _state: &(), request: ModelRequest) -> Result<ModelStream> {
        let (opened, credential) = SENT_CREDENTIAL
            .scope(std::cell::Cell::new(None), async {
                let opened = self.model.stream(&(), self.prepare_request(request)).await;
                (opened, SENT_CREDENTIAL.with(std::cell::Cell::get))
            })
            .await;
        let stream = match opened {
            Ok(stream) => stream,
            Err(error) => {
                self.observe_error(&error, TinyHumansFailurePoint::StreamOpen);
                self.observe_auth_error(&error, TinyHumansOperation::Stream, credential);
                return Err(error);
            }
        };
        let legacy_handler = self.auth_failure_handler.clone();
        let managed_handler = self.managed_auth_failure_handler.clone();
        let error_observer = self.error_observer.clone();
        Ok(stream.map_items(move |item| match item {
            ModelStreamItem::Completed(response) => {
                ModelStreamItem::Completed(project_tinyhumans_envelope(response))
            }
            ModelStreamItem::ProviderFailed(error) => {
                if let Some(observer) = &error_observer {
                    observer(
                        &Error::Provider(Box::new(error.clone())),
                        TinyHumansFailurePoint::StreamItem,
                    );
                }
                if is_auth_failure(&error) {
                    if let Some(handler) = &legacy_handler {
                        handler(&error, TinyHumansOperation::Stream);
                    }
                    if let Some(handler) = &managed_handler {
                        handler(
                            &error,
                            TinyHumansAuthFailure {
                                operation: TinyHumansOperation::Stream,
                                kind: auth_failure_kind(&error),
                                credential: credential.and_then(|sent| sent.kind),
                                credential_fingerprint: credential.map(|sent| sent.fingerprint),
                            },
                        );
                    }
                }
                ModelStreamItem::ProviderFailed(error)
            }
            other => other,
        }))
    }
}

fn is_auth_failure(error: &ProviderError) -> bool {
    let code = error
        .code
        .as_deref()
        .unwrap_or_default()
        .to_ascii_lowercase();
    let message = error.message.to_ascii_lowercase();
    error.status == Some(401)
        || error.status == Some(403)
        || matches!(
            code.as_str(),
            "unauthorized" | "authentication_error" | "invalid_api_key"
        )
        || message.contains("unauthorized")
        || message.contains("invalid api key")
}

fn auth_failure_kind(error: &ProviderError) -> TinyHumansAuthFailureKind {
    match error.status {
        Some(401) => TinyHumansAuthFailureKind::Unauthorized,
        Some(403) => TinyHumansAuthFailureKind::Forbidden,
        _ => TinyHumansAuthFailureKind::Other,
    }
}

fn is_provider_not_configured_error(error: &ProviderError) -> bool {
    if error.status != Some(400) {
        return false;
    }
    let message = error.message.to_ascii_lowercase();
    let bad_request = error
        .code
        .as_deref()
        .is_some_and(|code| code.eq_ignore_ascii_case("BAD_REQUEST"));
    message.contains("api key not configured for provider")
        || (bad_request && message.contains("not configured for provider"))
}

fn resolve_model(model: &str, default_model: &str) -> String {
    let model = model.trim();
    if model.is_empty() {
        default_model.trim().to_owned()
    } else {
        model.to_owned()
    }
}

struct ResolvedBearer(KindedResolver);

#[async_trait]
impl BearerSource for ResolvedBearer {
    async fn current(&self) -> Result<Option<String>> {
        let (bearer, kind) = (self.0)().await?;
        // Outside a dispatch scope (e.g. the readiness probe) there is nothing
        // to attribute, so a missing task-local is fine.
        let sent = SentCredential {
            kind,
            fingerprint: TinyHumansCredentialFingerprint::from_bearer(&bearer),
        };
        let _ = SENT_CREDENTIAL.try_with(|credential| credential.set(Some(sent)));
        Ok(Some(bearer))
    }
}

fn pooled_http_client() -> reqwest::Client {
    static CLIENT: OnceLock<reqwest::Client> = OnceLock::new();
    CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(30))
                .pool_idle_timeout(Duration::from_secs(50))
                .tcp_keepalive(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new())
        })
        .clone()
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
