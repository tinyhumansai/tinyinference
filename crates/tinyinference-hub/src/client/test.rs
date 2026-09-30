//! Tests for the chat model the hub hands out.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;
use tinyinference_llm::MockModel;
use tinyinference_llm::error::{Error, Result as LlmResult};
use tinyinference_llm::model::{
    ChatModel, ModelRequest, ModelResponse, ModelStream, ProviderError,
};

use super::*;
use crate::config::ProviderDraft;
use crate::credential::{CredentialOrigin, TokenSourceAdapter};
use crate::error::{HubError, ReasonCode};
use crate::health::ProviderHealth;
use crate::hub::fixtures::{Bed, KEY, model, slug};
use crate::hub::{ManagedConfig, ProviderPatch};
use crate::ports::{HealthStore, PortError, TokenSource};
use crate::route::{ProviderRoute, RouteTarget, TurnQuery};
use crate::secret::Secret;
use crate::taxonomy::{AuthStyle, CliKind, ProviderGroup};

/// One scripted answer of a fake model.
type Answer = Box<dyn Fn() -> LlmResult<ModelResponse> + Send + Sync>;

struct Fake {
    answers: Mutex<VecDeque<Answer>>,
    calls: AtomicUsize,
}

impl Fake {
    fn new(answers: Vec<Answer>) -> Arc<Self> {
        Arc::new(Self {
            answers: Mutex::new(answers.into()),
            calls: AtomicUsize::new(0),
        })
    }
}

#[async_trait]
impl ChatModel<()> for Fake {
    async fn invoke(&self, _: &(), _: ModelRequest) -> LlmResult<ModelResponse> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let next = self.answers.lock().unwrap().pop_front();
        match next {
            Some(answer) => answer(),
            None => Ok(MockModel::text_response("default")),
        }
    }

    async fn stream(&self, state: &(), request: ModelRequest) -> LlmResult<ModelStream> {
        // The trait default: an invoke replayed as a stream.
        let response = self.invoke(state, request).await?;
        Ok(ModelStream::new(Box::pin(futures::stream::iter(vec![
            tinyinference_llm::model::ModelStreamItem::Completed(response),
        ]))))
    }
}

fn ok(text: &'static str) -> Answer {
    Box::new(move || Ok(MockModel::text_response(text)))
}

fn provider_error(status: u16, code: &'static str, message: &'static str) -> Answer {
    Box::new(move || {
        Err(Error::Provider(Box::new(ProviderError {
            provider: "openai".into(),
            status: Some(status),
            code: Some(code.into()),
            message: message.into(),
            ..ProviderError::default()
        })))
    })
}

/// What a build saw: the endpoint, the extra headers, the Responses flag.
type SpecSeen = (String, Vec<(String, String)>, bool);

/// A factory that hands out the same fake and remembers which key each build saw.
#[derive(Debug)]
struct Recording {
    fake: Arc<FakeHandle>,
    keys: Mutex<Vec<Option<String>>>,
    specs: Mutex<Vec<SpecSeen>>,
}

struct FakeHandle(Arc<Fake>);

impl std::fmt::Debug for FakeHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Fake")
    }
}

#[async_trait]
impl ChatModel<()> for FakeHandle {
    async fn invoke(&self, state: &(), request: ModelRequest) -> LlmResult<ModelResponse> {
        self.0.invoke(state, request).await
    }

    async fn stream(&self, state: &(), request: ModelRequest) -> LlmResult<ModelStream> {
        self.0.stream(state, request).await
    }
}

impl ModelFactory for Recording {
    fn build(&self, spec: &ModelSpec<'_>) -> Result<Arc<dyn ChatModel<()>>, HubError> {
        self.keys.lock().unwrap().push(spec.key.map(str::to_string));
        self.specs.lock().unwrap().push((
            spec.turn.base_url.clone(),
            spec.extra_headers.to_vec(),
            spec.responses_api,
        ));
        Ok(Arc::new(FakeHandle(self.fake.0.clone())))
    }
}

fn recording(answers: Vec<Answer>) -> (Arc<Recording>, Arc<Fake>) {
    let fake = Fake::new(answers);
    (
        Arc::new(Recording {
            fake: Arc::new(FakeHandle(fake.clone())),
            keys: Mutex::new(Vec::new()),
            specs: Mutex::new(Vec::new()),
        }),
        fake,
    )
}

async fn openai_bed(factory: Arc<dyn ModelFactory>) -> (Bed, crate::route::ResolvedTurn) {
    let bed = Bed::with(|b| b.model_factory(factory));
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    (bed, turn)
}

// ---- the default factory -----------------------------------------------------------------

fn spec_turn(
    kind: &str,
    group: ProviderGroup,
    protocol: Protocol,
    base: &str,
    auth: AuthStyle,
) -> crate::route::ResolvedTurn {
    crate::route::ResolvedTurn {
        slug: slug(kind),
        kind: kind.into(),
        group,
        base_url: base.into(),
        model: Some(model("m")),
        protocol,
        auth,
        via: crate::route::ResolvedVia::Default,
        origin: None,
        temperature: crate::route::Temperature::new(0.2),
        cli: None,
    }
}

#[test]
fn client_builds_for_each_protocol() {
    let factory = LlmModelFactory::new();
    let build = |turn: &crate::route::ResolvedTurn| {
        factory.build(&ModelSpec {
            turn,
            key: Some(KEY),
            extra_headers: &[("x-title".into(), "t".into())],
            responses_api: false,
        })
    };
    for turn in [
        spec_turn(
            "openai",
            ProviderGroup::Cloud,
            Protocol::OpenAiChat,
            "https://api.openai.com/v1",
            AuthStyle::Bearer,
        ),
        spec_turn(
            "tinyhumans",
            ProviderGroup::Managed,
            Protocol::OpenAiChat,
            "https://api.tinyhumans.test/x",
            AuthStyle::SessionJwt,
        ),
        spec_turn(
            "custom",
            ProviderGroup::Custom,
            Protocol::OpenAiChat,
            "https://gw.test/v1",
            AuthStyle::Custom("api-key".into()),
        ),
        spec_turn(
            "anthropic",
            ProviderGroup::Cloud,
            Protocol::AnthropicMessages,
            "https://api.anthropic.com/v1",
            AuthStyle::Anthropic,
        ),
        // Anthropic on a proxy is spoken to as an OpenAI-compatible server.
        spec_turn(
            "anthropic",
            ProviderGroup::Cloud,
            Protocol::AnthropicMessages,
            "https://proxy.acme.test/v1",
            AuthStyle::Anthropic,
        ),
        spec_turn(
            "ollama",
            ProviderGroup::Local,
            Protocol::OpenAiChat,
            "http://localhost:11434/v1",
            AuthStyle::None,
        ),
        spec_turn(
            "mistral",
            ProviderGroup::Cloud,
            Protocol::OpenAiChat,
            "https://api.mistral.ai/v1",
            AuthStyle::XApiKey,
        ),
    ] {
        let model = build(&turn).unwrap_or_else(|e| panic!("{}: {e:?}", turn.kind));
        drop(model);
    }
    let responses = ModelSpec {
        turn: &spec_turn(
            "openai",
            ProviderGroup::Cloud,
            Protocol::OpenAiResponses,
            "https://api.openai.com/v1",
            AuthStyle::Bearer,
        ),
        key: None,
        extra_headers: &[],
        responses_api: true,
    };
    assert!(
        factory.build(&responses).is_ok(),
        "a keyless build is allowed; the call fails, not the build"
    );
    assert!(format!("{responses:?}").contains("responses_api"));
    let cli = spec_turn(
        "claude-code",
        ProviderGroup::Cli,
        Protocol::CliStream,
        "",
        AuthStyle::None,
    );
    assert!(matches!(build(&cli), Err(HubError::Unsupported { .. })));
    let mut modelless = spec_turn(
        "openai",
        ProviderGroup::Cloud,
        Protocol::OpenAiChat,
        "https://api.openai.com/v1",
        AuthStyle::Bearer,
    );
    modelless.model = None;
    assert!(matches!(
        build(&modelless),
        Err(HubError::Unsupported { .. })
    ));
    assert!(
        !format!(
            "{:?}",
            ModelSpec {
                turn: &modelless,
                key: Some(KEY),
                extra_headers: &[],
                responses_api: false
            }
        )
        .contains(KEY)
    );
    assert_eq!(format!("{factory:?}"), "LlmModelFactory");
}

// ---- the hub model -----------------------------------------------------------------------

#[tokio::test]
async fn client_a_success_is_built_once_reported_as_health_and_never_leaks_the_key() {
    let (factory, fake) = recording(vec![ok("one"), ok("two")]);
    let (bed, turn) = openai_bed(factory.clone()).await;
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    assert!(chat.profile().is_none());
    let identity = chat.cache_identity().unwrap();
    assert_eq!(identity, "hub:openai:openai:gpt-x");
    assert!(!identity.contains(KEY));
    let first = chat.invoke(&(), ModelRequest::default()).await.unwrap();
    assert_eq!(first.text(), "one");
    chat.invoke(&(), ModelRequest::default()).await.unwrap();
    assert_eq!(fake.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        factory.keys.lock().unwrap().as_slice(),
        [Some(KEY.to_string())],
        "built once while the key is unchanged"
    );
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("openai"))
            .await
            .unwrap()
            .health,
        ProviderHealth::Ok,
        "the turn fed health (D8)"
    );
    // The spec carried the OpenAI Responses flag for OpenAI's own host.
    assert!(factory.specs.lock().unwrap()[0].2);
}

#[tokio::test]
async fn sim_key_rotation_next_call_the_next_turn_uses_the_new_key() {
    let (factory, _fake) = recording(vec![]);
    let (bed, turn) = openai_bed(factory.clone()).await;
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    chat.invoke(&(), ModelRequest::default()).await.unwrap();
    bed.hub
        .edit(
            &bed.scope,
            &slug("openai"),
            ProviderPatch::new().key(Secret::new("sk-rotated")),
        )
        .await
        .unwrap();
    chat.invoke(&(), ModelRequest::default()).await.unwrap();
    chat.invoke(&(), ModelRequest::default()).await.unwrap();
    assert_eq!(
        factory.keys.lock().unwrap().as_slice(),
        [Some(KEY.to_string()), Some("sk-rotated".to_string())],
        "rebuilt exactly when the credential changed"
    );
}

#[tokio::test]
async fn client_a_rejected_key_is_reported_dropped_and_recorded() {
    let (factory, _fake) = recording(vec![provider_error(
        401,
        "invalid_api_key",
        "Incorrect API key provided",
    )]);
    let (bed, turn) = openai_bed(factory.clone()).await;
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    let error = chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    assert!(matches!(error, Error::Provider(_)));
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("openai"))
            .await
            .unwrap()
            .health,
        ProviderHealth::Down(ReasonCode::Auth)
    );
    // The client that held the rejected key was dropped: the next call rebuilds.
    chat.invoke(&(), ModelRequest::default()).await.unwrap();
    assert_eq!(factory.keys.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn client_only_provider_failures_are_health_and_a_bad_request_is_not() {
    let (factory, _fake) = recording(vec![
        Box::new(|| Err(Error::Validation("messages must not be empty".into()))),
        Box::new(|| Err(Error::Unsupported("nope".into()))),
        provider_error(503, "overloaded", "The server is overloaded"),
        Box::new(|| Err(Error::Model("connection reset by peer".into()))),
    ]);
    let (bed, turn) = openai_bed(factory).await;
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    for _ in 0..2 {
        chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    }
    assert_eq!(
        bed.ports
            .health
            .get(&bed.scope, &slug("openai"))
            .await
            .unwrap(),
        None,
        "a caller bug says nothing about the provider"
    );
    chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    let snapshot = bed.hub.health(&bed.scope, &slug("openai")).await.unwrap();
    assert_eq!(snapshot.snapshot.consecutive_failures, 2);
}

#[derive(Debug)]
struct Rotating {
    token: Mutex<Option<String>>,
    invalidated: AtomicUsize,
}

#[async_trait]
impl TokenSource for Rotating {
    async fn token(&self, _: &crate::ids::ScopeKey) -> Result<Option<Secret>, PortError> {
        Ok(self.token.lock().unwrap().clone().map(Secret::new))
    }

    fn invalidate(&self, _: &crate::ids::ScopeKey) {
        self.invalidated.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn client_managed_signed_out_is_typed_and_a_rejected_token_reaches_its_source() {
    let tokens = Arc::new(Rotating {
        token: Mutex::new(None),
        invalidated: AtomicUsize::new(0),
    });
    let (factory, _fake) = recording(vec![provider_error(401, "unauthorized", "invalid token")]);
    let bed = Bed::with(|b| {
        b.model_factory(factory).managed(
            ManagedConfig::new("https://api.tinyhumans.test/x").source(TokenSourceAdapter::new(
                tokens.clone(),
                CredentialOrigin::SessionJwt,
            )),
        )
    });
    let turn = crate::route::ResolvedTurn {
        slug: slug("tinyhumans"),
        kind: "tinyhumans".into(),
        group: ProviderGroup::Managed,
        base_url: "https://api.tinyhumans.test/x".into(),
        model: Some(model("m")),
        protocol: Protocol::OpenAiChat,
        auth: AuthStyle::Bearer,
        via: crate::route::ResolvedVia::Default,
        origin: None,
        temperature: None,
        cli: None,
    };
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    let error = chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    let Error::Provider(provider) = error else {
        panic!("{error:?}")
    };
    assert_eq!(provider.code.as_deref(), Some("signed_out"));
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("tinyhumans"))
            .await
            .unwrap()
            .health,
        ProviderHealth::SignedOut
    );
    *tokens.token.lock().unwrap() = Some("jwt".into());
    chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    assert_eq!(tokens.invalidated.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn client_a_stale_rejection_does_not_refresh_a_token_rotated_meanwhile() {
    let tokens = Arc::new(Rotating {
        token: Mutex::new(Some("jwt".into())),
        invalidated: AtomicUsize::new(0),
    });
    let (factory, _fake) = recording(vec![]);
    let bed = Bed::with(|b| {
        b.model_factory(factory).managed(
            ManagedConfig::new("https://api.tinyhumans.test/x").source(TokenSourceAdapter::new(
                tokens.clone(),
                CredentialOrigin::SessionJwt,
            )),
        )
    });
    let turn = crate::route::ResolvedTurn {
        slug: slug("tinyhumans"),
        kind: "tinyhumans".into(),
        group: ProviderGroup::Managed,
        base_url: "https://api.tinyhumans.test/x".into(),
        model: Some(model("m")),
        protocol: Protocol::OpenAiChat,
        auth: AuthStyle::Bearer,
        via: crate::route::ResolvedVia::Default,
        origin: None,
        temperature: None,
        cli: None,
    };
    let chat = super::model::HubModel::new(bed.hub.clone(), bed.scope.clone(), turn);
    let rejected = Error::Provider(Box::new(ProviderError {
        provider: "tinyhumans".into(),
        status: Some(401),
        message: "invalid token".into(),
        ..ProviderError::default()
    }));
    let stale = bed.hub.inner.health.epoch(&bed.scope, &slug("tinyhumans"));
    // The credential changes (which bumps the epoch) while the request is out.
    bed.hub.forget_health(&bed.scope, &slug("tinyhumans")).await;
    let origin = CredentialOrigin::SessionJwt;
    let jwt = Secret::new("jwt").id();
    chat.observe_err_for_test(&rejected, stale, Some((&origin, jwt)))
        .await;
    assert_eq!(
        tokens.invalidated.load(Ordering::SeqCst),
        0,
        "a stale 401 refreshes nothing"
    );
    let now = bed.hub.inner.health.epoch(&bed.scope, &slug("tinyhumans"));
    chat.observe_err_for_test(&rejected, now, Some((&origin, jwt)))
        .await;
    assert_eq!(tokens.invalidated.load(Ordering::SeqCst), 1);
}

/// A host token source that honours token identity: it refreshes only when the
/// rejected token is the one it would hand out now.
#[derive(Debug)]
struct Identifying {
    generation: AtomicUsize,
    refreshes: AtomicUsize,
    /// Calls to the plain (unattributed) `invalidate`.
    plain: AtomicUsize,
}

impl Identifying {
    fn current(&self) -> String {
        format!("jwt-{}", self.generation.load(Ordering::SeqCst))
    }
}

#[async_trait]
impl TokenSource for Identifying {
    async fn token(&self, _: &crate::ids::ScopeKey) -> Result<Option<Secret>, PortError> {
        Ok(Some(Secret::new(self.current())))
    }

    fn invalidate(&self, _: &crate::ids::ScopeKey) {
        self.plain.fetch_add(1, Ordering::SeqCst);
    }

    fn invalidate_rejected(&self, _: &crate::ids::ScopeKey, rejected: crate::secret::SecretId) {
        if Secret::new(self.current()).id() == rejected {
            self.refreshes.fetch_add(1, Ordering::SeqCst);
            self.generation.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[tokio::test]
async fn client_two_rejections_of_one_token_refresh_a_rotating_source_once() {
    // Finding 3.3: `invalidate` had no token identity, so the second of two
    // requests that used token A and both got a 401 discarded the fresh token B.
    let tokens = Arc::new(Identifying {
        generation: AtomicUsize::new(0),
        refreshes: AtomicUsize::new(0),
        plain: AtomicUsize::new(0),
    });
    let (factory, _fake) = recording(vec![]);
    let bed = Bed::with(|b| {
        b.model_factory(factory).managed(
            ManagedConfig::new("https://api.tinyhumans.test/x").source(TokenSourceAdapter::new(
                tokens.clone(),
                CredentialOrigin::SessionJwt,
            )),
        )
    });
    let turn = crate::route::ResolvedTurn {
        slug: slug("tinyhumans"),
        kind: "tinyhumans".into(),
        group: ProviderGroup::Managed,
        base_url: "https://api.tinyhumans.test/x".into(),
        model: Some(model("m")),
        protocol: Protocol::OpenAiChat,
        auth: AuthStyle::Bearer,
        via: crate::route::ResolvedVia::Default,
        origin: None,
        temperature: None,
        cli: None,
    };
    let chat = super::model::HubModel::new(bed.hub.clone(), bed.scope.clone(), turn);
    let rejected = Error::Provider(Box::new(ProviderError {
        provider: "tinyhumans".into(),
        status: Some(401),
        message: "invalid token".into(),
        ..ProviderError::default()
    }));
    let origin = CredentialOrigin::SessionJwt;
    let token_a = Secret::new("jwt-0").id();
    let epoch = bed.hub.inner.health.epoch(&bed.scope, &slug("tinyhumans"));
    chat.observe_err_for_test(&rejected, epoch, Some((&origin, token_a)))
        .await;
    chat.observe_err_for_test(&rejected, epoch, Some((&origin, token_a)))
        .await;
    assert_eq!(
        tokens.refreshes.load(Ordering::SeqCst),
        1,
        "the second rejection of the same stale token is ignored"
    );
    assert_eq!(tokens.current(), "jwt-1");
    // A rejection of the fresh token does refresh it.
    chat.observe_err_for_test(&rejected, epoch, Some((&origin, Secret::new("jwt-1").id())))
        .await;
    assert_eq!(tokens.refreshes.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn client_a_source_that_ignores_identity_still_gets_the_old_invalidate() {
    let tokens = Arc::new(Rotating {
        token: Mutex::new(Some("jwt".into())),
        invalidated: AtomicUsize::new(0),
    });
    let adapter = TokenSourceAdapter::new(tokens.clone(), CredentialOrigin::SessionJwt);
    crate::credential::CredentialSource::invalidate_rejected(
        &adapter,
        &crate::ids::ScopeKey::new("s"),
        Secret::new("anything").id(),
    );
    assert_eq!(tokens.invalidated.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn client_an_unreadable_credential_store_is_a_retryable_error_never_a_keyless_call() {
    let (factory, _fake) = recording(vec![]);
    let (bed, turn) = openai_bed(factory.clone()).await;
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    bed.ports
        .credentials
        .inject(crate::ports::memory::CredentialFault::Read);
    let error = chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    let Error::Provider(provider) = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        (provider.code.as_deref(), provider.retryable),
        (Some("store_unreadable"), true)
    );
    assert!(
        factory.keys.lock().unwrap().is_empty(),
        "nothing was built or sent"
    );
}

#[tokio::test]
async fn client_a_model_kept_after_its_key_was_cleared_fails_closed() {
    let (factory, _fake) = recording(vec![]);
    let (bed, turn) = openai_bed(factory.clone()).await;
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    chat.invoke(&(), ModelRequest::default()).await.unwrap();
    bed.hub
        .clear_key(&bed.scope, &slug("openai"), crate::hub::Confirm::in_use())
        .await
        .unwrap();
    let built = factory.keys.lock().unwrap().len();
    let error = chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    let Error::Provider(provider) = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        (provider.code.as_deref(), provider.retryable),
        (Some("no_key"), false)
    );
    assert_eq!(
        factory.keys.lock().unwrap().len(),
        built,
        "nothing was built, so nothing could be sent"
    );
    // A kind that takes no key is unaffected.
    let bed = Bed::with(|b| b.model_factory(recording(vec![]).0));
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("ollama").with_model(model("llama3")),
        )
        .await
        .unwrap();
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    chat.invoke(&(), ModelRequest::default()).await.unwrap();
}

#[tokio::test]
async fn client_the_usage_meta_key_is_mirrored_both_ways() {
    let legacy: Answer = Box::new(|| {
        let mut response = MockModel::text_response("x");
        response.raw =
            Some(json!({"openhuman_usage_meta": {"charged_amount_usd": 0.5}, "id": "r1"}));
        Ok(response)
    });
    let neutral: Answer = Box::new(|| {
        let mut response = MockModel::text_response("x");
        response.raw = Some(json!({"usage_meta": {"charged_amount_usd": 0.25}}));
        Ok(response)
    });
    let both: Answer = Box::new(|| {
        let mut response = MockModel::text_response("x");
        response.raw = Some(json!({"usage_meta": {"a": 1}, "openhuman_usage_meta": {"a": 2}}));
        Ok(response)
    });
    let plain: Answer = Box::new(|| {
        let mut response = MockModel::text_response("x");
        response.raw = Some(json!("not an object"));
        Ok(response)
    });
    let (factory, _fake) = recording(vec![legacy, neutral, both, plain, ok("no raw")]);
    let (bed, turn) = openai_bed(factory).await;
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    let raw = |r: ModelResponse| r.raw.unwrap();
    let one = raw(chat.invoke(&(), ModelRequest::default()).await.unwrap());
    assert_eq!(one["usage_meta"], one["openhuman_usage_meta"]);
    assert_eq!(one["usage_meta"]["charged_amount_usd"], 0.5);
    assert_eq!(one["id"], "r1", "the rest of the payload is untouched");
    let two = raw(chat.invoke(&(), ModelRequest::default()).await.unwrap());
    assert_eq!(two["openhuman_usage_meta"]["charged_amount_usd"], 0.25);
    let three = raw(chat.invoke(&(), ModelRequest::default()).await.unwrap());
    assert_eq!(
        (
            three["usage_meta"]["a"].clone(),
            three["openhuman_usage_meta"]["a"].clone()
        ),
        (json!(1), json!(2)),
        "neither is overwritten"
    );
    assert_eq!(
        raw(chat.invoke(&(), ModelRequest::default()).await.unwrap()),
        json!("not an object")
    );
    assert!(
        chat.invoke(&(), ModelRequest::default())
            .await
            .unwrap()
            .raw
            .is_none()
    );
}

#[tokio::test]
async fn client_a_stream_that_starts_is_a_turn_that_worked_and_one_that_cannot_is_a_failure() {
    let (factory, _fake) = recording(vec![
        ok("s"),
        provider_error(429, "rate_limit_exceeded", "slow down"),
    ]);
    let (bed, turn) = openai_bed(factory).await;
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    let _stream = chat.stream(&(), ModelRequest::default()).await.unwrap();
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("openai"))
            .await
            .unwrap()
            .health,
        ProviderHealth::Ok
    );
    assert!(chat.stream(&(), ModelRequest::default()).await.is_err());
    let snapshot = bed.hub.health(&bed.scope, &slug("openai")).await.unwrap();
    assert_eq!(snapshot.snapshot.consecutive_failures, 1);
}

#[tokio::test]
async fn client_chat_model_refuses_what_it_cannot_serve() {
    let bed = Bed::new();
    let mut turn = spec_turn(
        "openai",
        ProviderGroup::Cloud,
        Protocol::OpenAiChat,
        "https://api.openai.com/v1",
        AuthStyle::Bearer,
    );
    turn.model = None;
    assert!(matches!(
        bed.hub.chat_model(&bed.scope, &turn).await,
        Err(HubError::Unresolved(crate::error::Unresolved::NoModel(_)))
    ));
    let cli = TurnQuery::new().with_override(ProviderRoute::new(RouteTarget::Cli(CliKind::Codex)));
    let resolved = bed.hub.resolve_for_turn(&bed.scope, &cli).await.unwrap();
    assert!(matches!(
        bed.hub.chat_model(&bed.scope, &resolved).await,
        Err(HubError::Unsupported { .. })
    ));
}

#[tokio::test]
async fn client_the_default_factory_builds_a_real_model_without_sending_anything() {
    let bed = Bed::new();
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai")
                .with_key(Secret::new(KEY))
                .with_model(model("gpt-x")),
        )
        .await
        .unwrap();
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    assert_eq!(chat.cache_identity().unwrap(), "hub:openai:openai:gpt-x");
    assert_eq!(bed.ports.http.request_count(), 0);
}

#[test]
fn client_product_and_kind_headers_reach_only_first_party_hosts() {
    let bed = Bed::with(|b| {
        b.managed(
            ManagedConfig::new("https://api.tinyhumans.ai/x")
                .product_header("x-sdk-name", "opencompany"),
        )
    });
    let managed = spec_turn(
        "tinyhumans",
        ProviderGroup::Managed,
        Protocol::OpenAiChat,
        "https://api.tinyhumans.ai/x",
        AuthStyle::Bearer,
    );
    let third_party = spec_turn(
        "openrouter",
        ProviderGroup::Cloud,
        Protocol::OpenAiChat,
        "https://openrouter.ai/api/v1",
        AuthStyle::Bearer,
    );
    let first = bed.hub.request_headers(&managed);
    assert!(first.iter().any(|(n, _)| n == "x-sdk-name"));
    let other = bed.hub.request_headers(&third_party);
    assert!(
        !other.iter().any(|(n, _)| n == "x-sdk-name"),
        "the product header never leaves first-party hosts"
    );
    assert!(
        other
            .iter()
            .any(|(n, _)| n.eq_ignore_ascii_case("http-referer")),
        "OpenRouter's attribution headers ride"
    );
    assert!(bed.hub.serves_responses_api(&spec_turn(
        "openai",
        ProviderGroup::Cloud,
        Protocol::OpenAiChat,
        "https://api.openai.com/v1",
        AuthStyle::Bearer
    )));
    assert!(!bed.hub.serves_responses_api(&spec_turn(
        "openai",
        ProviderGroup::Cloud,
        Protocol::OpenAiChat,
        "https://proxy.test/v1",
        AuthStyle::Bearer
    )));
    assert!(!bed.hub.serves_responses_api(&spec_turn(
        "groq",
        ProviderGroup::Cloud,
        Protocol::OpenAiChat,
        "https://api.openai.com/v1",
        AuthStyle::Bearer
    )));
}

#[tokio::test]
async fn client_a_kept_model_fails_closed_after_its_provider_moved_was_disabled_or_removed() {
    let (factory, _fake) = recording(vec![]);
    let bed = Bed::with(|b| b.model_factory(factory.clone()));
    bed.hub
        .add(
            &bed.scope,
            crate::config::ProviderDraft::new("custom")
                .with_label("Acme")
                .with_base_url("https://llm.acme.test/v1")
                .with_key(Secret::new("sk-not-a-real-key"))
                .with_model(model("m")),
        )
        .await
        .unwrap();
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    chat.invoke(&(), ModelRequest::default()).await.unwrap();
    // The operator moves the provider to another origin with a new key.
    bed.hub
        .edit(
            &bed.scope,
            &slug("acme"),
            crate::hub::ProviderPatch::new()
                .base_url("https://other.test/v1")
                .key(Secret::new("sk-second-fake")),
        )
        .await
        .unwrap();
    let built = factory.keys.lock().unwrap().len();
    let error = chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    let Error::Provider(provider) = error else {
        panic!("{error:?}")
    };
    assert_eq!(provider.code.as_deref(), Some("stale_route"));
    assert_eq!(
        factory.keys.lock().unwrap().len(),
        built,
        "nothing was built or sent"
    );
    // A model resolved afresh works, and a removal stops it too.
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    let fresh = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    fresh.invoke(&(), ModelRequest::default()).await.unwrap();
    bed.hub
        .remove(&bed.scope, &slug("acme"), crate::hub::Confirm::in_use())
        .await
        .unwrap();
    let error = fresh
        .invoke(&(), ModelRequest::default())
        .await
        .unwrap_err();
    assert!(matches!(error, Error::Provider(_)));
}

#[tokio::test]
async fn client_a_signed_out_mark_does_not_outlive_a_sign_in_the_hub_was_not_told_about() {
    // Finding 4.8: the host signs a user in through its own token source, so no
    // hub operation bumps the epoch. The health written while signed out must
    // stop reading "signed out" once a credential answers.
    let tokens = Arc::new(Rotating {
        token: Mutex::new(None),
        invalidated: AtomicUsize::new(0),
    });
    let (factory, _fake) = recording(vec![]);
    let bed = Bed::with(|b| {
        b.model_factory(factory).managed(
            ManagedConfig::new("https://api.tinyhumans.test/x").source(TokenSourceAdapter::new(
                tokens.clone(),
                CredentialOrigin::SessionJwt,
            )),
        )
    });
    let turn = crate::route::ResolvedTurn {
        slug: slug("tinyhumans"),
        kind: "tinyhumans".into(),
        group: ProviderGroup::Managed,
        base_url: "https://api.tinyhumans.test/x".into(),
        model: Some(model("m")),
        protocol: Protocol::OpenAiChat,
        auth: AuthStyle::Bearer,
        via: crate::route::ResolvedVia::Default,
        origin: None,
        temperature: None,
        cli: None,
    };
    let chat = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    chat.invoke(&(), ModelRequest::default()).await.unwrap_err();
    let health = |bed: &Bed| {
        let hub = bed.hub.clone();
        let scope = bed.scope.clone();
        async move { hub.health(&scope, &slug("tinyhumans")).await.unwrap() }
    };
    assert_eq!(health(&bed).await.health, ProviderHealth::SignedOut);
    // The user signs in; the hub hears nothing.
    *tokens.token.lock().unwrap() = Some("jwt".into());
    let after = health(&bed).await;
    assert_eq!(after.health, ProviderHealth::Unknown);
    assert_eq!(after.snapshot.health, ProviderHealth::Unknown);
    // And signing out again reads signed out again.
    *tokens.token.lock().unwrap() = None;
    assert_eq!(health(&bed).await.health, ProviderHealth::SignedOut);
}

#[tokio::test]
async fn client_a_rejection_the_host_reports_names_no_token_because_the_hub_does_not_know_which() {
    // `record_outcome` is told a turn was rejected, not which token it used; the
    // token answering now may be a newer one. Naming it would make a source that
    // honours identity treat the rejection as current and discard a fresh token,
    // so the source is told plainly and refreshes as it always did.
    let tokens = Arc::new(Identifying {
        generation: AtomicUsize::new(1),
        refreshes: AtomicUsize::new(0),
        plain: AtomicUsize::new(0),
    });
    let (factory, _fake) = recording(vec![]);
    let bed = Bed::with(|b| {
        b.model_factory(factory).managed(
            ManagedConfig::new("https://api.tinyhumans.test/x").source(TokenSourceAdapter::new(
                tokens.clone(),
                CredentialOrigin::SessionJwt,
            )),
        )
    });
    let mut failure =
        crate::error::ProviderFailure::new(ReasonCode::Auth, crate::error::Retry::Never);
    failure.status = Some(401);
    bed.hub
        .record_outcome(
            &bed.scope,
            &slug("tinyhumans"),
            crate::health::Outcome::Failed(failure),
        )
        .await
        .unwrap();
    assert_eq!(tokens.plain.load(Ordering::SeqCst), 1);
    assert_eq!(
        tokens.refreshes.load(Ordering::SeqCst),
        0,
        "no id was named"
    );
}
