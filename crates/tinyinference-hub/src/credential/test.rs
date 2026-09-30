//! Tests for the credential chain and its sources.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;

use super::*;
use crate::error::{HubError, PortName};
use crate::ids::{ScopeKey, Slug};
use crate::ports::memory::{CredentialFault, MapEnv, MemoryCredentials};
use crate::ports::{Clock, CredentialStore, PortError, TokenSource};
use crate::secret::Secret;
use crate::testkit::FakeClock;

fn scope() -> ScopeKey {
    ScopeKey::new("company:acme")
}

fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

fn store() -> Arc<MemoryCredentials> {
    Arc::new(MemoryCredentials::new())
}

/// A source that counts how often it is consulted.
#[derive(Debug)]
struct Counting {
    answer: Result<Option<&'static str>, ()>,
    origin: CredentialOrigin,
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl CredentialSource for Counting {
    fn origin(&self) -> CredentialOrigin {
        self.origin.clone()
    }
    async fn resolve(&self, _s: &ScopeKey, _p: &Slug) -> Result<Option<Secret>, PortError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.answer {
            Ok(value) => Ok(value.map(Secret::new)),
            Err(()) => Err(PortError::unavailable("down")),
        }
    }
}

fn counting(
    answer: Result<Option<&'static str>, ()>,
    origin: CredentialOrigin,
) -> (Counting, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        Counting {
            answer,
            origin,
            calls: calls.clone(),
        },
        calls,
    )
}

#[tokio::test]
async fn credential_the_first_source_that_answers_wins_and_reports_its_origin() {
    let (first, first_calls) = counting(Ok(Some("sk-first")), CredentialOrigin::ProviderKey);
    let (second, second_calls) = counting(Ok(Some("sk-second")), CredentialOrigin::AccountKey);
    let chain = CredentialChain::new().with(first).with(second);
    let (secret, origin) = chain
        .resolve(&scope(), &slug("tinyhumans"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(secret.expose(), "sk-first");
    assert_eq!(origin, CredentialOrigin::ProviderKey);
    assert_eq!(first_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        second_calls.load(Ordering::SeqCst),
        0,
        "later sources are not consulted"
    );
}

#[tokio::test]
async fn credential_a_source_with_nothing_is_skipped() {
    let (none, none_calls) = counting(Ok(None), CredentialOrigin::ProviderKey);
    let (some, _) = counting(Ok(Some("sk-account")), CredentialOrigin::AccountKey);
    let chain = CredentialChain::new().with(none).with(some);
    let (secret, origin) = chain
        .resolve(&scope(), &slug("tinyhumans"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (secret.expose(), origin),
        ("sk-account", CredentialOrigin::AccountKey)
    );
    assert_eq!(none_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn credential_a_blank_value_is_not_a_credential_and_does_not_shadow_the_next() {
    let (blank, _) = counting(Ok(Some("")), CredentialOrigin::ProviderKey);
    let (spaces, _) = counting(Ok(Some("   ")), CredentialOrigin::AccountKey);
    let (real, _) = counting(Ok(Some("sk-instance")), CredentialOrigin::InstanceIdentity);
    let chain = CredentialChain::new().with(blank).with(spaces).with(real);
    let (secret, origin) = chain
        .resolve(&scope(), &slug("tinyhumans"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(secret.expose(), "sk-instance");
    assert_eq!(origin, CredentialOrigin::InstanceIdentity);
}

#[tokio::test]
async fn credential_an_empty_chain_and_an_all_none_chain_resolve_nothing() {
    assert!(CredentialChain::new().is_empty());
    assert!(
        CredentialChain::new()
            .resolve(&scope(), &slug("a"))
            .await
            .unwrap()
            .is_none()
    );
    let (a, _) = counting(Ok(None), CredentialOrigin::ProviderKey);
    let (b, _) = counting(Ok(Some("")), CredentialOrigin::AccountKey);
    let chain = CredentialChain::new().with(a).with(b);
    assert!(!chain.is_empty());
    assert!(chain.resolve(&scope(), &slug("a")).await.unwrap().is_none());
}

#[tokio::test]
async fn credential_an_unreadable_source_stops_the_chain_and_never_falls_through() {
    let (broken, _) = counting(Err(()), CredentialOrigin::ProviderKey);
    let (managed, managed_calls) = counting(Ok(Some("sk-account")), CredentialOrigin::AccountKey);
    let chain = CredentialChain::new().with(broken).with(managed);
    let error = chain
        .resolve(&scope(), &slug("tinyhumans"))
        .await
        .unwrap_err();
    match error {
        HubError::StoreUnreadable { port, .. } => assert_eq!(port, PortName::Credentials),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        managed_calls.load(Ordering::SeqCst),
        0,
        "an outage must not spend the operator's managed account"
    );
}

#[tokio::test]
async fn credential_the_store_source_reads_the_per_provider_slot() {
    let creds = store();
    creds
        .set(&scope(), "provider/openai/key", Secret::new("sk-openai"))
        .await
        .unwrap();
    creds
        .set(&scope(), "provider/groq/key", Secret::new("gsk-groq"))
        .await
        .unwrap();
    let chain = CredentialChain::new().with(StoreSource::provider_key(creds));
    let openai = chain
        .resolve(&scope(), &slug("openai"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(openai.0.expose(), "sk-openai");
    assert_eq!(openai.1, CredentialOrigin::ProviderKey);
    let groq = chain
        .resolve(&scope(), &slug("groq"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(groq.0.expose(), "gsk-groq");
    assert!(
        chain
            .resolve(&scope(), &slug("mistral"))
            .await
            .unwrap()
            .is_none()
    );
    assert!(
        chain
            .resolve(&ScopeKey::new("other"), &slug("openai"))
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn credential_the_store_source_reports_an_outage_as_an_error_not_as_no_key() {
    let creds = store();
    creds.inject(CredentialFault::Read);
    let chain = CredentialChain::new().with(StoreSource::provider_key(creds));
    assert!(matches!(
        chain.resolve(&scope(), &slug("openai")).await,
        Err(HubError::StoreUnreadable {
            port: PortName::Credentials,
            ..
        })
    ));
}

#[tokio::test]
async fn credential_a_fixed_slot_serves_every_provider() {
    let creds = store();
    creds
        .set(&scope(), "company/key", Secret::new("sk-company"))
        .await
        .unwrap();
    let source = StoreSource::fixed_slot(creds, "company/key", CredentialOrigin::AccountKey);
    let chain = CredentialChain::new().with(source);
    for name in ["tinyhumans", "anything"] {
        let (secret, origin) = chain.resolve(&scope(), &slug(name)).await.unwrap().unwrap();
        assert_eq!(
            (secret.expose(), origin),
            ("sk-company", CredentialOrigin::AccountKey)
        );
    }
}

#[tokio::test]
async fn credential_the_opencompany_chain_switches_origin_as_keys_come_and_go() {
    let creds = store();
    let chain = CredentialChain::new()
        .with(StoreSource::provider_key(creds.clone()))
        .with(StoreSource::fixed_slot(
            creds.clone(),
            "company/key",
            CredentialOrigin::AccountKey,
        ))
        .with(StaticSource::new(Secret::new("instance-token")));
    let managed = slug("tinyhumans");
    let origin = || async { chain.resolve(&scope(), &managed).await.unwrap().unwrap().1 };
    assert_eq!(origin().await, CredentialOrigin::Static);
    creds
        .set(&scope(), "company/key", Secret::new("sk-company"))
        .await
        .unwrap();
    assert_eq!(origin().await, CredentialOrigin::AccountKey);
    creds
        .set(
            &scope(),
            "provider/tinyhumans/key",
            Secret::new("sk-pasted"),
        )
        .await
        .unwrap();
    assert_eq!(origin().await, CredentialOrigin::ProviderKey);
    creds
        .delete(&scope(), "provider/tinyhumans/key")
        .await
        .unwrap();
    assert_eq!(origin().await, CredentialOrigin::AccountKey);
    creds.delete(&scope(), "company/key").await.unwrap();
    assert_eq!(origin().await, CredentialOrigin::Static);
}

#[tokio::test]
async fn credential_an_environment_source_reads_the_named_variable_only() {
    let env = Arc::new(
        MapEnv::new()
            .with("OPENAI_API_KEY", "sk-env")
            .with("EMPTY", ""),
    );
    let chain = CredentialChain::new()
        .with(EnvVarSource::new(env.clone(), "EMPTY"))
        .with(EnvVarSource::new(env.clone(), "MISSING"))
        .with(EnvVarSource::new(env, "OPENAI_API_KEY"));
    let (secret, origin) = chain
        .resolve(&scope(), &slug("openai"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(secret.expose(), "sk-env");
    assert_eq!(origin, CredentialOrigin::Env("OPENAI_API_KEY".into()));
    assert_eq!(
        origin.describe(),
        "Using the OPENAI_API_KEY environment variable"
    );
}

#[tokio::test]
async fn credential_a_static_source_always_answers() {
    let chain = CredentialChain::new().with(StaticSource::new(Secret::new("sk-not-a-real-key")));
    for _ in 0..3 {
        let (secret, origin) = chain.resolve(&scope(), &slug("x")).await.unwrap().unwrap();
        assert_eq!(
            (secret.expose(), origin),
            ("sk-not-a-real-key", CredentialOrigin::Static)
        );
    }
}

/// A token that changes every fake minute and records invalidations.
#[derive(Debug)]
struct RotatingToken {
    clock: FakeClock,
    invalidated: Mutex<Vec<ScopeKey>>,
    signed_out: Mutex<bool>,
    down: Mutex<bool>,
}

#[async_trait]
impl TokenSource for RotatingToken {
    async fn token(&self, _scope: &ScopeKey) -> Result<Option<Secret>, PortError> {
        if *self.down.lock().unwrap() {
            return Err(PortError::unavailable("token service unreachable"));
        }
        if *self.signed_out.lock().unwrap() {
            return Ok(None);
        }
        let minute = (self.clock.wall_ms() - FakeClock::START_WALL_MS) / 60_000;
        Ok(Some(Secret::new(format!("jwt-{minute}"))))
    }
    fn invalidate(&self, scope: &ScopeKey) {
        self.invalidated.lock().unwrap().push(scope.clone());
    }
}

fn rotating(clock: &FakeClock) -> Arc<RotatingToken> {
    Arc::new(RotatingToken {
        clock: clock.clone(),
        invalidated: Mutex::new(Vec::new()),
        signed_out: Mutex::new(false),
        down: Mutex::new(false),
    })
}

#[tokio::test]
async fn credential_a_rotating_token_is_read_again_on_every_call() {
    let clock = FakeClock::new();
    let token = rotating(&clock);
    let chain = CredentialChain::new().with(TokenSourceAdapter::new(
        token,
        CredentialOrigin::InstanceIdentity,
    ));
    let managed = slug("tinyhumans");
    let mut seen = Vec::new();
    for _ in 0..4 {
        let (secret, origin) = chain.resolve(&scope(), &managed).await.unwrap().unwrap();
        assert_eq!(origin, CredentialOrigin::InstanceIdentity);
        seen.push(secret.expose().to_string());
        clock.advance(Duration::from_secs(60));
    }
    assert_eq!(
        seen,
        ["jwt-0", "jwt-1", "jwt-2", "jwt-3"],
        "no stale token is ever replayed"
    );
}

#[tokio::test]
async fn credential_a_signed_out_token_source_resolves_nothing_and_a_down_one_errors() {
    let clock = FakeClock::new();
    let token = rotating(&clock);
    let chain = CredentialChain::new().with(TokenSourceAdapter::new(
        token.clone(),
        CredentialOrigin::SessionJwt,
    ));
    *token.signed_out.lock().unwrap() = true;
    assert!(
        chain
            .resolve(&scope(), &slug("tinyhumans"))
            .await
            .unwrap()
            .is_none()
    );
    *token.down.lock().unwrap() = true;
    assert!(matches!(
        chain.resolve(&scope(), &slug("tinyhumans")).await,
        Err(HubError::StoreUnreadable {
            port: PortName::Token,
            ..
        })
    ));
}

#[test]
fn credential_invalidation_reaches_rotating_sources_only() {
    let clock = FakeClock::new();
    let token = rotating(&clock);
    let (plain, _) = counting(Ok(None), CredentialOrigin::ProviderKey);
    let chain = CredentialChain::new()
        .with(plain)
        .with(TokenSourceAdapter::new(
            token.clone(),
            CredentialOrigin::SessionJwt,
        ));
    chain.invalidate(&scope());
    assert_eq!(token.invalidated.lock().unwrap().as_slice(), [scope()]);
}

#[tokio::test]
async fn credential_the_legacy_flat_slot_applies_only_inside_its_gate() {
    let creds = store();
    creds
        .set(&scope(), "inference/key", Secret::new("sk-legacy"))
        .await
        .unwrap();
    let legacy = LegacyFlatSlot::new(creds.clone(), "inference/key", |_scope, slug| {
        slug.as_str() == "tinyhumans"
    });
    let chain = CredentialChain::new().with(legacy);
    let (secret, origin) = chain
        .resolve(&scope(), &slug("tinyhumans"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (secret.expose(), origin),
        ("sk-legacy", CredentialOrigin::ProviderKey)
    );
    assert!(
        chain
            .resolve(&scope(), &slug("openai"))
            .await
            .unwrap()
            .is_none(),
        "outside the gate the flat slot is invisible"
    );
    creds.inject(CredentialFault::Read);
    assert!(
        chain
            .resolve(&scope(), &slug("openai"))
            .await
            .unwrap()
            .is_none(),
        "the gate is checked before the store"
    );
    assert!(chain.resolve(&scope(), &slug("tinyhumans")).await.is_err());
}

#[test]
fn credential_no_debug_output_carries_a_secret() {
    let clock = FakeClock::new();
    let creds = store();
    let env = Arc::new(MapEnv::new().with("K", "sk-env-fake"));
    let chain = CredentialChain::new()
        .with(StaticSource::new(Secret::new("sk-static-fake")))
        .with(StoreSource::provider_key(creds.clone()))
        .with(EnvVarSource::new(env, "K"))
        .with(TokenSourceAdapter::new(
            rotating(&clock),
            CredentialOrigin::SessionJwt,
        ))
        .with(LegacyFlatSlot::new(creds, "inference/key", |_, _| true));
    let debug = format!("{chain:?}");
    for leak in ["sk-static-fake", "sk-env-fake"] {
        assert!(!debug.contains(leak), "{debug}");
    }
    assert!(debug.contains("Env"), "{debug}");
    assert_eq!(chain.origins().len(), 5);
}

#[test]
fn credential_origins_have_stable_wire_forms_and_user_copy() {
    let json = |o: &CredentialOrigin| serde_json::to_string(o).unwrap();
    assert_eq!(
        json(&CredentialOrigin::AccountKey),
        r#"{"origin":"account_key"}"#
    );
    assert_eq!(
        json(&CredentialOrigin::Env("K".into())),
        r#"{"origin":"env","name":"K"}"#
    );
    let back: CredentialOrigin = serde_json::from_str(r#"{"origin":"session_jwt"}"#).unwrap();
    assert_eq!(back, CredentialOrigin::SessionJwt);
    let copy = |o: CredentialOrigin| o.to_string();
    assert_eq!(
        copy(CredentialOrigin::ProviderKey),
        "Using the key you added"
    );
    assert_eq!(copy(CredentialOrigin::AccountKey), "Using company key");
    assert_eq!(
        copy(CredentialOrigin::InstanceIdentity),
        "Instance identity"
    );
    assert_eq!(copy(CredentialOrigin::SessionJwt), "Signed in");
    assert_eq!(
        copy(CredentialOrigin::Keychain),
        "Using the system keychain"
    );
    assert_eq!(copy(CredentialOrigin::Static), "Using a fixed key");
    assert_eq!(
        copy(CredentialOrigin::OAuth),
        "Signed in with a browser login"
    );
}

/// A token source that does not override `invalidate`.
#[derive(Debug)]
struct PlainToken;

#[async_trait]
impl TokenSource for PlainToken {
    async fn token(&self, _scope: &ScopeKey) -> Result<Option<Secret>, PortError> {
        Ok(Some(Secret::new("plain")))
    }
}

#[test]
fn credential_the_default_invalidation_is_a_no_op() {
    let chain = CredentialChain::new().with(TokenSourceAdapter::new(
        Arc::new(PlainToken),
        CredentialOrigin::OAuth,
    ));
    chain.invalidate(&scope());
    PlainToken.invalidate(&scope());
}

#[tokio::test]
async fn credential_boxed_sources_join_the_chain_in_order() {
    let boxed: Box<dyn CredentialSource> = Box::new(StaticSource::new(Secret::new("sk-boxed")));
    let chain = CredentialChain::new().with_boxed(boxed);
    let (secret, origin) = chain.resolve(&scope(), &slug("x")).await.unwrap().unwrap();
    assert_eq!(
        (secret.expose(), origin),
        ("sk-boxed", CredentialOrigin::Static)
    );
}

#[test]
fn credential_every_source_has_a_debug_form_that_names_it_and_hides_values() {
    let creds = store();
    let env = Arc::new(MapEnv::new().with("K", "sk-env-fake"));
    let clock = FakeClock::new();
    let forms = [
        format!("{:?}", StoreSource::provider_key(creds.clone())),
        format!(
            "{:?}",
            StoreSource::fixed_slot(creds.clone(), "company/key", CredentialOrigin::AccountKey)
        ),
        format!("{:?}", EnvVarSource::new(env, "K")),
        format!("{:?}", StaticSource::new(Secret::new("sk-static-fake"))),
        format!(
            "{:?}",
            TokenSourceAdapter::new(rotating(&clock), CredentialOrigin::SessionJwt)
        ),
        format!(
            "{:?}",
            LegacyFlatSlot::new(creds, "inference/key", |_, _| true)
        ),
    ];
    let names = [
        "StoreSource",
        "StoreSource",
        "EnvVarSource",
        "StaticSource",
        "TokenSourceAdapter",
        "LegacyFlatSlot",
    ];
    for (form, name) in forms.iter().zip(names) {
        assert!(form.starts_with(name), "{form}");
        assert!(
            !form.contains("sk-env-fake") && !form.contains("sk-static-fake"),
            "{form}"
        );
    }
    assert!(forms[1].contains("company/key") && forms[5].contains("inference/key"));
}

mod chain_props {
    use proptest::prelude::*;

    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum Answer {
        Nothing,
        Blank,
        Key(u8),
        Down,
    }

    fn answer() -> impl Strategy<Value = Answer> {
        prop_oneof![
            Just(Answer::Nothing),
            Just(Answer::Blank),
            (0u8..9).prop_map(Answer::Key),
            Just(Answer::Down),
        ]
    }

    #[derive(Debug)]
    struct Scripted(Answer, u8);

    #[async_trait]
    impl CredentialSource for Scripted {
        fn origin(&self) -> CredentialOrigin {
            CredentialOrigin::Env(format!("SOURCE_{}", self.1))
        }
        async fn resolve(&self, _s: &ScopeKey, _p: &Slug) -> Result<Option<Secret>, PortError> {
            match self.0 {
                Answer::Nothing => Ok(None),
                Answer::Blank => Ok(Some(Secret::new("  "))),
                Answer::Key(n) => Ok(Some(Secret::new(format!("sk-fake-{n}")))),
                Answer::Down => Err(PortError::unavailable("down")),
            }
        }
    }

    proptest! {
        /// The chain's answer is exactly: the first non-blank key, unless a
        /// source before it errored, in which case the error and no key.
        #[test]
        fn credential_prop_the_chain_is_first_key_or_first_error(answers in proptest::collection::vec(answer(), 0..8)) {
            let mut chain = CredentialChain::new();
            for (i, a) in answers.iter().enumerate() {
                chain = chain.with(Scripted(*a, u8::try_from(i).unwrap()));
            }
            let got = futures::executor::block_on(chain.resolve(&scope(), &slug("x")));
            let expected = answers.iter().enumerate().find_map(|(i, a)| match a {
                Answer::Key(n) => Some(Ok((format!("sk-fake-{n}"), format!("SOURCE_{i}")))),
                Answer::Down => Some(Err(())),
                _ => None,
            });
            match (got, expected) {
                (Ok(Some((secret, CredentialOrigin::Env(name)))), Some(Ok((key, source)))) => {
                    prop_assert_eq!(secret.expose(), key);
                    prop_assert_eq!(name, source);
                }
                (Ok(None), None) => {}
                (Err(HubError::StoreUnreadable { .. }), Some(Err(()))) => {}
                (got, expected) => prop_assert!(false, "got {got:?}, expected {expected:?}"),
            }
        }
    }
}

#[tokio::test]
async fn credential_invalidating_an_origin_reaches_only_the_sources_that_answered_as_it() {
    let clock = FakeClock::new();
    let token = rotating(&clock);
    let creds = store();
    let chain = CredentialChain::new()
        .with(StoreSource::provider_key(creds.clone()))
        .with(TokenSourceAdapter::new(
            token.clone(),
            CredentialOrigin::InstanceIdentity,
        ));
    creds
        .set(
            &scope(),
            &slug("tinyhumans").key_slot(),
            Secret::new("sk-pasted"),
        )
        .await
        .unwrap();
    let (_, origin) = chain
        .resolve(&scope(), &slug("tinyhumans"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(origin, CredentialOrigin::ProviderKey);
    // The pasted key was rejected: the healthy rotating token is left alone.
    chain.invalidate_origin(&scope(), &origin);
    assert!(token.invalidated.lock().unwrap().is_empty());
    chain.invalidate_origin(&scope(), &CredentialOrigin::InstanceIdentity);
    assert_eq!(token.invalidated.lock().unwrap().len(), 1);
}
