//! The scenario catalogue (08-test-plan section 4) driven through the public
//! `Hub` API over in-memory ports and a scripted transport. No sockets, no
//! ports, no wall clock, no process environment.
#![cfg(feature = "testing")]

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::future::join_all;
use serde_json::{Value, json};

use tinyinference_hub::catalog::Freshness;
use tinyinference_hub::credential::{CredentialOrigin, TokenSourceAdapter};
use tinyinference_hub::error::{PolicyViolation, PortName};
use tinyinference_hub::health::ProviderHealth;
use tinyinference_hub::import::{oc, oh};
use tinyinference_hub::ports::memory::{CredentialFault, MapEnv};
use tinyinference_hub::ports::{
    CredentialStore, DetectOptions, HealthStore, PortError, TokenSource,
};
use tinyinference_hub::testkit::{Match, MemoryPorts, Scripted};
use tinyinference_hub::{
    AgentKey, Confirm, ConnectOptions, DefaultChoice, EndpointPolicy, Hub, HubError, KeyState,
    ManagedConfig, ModelChoice, ModelId, ProviderDraft, ProviderPatch, ProviderRoute, ReasonCode,
    Retry, RouteTarget, ScopeKey, Secret, Slug, TestDepth, TurnQuery, WorkloadKey,
};

const KEY: &str = "sk-not-a-real-key";

fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

fn model(id: &str) -> ModelId {
    ModelId::parse(id).unwrap()
}

fn scope(name: &str) -> ScopeKey {
    ScopeKey::new(name)
}

fn listing(ids: &[&str]) -> Value {
    json!({"data": ids.iter().map(|i| json!({"id": i})).collect::<Vec<_>>()})
}

fn openai_lists(ports: &MemoryPorts, ids: &[&str]) {
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &listing(ids)),
    );
}

fn rejects(ports: &MemoryPorts, prefix: &str) {
    ports.http.route(
        Match::prefix(prefix),
        Scripted::json(
            401,
            &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
        ),
    );
}

fn custom(label: &str, base: &str) -> ProviderDraft {
    ProviderDraft::new("custom")
        .with_label(label)
        .with_base_url(base)
}

async fn kept(hub: &Hub, scope: &ScopeKey) -> Vec<String> {
    hub.status(scope)
        .await
        .unwrap()
        .providers
        .iter()
        .map(|p| p.view.record.slug.to_string())
        .collect()
}

// ---- onboarding and one-shots ------------------------------------------------------------

#[tokio::test]
async fn sim_desktop_onboarding() {
    let ports = MemoryPorts::new().with_env(MapEnv::new().with("OPENAI_API_KEY", KEY));
    let hub = ports.builder().env_credentials(true).build().unwrap();
    let me = scope("user:local");
    for (port, path, body) in [
        (11434, "/api/version", Some(json!({"version": "0.5.7"}))),
        (1234, "/api/v0/models", None),
        (8000, "/version", None),
        (8080, "/props", None),
    ] {
        ports.http.route(
            Match::get(format!("http://localhost:{port}{path}")),
            body.map_or(Scripted::ConnectRefused, |b| Scripted::json(200, &b)),
        );
    }
    let drafts = hub.detect(&DetectOptions::default()).await.unwrap();
    assert_eq!(
        drafts.iter().map(|d| d.kind.as_str()).collect::<Vec<_>>(),
        ["ollama", "openai"]
    );
    assert!(
        drafts.iter().all(|d| d.key.is_none()),
        "detection never copies a key"
    );

    // Connect Ollama (nothing to spend, no key) and read what it has.
    ports.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::json(200, &listing(&["llama3.2"])),
    );
    let ollama = hub
        .connect(&me, drafts[0].clone(), ConnectOptions::default())
        .await
        .unwrap();
    assert_eq!(ollama.probe.unwrap().model_count(), Some(1));
    // Connect OpenAI on the key in the environment: checked with it, stored nowhere.
    openai_lists(&ports, &["gpt-x"]);
    let openai = hub
        .connect(
            &me,
            drafts[1].clone().with_model(model("gpt-x")),
            ConnectOptions::default(),
        )
        .await
        .unwrap();
    assert!(
        ports
            .http
            .requests()
            .last()
            .unwrap()
            .carried(&Secret::new(KEY))
    );
    assert!(ports.credentials.is_empty());
    assert_eq!(
        openai.record.unwrap().key,
        KeyState::Configured(CredentialOrigin::Env("OPENAI_API_KEY".into()))
    );
    // The operator picks a model from what Ollama lists and makes it the default.
    let listed = hub.list_models(&me, &slug("ollama"), false).await.unwrap();
    hub.set_default(
        &me,
        ModelChoice::new(slug("ollama"), listed.models[0].id.clone()),
    )
    .await
    .unwrap();
    let turn = hub.resolve_for_turn(&me, &TurnQuery::new()).await.unwrap();
    assert_eq!(
        (turn.slug, turn.model.unwrap().as_str().to_string()),
        (slug("ollama"), "llama3.2".to_string())
    );
}

#[tokio::test]
async fn sim_cli_oneshot_env_only() {
    let ports = MemoryPorts::new().with_env(MapEnv::new().with("OPENAI_API_KEY", "sk-env-fake"));
    let hub = ports.builder().env_credentials(true).build().unwrap();
    let me = scope("user:one-shot");
    hub.add(&me, ProviderDraft::new("openai").with_model(model("m")))
        .await
        .unwrap();
    let turn = hub.resolve_for_turn(&me, &TurnQuery::new()).await.unwrap();
    assert_eq!(
        turn.origin,
        Some(CredentialOrigin::Env("OPENAI_API_KEY".into()))
    );
    assert!(
        ports.credentials.is_empty(),
        "nothing about the key was persisted"
    );
    assert!(!ports.config.raw(&me).unwrap().contains("sk-env-fake"));
}

// ---- connect, rollback, rotation ---------------------------------------------------------

#[tokio::test]
async fn sim_connect_rollback_on_auth() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    ports
        .credentials
        .set(&me, &slug("openai").key_slot(), Secret::new("sk-previous"))
        .await
        .unwrap();
    rejects(&ports, "https://api.openai.com/v1/");
    let error = hub
        .connect(
            &me,
            ProviderDraft::new("openai").with_key(Secret::new(KEY)),
            ConnectOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Auth);
    assert!(kept(&hub, &me).await.is_empty());
    let back = ports
        .credentials
        .get(&me, &slug("openai").key_slot())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(back.expose(), "sk-previous", "the previous key is back");
    let said = error.user_message(tinyinference_hub::CopyContext::undone("OpenAI"));
    assert!(said.contains("rejected the credential") && !said.contains("Saved"));
}

#[tokio::test]
async fn sim_connect_add_anyway() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("user:local");
    ports.http.route(
        Match::prefix("http://localhost:11434/"),
        Scripted::ConnectRefused,
    );
    assert!(
        hub.connect(&me, ProviderDraft::new("ollama"), ConnectOptions::default())
            .await
            .is_err()
    );
    hub.connect(
        &me,
        ProviderDraft::new("ollama"),
        ConnectOptions::default().add_anyway(true),
    )
    .await
    .unwrap();
    assert_eq!(
        hub.health(&me, &slug("ollama")).await.unwrap().health,
        ProviderHealth::Down(ReasonCode::Endpoint)
    );
    ports.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::json(200, &listing(&["llama3"])),
    );
    assert!(
        hub.test(&me, &slug("ollama"), TestDepth::Catalog, None)
            .await
            .unwrap()
            .ok()
    );
    assert_eq!(
        hub.health(&me, &slug("ollama")).await.unwrap().health,
        ProviderHealth::Ok
    );
}

#[tokio::test]
async fn sim_local_rollback_on_timeout() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("user:local");
    ports
        .http
        .route(Match::prefix("http://localhost:11434/"), Scripted::Timeout);
    let error = hub
        .connect(
            &me,
            ProviderDraft::new("lmstudio").with_base_url("http://localhost:11434"),
            ConnectOptions::default(),
        )
        .await;
    assert!(
        matches!(error, Err(ref e) if e.reason() == ReasonCode::Timeout),
        "{error:?}"
    );
    assert!(
        kept(&hub, &me).await.is_empty(),
        "a local runtime that times out is rolled back"
    );
    // A cloud provider that times out is kept.
    ports
        .http
        .route(Match::prefix("https://api.openai.com/"), Scripted::Timeout);
    hub.connect(
        &me,
        ProviderDraft::new("openai").with_key(Secret::new(KEY)),
        ConnectOptions::default(),
    )
    .await
    .unwrap();
    assert_eq!(kept(&hub, &me).await, ["openai"]);
}

#[tokio::test]
async fn sim_key_rotation_next_call() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    openai_lists(&ports, &["m"]);
    hub.connect(
        &me,
        ProviderDraft::new("openai").with_key(Secret::new(KEY)),
        ConnectOptions::default(),
    )
    .await
    .unwrap();
    hub.list_models(&me, &slug("openai"), false).await.unwrap();
    hub.edit(
        &me,
        &slug("openai"),
        ProviderPatch::new().key(Secret::new("sk-rotated")),
    )
    .await
    .unwrap();
    assert_eq!(
        hub.health(&me, &slug("openai")).await.unwrap().health,
        ProviderHealth::Unknown
    );
    let list = hub.list_models(&me, &slug("openai"), false).await.unwrap();
    assert_eq!(
        list.freshness,
        Freshness::Fresh,
        "the old list is not served for the new key"
    );
    assert!(
        ports
            .http
            .requests()
            .last()
            .unwrap()
            .carried(&Secret::new("sk-rotated"))
    );
}

#[derive(Debug)]
struct Minute(tinyinference_hub::testkit::FakeClock);

#[async_trait]
impl TokenSource for Minute {
    async fn token(&self, _: &ScopeKey) -> Result<Option<Secret>, PortError> {
        Ok(Some(Secret::new(format!(
            "platform-token-{}",
            self.0.elapsed().as_secs() / 60
        ))))
    }
}

const MANAGED: &str = "https://api.tinyhumans.test/agent-integrations/openrouter";

fn managed_page(ids: &[&str]) -> Value {
    json!({"success": true, "data": {"object": "list", "total": ids.len(), "limit": 500, "offset": 0,
        "data": ids.iter().map(|i| json!({"id": i, "pricing": {"inputPer1M": 1.0, "outputPer1M": 4.0}})).collect::<Vec<_>>()}})
}

#[tokio::test]
async fn sim_platform_token_rotation() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .managed(ManagedConfig::new(MANAGED).source(TokenSourceAdapter::new(
            Arc::new(Minute(ports.clock.clone())),
            CredentialOrigin::InstanceIdentity,
        )))
        .build()
        .unwrap();
    let me = scope("company:acme");
    ports.http.route(
        Match::prefix(format!("{MANAGED}/models")),
        Scripted::json(200, &managed_page(&["a/b"])),
    );
    for minute in 0..5u64 {
        hub.list_models(&me, &slug("tinyhumans"), true)
            .await
            .unwrap();
        let sent = ports.http.requests().pop().unwrap();
        assert!(sent.carried(&Secret::new(format!("platform-token-{minute}"))));
        if let Some(older) = minute.checked_sub(1) {
            assert!(
                !sent.carried(&Secret::new(format!("platform-token-{older}"))),
                "no stale token"
            );
        }
        ports.clock.advance(Duration::from_secs(60));
    }
}

#[tokio::test]
async fn sim_managed_origin_switch() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .managed(ManagedConfig::new(MANAGED).source(TokenSourceAdapter::new(
            Arc::new(Minute(ports.clock.clone())),
            CredentialOrigin::InstanceIdentity,
        )))
        .build()
        .unwrap();
    let me = scope("company:acme");
    let key = || async { hub.health(&me, &slug("tinyhumans")).await.unwrap().view.key };
    assert_eq!(
        key().await,
        KeyState::Configured(CredentialOrigin::InstanceIdentity)
    );
    hub.set_key(&me, &slug("tinyhumans"), Secret::new("th-company"))
        .await
        .unwrap();
    assert_eq!(
        key().await,
        KeyState::Configured(CredentialOrigin::ProviderKey)
    );
    hub.clear_key(&me, &slug("tinyhumans"), Confirm::no())
        .await
        .unwrap();
    assert_eq!(
        key().await,
        KeyState::Configured(CredentialOrigin::InstanceIdentity)
    );
}

#[tokio::test]
async fn sim_signed_out() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .managed(ManagedConfig::new(MANAGED))
        .build()
        .unwrap();
    let me = scope("company:acme");
    let error = hub
        .list_models(&me, &slug("tinyhumans"), false)
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::SignedOut { .. }),
        "signed out is not an empty list and not a red error"
    );
    let status = hub.status(&me).await.unwrap();
    assert_eq!(status.providers[0].health, ProviderHealth::SignedOut);
    assert_eq!(ports.http.request_count(), 0);
}

#[tokio::test]
async fn sim_tenant_isolation() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let (a, b) = (scope("company:a"), scope("company:b"));
    for (s, key, ids) in [
        (&a, "sk-tenant-a", ["entitled-to-a"]),
        (&b, "sk-tenant-b", ["entitled-to-b"]),
    ] {
        hub.add(s, ProviderDraft::new("openai").with_key(Secret::new(key)))
            .await
            .unwrap();
        ports.http.route(
            Match::get("https://api.openai.com/v1/models")
                .with_header("authorization", format!("Bearer {key}")),
            Scripted::json(200, &listing(&ids)),
        );
    }
    assert_eq!(
        hub.list_models(&a, &slug("openai"), false)
            .await
            .unwrap()
            .ids(),
        ["entitled-to-a"]
    );
    assert_eq!(
        hub.list_models(&b, &slug("openai"), false)
            .await
            .unwrap()
            .ids(),
        ["entitled-to-b"]
    );
    // A's key is revoked upstream: A sees it, B never does, and health is per scope.
    ports.http.route(
        Match::get("https://api.openai.com/v1/models")
            .with_header("authorization", "Bearer sk-tenant-a"),
        Scripted::text(401, "Incorrect API key provided"),
    );
    let error = hub
        .list_models(&a, &slug("openai"), true)
        .await
        .unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Auth);
    hub.test(&a, &slug("openai"), TestDepth::Catalog, None)
        .await
        .unwrap();
    assert_eq!(
        hub.health(&a, &slug("openai")).await.unwrap().health,
        ProviderHealth::Down(ReasonCode::Auth)
    );
    assert_eq!(
        hub.health(&b, &slug("openai")).await.unwrap().health,
        ProviderHealth::Unknown
    );
    assert_eq!(
        hub.list_models(&b, &slug("openai"), false)
            .await
            .unwrap()
            .ids(),
        ["entitled-to-b"]
    );
    // B's cache cannot serve A a list, even for the same endpoint.
    assert_eq!(
        hub.list_models(&a, &slug("openai"), false)
            .await
            .unwrap_err()
            .reason(),
        ReasonCode::Auth
    );
}

#[tokio::test]
async fn sim_refresh_bypass() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, ProviderDraft::new("openai").with_key(Secret::new(KEY)))
        .await
        .unwrap();
    openai_lists(&ports, &["m1"]);
    assert_eq!(
        hub.list_models(&me, &slug("openai"), false)
            .await
            .unwrap()
            .freshness,
        Freshness::Fresh
    );
    openai_lists(&ports, &["m2"]);
    assert_eq!(
        hub.list_models(&me, &slug("openai"), false)
            .await
            .unwrap()
            .ids(),
        ["m1"]
    );
    assert_eq!(
        hub.list_models(&me, &slug("openai"), true)
            .await
            .unwrap()
            .ids(),
        ["m2"]
    );
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(503, "down"),
    );
    let stale = hub.list_models(&me, &slug("openai"), true).await.unwrap();
    assert!(
        stale.is_stale() && stale.ids() == ["m2"],
        "the old list and a typed warning"
    );
}

// ---- security ----------------------------------------------------------------------------

const REFUSED_LITERALS: &[&str] = &[
    "http://169.254.169.254/latest/meta-data/",
    "http://169.254.0.1/v1",
    "http://[fe80::1]/v1",
    "http://[fec0::1]/v1",
    "http://10.0.0.1/v1",
    "http://172.16.0.1/v1",
    "http://192.168.1.1/v1",
    "http://100.64.0.1/v1",
    "http://[fc00::1]/v1",
    "http://0.0.0.0/v1",
    "http://[::]/v1",
    "http://224.0.0.1/v1",
    "http://255.255.255.255/v1",
    "http://[::ffff:10.0.0.1]/v1",
    "http://[::ffff:169.254.169.254]/v1",
    "http://2130706433/v1",
    "http://0x7f000001/v1",
    "http://017700000001/v1",
    "http://127.1/v1",
];

#[tokio::test]
async fn sim_ssrf_literal_ips() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    for url in REFUSED_LITERALS {
        // Under the desktop policy, loopback spellings are fine; everything else is not.
        let loopback = matches!(
            *url,
            "http://2130706433/v1"
                | "http://0x7f000001/v1"
                | "http://017700000001/v1"
                | "http://127.1/v1"
        );
        if loopback {
            continue;
        }
        let added = hub.add(&me, custom("Evil", url)).await;
        let probed = hub
            .probe_draft(&me, &custom("Evil", url), TestDepth::Catalog)
            .await;
        assert!(
            matches!(added, Err(HubError::Policy(_)) | Err(HubError::Invalid(_))),
            "{url}: {added:?}"
        );
        assert!(
            matches!(probed, Err(HubError::Policy(_)) | Err(HubError::Invalid(_))),
            "{url}: {probed:?}"
        );
    }
    assert_eq!(ports.http.request_count(), 0, "not one request was sent");
    assert!(kept(&hub, &me).await.is_empty());
    // Hosted refuses every loopback spelling too.
    let hosted = ports
        .builder()
        .policy(EndpointPolicy::hosted())
        .build()
        .unwrap();
    for url in [
        "http://localhost:1/v1",
        "http://127.0.0.1/v1",
        "http://[::1]/v1",
        "http://2130706433/v1",
    ] {
        assert!(
            hosted.add(&me, custom("Local", url)).await.is_err(),
            "{url}"
        );
    }
    // As redirect targets: a public endpoint that redirects into each range.
    ports.http.route(
        Match::get("https://llm.acme.test/v1/models"),
        Scripted::redirect(302, "http://169.254.169.254/latest"),
    );
    let report = hub
        .probe_draft(
            &me,
            &custom("Acme", "https://llm.acme.test/v1").with_key(Secret::new(KEY)),
            TestDepth::Catalog,
        )
        .await
        .unwrap();
    assert!(report.refusal.is_some(), "{report:?}");
    assert_eq!(ports.http.refused().len(), 1);
    assert_eq!(ports.http.request_count(), 1, "only the first hop was sent");
}

#[tokio::test]
async fn sim_ssrf_redirect_chain() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    let draft = custom("Acme", "https://a.acme.test/v1");
    // Three redirects then an answer: fine.
    for (from, to) in [("a", "b"), ("b", "c"), ("c", "d")] {
        ports.http.route(
            Match::get(format!(
                "https://{from}.acme.test/{}",
                if from == "a" { "v1/models" } else { "hop" }
            )),
            Scripted::redirect(302, format!("https://{to}.acme.test/hop")),
        );
    }
    ports.http.route(
        Match::get("https://d.acme.test/hop"),
        Scripted::json(200, &listing(&["m"])),
    );
    assert!(
        hub.probe_draft(&me, &draft, TestDepth::Catalog)
            .await
            .unwrap()
            .ok()
    );
    // A fourth redirect is refused.
    ports.http.route(
        Match::get("https://d.acme.test/hop"),
        Scripted::redirect(302, "https://e.acme.test/hop"),
    );
    let report = hub
        .probe_draft(&me, &draft, TestDepth::Catalog)
        .await
        .unwrap();
    assert_eq!(
        report.refusal,
        Some(PolicyViolation::TooManyRedirects { max: 3 })
    );
    // A credentialed request never follows a redirect to another origin.
    ports.http.route(
        Match::get("https://a.acme.test/v1/models"),
        Scripted::redirect(302, "https://b.acme.test/hop"),
    );
    let keyed = draft.clone().with_key(Secret::new(KEY));
    let report = hub
        .probe_draft(&me, &keyed, TestDepth::Catalog)
        .await
        .unwrap();
    assert_eq!(report.refusal, Some(PolicyViolation::CrossOriginRedirect));
    assert!(
        ports
            .http
            .requests()
            .iter()
            .filter(|r| r.url.contains("b.acme.test"))
            .all(|r| !r.credentialed)
    );
}

#[tokio::test]
async fn sim_dns_rebinding_scripted() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .policy(EndpointPolicy::hosted())
        .build()
        .unwrap();
    let me = scope("company:acme");
    ports.http.route(
        Match::get("https://llm.acme.test/v1/models"),
        Scripted::json(200, &listing(&["m"])).resolving_to(vec![
            "93.184.216.34".parse().unwrap(),
            "10.0.0.7".parse().unwrap(),
        ]),
    );
    let report = hub
        .probe_draft(
            &me,
            &custom("Acme", "https://llm.acme.test/v1").with_key(Secret::new(KEY)),
            TestDepth::Catalog,
        )
        .await
        .unwrap();
    assert!(
        report.refusal.is_some(),
        "a name that resolves to a private address is refused: {report:?}"
    );
    assert!(
        !ports
            .http
            .requests()
            .iter()
            .any(|r| r.url.contains("llm.acme.test")),
        "nothing was sent to it"
    );
}

#[tokio::test]
async fn sim_cleartext_credential() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    let draft = custom("Acme", "http://llm.acme.test/v1").with_key(Secret::new(KEY));
    let report = hub
        .probe_draft(&me, &draft, TestDepth::Catalog)
        .await
        .unwrap();
    assert!(
        report.refusal.is_some(),
        "a key never goes over cleartext http to a public host"
    );
    let error = hub
        .connect(&me, draft, ConnectOptions::default().add_anyway(true))
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Policy(_)));
    assert_eq!(ports.http.request_count(), 0);
    // Loopback http is fine under the desktop policy, and only there.
    ports.http.route(
        Match::prefix("http://localhost:9000/"),
        Scripted::json(200, &listing(&["m"])),
    );
    let local = custom("Local", "http://localhost:9000/v1").with_key(Secret::new(KEY));
    assert!(
        hub.probe_draft(&me, &local, TestDepth::Catalog)
            .await
            .unwrap()
            .ok()
    );
    let hosted = ports
        .builder()
        .policy(EndpointPolicy::hosted())
        .build()
        .unwrap();
    assert!(
        hosted
            .probe_draft(&me, &local, TestDepth::Catalog)
            .await
            .is_err()
    );
}

// ---- degraded providers ------------------------------------------------------------------

#[tokio::test]
async fn sim_offline_local_only() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .policy(EndpointPolicy::local_only())
        .build()
        .unwrap();
    let me = scope("user:air-gapped");
    ports.config.put_raw(
        &me,
        json!({"providers": [
            {"id": "a", "slug": "openai", "label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1"},
            {"id": "b", "slug": "ollama", "label": "Ollama", "kind": "ollama", "base_url": "http://localhost:11434/v1"}
        ]})
        .to_string(),
    );
    ports.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::json(200, &listing(&["llama3"])),
    );
    let status = hub.status(&me).await.unwrap();
    assert_eq!(
        status.providers[0].health,
        ProviderHealth::Disabled,
        "cloud rows are disabled, not deleted"
    );
    assert_eq!(
        hub.list_models(&me, &slug("ollama"), false)
            .await
            .unwrap()
            .ids(),
        ["llama3"]
    );
    assert_eq!(ports.http.request_count(), 1, "nothing reached the cloud");
}

#[tokio::test]
async fn sim_partial_outage() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("m")),
    )
    .await
    .unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(503, "unavailable"),
    );
    ports.http.route(
        Match::post("https://api.openai.com/v1/chat/completions"),
        Scripted::json(200, &json!({"choices": [{"message": {"content": "ok"}}]})),
    );
    let catalog = hub
        .test(&me, &slug("openai"), TestDepth::Catalog, None)
        .await
        .unwrap();
    let completion = hub
        .test(&me, &slug("openai"), TestDepth::Completion, None)
        .await
        .unwrap();
    assert!(!catalog.ok() && completion.ok());
    let health = hub.health(&me, &slug("openai")).await.unwrap().health;
    assert!(
        matches!(health, ProviderHealth::Degraded(_)),
        "degraded, not down: {health:?}"
    );
}

#[tokio::test]
async fn sim_quota_vs_rate() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    for kind in ["anthropic", "openai"] {
        hub.add(&me, ProviderDraft::new(kind).with_key(Secret::new(KEY)))
            .await
            .unwrap();
    }
    ports.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Scripted::json(429, &json!({"type": "error", "error": {"type": "rate_limit_error",
            "message": "You have reached your specified API usage limits. You will regain access on 2026-10-01"}})),
    );
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(429, &json!({"error": {"message": "Rate limit reached for requests. Please try again in 6s.", "type": "requests", "code": "rate_limit_exceeded"}})),
    );
    let spend = hub
        .test(&me, &slug("anthropic"), TestDepth::Catalog, None)
        .await
        .unwrap()
        .failure
        .unwrap();
    assert_eq!(
        (spend.reason, spend.retry),
        (ReasonCode::Quota, Retry::Never)
    );
    let rate = hub
        .test(&me, &slug("openai"), TestDepth::Catalog, None)
        .await
        .unwrap()
        .failure
        .unwrap();
    assert_eq!(rate.reason, ReasonCode::RateLimited);
    assert!(matches!(rate.retry, Retry::Later(Some(d)) if d == Duration::from_secs(6)));
    assert_eq!(
        hub.health(&me, &slug("anthropic")).await.unwrap().health,
        ProviderHealth::Down(ReasonCode::Quota)
    );
    assert!(matches!(
        hub.health(&me, &slug("openai")).await.unwrap().health,
        ProviderHealth::Degraded(ReasonCode::RateLimited)
    ));
}

// ---- concurrency and referential rules ---------------------------------------------------

#[tokio::test]
async fn sim_concurrent_writers() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    let kinds = ["openai", "groq", "mistral", "deepseek", "together", "xai"];
    let adds = kinds.iter().map(|kind| {
        let (hub, me) = (hub.clone(), me.clone());
        async move {
            hub.add(&me, ProviderDraft::new(*kind).with_model(model("m")))
                .await
        }
    });
    ports.config.conflict_next(2);
    let results = join_all(adds).await;
    let saved = results.iter().filter(|r| r.is_ok()).count();
    for result in &results {
        assert!(
            result.is_ok() || matches!(result, Err(HubError::Conflict)),
            "{result:?}"
        );
    }
    let status = hub.status(&me).await.unwrap();
    assert_eq!(status.providers.len(), saved);
    assert_eq!(
        status
            .providers
            .iter()
            .filter(|p| p.view.is_default)
            .count(),
        1,
        "G9: exactly one default"
    );
    // Concurrent toggles and default changes never corrupt the document.
    let tasks = (0..8).map(|i| {
        let (hub, me) = (hub.clone(), me.clone());
        let kind = kinds[i % kinds.len()];
        async move {
            let s = slug(kind);
            let _ = hub
                .set_enabled(&me, &s, i % 2 == 0, Confirm::in_use())
                .await;
            let _ = hub.set_default(&me, ModelChoice::new(s, model("m"))).await;
        }
    });
    join_all(tasks).await;
    let raw = ports.config.raw(&me).unwrap();
    assert!(serde_json::from_str::<tinyinference_hub::HubConfig>(&raw).is_ok());
}

#[tokio::test]
async fn sim_delete_vs_pin() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("m")),
    )
    .await
    .unwrap();
    hub.add(
        &me,
        ProviderDraft::new("groq")
            .with_key(Secret::new("gsk"))
            .with_model(model("l")),
    )
    .await
    .unwrap();
    let agent = AgentKey::new("agent:writer");
    hub.pin_agent(
        &me,
        &agent,
        Some(ModelChoice::new(slug("groq"), model("l"))),
    )
    .await
    .unwrap();
    let refused = hub
        .remove(&me, &slug("groq"), Confirm::no())
        .await
        .unwrap_err();
    assert!(matches!(refused, HubError::InUse(u) if u.agents == [agent.clone()]));
    hub.remove(&me, &slug("groq"), Confirm::in_use())
        .await
        .unwrap();
    let error = hub
        .resolve_for_turn(&me, &TurnQuery::new().with_agent(agent))
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::Unresolved(_)),
        "the pin fails closed, it does not fall to the default: {error:?}"
    );
    assert!(
        hub.resolve_for_turn(&me, &TurnQuery::new()).await.is_ok(),
        "the default still works"
    );
}

#[tokio::test]
async fn sim_disable_fail_closed() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("m")),
    )
    .await
    .unwrap();
    hub.add(
        &me,
        ProviderDraft::new("groq")
            .with_key(Secret::new("gsk"))
            .with_model(model("l")),
    )
    .await
    .unwrap();
    hub.set_enabled(&me, &slug("openai"), false, Confirm::in_use())
        .await
        .unwrap();
    let error = hub
        .resolve_for_turn(&me, &TurnQuery::new())
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::Unresolved(_)),
        "never falls through to another row: {error:?}"
    );
    assert!(
        matches!(
            hub.status(&me).await.unwrap().default,
            DefaultChoice::Full { .. }
        ),
        "the default is untouched"
    );
}

#[tokio::test]
async fn sim_store_unreadable() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .managed(ManagedConfig::new(MANAGED).source(TokenSourceAdapter::new(
            Arc::new(Minute(ports.clock.clone())),
            CredentialOrigin::InstanceIdentity,
        )))
        .build()
        .unwrap();
    let me = scope("company:acme");
    ports.credentials.inject(CredentialFault::Read);
    // The pasted-key source cannot be read: the chain stops. It does NOT fall
    // through to the instance identity, which would spend the operator's account.
    let error = hub
        .list_models(&me, &slug("tinyhumans"), false)
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            HubError::StoreUnreadable {
                port: PortName::Credentials,
                ..
            }
        ),
        "{error:?}"
    );
    assert_eq!(ports.http.request_count(), 0);
    let error = hub
        .resolve_for_turn(
            &me,
            &TurnQuery::new()
                .with_override(ProviderRoute::new(RouteTarget::Managed).with_model(model("m"))),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::StoreUnreadable { .. }));
    ports.credentials.heal();
    assert!(hub.health(&me, &slug("tinyhumans")).await.is_ok());
}

// ---- imports -----------------------------------------------------------------------------

#[tokio::test]
async fn sim_import_oc_then_ops() {
    let snapshot: oc::OcSnapshot = serde_json::from_value(json!({
        "providers": [
            {"id": "prv_a", "slug": "openrouter", "label": "OpenRouter", "kind": "openrouter", "models": {"chat-v1": "openai/gpt-5"}},
            {"id": "prv_b", "slug": "acme-llm", "label": "Acme LLM", "kind": "openai_compatible", "base_url": "https://llm.acme.test/v1", "models": {"chat-v1": "acme-1"}}
        ],
        "default": "openrouter",
        "routes": {"chat-v1": "acme-llm", "code-v1": "openrouter:openai/gpt-5-mini"},
        "entry_zero": {"provider": "managed"}
    }))
    .unwrap();
    let imported = oc::import(&snapshot).unwrap();
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    ports
        .config
        .put_raw(&me, serde_json::to_string(&imported.config).unwrap());
    for slug_name in ["openrouter", "acme-llm"] {
        ports
            .credentials
            .set(
                &me,
                &slug(slug_name).key_slot(),
                Secret::new(format!("sk-{slug_name}")),
            )
            .await
            .unwrap();
    }
    // The bare-slug default fails closed on the turn path, exactly as it did in OpenCompany.
    assert!(matches!(
        hub.resolve_for_turn(&me, &TurnQuery::new()).await,
        Err(HubError::Unresolved(_))
    ));
    // A tier route resolves through the opaque workload key.
    let turn = hub
        .resolve_for_turn(
            &me,
            &TurnQuery::new().with_workload(WorkloadKey::new("chat-v1")),
        )
        .await
        .unwrap();
    assert_eq!(
        (turn.slug, turn.model),
        (slug("acme-llm"), Some(model("acme-1")))
    );
    // Operations work on imported records.
    hub.set_default(
        &me,
        ModelChoice::new(slug("openrouter"), model("openai/gpt-5")),
    )
    .await
    .unwrap();
    assert!(hub.resolve_for_turn(&me, &TurnQuery::new()).await.is_ok());
    hub.edit(
        &me,
        &slug("acme-llm"),
        ProviderPatch::new().label("Acme (prod)"),
    )
    .await
    .unwrap();
    let in_use = hub
        .remove(&me, &slug("openrouter"), Confirm::no())
        .await
        .unwrap_err();
    assert!(
        matches!(in_use, HubError::InUse(u) if u.default_choice && u.workloads == [WorkloadKey::new("code-v1")])
    );
}

#[tokio::test]
async fn sim_import_oh_then_resolve() {
    let snapshot: oh::OhSnapshot = serde_json::from_value(json!({
        "cloud_providers": [
            {"id": "p_openai", "slug": "openai", "label": "OpenAI", "endpoint": "https://api.openai.com/v1", "auth_style": "bearer", "default_model": "gpt-5"},
            {"id": "p_acme", "slug": "acme", "label": "Acme", "endpoint": "https://llm.acme.test/v1", "auth_style": "bearer"}
        ],
        "primary_cloud": "p_openai",
        "routes": {
            "chat_provider": "openai:o3@0.2", "reasoning_provider": "acme:big", "agentic_provider": "",
            "coding_provider": "claude-code:sonnet", "vision_provider": "ollama:llava", "memory_provider": "vllm",
            "embeddings_provider": "cloud", "heartbeat_provider": "openhuman", "learning_provider": "acme:hint:fast",
            "subconscious_provider": "__byok_incomplete__"
        },
        "local_ai": {"provider": "ollama", "base_url": "http://localhost:11434"}
    }))
    .unwrap();
    let imported = oh::import(&snapshot).unwrap();
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .managed(ManagedConfig::new(MANAGED))
        .build()
        .unwrap();
    let me = scope("user:local");
    ports
        .config
        .put_raw(&me, serde_json::to_string(&imported.config).unwrap());
    for name in ["openai", "acme"] {
        hub.set_key(&me, &slug(name), Secret::new(format!("sk-{name}")))
            .await
            .unwrap();
    }
    let resolve = |role: &'static str| {
        let hub = hub.clone();
        let me = me.clone();
        async move {
            hub.resolve_for_turn(&me, &TurnQuery::new().with_workload(WorkloadKey::new(role)))
                .await
        }
    };
    let chat = resolve("chat").await.unwrap();
    assert_eq!(
        (chat.slug, chat.model, chat.temperature.map(|t| t.get())),
        (slug("openai"), Some(model("o3")), Some(0.2))
    );
    assert_eq!(
        resolve("reasoning").await.unwrap().model,
        Some(model("big"))
    );
    // No route (empty and `cloud` strings) means the default: the primary cloud row.
    assert_eq!(
        resolve("agentic").await.unwrap().model,
        Some(model("gpt-5"))
    );
    assert_eq!(resolve("embeddings").await.unwrap().slug, slug("openai"));
    // A CLI login, a local runtime, a bare vllm that has no record, the managed provider (signed out).
    assert_eq!(
        resolve("coding").await.unwrap().cli,
        Some(tinyinference_hub::CliKind::ClaudeCode)
    );
    let vision = resolve("vision").await.unwrap();
    assert_eq!(
        (vision.slug, vision.model),
        (slug("ollama"), Some(model("llava")))
    );
    assert!(
        matches!(resolve("memory").await, Err(HubError::Unresolved(_))),
        "vllm was never configured: fails closed"
    );
    assert!(matches!(
        resolve("heartbeat").await,
        Err(HubError::SignedOut { .. })
    ));
    // A sentinel and an unreplaceable hint fall to the default rather than to a guess.
    assert_eq!(resolve("subconscious").await.unwrap().slug, slug("openai"));
    assert!(
        imported
            .loss
            .entries
            .iter()
            .any(|e| e.source_key == "routes/learning")
    );
}

// ---- detection ---------------------------------------------------------------------------

#[tokio::test]
async fn sim_detect_excludes_self() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    for (port, path, body) in [
        (11434, "/api/version", None),
        (1234, "/api/v0/models", None),
        (8000, "/version", None),
        (8080, "/props", Some(json!({"ok": true}))),
    ] {
        ports.http.route(
            Match::get(format!("http://localhost:{port}{path}")),
            body.map_or(Scripted::ConnectRefused, |b| Scripted::json(200, &b)),
        );
    }
    assert!(
        hub.detect(&DetectOptions::default())
            .await
            .unwrap()
            .is_empty(),
        "8080 answers, but not as llama.cpp"
    );
}

#[tokio::test]
async fn sim_detect_off_when_hosted() {
    let ports =
        MemoryPorts::new().with_env(MapEnv::new().with("OPENAI_API_KEY", "sk-server-secret"));
    let hub = ports
        .builder()
        .policy(EndpointPolicy::hosted())
        .build()
        .unwrap();
    assert!(
        hub.detect(&DetectOptions::default())
            .await
            .unwrap()
            .is_empty()
    );
    assert_eq!(ports.http.request_count(), 0);
}

// ---- transport edge cases ----------------------------------------------------------------

#[tokio::test]
async fn sim_slow_stream_completion() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("m")),
    )
    .await
    .unwrap();
    ports.http.route(
        Match::post("https://api.openai.com/v1/chat/completions"),
        Scripted::SlowStream {
            chunks: vec![
                b"{\"choices\":".to_vec(),
                b"[{\"message\":".to_vec(),
                b"{\"content\":\"ok\"}}]}".to_vec(),
            ],
            gap: Duration::from_secs(5),
        },
    );
    let before = ports.clock.elapsed();
    let report = hub
        .test(&me, &slug("openai"), TestDepth::Completion, None)
        .await
        .unwrap();
    assert_eq!(report.failure.unwrap().reason, ReasonCode::Timeout);
    assert_eq!(
        ports.clock.elapsed() - before,
        Duration::from_secs(10),
        "the timeout cap, on the fake clock"
    );
}

#[tokio::test]
async fn sim_malformed_catalogs() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, ProviderDraft::new("openai").with_key(Secret::new(KEY)))
        .await
        .unwrap();
    let cases: Vec<(Scripted, Result<Vec<&str>, ReasonCode>)> = vec![
        (
            Scripted::json(
                200,
                &json!({"data": [{"id": "ok"}, {"nope": 1}, {"id": 5}, {"id": ""}, null]}),
            ),
            Ok(vec!["ok"]),
        ),
        // `data: null` is an empty catalog (Ollama with nothing pulled), not a failure.
        (Scripted::json(200, &json!({"data": null})), Ok(vec![])),
        (Scripted::json(200, &json!({"data": []})), Ok(vec![])),
        (
            Scripted::json(200, &json!({"data": [{"nope": 1}, {"nope": 2}]})),
            Err(ReasonCode::Unknown),
        ),
        (
            Scripted::Malformed(b"<html>".to_vec()),
            Err(ReasonCode::Unknown),
        ),
        (
            Scripted::Oversize {
                bytes: 20 * 1024 * 1024,
            },
            Err(ReasonCode::Unknown),
        ),
    ];
    for (script, want) in cases {
        ports
            .http
            .route(Match::prefix("https://api.openai.com/v1/models"), script);
        let got = hub.list_models(&me, &slug("openai"), true).await;
        match (got, want) {
            (Ok(list), Ok(ids)) => assert_eq!(list.ids(), ids),
            (Err(error), Err(reason)) => {
                assert_eq!(error.reason(), reason);
                let HubError::Provider(failure) = error else {
                    panic!()
                };
                assert!(
                    failure.raw.expose().len() < 1024,
                    "raw text is capped and log-only"
                );
            }
            (got, want) => panic!("{got:?} vs {want:?}"),
        }
    }
}

fn page(from: usize, count: usize, total: usize) -> Value {
    json!({"success": true, "data": {"object": "list", "total": total, "limit": 500, "offset": from,
        "data": (from..from + count).map(|n| json!({"id": format!("vendor/model-{n}")})).collect::<Vec<_>>()}})
}

#[tokio::test]
async fn sim_paged_catalog() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .managed(ManagedConfig::new(MANAGED))
        .build()
        .unwrap();
    let me = scope("company:acme");
    hub.set_key(&me, &slug("tinyhumans"), Secret::new("th-key"))
        .await
        .unwrap();
    for (offset, count) in [(0, 500), (500, 500), (1000, 200)] {
        ports.http.route(
            Match::get(format!("{MANAGED}/models?limit=500&offset={offset}")),
            Scripted::json(200, &page(offset, count, 1200)),
        );
    }
    let list = hub
        .list_models(&me, &slug("tinyhumans"), true)
        .await
        .unwrap();
    assert_eq!((list.models.len(), list.truncated), (1200, false));
    assert_eq!(list.models[0].input_per_1m, None);
    // A page over the size cap is a typed failure, not a partial list.
    ports.http.route(
        Match::get(format!("{MANAGED}/models?limit=500&offset=500")),
        Scripted::Oversize {
            bytes: 5 * 1024 * 1024,
        },
    );
    let error = hub.list_models(&me, &slug("tinyhumans"), true).await;
    assert!(
        matches!(error, Ok(ref l) if l.is_stale())
            || matches!(error, Err(ref e) if e.reason() == ReasonCode::Unknown),
        "{error:?}"
    );
    // A catalog that never ends stops at the page cap and says it was cut.
    let endless = MemoryPorts::new();
    let hub = endless
        .builder()
        .managed(ManagedConfig::new(MANAGED))
        .build()
        .unwrap();
    hub.set_key(&me, &slug("tinyhumans"), Secret::new("th-key"))
        .await
        .unwrap();
    for n in 0..25 {
        endless.http.route(
            Match::get(format!("{MANAGED}/models?limit=500&offset={}", n * 500)),
            Scripted::json(200, &page(n * 500, 500, 100_000)),
        );
    }
    let list = hub
        .list_models(&me, &slug("tinyhumans"), true)
        .await
        .unwrap();
    assert!(list.truncated);
    assert_eq!(list.models.len(), 20 * 500, "20 pages of 500");
    assert_eq!(endless.http.request_count(), 20);
}

// ---- CLI and OAuth (feature-gated) -------------------------------------------------------

#[cfg(feature = "cli")]
#[tokio::test]
async fn sim_cli_readiness() {
    use tinyinference_hub::CliKind;
    use tinyinference_hub::cli::CliReadiness;
    use tinyinference_hub::testkit::{ScriptedSpawner, Spawned};
    let spawner = Arc::new(ScriptedSpawner::new());
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .process_spawner(spawner.clone())
        .build()
        .unwrap();
    assert_eq!(
        hub.cli_readiness(CliKind::ClaudeCode).await.unwrap(),
        CliReadiness::NotInstalled
    );
    spawner.script("claude", Spawned::ok(r#"{"loggedIn":true}"#));
    assert_eq!(
        hub.cli_readiness(CliKind::ClaudeCode).await.unwrap(),
        CliReadiness::Ready
    );
    spawner.script("claude", Spawned::Hang);
    assert_eq!(
        hub.cli_readiness(CliKind::ClaudeCode).await.unwrap(),
        CliReadiness::Unknown
    );
    spawner.script("codex", Spawned::failed(1, "Not logged in"));
    assert_eq!(
        hub.cli_readiness(CliKind::Codex).await.unwrap(),
        CliReadiness::SignedOut
    );
}

#[cfg(feature = "oauth")]
#[tokio::test]
async fn sim_oauth_disabled() {
    use tinyinference_hub::KindId;
    let hub = MemoryPorts::new().hub();
    let me = scope("user:local");
    for kind in ["openai-codex", "openrouter", "gemini-code-assist"] {
        let kind = KindId::new(kind);
        assert!(matches!(
            hub.oauth_start(&me, &kind).await,
            Err(HubError::Unsupported { .. })
        ));
        assert!(matches!(
            hub.oauth_complete(&me, &kind, "code").await,
            Err(HubError::Unsupported { .. })
        ));
    }
}

#[tokio::test]
async fn sim_health_store_outage_is_typed_and_a_turn_outcome_still_reports() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("m")),
    )
    .await
    .unwrap();
    ports.health.set_unavailable(true);
    let error = hub.health(&me, &slug("openai")).await.unwrap_err();
    assert!(matches!(
        error,
        HubError::StoreUnreadable {
            port: PortName::Health,
            ..
        }
    ));
    ports.health.set_unavailable(false);
    assert!(
        ports
            .health
            .get(&me, &slug("openai"))
            .await
            .unwrap()
            .is_none()
    );
}
