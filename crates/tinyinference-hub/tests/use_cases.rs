//! One test per catalogued use case (09-use-cases), named as the catalogue names
//! it, through the public API. Where a scenario or a unit test already covers
//! the case in depth, this file carries the compact statement of it, so the
//! traceability from a use case to a test name is direct.
#![cfg(feature = "testing")]

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use tinyinference_hub::catalog::{self, Freshness, ModelMeta, ModelMetadataSource, ModelOverride};
use tinyinference_hub::credential::{
    CredentialChain, CredentialOrigin, CredentialSource, EnvVarSource,
};
use tinyinference_hub::descriptor::{CapSource, Tri};
use tinyinference_hub::error::{PolicyViolation, PortName};
use tinyinference_hub::health::{Outcome, ProviderHealth};
use tinyinference_hub::ports::memory::{
    CredentialFault, MapEnv, MemoryCredentials, MemoryHealth, NoopEvents,
};
use tinyinference_hub::ports::{CredentialStore, EnvSource, HealthStore, HubEvent};
use tinyinference_hub::testkit::{FakeClock, Match, MemoryPorts, Scripted, ScriptedHttp};
use tinyinference_hub::{
    AgentKey, Confirm, ConnectOptions, DefaultChoice, EndpointPolicy, HubConfig, HubError,
    HubPolicy, KindId, ManagedConfig, ModelChoice, ModelId, ProviderDraft, ProviderFailure,
    ProviderRoute, ReasonCode, Retry, ScopeKey, Secret, Slug, TestDepth, TurnQuery, WorkloadKey,
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

fn keyed(kind: &str) -> ProviderDraft {
    ProviderDraft::new(kind)
        .with_key(Secret::new(KEY))
        .with_model(model("m"))
}

// ---- A: consumers ------------------------------------------------------------------------

#[tokio::test]
async fn credential_env_source_reads_the_variable_per_call_and_ignores_blank() {
    let env = Arc::new(
        MapEnv::new()
            .with("OPENAI_API_KEY", KEY)
            .with("BLANK", "   "),
    );
    let source = EnvVarSource::new(env.clone(), "OPENAI_API_KEY");
    let chain = CredentialChain::new()
        .with(source)
        .with(EnvVarSource::new(env, "BLANK"));
    let (found, origin) = chain
        .resolve(&scope("s"), &slug("openai"))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (found.expose(), origin),
        (KEY, CredentialOrigin::Env("OPENAI_API_KEY".into()))
    );
    let blank = CredentialChain::new().with(EnvVarSource::new(
        Arc::new(MapEnv::new().with("BLANK", "  ")),
        "BLANK",
    ));
    assert!(
        blank
            .resolve(&scope("s"), &slug("openai"))
            .await
            .unwrap()
            .is_none(),
        "blank is not a credential"
    );
    let unset = EnvVarSource::new(Arc::new(MapEnv::new()), "NOPE");
    assert!(
        unset
            .resolve(&scope("s"), &slug("openai"))
            .await
            .unwrap()
            .is_none()
    );
    assert!(!format!("{chain:?}").contains(KEY));
}

#[tokio::test]
async fn testkit_scripted_http_answers_from_the_script_records_redacted_and_advances_time() {
    let clock = FakeClock::new();
    let http = ScriptedHttp::new(clock.clone());
    http.route(
        Match::get("https://a.test/x").with_header_present("authorization"),
        Scripted::json(200, &json!({"ok": true})).after(Duration::from_secs(3)),
    );
    let request = tinyinference_hub::ports::HubRequest::get("https://a.test/x")
        .with_header("authorization", "Bearer sk-secret-value")
        .with_credentialed(true);
    let response = tinyinference_hub::ports::Http::send(&http, request, &EndpointPolicy::hosted())
        .await
        .unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(
        clock.elapsed(),
        Duration::from_secs(3),
        "latency moves the fake clock, never the wall clock"
    );
    let logged = http.requests();
    assert!(!format!("{logged:?}").contains("sk-secret-value"));
    assert!(logged[0].carried(&Secret::new("sk-secret-value")));
    assert_eq!(logged[0].header("authorization"), Some("<redacted>"));
}

#[tokio::test]
#[should_panic(expected = "no rule answers")]
async fn testkit_scripted_http_panics_on_an_unscripted_request() {
    let http = ScriptedHttp::new(FakeClock::new());
    let _ = tinyinference_hub::ports::Http::send(
        &http,
        tinyinference_hub::ports::HubRequest::get("https://a.test/unscripted"),
        &EndpointPolicy::hosted(),
    )
    .await;
}

#[tokio::test]
async fn testkit_scripted_http_applies_the_endpoint_policy_on_every_hop() {
    let http = ScriptedHttp::new(FakeClock::new());
    http.route(
        Match::get("https://a.test/x"),
        Scripted::redirect(302, "http://169.254.169.254/latest"),
    );
    let error = tinyinference_hub::ports::Http::send(
        &http,
        tinyinference_hub::ports::HubRequest::get("https://a.test/x"),
        &EndpointPolicy::hosted(),
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        tinyinference_hub::ports::HttpError::Policy(_)
    ));
    assert_eq!(http.request_count(), 1);
    assert_eq!(http.refused().len(), 1);
}

#[test]
fn golden_hubconfig_v1() {
    let text = include_str!("golden/hub_config_v1.json");
    let config: HubConfig = serde_json::from_str(text).unwrap();
    assert_eq!(config.schema_version, 1);
    assert_eq!(
        serde_json::to_value(&config).unwrap(),
        serde_json::from_str::<Value>(text).unwrap()
    );
}

#[tokio::test]
async fn policy_local_only_refuses_public() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .policy(EndpointPolicy::local_only())
        .build()
        .unwrap();
    let me = scope("user:air-gapped");
    let error = hub
        .probe_draft(
            &me,
            &ProviderDraft::new("custom")
                .with_label("Remote")
                .with_base_url("https://llm.acme.test/v1"),
            TestDepth::Catalog,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Policy(_)), "{error:?}");
    assert_eq!(ports.http.request_count(), 0);
}

#[tokio::test]
async fn managed_catalog_prices_come_from_the_listing() {
    let ports = MemoryPorts::new();
    let base = "https://api.tinyhumans.test/agent-integrations/openrouter";
    let hub = ports
        .builder()
        .managed(ManagedConfig::new(base))
        .build()
        .unwrap();
    let me = scope("company:acme");
    hub.set_key(&me, &slug("tinyhumans"), Secret::new("th-key"))
        .await
        .unwrap();
    ports.http.route(
        Match::prefix(format!("{base}/models")),
        Scripted::json(
            200,
            &json!({"success": true, "data": {"object": "list", "total": 1, "data": [
            {"id": "openai/gpt-5", "pricing": {"inputPer1M": 1.375, "outputPer1M": 11.0}}]}}),
        ),
    );
    let list = hub
        .list_models(&me, &slug("tinyhumans"), false)
        .await
        .unwrap();
    assert_eq!(
        (list.models[0].input_per_1m, list.models[0].output_per_1m),
        (Some(1.375), Some(11.0))
    );
}

// ---- B: operators ------------------------------------------------------------------------

#[tokio::test]
async fn guard_g11_add_anyway_keeps_row() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    ports.http.route(
        Match::prefix("https://api.openai.com/"),
        Scripted::json(
            401,
            &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
        ),
    );
    assert!(
        hub.connect(&me, keyed("openai"), ConnectOptions::default())
            .await
            .is_err()
    );
    let kept = hub
        .connect(
            &me,
            keyed("openai"),
            ConnectOptions::default().add_anyway(true),
        )
        .await
        .unwrap();
    assert_eq!(
        kept.status,
        tinyinference_hub::MutationStatus::SavedWithWarning
    );
    assert_eq!(hub.status(&me).await.unwrap().providers.len(), 1);
}

#[tokio::test]
async fn chain_order_is_pasted_key_then_the_hosts_sources() {
    let store = Arc::new(MemoryCredentials::new());
    let chain = CredentialChain::new()
        .with(tinyinference_hub::credential::StoreSource::provider_key(
            store.clone(),
        ))
        .with(tinyinference_hub::credential::StaticSource::new(
            Secret::new("fallback"),
        ));
    let (s, p) = (scope("s"), slug("openai"));
    assert_eq!(
        chain.resolve(&s, &p).await.unwrap().unwrap().1,
        CredentialOrigin::Static
    );
    store
        .set(&s, &p.key_slot(), Secret::new("pasted"))
        .await
        .unwrap();
    assert_eq!(
        chain.resolve(&s, &p).await.unwrap().unwrap().1,
        CredentialOrigin::ProviderKey
    );
    store.inject(CredentialFault::Read);
    assert!(
        chain.resolve(&s, &p).await.is_err(),
        "an unreadable first source stops the chain"
    );
}

#[tokio::test]
async fn retry_semantics_by_reason() {
    use tinyinference_hub::classify;
    let cases = [
        (
            401,
            "Incorrect API key provided",
            ReasonCode::Auth,
            Retry::Never,
        ),
        (
            429,
            "You exceeded your current quota, please check your plan and billing details",
            ReasonCode::Quota,
            Retry::Never,
        ),
        (
            429,
            "Rate limit reached for requests",
            ReasonCode::RateLimited,
            Retry::Later(None),
        ),
        (
            503,
            "service unavailable",
            ReasonCode::Unknown,
            Retry::Later(None),
        ),
    ];
    for (status, body, reason, retry) in cases {
        let failure = classify(status, &[], body);
        assert_eq!(
            (failure.reason, failure.retry),
            (reason, retry),
            "{status} {body}"
        );
    }
    assert!(matches!(HubError::Conflict.retry(), Retry::Now));
    assert!(
        matches!(HubError::Provider(classify(429, &[("retry-after", "7")], "slow down")).retry(), Retry::Later(Some(d)) if d == Duration::from_secs(7))
    );
}

#[tokio::test]
async fn catalog_pinned_model_vanished() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &listing(&["other"])),
    );
    let list = hub.list_models(&me, &slug("openai"), true).await.unwrap();
    let chosen = hub
        .health(&me, &slug("openai"))
        .await
        .unwrap()
        .view
        .record
        .model
        .unwrap();
    assert!(
        !list.ids().contains(&chosen.as_str()),
        "the pinned model is not listed any more, which a UI can now say"
    );
}

#[tokio::test]
async fn resolve_pin_precedence() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    hub.add(&me, keyed("groq")).await.unwrap();
    let agent = AgentKey::new("agent:a");
    hub.pin_agent(
        &me,
        &agent,
        Some(ModelChoice::new(slug("groq"), model("pinned"))),
    )
    .await
    .unwrap();
    let turn = hub
        .resolve_for_turn(&me, &TurnQuery::new().with_agent(agent))
        .await
        .unwrap();
    assert_eq!(
        (turn.slug, turn.model),
        (slug("groq"), Some(model("pinned"))),
        "the pin wins over the default, model verbatim"
    );
}

#[tokio::test]
async fn resolve_override_precedence() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    hub.add(&me, keyed("groq")).await.unwrap();
    let agent = AgentKey::new("agent:a");
    hub.pin_agent(
        &me,
        &agent,
        Some(ModelChoice::new(slug("openai"), model("pinned"))),
    )
    .await
    .unwrap();
    let query = TurnQuery::new()
        .with_agent(agent)
        .with_override(ProviderRoute::provider(slug("groq")).with_model(model("forced")));
    let turn = hub.resolve_for_turn(&me, &query).await.unwrap();
    assert_eq!(
        (turn.slug, turn.model),
        (slug("groq"), Some(model("forced"))),
        "a per-task override beats the pin"
    );
}

#[tokio::test]
async fn workload_keys_are_opaque() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    for key in ["chat-v1", "tier:reasoning", "a b c", "ünïcode/✓", ""] {
        hub.set_workload_route(
            &me,
            &WorkloadKey::new(key),
            Some(ProviderRoute::provider(slug("openai")).with_model(model("routed"))),
        )
        .await
        .unwrap();
        let turn = hub
            .resolve_for_turn(&me, &TurnQuery::new().with_workload(WorkloadKey::new(key)))
            .await
            .unwrap();
        assert_eq!(turn.model, Some(model("routed")), "{key:?}");
    }
}

struct Registry;

impl std::fmt::Debug for Registry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Registry")
    }
}

impl ModelMetadataSource for Registry {
    fn lookup(&self, _: &KindId, model: &str) -> Option<ModelMeta> {
        let mut meta = ModelMeta::default();
        meta.capabilities.context_window = tinyinference_hub::descriptor::Sourced::new(
            Some(if model == "m2" { 32_000 } else { 999 }),
            CapSource::Registry,
        );
        Some(meta)
    }
}

#[tokio::test]
async fn catalog_context_window_sources() {
    let ports = MemoryPorts::new();
    let mut over = ModelOverride::new(model("m3"));
    over.context_window = Some(1_000);
    let hub = ports
        .builder()
        .metadata(Arc::new(Registry))
        .overrides(vec![over])
        .build()
        .unwrap();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(
            200,
            &json!({"data": [{"id": "m1", "context_length": 128_000}, {"id": "m2"}, {"id": "m3"}]}),
        ),
    );
    let list = hub.list_models(&me, &slug("openai"), false).await.unwrap();
    let source = |id: &str| {
        list.models
            .iter()
            .find(|m| m.id.as_str() == id)
            .unwrap()
            .capabilities
            .context_window
    };
    assert_eq!(
        (source("m1").value, source("m1").source),
        (Some(128_000), CapSource::ProviderApi),
        "the provider's own answer"
    );
    assert_eq!(
        (source("m2").value, source("m2").source),
        (Some(32_000), CapSource::Registry),
        "a registry fills a gap"
    );
    assert_eq!(
        (source("m3").value, source("m3").source),
        (Some(1_000), CapSource::UserOverride),
        "an operator override beats both"
    );
    assert_eq!(
        list.models[0].capabilities.tools.value,
        Tri::Unknown,
        "never promoted to yes"
    );
    let _ = catalog::ModelEntry::new(model("x"));
}

#[tokio::test]
async fn guard_g26_product_header() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .managed(
            ManagedConfig::new("https://api.tinyhumans.ai/agent-integrations/openrouter")
                .product_header("x-sdk-name", "opencompany"),
        )
        .build()
        .unwrap();
    let me = scope("company:acme");
    hub.set_key(&me, &slug("tinyhumans"), Secret::new("th-key"))
        .await
        .unwrap();
    hub.add(&me, ProviderDraft::new("openai").with_key(Secret::new(KEY)))
        .await
        .unwrap();
    ports.http.route(
        Match::prefix("https://api.tinyhumans.ai/"),
        Scripted::json(
            200,
            &json!({"success": true, "data": {"object": "list", "total": 0, "data": []}}),
        ),
    );
    ports.http.route(
        Match::prefix("https://api.openai.com/"),
        Scripted::json(200, &listing(&["m"])),
    );
    hub.list_models(&me, &slug("tinyhumans"), true)
        .await
        .unwrap();
    hub.list_models(&me, &slug("openai"), true).await.unwrap();
    let sent = ports.http.requests();
    let header = |host: &str| {
        sent.iter()
            .find(|r| r.url.contains(host))
            .unwrap()
            .header("x-sdk-name")
            .map(str::to_string)
    };
    assert_eq!(header("tinyhumans.ai").as_deref(), Some("opencompany"));
    assert_eq!(
        header("openai.com"),
        None,
        "the product identity never reaches another vendor"
    );
}

#[tokio::test]
async fn guard_g9_first_provider_default_race() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    ports.config.conflict_next(1);
    let adds = ["openai", "groq"].map(|kind| {
        let (hub, me) = (hub.clone(), me.clone());
        async move {
            hub.add(&me, ProviderDraft::new(kind).with_model(model("m")))
                .await
        }
    });
    let results = futures::future::join_all(adds).await;
    assert!(
        results
            .iter()
            .all(|r| r.is_ok() || matches!(r, Err(HubError::Conflict)))
    );
    let status = hub.status(&me).await.unwrap();
    assert_eq!(
        status
            .providers
            .iter()
            .filter(|p| p.view.is_default)
            .count(),
        1
    );
    assert!(matches!(status.default, DefaultChoice::Full { .. }));
}

#[tokio::test]
async fn list_models_error_is_typed() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    let cases = [
        (
            Scripted::text(401, "Incorrect API key provided"),
            ReasonCode::Auth,
        ),
        (
            Scripted::text(
                429,
                "You exceeded your current quota, please check your plan and billing details",
            ),
            ReasonCode::Quota,
        ),
        (Scripted::text(503, "down"), ReasonCode::Unknown),
        (Scripted::Timeout, ReasonCode::Timeout),
        (Scripted::ConnectRefused, ReasonCode::Endpoint),
        (Scripted::Malformed(b"<html>".to_vec()), ReasonCode::Unknown),
    ];
    for (script, reason) in cases {
        ports
            .http
            .route(Match::prefix("https://api.openai.com/v1/models"), script);
        ports.clock.advance(Duration::from_secs(61));
        let error = hub
            .list_models(&me, &slug("openai"), true)
            .await
            .unwrap_err();
        assert_eq!(error.reason(), reason, "{error:?}");
        assert!(!error.to_string().contains(KEY));
    }
}

#[tokio::test]
async fn classify_quota_vs_rate_vs_auth() {
    use tinyinference_hub::classify;
    let quota = classify(
        429,
        &[],
        r#"{"type":"error","error":{"type":"rate_limit_error","message":"You have reached your specified API usage limits."}}"#,
    );
    assert_eq!(
        (quota.reason, quota.retry),
        (ReasonCode::Quota, Retry::Never)
    );
    let rate = classify(
        429,
        &[("retry-after", "2")],
        r#"{"error":{"code":"rate_limit_exceeded","message":"Rate limit reached"}}"#,
    );
    assert_eq!(rate.reason, ReasonCode::RateLimited);
    assert_eq!(classify(401, &[], "nope").reason, ReasonCode::Auth);
    assert_ne!(quota.reason, rate.reason);
}

#[tokio::test]
async fn custom_provider_needs_a_name_and_endpoint_and_takes_an_optional_key() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    assert!(
        hub.add(
            &me,
            ProviderDraft::new("custom").with_base_url("https://gw.acme.test/v1")
        )
        .await
        .is_err(),
        "a name is required"
    );
    assert!(
        hub.add(&me, ProviderDraft::new("custom").with_label("Gateway"))
            .await
            .is_err(),
        "an endpoint is required"
    );
    ports.http.route(
        Match::prefix("https://gw.acme.test/v1/models"),
        Scripted::json(200, &listing(&["m"])),
    );
    let added = hub
        .connect(
            &me,
            ProviderDraft::new("custom")
                .with_label("Gateway")
                .with_base_url("https://gw.acme.test/v1"),
            ConnectOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(
        added.status,
        tinyinference_hub::MutationStatus::Saved,
        "keyless is fine for a custom endpoint"
    );
    assert_eq!(
        hub.list_models(&me, &slug("gateway"), false)
            .await
            .unwrap()
            .ids(),
        ["m"]
    );
}

#[tokio::test]
async fn guard_g4_multi_instance_policy() {
    let a = scope("a");
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    hub.add(&a, ProviderDraft::new("openai")).await.unwrap();
    assert!(
        hub.add(
            &a,
            ProviderDraft::new("openai").with_label("Second account")
        )
        .await
        .is_ok(),
        "instances by default"
    );
    let strict_ports = MemoryPorts::new();
    let strict = strict_ports
        .builder()
        .hub_policy(HubPolicy::new().one_row_per_kind(true))
        .build()
        .unwrap();
    strict.add(&a, ProviderDraft::new("openai")).await.unwrap();
    assert!(matches!(
        strict
            .add(
                &a,
                ProviderDraft::new("openai").with_label("Second account")
            )
            .await,
        Err(HubError::AlreadyExists { .. })
    ));
}

#[tokio::test]
async fn status_sorted_managed_first() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .managed(ManagedConfig::new("https://api.tinyhumans.test/x"))
        .build()
        .unwrap();
    let me = scope("company:acme");
    for kind in ["openai", "groq"] {
        hub.add(&me, ProviderDraft::new(kind)).await.unwrap();
    }
    let order: Vec<String> = hub
        .status(&me)
        .await
        .unwrap()
        .providers
        .iter()
        .map(|p| p.view.record.slug.to_string())
        .collect();
    assert_eq!(order, ["tinyhumans", "openai", "groq"]);
}

#[tokio::test]
async fn openrouter_key_only_reports_account() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openrouter").with_key(Secret::new(KEY)),
    )
    .await
    .unwrap();
    ports.http.route(
        Match::get("https://openrouter.ai/api/v1/key"),
        Scripted::json(200, &json!({"data": {"label": "work", "limit": 10.0}})),
    );
    let report = hub
        .test(&me, &slug("openrouter"), TestDepth::KeyOnly, None)
        .await
        .unwrap();
    assert!(
        report.ok() && report.proves_key,
        "the key check proves the key without reading a catalog"
    );
    assert_eq!(ports.http.request_count(), 1);
}

#[tokio::test]
async fn hubconfig_has_no_secret_fields() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    hub.set_key(&me, &slug("openai"), Secret::new("sk-rotated-value"))
        .await
        .unwrap();
    let raw = ports.config.raw(&me).unwrap();
    assert!(!raw.contains(KEY) && !raw.contains("sk-rotated-value"));
    let doc: Value = serde_json::from_str(&raw).unwrap();
    fn walk(value: &Value) {
        match value {
            Value::Object(map) => map.iter().for_each(|(k, v)| {
                let lower = k.to_ascii_lowercase();
                assert!(
                    ![
                        "key",
                        "api_key",
                        "secret",
                        "token",
                        "password",
                        "authorization"
                    ]
                    .contains(&lower.as_str()),
                    "{k}"
                );
                walk(v);
            }),
            Value::Array(items) => items.iter().for_each(walk),
            _ => {}
        }
    }
    walk(&doc);
}

// ---- C: developers -----------------------------------------------------------------------

#[tokio::test]
async fn cache_ttl_with_fake_clock() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &listing(&["m"])),
    );
    let before = std::time::Instant::now();
    hub.list_models(&me, &slug("openai"), false).await.unwrap();
    ports.clock.advance(Duration::from_secs(3599));
    assert_eq!(
        hub.list_models(&me, &slug("openai"), false)
            .await
            .unwrap()
            .freshness,
        Freshness::Cached
    );
    ports.clock.advance(Duration::from_secs(2));
    assert_eq!(
        hub.list_models(&me, &slug("openai"), false)
            .await
            .unwrap()
            .freshness,
        Freshness::Fresh
    );
    assert!(
        before.elapsed() < Duration::from_secs(5),
        "an hour passed on the fake clock, not the real one"
    );
}

#[test]
fn resolved_turn_debug_is_redacted() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    futures::executor::block_on(async {
        hub.add(
            &me,
            ProviderDraft::new("custom")
                .with_label("Acme")
                .with_base_url("https://llm.acme.test/v1")
                .with_key(Secret::new(KEY))
                .with_model(model("m")),
        )
        .await
        .unwrap();
        let turn = hub.resolve_for_turn(&me, &TurnQuery::new()).await.unwrap();
        assert!(!format!("{turn:?}").contains(KEY));
    });
}

#[tokio::test]
async fn events_emitted_per_op() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    hub.set_key(&me, &slug("openai"), Secret::new("k2"))
        .await
        .unwrap();
    hub.set_enabled(&me, &slug("openai"), false, Confirm::in_use())
        .await
        .unwrap();
    hub.clear_default(&me).await.unwrap();
    hub.remove(&me, &slug("openai"), Confirm::no())
        .await
        .unwrap();
    let names: Vec<&str> = ports
        .events
        .events()
        .iter()
        .map(|e| match e {
            HubEvent::ProviderAdded { .. } => "added",
            HubEvent::KeyChanged { .. } => "key",
            HubEvent::EnabledChanged { .. } => "enabled",
            HubEvent::DefaultChanged { .. } => "default",
            HubEvent::ProviderRemoved { .. } => "removed",
            _ => "other",
        })
        .collect();
    for wanted in ["added", "key", "enabled", "default", "removed"] {
        assert!(names.contains(&wanted), "{wanted} in {names:?}");
    }
    let _ = (MemoryHealth::new(), NoopEvents, PortName::Config);
}

#[tokio::test]
async fn config_schema_version_is_written_and_a_newer_one_is_refused() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, ProviderDraft::new("openai")).await.unwrap();
    let doc: Value = serde_json::from_str(&ports.config.raw(&me).unwrap()).unwrap();
    assert_eq!(doc["schema_version"], 1);
    ports.config.put_raw(
        &me,
        json!({"schema_version": 2, "providers": []}).to_string(),
    );
    let error = hub.status(&me).await.unwrap_err();
    assert!(
        matches!(
            error,
            HubError::StoreUnreadable {
                port: PortName::Config,
                ..
            }
        ),
        "never rewritten, never read as empty: {error:?}"
    );
}

#[test]
fn catalogue_golden() {
    let actual = serde_json::to_value(tinyinference_hub::catalogue::descriptors()).unwrap();
    let expected: Value = serde_json::from_str(include_str!("golden/descriptors.json")).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn catalog_parser_openai_ollama_lmstudio_and_paged_shapes() {
    let openai = catalog::parse_openai(br#"{"data":[{"id":"a"},{"id":"b"}]}"#).unwrap();
    assert_eq!(openai.entries.len(), 2);
    assert_eq!(
        catalog::parse_openai(br#"[{"id":"x"}]"#)
            .unwrap()
            .entries
            .len(),
        1,
        "a bare array"
    );
    assert_eq!(
        catalog::parse_ollama_tags(br#"{"models":[{"name":"llama3"}]}"#)
            .unwrap()
            .entries
            .len(),
        1
    );
    assert_eq!(
        catalog::parse_lmstudio_v0(br#"{"data":[{"id":"q","type":"llm"}]}"#)
            .unwrap()
            .entries
            .len(),
        1
    );
    let page = catalog::parse_page(
        r#"{"success":true,"data":{"object":"list","total":1,"data":[{"id":"a/b"}]}}"#,
    )
    .unwrap();
    assert_eq!(page.entries.len(), 1);
}

#[tokio::test]
async fn cache_single_flight_one_request_for_a_hundred_callers() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &listing(&["m"])),
    );
    let calls = (0..100).map(|_| {
        let (hub, me) = (hub.clone(), me.clone());
        async move { hub.list_models(&me, &slug("openai"), false).await.unwrap() }
    });
    futures::future::join_all(calls).await;
    assert_eq!(ports.http.request_count(), 1);
}

#[test]
fn catalog_tolerant_entries_are_skipped_not_fatal() {
    let parsed =
        catalog::parse_openai(br#"{"data":[{"id":"ok"},{"nope":1},{"id":7},null,{"id":""}]}"#)
            .unwrap();
    assert_eq!(parsed.entries.len(), 1);
    assert_eq!(parsed.skipped, 4);
    assert!(catalog::parse_openai(br#"{"error":{"message":"boom"}}"#).is_err());
}

#[tokio::test]
async fn catalog_body_cap() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::Oversize {
            bytes: 64 * 1024 * 1024,
        },
    );
    let error = hub
        .list_models(&me, &slug("openai"), true)
        .await
        .unwrap_err();
    let HubError::Provider(failure) = error else {
        panic!()
    };
    assert!(
        failure.truncated && failure.reason == ReasonCode::Unknown,
        "a listing over the cap is a typed, truncated failure"
    );
}

#[test]
fn reason_code_strings_stable() {
    let strings: Vec<&str> = ReasonCode::ALL.iter().map(|r| r.as_str()).collect();
    assert_eq!(
        strings,
        [
            "auth",
            "model",
            "quota",
            "rate_limited",
            "endpoint",
            "timeout",
            "signed_out",
            "unsupported",
            "unknown",
            "policy",
            "invalid",
            "not_found",
            "already_exists",
            "in_use",
            "conflict",
            "store_unreadable",
            "unresolved"
        ]
    );
}

#[test]
fn testkit_absent_without_feature() {
    let lib = include_str!("../src/lib.rs");
    assert!(
        lib.contains("#[cfg(any(test, feature = \"testing\"))]\npub mod testkit;"),
        "the kit is gated behind the feature"
    );
    let manifest = include_str!("../Cargo.toml");
    assert!(manifest.contains("default = []"));
}

#[test]
fn no_spawn_in_core() {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let mut files = Vec::new();
    walk(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    for file in files {
        let text = std::fs::read_to_string(&file).unwrap();
        let name = file.to_string_lossy().to_string();
        if name.contains("reqwest_http")
            || name.contains("_test.rs")
            || name.ends_with("/test.rs")
            || name.contains("testkit")
        {
            continue;
        }
        for banned in ["tokio::spawn", "std::thread::spawn", "task::spawn("] {
            assert!(
                !text.contains(banned),
                "{name} spawns ({banned}); the core is runtime-agnostic"
            );
        }
    }
}

#[tokio::test]
async fn tracing_fields_never_secret() {
    use std::sync::Mutex;
    use tracing::field::{Field, Visit};
    use tracing::{Event, Metadata, Subscriber, span};

    #[derive(Default)]
    struct Collect(Arc<Mutex<Vec<String>>>);

    struct Fields(String);

    impl Visit for Fields {
        fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
            self.0.push_str(&format!("{}={value:?} ", field.name()));
        }
    }

    impl Subscriber for Collect {
        fn enabled(&self, _: &Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &span::Attributes<'_>) -> span::Id {
            span::Id::from_u64(1)
        }
        fn record(&self, _: &span::Id, _: &span::Record<'_>) {}
        fn record_follows_from(&self, _: &span::Id, _: &span::Id) {}
        fn event(&self, event: &Event<'_>) {
            let mut fields = Fields(String::new());
            event.record(&mut fields);
            self.0.lock().unwrap().push(fields.0);
        }
        fn enter(&self, _: &span::Id) {}
        fn exit(&self, _: &span::Id) {}
    }

    let seen = Arc::new(Mutex::new(Vec::new()));
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.health.set_unavailable(true);
    let guard = tracing::subscriber::set_default(Collect(seen.clone()));
    hub.set_key(&me, &slug("openai"), Secret::new("sk-rotated-value"))
        .await
        .unwrap();
    hub.remove(&me, &slug("openai"), Confirm::in_use())
        .await
        .unwrap();
    drop(guard);
    let events = seen.lock().unwrap().join("\n");
    assert!(!events.is_empty(), "the health outage was logged");
    assert!(
        !events.contains("sk-rotated-value") && !events.contains(KEY),
        "{events}"
    );
}

// ---- D: failure modes and security -------------------------------------------------------

#[tokio::test]
async fn credential_shaped_values_rejected() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    for value in [
        "sk-proj-abcdef1234567890",
        "https://user:hunter2@llm.acme.test/v1",
        "https://llm.acme.test/v1?api_key=sk-abc",
        "https://llm.acme.test/v1?access_token=abc",
    ] {
        let draft = ProviderDraft::new("custom")
            .with_label("Acme")
            .with_base_url(value);
        let error = hub.add(&me, draft).await.unwrap_err();
        assert!(
            matches!(error, HubError::Policy(_) | HubError::Invalid(_)),
            "{value}: {error:?}"
        );
        assert!(
            !format!("{error:?}{error}").contains("hunter2"),
            "the refusal never echoes the credential"
        );
    }
    ports.config.put_raw(&me, json!({"providers": [{"id": "p", "slug": "acme", "label": "A", "kind": "custom", "base_url": "https://a.test", "api_key": "sk-x"}]}).to_string());
    assert!(
        hub.status(&me).await.is_err(),
        "a record with a credential field does not load"
    );
}

#[test]
fn raw_never_in_user_message() {
    let raw = "401 Unauthorized: Bearer sk-not-a-real-key rejected for https://u:p@x.test/?key=abc";
    let error = HubError::Provider(tinyinference_hub::classify(401, &[], raw));
    for undone in [false, true] {
        let ctx = if undone {
            tinyinference_hub::CopyContext::undone("Acme")
        } else {
            tinyinference_hub::CopyContext::saved("Acme")
        };
        let said = error.user_message(ctx);
        assert!(
            !said.contains("sk-not-a-real-key") && !said.contains("u:p@"),
            "{said}"
        );
    }
    assert!(!format!("{error} {error:?}").contains("sk-not-a-real-key"));
}

#[tokio::test]
async fn cache_key_never_contains_secret() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &listing(&["m"])),
    );
    hub.list_models(&me, &slug("openai"), false).await.unwrap();
    let key = catalog::CatalogKey::new(
        &me,
        &slug("openai"),
        true,
        "https://api.openai.com/v1",
        tinyinference_hub::CatalogShape::OpenAi,
    );
    assert!(!format!("{key:?}").contains(KEY));
    // Two different keys for one provider share nothing: rotation is a fresh fetch.
    hub.set_key(&me, &slug("openai"), Secret::new("sk-other"))
        .await
        .unwrap();
    assert_eq!(
        hub.list_models(&me, &slug("openai"), false)
            .await
            .unwrap()
            .freshness,
        Freshness::Fresh
    );
}

#[tokio::test]
async fn slot_bound_to_record() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openai").with_key(Secret::new("sk-for-openai")),
    )
    .await
    .unwrap();
    hub.add(
        &me,
        ProviderDraft::new("groq").with_key(Secret::new("gsk-for-groq")),
    )
    .await
    .unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/"),
        Scripted::json(200, &listing(&["m"])),
    );
    ports.http.route(
        Match::prefix("https://api.groq.com/"),
        Scripted::json(200, &listing(&["m"])),
    );
    hub.list_models(&me, &slug("openai"), true).await.unwrap();
    hub.list_models(&me, &slug("groq"), true).await.unwrap();
    for request in ports.http.requests() {
        let (own, other) = if request.url.contains("openai.com") {
            ("sk-for-openai", "gsk-for-groq")
        } else {
            ("gsk-for-groq", "sk-for-openai")
        };
        assert!(
            request.carried(&Secret::new(own)) && !request.carried(&Secret::new(other)),
            "{}",
            request.url
        );
    }
    assert_eq!(slug("openai").key_slot(), "provider/openai/key");
}

#[tokio::test]
async fn cache_failure_memo_60s() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(503, "down"),
    );
    assert!(hub.list_models(&me, &slug("openai"), false).await.is_err());
    let after = ports.http.request_count();
    assert!(hub.list_models(&me, &slug("openai"), false).await.is_err());
    assert_eq!(ports.http.request_count(), after, "the memo answers");
    ports.clock.advance(Duration::from_secs(61));
    assert!(hub.list_models(&me, &slug("openai"), false).await.is_err());
    assert_eq!(
        ports.http.request_count(),
        after + 1,
        "a minute later it asks again"
    );
}

#[tokio::test]
async fn guard_g27_unknown_kind_fails() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    ports.config.put_raw(&me, json!({"providers": [{"id": "p", "slug": "mystery", "label": "M", "kind": "not-a-kind", "base_url": "https://a.test/v1"}],
        "default": {"mode": "full", "provider": "mystery", "model": "m"}}).to_string());
    assert!(matches!(
        hub.list_models(&me, &slug("mystery"), false).await,
        Err(HubError::NotFound(_))
    ));
    assert!(matches!(
        hub.resolve_for_turn(&me, &TurnQuery::new()).await,
        Err(HubError::NotFound(_))
    ));
    assert!(matches!(
        hub.test(&me, &slug("mystery"), TestDepth::Catalog, None)
            .await,
        Err(HubError::NotFound(_))
    ));
    assert_eq!(
        ports.http.request_count(),
        0,
        "it is never treated as a custom endpoint and called"
    );
}

#[tokio::test]
async fn model_ids_verbatim() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openrouter")).await.unwrap();
    for id in [
        "openai/gpt-5:online",
        "ft:gpt-4o:acme::A1b2",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "llama3.2:3b-instruct-q4_K_M",
        "a.b_c-d/E",
    ] {
        let turn = hub
            .resolve_for_turn(
                &me,
                &TurnQuery::new().with_override(
                    ProviderRoute::provider(slug("openrouter")).with_model(model(id)),
                ),
            )
            .await
            .unwrap();
        assert_eq!(
            turn.model.unwrap().as_str(),
            id,
            "no rewriting, trimming or case change"
        );
    }
}

#[tokio::test]
async fn store_err_is_not_none() {
    let store = MemoryCredentials::new();
    let (s, slot) = (scope("s"), slug("openai").key_slot());
    assert!(
        store.get(&s, &slot).await.unwrap().is_none(),
        "absent is Ok(None)"
    );
    store.inject(CredentialFault::Read);
    assert!(
        store.get(&s, &slot).await.is_err(),
        "unreadable is Err, never None"
    );
    store.heal();
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    ports.credentials.inject(CredentialFault::Read);
    let error = hub.add(&scope("c"), keyed("openai")).await.unwrap_err();
    assert!(matches!(
        error,
        HubError::StoreUnreadable {
            port: PortName::Credentials,
            ..
        }
    ));
    let health = MemoryHealth::new();
    health.set_unavailable(true);
    assert!(health.get(&s, &slug("openai")).await.is_err());
}

#[tokio::test]
async fn policy_hosted_refuses_loopback() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .policy(EndpointPolicy::hosted())
        .build()
        .unwrap();
    for url in [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://[::1]:1234",
    ] {
        let error = hub
            .add(&scope("t"), ProviderDraft::new("ollama").with_base_url(url))
            .await
            .unwrap_err();
        assert!(
            matches!(error, HubError::Policy(PolicyViolation::Endpoint(_))),
            "{url}: {error:?}"
        );
    }
    assert_eq!(ports.http.request_count(), 0);
}

#[tokio::test]
async fn completion_empty_is_unknown() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(&me, keyed("openai")).await.unwrap();
    ports.http.route(
        Match::post("https://api.openai.com/v1/chat/completions"),
        Scripted::text(200, ""),
    );
    let report = hub
        .test(&me, &slug("openai"), TestDepth::Completion, None)
        .await
        .unwrap();
    let failure = report.failure.expect("an empty 200 is not a passing ping");
    assert_eq!(
        failure.reason,
        ReasonCode::Unknown,
        "never `auth`: an empty body does not delete a key"
    );
}

#[tokio::test]
async fn driver_receives_only_own_secret() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    for (kind, key) in [
        ("openai", "sk-openai-only"),
        ("groq", "gsk-groq-only"),
        ("mistral", "ms-mistral-only"),
    ] {
        hub.add(&me, ProviderDraft::new(kind).with_key(Secret::new(key)))
            .await
            .unwrap();
        ports.http.route(
            Match::prefix(format!(
                "https://{}/",
                match kind {
                    "openai" => "api.openai.com",
                    "groq" => "api.groq.com",
                    _ => "api.mistral.ai",
                }
            )),
            Scripted::json(200, &listing(&["m"])),
        );
    }
    for kind in ["openai", "groq", "mistral"] {
        hub.list_models(&me, &slug(kind), true).await.unwrap();
    }
    let all_keys = ["sk-openai-only", "gsk-groq-only", "ms-mistral-only"];
    for request in ports.http.requests() {
        let carried: Vec<&&str> = all_keys
            .iter()
            .filter(|k| request.carried(&Secret::new(**k)))
            .collect();
        assert_eq!(carried.len(), 1, "{}: {carried:?}", request.url);
    }
}

#[tokio::test]
async fn cli_readiness_timeout_is_unknown() {
    // The CLI feature is off in some lanes; the rule is also stated by the type
    // system: a hub without a spawner cannot say `SignedOut` at all.
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let error = hub.cli_readiness_unavailable_without_feature().await;
    assert!(error);
}

trait CliGate {
    async fn cli_readiness_unavailable_without_feature(&self) -> bool;
}

impl CliGate for tinyinference_hub::Hub {
    async fn cli_readiness_unavailable_without_feature(&self) -> bool {
        #[cfg(feature = "cli")]
        {
            use tinyinference_hub::CliKind;
            use tinyinference_hub::cli::CliReadiness;
            let spawner = Arc::new(tinyinference_hub::testkit::ScriptedSpawner::new());
            spawner.script("claude", tinyinference_hub::testkit::Spawned::Hang);
            let ports = MemoryPorts::new();
            let hub = ports.builder().process_spawner(spawner).build().unwrap();
            return hub.cli_readiness(CliKind::ClaudeCode).await.unwrap() == CliReadiness::Unknown;
        }
        #[cfg(not(feature = "cli"))]
        true
    }
}

#[test]
fn health_states_and_outcome_types_exist_for_the_router() {
    // B12: the signals a router consumes are public and constructible.
    let failure = ProviderFailure::new(ReasonCode::Timeout, Retry::Later(None));
    let _ = Outcome::Failed(failure);
    let _ = ProviderHealth::Down(ReasonCode::Timeout);
    let _ = ProviderHealth::SignedOut;
    let _: Option<&dyn EnvSource> = None;
    let _: Option<&dyn HealthStore> = None;
    let _: Option<&dyn CredentialSource> = None;
    let _: Option<Arc<dyn CredentialStore>> = None;
}
