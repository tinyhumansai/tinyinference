//! Guards G20, G24 and G25 of 04-operation-matrix, through the public `Hub`.
#![cfg(feature = "testing")]

use std::time::Duration;

use futures::future::join_all;
use serde_json::json;

use tinyinference_hub::catalog::Freshness;
use tinyinference_hub::health::{Outcome, ProviderHealth};
use tinyinference_hub::ports::HubEvent;
use tinyinference_hub::testkit::{Match, MemoryPorts, Scripted};
use tinyinference_hub::{
    AgentKey, Confirm, HubError, HubPolicy, ModelChoice, ModelId, ProviderDraft, ProviderFailure,
    ProviderRoute, ReasonCode, Retry, ScopeKey, Secret, Slug, TurnQuery, WorkloadKey,
};

fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

fn model(id: &str) -> ModelId {
    ModelId::parse(id).unwrap()
}

fn listing(ids: &[&str]) -> serde_json::Value {
    json!({"data": ids.iter().map(|i| json!({"id": i})).collect::<Vec<_>>()})
}

#[tokio::test]
async fn guard_g20_the_cache_is_partitioned_short_on_failure_long_on_success_and_single_flight() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = ScopeKey::new("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openai").with_key(Secret::new("sk-fake")),
    )
    .await
    .unwrap();
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &listing(&["m"])),
    );

    // Single flight: a hundred concurrent callers make one request.
    let calls = (0..100).map(|_| {
        let (hub, me) = (hub.clone(), me.clone());
        async move { hub.list_models(&me, &slug("openai"), false).await }
    });
    for result in join_all(calls).await {
        assert!(result.is_ok());
    }
    assert_eq!(ports.http.request_count(), 1, "single flight");

    // Success is remembered for an hour, not longer.
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
    assert_eq!(ports.http.request_count(), 2);

    // Failure is remembered for a minute (and only a minute).
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(503, "down"),
    );
    ports.clock.advance(Duration::from_secs(3601));
    assert!(
        hub.list_models(&me, &slug("openai"), false)
            .await
            .unwrap()
            .is_stale()
    );
    let after = ports.http.request_count();
    assert!(
        hub.list_models(&me, &slug("openai"), false)
            .await
            .unwrap()
            .is_stale()
    );
    assert_eq!(
        ports.http.request_count(),
        after,
        "the failure memo answers"
    );
    ports.clock.advance(Duration::from_secs(61));
    hub.list_models(&me, &slug("openai"), false).await.unwrap();
    assert_eq!(
        ports.http.request_count(),
        after + 1,
        "a minute later it asks again"
    );

    // A rejected key is never remembered. (The failure memo above has to lapse
    // first, or it would answer instead of the provider.)
    ports.clock.advance(Duration::from_secs(61));
    let after = ports.http.request_count() - 1;
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(401, "Incorrect API key provided"),
    );
    for n in 1..=3 {
        let error = hub
            .list_models(&me, &slug("openai"), false)
            .await
            .unwrap_err();
        assert_eq!(error.reason(), ReasonCode::Auth);
        assert_eq!(
            ports.http.request_count(),
            after + 1 + n,
            "each call asked: nothing was cached"
        );
    }

    // An empty catalog is remembered only briefly (a minute), never for an hour.
    ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &listing(&[])),
    );
    hub.list_models(&me, &slug("openai"), true).await.unwrap();
    let asked = ports.http.request_count();
    ports.clock.advance(Duration::from_secs(61));
    hub.list_models(&me, &slug("openai"), false).await.unwrap();
    assert_eq!(ports.http.request_count(), asked + 1);
}

#[tokio::test]
async fn guard_g24_health_is_keyed_on_slug_latched_per_change_and_cleared_on_removal_and_rotation()
{
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = ScopeKey::new("company:acme");
    for kind in ["openai", "groq"] {
        hub.add(
            &me,
            ProviderDraft::new(kind)
                .with_key(Secret::new("sk-fake"))
                .with_model(model("m")),
        )
        .await
        .unwrap();
    }
    let timeout = ProviderFailure::new(ReasonCode::Timeout, Retry::Later(None));
    for _ in 0..3 {
        hub.record_outcome(&me, &slug("openai"), Outcome::Failed(timeout.clone()))
            .await
            .unwrap();
    }
    // Keyed on the slug: the other provider is untouched.
    assert_eq!(
        hub.health(&me, &slug("groq")).await.unwrap().health,
        ProviderHealth::Unknown
    );
    // Latched: three failures in a row are one change to Degraded then one to Down,
    // and repeating the same state adds nothing.
    for _ in 0..5 {
        hub.record_outcome(&me, &slug("openai"), Outcome::Failed(timeout.clone()))
            .await
            .unwrap();
    }
    let changes: Vec<_> = ports
        .events
        .events()
        .into_iter()
        .filter(|e| matches!(e, HubEvent::HealthChanged { slug: s, .. } if *s == slug("openai")))
        .collect();
    assert_eq!(changes.len(), 2, "{changes:?}");
    // Cleared by rotation and by removal.
    hub.set_key(&me, &slug("openai"), Secret::new("sk-new"))
        .await
        .unwrap();
    assert_eq!(
        hub.health(&me, &slug("openai")).await.unwrap().health,
        ProviderHealth::Unknown
    );
    hub.record_outcome(&me, &slug("groq"), Outcome::Failed(timeout))
        .await
        .unwrap();
    hub.clear_default(&me).await.unwrap();
    hub.remove(&me, &slug("groq"), Confirm::no()).await.unwrap();
    hub.add(&me, ProviderDraft::new("groq")).await.unwrap();
    assert_eq!(
        hub.health(&me, &slug("groq")).await.unwrap().health,
        ProviderHealth::Unknown
    );
}

#[tokio::test]
async fn guard_g25_the_model_on_the_wire_is_never_a_hosts_tier_word() {
    let ports = MemoryPorts::new();
    let hub = ports
        .builder()
        .hub_policy(HubPolicy::new().reserved_model_words(["chat-v1", "reasoning-v1"]))
        .build()
        .unwrap();
    let me = ScopeKey::new("company:acme");
    let tier = || model("chat-v1");
    assert!(matches!(
        hub.add(&me, ProviderDraft::new("openai").with_model(tier()))
            .await,
        Err(HubError::Invalid(_))
    ));
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new("sk-fake"))
            .with_model(model("gpt-5")),
    )
    .await
    .unwrap();
    let s = slug("openai");
    assert!(
        hub.edit(
            &me,
            &s,
            tinyinference_hub::ProviderPatch::new().model(tier())
        )
        .await
        .is_err()
    );
    assert!(
        hub.set_default(&me, ModelChoice::new(s.clone(), tier()))
            .await
            .is_err()
    );
    assert!(
        hub.pin_agent(
            &me,
            &AgentKey::new("a"),
            Some(ModelChoice::new(s.clone(), tier()))
        )
        .await
        .is_err()
    );
    assert!(
        hub.set_workload_route(
            &me,
            &WorkloadKey::new("w"),
            Some(ProviderRoute::provider(s.clone()).with_model(tier()))
        )
        .await
        .is_err()
    );
    // A stored document that somehow carries one still cannot reach the wire.
    ports.config.put_raw(
        &me,
        json!({
            "providers": [{"id": "p", "slug": "openai", "label": "OpenAI", "kind": "openai",
                "base_url": "https://api.openai.com/v1", "model": "reasoning-v1"}],
            "default": {"mode": "full", "provider": "openai", "model": "reasoning-v1"}
        })
        .to_string(),
    );
    assert!(matches!(
        hub.resolve_for_turn(&me, &TurnQuery::new()).await,
        Err(HubError::Invalid(_))
    ));
}
