//! Tests for probes, model lists, health, status and re-testing.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use crate::catalog::{Freshness, ModelMeta, ModelMetadataSource, ModelOverride};
use crate::config::ProviderDraft;
use crate::credential::{CredentialOrigin, TokenSourceAdapter};
use crate::descriptor::{CapSource, Tri};
use crate::error::{HubError, InvalidInput, Operation, ProviderFailure, ReasonCode, Retry};
use crate::health::{Outcome, ProviderHealth};
use crate::hub::fixtures::{Bed, KEY, model, models_body, slug};
use crate::hub::{Confirm, ConnectOptions, KeyState, ManagedConfig};
use crate::ids::{KindId, ScopeKey};
use crate::ports::{HealthStore, HubEvent, PortError, TokenSource};
use crate::secret::Secret;
use crate::taxonomy::TestDepth;
use crate::testkit::{Match, Scripted};

const MANAGED: &str = "https://api.tinyhumans.test/agent-integrations/openrouter";

fn managed_page(ids: &[&str]) -> serde_json::Value {
    json!({"success": true, "data": {"object": "list", "total": ids.len(), "limit": 500, "offset": 0,
        "data": ids.iter().map(|i| json!({"id": i, "pricing": {"inputPer1M": 1.5, "outputPer1M": 6.0}})).collect::<Vec<_>>()}})
}

fn managed_bed() -> Bed {
    Bed::with(|b| b.managed(ManagedConfig::new(MANAGED)))
}

// ---- list_models -------------------------------------------------------------------------

#[tokio::test]
async fn ops_list_models_caches_refreshes_and_serves_stale_with_a_typed_warning() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    bed.openai_lists(&["m1"]);
    let s = slug("openai");
    let first = bed.hub.list_models(&bed.scope, &s, false).await.unwrap();
    assert_eq!(
        (first.freshness.clone(), first.ids()),
        (Freshness::Fresh, vec!["m1"])
    );
    bed.openai_lists(&["m2"]);
    let cached = bed.hub.list_models(&bed.scope, &s, false).await.unwrap();
    assert_eq!(
        (cached.freshness.clone(), cached.ids()),
        (Freshness::Cached, vec!["m1"])
    );
    let refreshed = bed.hub.list_models(&bed.scope, &s, true).await.unwrap();
    assert_eq!(refreshed.ids(), vec!["m2"]);
    bed.ports.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(503, "unavailable"),
    );
    let stale = bed.hub.list_models(&bed.scope, &s, true).await.unwrap();
    assert!(stale.is_stale() && stale.ids() == vec!["m2"]);
    let events = bed.ports.events.events();
    assert!(
        events
            .iter()
            .any(|e| matches!(e, HubEvent::CatalogFetched { models: 1, .. }))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        HubEvent::CatalogServedStale {
            reason: ReasonCode::Unknown,
            ..
        }
    )));
}

#[tokio::test]
async fn ops_a_rejected_key_is_reported_never_remembered_and_never_hidden_by_an_old_list() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    bed.openai_lists(&["m1"]);
    let s = slug("openai");
    bed.hub.list_models(&bed.scope, &s, false).await.unwrap();
    bed.openai_rejects_key();
    let error = bed.hub.list_models(&bed.scope, &s, true).await.unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Auth);
    let before = bed.ports.http.request_count();
    // Not memoised: the next call asks again (and gets the same answer), it does
    // not replay the failure from the cache, and it does not serve the old list.
    let again = bed
        .hub
        .list_models(&bed.scope, &s, false)
        .await
        .unwrap_err();
    assert_eq!(again.reason(), ReasonCode::Auth);
    assert!(bed.ports.http.request_count() > before);
}

#[tokio::test]
async fn ops_list_models_errors_are_typed() {
    let bed = Bed::new();
    let s = slug("openai");
    assert!(matches!(
        bed.hub.list_models(&bed.scope, &s, false).await,
        Err(HubError::NotFound(_))
    ));
    // A keyed kind with no key is Invalid, not a network error.
    bed.hub
        .add(&bed.scope, ProviderDraft::new("openai"))
        .await
        .unwrap();
    assert!(matches!(
        bed.hub.list_models(&bed.scope, &s, false).await,
        Err(HubError::Invalid(InvalidInput::Empty(_)))
    ));
    assert_eq!(bed.ports.http.request_count(), 0);
    // A record whose endpoint the policy refuses is refused before any request.
    bed.ports.config.put_raw(
        &bed.scope,
        json!({"providers": [{"id": "p", "slug": "evil", "label": "Evil", "kind": "custom",
            "base_url": "http://169.254.169.254/v1"}]})
        .to_string(),
    );
    assert!(matches!(
        bed.hub.list_models(&bed.scope, &slug("evil"), false).await,
        Err(HubError::Policy(_))
    ));
    assert_eq!(bed.ports.http.request_count(), 0);
}

#[derive(Debug)]
struct Registry;

impl ModelMetadataSource for Registry {
    fn lookup(&self, _kind: &KindId, model: &str) -> Option<ModelMeta> {
        (model == "m1").then(|| {
            let mut meta = ModelMeta {
                display_name: Some("Model One".into()),
                ..ModelMeta::default()
            };
            meta.capabilities.tools =
                crate::descriptor::Sourced::new(Tri::Yes, CapSource::Registry);
            meta
        })
    }
}

#[tokio::test]
async fn ops_list_models_merges_registry_facts_and_operator_overrides() {
    let mut over = ModelOverride::new(model("m1"));
    over.context_window = Some(64_000);
    let mut added = ModelOverride::new(model("my-deployment"));
    added.display_name = Some("Mine".into());
    let bed = Bed::with(|b| b.metadata(Arc::new(Registry)).overrides(vec![over, added]));
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    bed.openai_lists(&["m1", "m2"]);
    let list = bed
        .hub
        .list_models(&bed.scope, &slug("openai"), false)
        .await
        .unwrap();
    let m1 = list.models.iter().find(|m| m.id.as_str() == "m1").unwrap();
    assert_eq!(m1.display_name.as_deref(), Some("Model One"));
    assert_eq!(m1.capabilities.tools.value, Tri::Yes);
    assert_eq!(
        m1.capabilities.context_window.source,
        CapSource::UserOverride
    );
    assert!(
        list.models.iter().any(|m| m.id.as_str() == "my-deployment"),
        "an override adds a model"
    );
    // The cached copy is never mutated by the merge (a second call sees the same).
    let again = bed
        .hub
        .list_models(&bed.scope, &slug("openai"), false)
        .await
        .unwrap();
    assert_eq!(again.models.len(), 3);
}

// ---- probe_draft / test ------------------------------------------------------------------

#[tokio::test]
async fn ops_probe_draft_reports_a_failure_as_a_fact_and_stores_nothing() {
    let bed = Bed::new();
    bed.openai_rejects_key();
    let report = bed
        .hub
        .probe_draft(&bed.scope, &bed.openai_draft(), TestDepth::Catalog)
        .await
        .unwrap();
    assert_eq!(report.failure.unwrap().reason, ReasonCode::Auth);
    assert!(bed.ports.config.raw(&bed.scope).is_none() && bed.ports.credentials.is_empty());
    assert_eq!(
        bed.ports
            .health
            .get(&bed.scope, &slug("openai"))
            .await
            .unwrap(),
        None,
        "a draft's probe is not health"
    );
    // A saved provider's key is never lent to a draft.
    bed.store_key("openai", "sk-saved").await;
    bed.openai_lists(&["m"]);
    assert!(matches!(
        bed.hub
            .probe_draft(
                &bed.scope,
                &ProviderDraft::new("openai"),
                TestDepth::Catalog
            )
            .await,
        Err(HubError::Invalid(_))
    ));
    assert_eq!(
        bed.ports.http.request_count(),
        1,
        "only the first probe sent anything"
    );
}

#[tokio::test]
async fn ops_probe_draft_validates_like_add() {
    let bed = Bed::new();
    assert!(matches!(
        bed.hub
            .probe_draft(
                &bed.scope,
                &ProviderDraft::new("openai"),
                TestDepth::Completion
            )
            .await,
        Err(HubError::Invalid(_))
    ));
    assert!(matches!(
        bed.hub
            .probe_draft(
                &bed.scope,
                &ProviderDraft::new("claude-code"),
                TestDepth::Catalog
            )
            .await,
        Err(HubError::Unsupported {
            op: Operation::ProbeDraft,
            ..
        })
    ));
    let custom = ProviderDraft::new("custom")
        .with_label("A")
        .with_base_url("http://10.1.1.1/v1");
    assert!(matches!(
        bed.hub
            .probe_draft(&bed.scope, &custom, TestDepth::Catalog)
            .await,
        Err(HubError::Policy(_))
    ));
}

#[tokio::test]
async fn ops_test_runs_at_each_depth_and_records_health() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let s = slug("openai");
    bed.openai_lists(&["m"]);
    bed.openai_chat_ok();
    assert!(
        bed.hub
            .test(&bed.scope, &s, TestDepth::Catalog, None)
            .await
            .unwrap()
            .ok()
    );
    let completion = bed
        .hub
        .test(&bed.scope, &s, TestDepth::Completion, Some(&model("other")))
        .await
        .unwrap();
    assert!(completion.ok());
    assert!(
        bed.ports
            .http
            .requests()
            .last()
            .unwrap()
            .body
            .as_deref()
            .unwrap()
            .contains("other")
    );
    assert!(matches!(
        bed.hub.test(&bed.scope, &s, TestDepth::KeyOnly, None).await,
        Err(HubError::Unsupported {
            op: Operation::Test(TestDepth::KeyOnly),
            ..
        })
    ));
    assert!(matches!(
        bed.hub
            .test(&bed.scope, &slug("nope"), TestDepth::Catalog, None)
            .await,
        Err(HubError::NotFound(_))
    ));
    assert_eq!(
        bed.hub.health(&bed.scope, &s).await.unwrap().health,
        ProviderHealth::Ok
    );
}

// ---- health, status ----------------------------------------------------------------------

#[tokio::test]
async fn ops_record_outcome_feeds_health_and_a_rejection_reaches_the_source() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let s = slug("openai");
    bed.hub
        .record_outcome(
            &bed.scope,
            &s,
            Outcome::Ok {
                latency: Duration::from_millis(40),
            },
        )
        .await
        .unwrap();
    assert_eq!(
        bed.hub.health(&bed.scope, &s).await.unwrap().health,
        ProviderHealth::Ok
    );
    let failure = ProviderFailure::new(ReasonCode::Timeout, Retry::Later(None));
    for _ in 0..3 {
        bed.hub
            .record_outcome(&bed.scope, &s, Outcome::Failed(failure.clone()))
            .await
            .unwrap();
    }
    assert_eq!(
        bed.hub.health(&bed.scope, &s).await.unwrap().health,
        ProviderHealth::Down(ReasonCode::Timeout)
    );
    assert!(matches!(
        bed.hub
            .record_outcome(
                &bed.scope,
                &slug("nope"),
                Outcome::Ok {
                    latency: Duration::ZERO
                }
            )
            .await,
        Err(HubError::NotFound(_))
    ));
}

#[tokio::test]
async fn ops_status_lists_managed_first_and_keeps_the_added_order() {
    let bed = Bed::with(|b| b.managed(ManagedConfig::new(MANAGED)));
    for kind in ["groq", "openai", "mistral"] {
        bed.hub
            .add(&bed.scope, ProviderDraft::new(kind))
            .await
            .unwrap();
    }
    let status = bed.hub.status(&bed.scope).await.unwrap();
    let order: Vec<&str> = status
        .providers
        .iter()
        .map(|p| p.view.record.slug.as_str())
        .collect();
    assert_eq!(order, ["tinyhumans", "groq", "openai", "mistral"]);
    assert_eq!(
        status.primary,
        Some(slug("tinyhumans")),
        "no default: the first enabled row"
    );
    assert_eq!(status.providers[0].health, ProviderHealth::SignedOut);
    assert_eq!(status.providers[1].health, ProviderHealth::Unknown);
    assert_eq!(status.providers[1].view.key, KeyState::Missing);
}

#[tokio::test]
async fn sim_offline_local_only_shows_cloud_rows_disabled_and_local_rows_working() {
    let bed = Bed::with(|b| b.policy(crate::policy::EndpointPolicy::local_only()));
    bed.ports.config.put_raw(
        &bed.scope,
        json!({"providers": [
            {"id": "a", "slug": "openai", "label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1"},
            {"id": "b", "slug": "ollama", "label": "Ollama", "kind": "ollama", "base_url": "http://localhost:11434/v1"}
        ]})
        .to_string(),
    );
    bed.ports.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::json(200, &models_body(&["llama3"])),
    );
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert_eq!(status.providers[0].health, ProviderHealth::Disabled);
    assert_eq!(status.providers[1].health, ProviderHealth::Unknown);
    assert_eq!(
        bed.hub
            .list_models(&bed.scope, &slug("ollama"), false)
            .await
            .unwrap()
            .ids(),
        ["llama3"]
    );
    assert!(matches!(
        bed.hub
            .list_models(&bed.scope, &slug("openai"), false)
            .await,
        Err(HubError::Invalid(_)) | Err(HubError::Policy(_))
    ));
}

// ---- managed (D1, D3, D5) ----------------------------------------------------------------

#[tokio::test]
async fn sim_signed_out_is_typed_never_an_empty_list_and_sends_nothing() {
    let bed = managed_bed();
    let s = slug("tinyhumans");
    let error = bed
        .hub
        .list_models(&bed.scope, &s, false)
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::SignedOut { .. }), "{error:?}");
    assert_eq!(
        bed.hub.health(&bed.scope, &s).await.unwrap().health,
        ProviderHealth::SignedOut
    );
    assert_eq!(bed.ports.http.request_count(), 0);
    let test = bed
        .hub
        .test(&bed.scope, &s, TestDepth::Catalog, None)
        .await
        .unwrap_err();
    assert!(matches!(test, HubError::SignedOut { .. }));
}

#[derive(Debug)]
struct Tokens {
    token: std::sync::Mutex<Option<String>>,
    invalidated: AtomicUsize,
}

#[async_trait]
impl TokenSource for Tokens {
    async fn token(&self, _scope: &ScopeKey) -> Result<Option<Secret>, PortError> {
        Ok(self.token.lock().unwrap().clone().map(Secret::new))
    }

    fn invalidate(&self, _scope: &ScopeKey) {
        self.invalidated.fetch_add(1, Ordering::SeqCst);
    }
}

#[tokio::test]
async fn sim_managed_origin_switch_shows_which_source_answers() {
    let tokens = Arc::new(Tokens {
        token: std::sync::Mutex::new(None),
        invalidated: AtomicUsize::new(0),
    });
    let bed = Bed::with(|b| {
        b.managed(ManagedConfig::new(MANAGED).source(TokenSourceAdapter::new(
            tokens.clone(),
            CredentialOrigin::InstanceIdentity,
        )))
    });
    let s = slug("tinyhumans");
    let key = |bed: &Bed| {
        let hub = bed.hub.clone();
        let scope = bed.scope.clone();
        async move {
            hub.health(&scope, &slug("tinyhumans"))
                .await
                .unwrap()
                .view
                .key
        }
    };
    assert_eq!(key(&bed).await, KeyState::Missing);
    *tokens.token.lock().unwrap() = Some("instance-token".into());
    assert_eq!(
        key(&bed).await,
        KeyState::Configured(CredentialOrigin::InstanceIdentity)
    );
    // A pasted key wins (D5: an ordinary key-set on the managed descriptor).
    bed.hub
        .set_key(&bed.scope, &s, Secret::new("th-pasted"))
        .await
        .unwrap();
    assert_eq!(
        key(&bed).await,
        KeyState::Configured(CredentialOrigin::ProviderKey)
    );
    bed.hub
        .clear_key(&bed.scope, &s, Confirm::no())
        .await
        .unwrap();
    assert_eq!(
        key(&bed).await,
        KeyState::Configured(CredentialOrigin::InstanceIdentity)
    );
    *tokens.token.lock().unwrap() = None;
    assert_eq!(key(&bed).await, KeyState::Missing);
    assert_eq!(
        bed.hub.health(&bed.scope, &s).await.unwrap().health,
        ProviderHealth::SignedOut
    );
}

#[tokio::test]
async fn sim_platform_token_rotation_no_stale_token_is_ever_sent() {
    struct Minute(crate::testkit::FakeClock);
    impl std::fmt::Debug for Minute {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Minute")
        }
    }
    #[async_trait]
    impl TokenSource for Minute {
        async fn token(&self, _scope: &ScopeKey) -> Result<Option<Secret>, PortError> {
            Ok(Some(Secret::new(format!(
                "token-{}",
                self.0.elapsed().as_secs() / 60
            ))))
        }
    }
    let ports = crate::testkit::MemoryPorts::new();
    let clock = ports.clock.clone();
    let hub = ports
        .builder()
        .managed(ManagedConfig::new(MANAGED).source(TokenSourceAdapter::new(
            Arc::new(Minute(clock.clone())),
            CredentialOrigin::InstanceIdentity,
        )))
        .build()
        .unwrap();
    let scope = ScopeKey::new("company:acme");
    ports.http.route(
        Match::prefix(format!("{MANAGED}/models")),
        Scripted::json(200, &managed_page(&["a/b"])),
    );
    for minute in 0..4u64 {
        let list = hub
            .list_models(&scope, &slug("tinyhumans"), true)
            .await
            .unwrap();
        assert_eq!(
            list.models[0].input_per_1m,
            Some(1.5),
            "managed prices come from the listing (D1)"
        );
        let sent = ports.http.requests().pop().unwrap();
        assert!(
            sent.carried(&Secret::new(format!("token-{minute}"))),
            "minute {minute}"
        );
        clock.advance(Duration::from_secs(60));
    }
}

#[tokio::test]
async fn ops_a_rejected_managed_token_is_invalidated_at_its_source() {
    let tokens = Arc::new(Tokens {
        token: std::sync::Mutex::new(Some("t".into())),
        invalidated: AtomicUsize::new(0),
    });
    let bed = Bed::with(|b| {
        b.managed(ManagedConfig::new(MANAGED).source(TokenSourceAdapter::new(
            tokens.clone(),
            CredentialOrigin::SessionJwt,
        )))
    });
    bed.ports.http.route(
        Match::prefix(format!("{MANAGED}/models")),
        Scripted::text(401, "Unauthorized: invalid token"),
    );
    let error = bed
        .hub
        .list_models(&bed.scope, &slug("tinyhumans"), false)
        .await
        .unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Auth);
    assert_eq!(
        tokens.invalidated.load(Ordering::SeqCst),
        1,
        "the source that answered was told"
    );
}

#[tokio::test]
async fn ops_an_openai_shaped_managed_backend_is_read_with_its_query() {
    let bed = Bed::with(|b| {
        b.managed(
            ManagedConfig::new("https://api.openhuman.test/openai/v1")
                .openai_shaped("catalog=openrouter"),
        )
    });
    bed.hub
        .set_key(&bed.scope, &slug("tinyhumans"), Secret::new("jwt"))
        .await
        .unwrap();
    bed.ports.http.route(
        Match::get("https://api.openhuman.test/openai/v1/models?catalog=openrouter"),
        Scripted::json(200, &models_body(&["x/y"])),
    );
    let list = bed
        .hub
        .list_models(&bed.scope, &slug("tinyhumans"), false)
        .await
        .unwrap();
    assert_eq!(list.ids(), ["x/y"]);
}

// ---- re-testing --------------------------------------------------------------------------

#[tokio::test]
async fn ops_retest_down_waits_then_clears_a_rejected_key_only_when_a_completion_passes() {
    let bed = Bed::new();
    bed.openai_rejects_key();
    bed.hub
        .connect(
            &bed.scope,
            bed.openai_draft(),
            ConnectOptions::default().add_anyway(true),
        )
        .await
        .unwrap();
    let s = slug("openai");
    assert_eq!(
        bed.hub.health(&bed.scope, &s).await.unwrap().health,
        ProviderHealth::Down(ReasonCode::Auth)
    );
    // Too soon: nothing runs.
    assert!(bed.hub.retest_down(&bed.scope).await.unwrap().is_empty());
    // Still rejected after the wait: refreshed, still down, and it waits again.
    bed.ports.clock.advance(Duration::from_secs(301));
    let first = bed.hub.retest_down(&bed.scope).await.unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].health, ProviderHealth::Down(ReasonCode::Auth));
    assert!(bed.hub.retest_down(&bed.scope).await.unwrap().is_empty());
    // The operator fixes the key upstream; the next scheduled re-test recovers.
    bed.openai_lists(&["m"]);
    bed.openai_chat_ok();
    bed.ports.clock.advance(Duration::from_secs(301));
    let second = bed.hub.retest_down(&bed.scope).await.unwrap();
    assert_eq!(second[0].health, ProviderHealth::Ok);
    assert!(second[0].failure.is_none());
    assert!(
        bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, HubEvent::Retested { ok: true, .. }))
    );
}

#[tokio::test]
async fn ops_retest_down_ignores_disabled_healthy_and_merely_degraded_providers() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let s = slug("openai");
    let failure = ProviderFailure::new(ReasonCode::Timeout, Retry::Later(None));
    for _ in 0..3 {
        bed.hub
            .record_outcome(&bed.scope, &s, Outcome::Failed(failure.clone()))
            .await
            .unwrap();
    }
    bed.ports.clock.advance(Duration::from_secs(3600));
    assert!(
        bed.hub.retest_down(&bed.scope).await.unwrap().is_empty(),
        "timeouts are not the re-test's job"
    );
    let auth = ProviderFailure::new(ReasonCode::Auth, Retry::Never);
    bed.hub
        .record_outcome(&bed.scope, &s, Outcome::Failed(auth))
        .await
        .unwrap();
    bed.ports.clock.advance(Duration::from_secs(3600));
    bed.hub
        .set_enabled(&bed.scope, &s, false, Confirm::in_use())
        .await
        .unwrap();
    assert!(
        bed.hub.retest_down(&bed.scope).await.unwrap().is_empty(),
        "disabled is skipped"
    );
    let _ = KEY;
}

#[tokio::test]
async fn ops_retest_down_skips_what_it_cannot_test() {
    let bed = Bed::new();
    // Three providers Down on a rejected key: one keyless now, one whose
    // credential store is unreadable, one whose kind offers no completion or
    // catalog depth (a stored CLI record).
    bed.ports.config.put_raw(
        &bed.scope,
        json!({"providers": [
            {"id": "a", "slug": "openai", "label": "OpenAI", "kind": "openai", "base_url": "https://api.openai.com/v1", "model": "m"},
            {"id": "b", "slug": "claude-code", "label": "Claude Code", "kind": "claude-code", "base_url": ""}
        ]})
        .to_string(),
    );
    let auth = ProviderFailure::new(ReasonCode::Auth, Retry::Never);
    for name in ["openai", "claude-code"] {
        bed.hub
            .record_outcome(&bed.scope, &slug(name), Outcome::Failed(auth.clone()))
            .await
            .unwrap();
    }
    bed.ports.clock.advance(Duration::from_secs(3600));
    // No key: skipped without a request.
    assert!(bed.hub.retest_down(&bed.scope).await.unwrap().is_empty());
    assert_eq!(bed.ports.http.request_count(), 0);
    // Unreadable credential store: skipped, not an error, not a keyless call.
    bed.store_key("openai", KEY).await;
    bed.ports
        .credentials
        .inject(crate::ports::memory::CredentialFault::Read);
    assert!(bed.hub.retest_down(&bed.scope).await.unwrap().is_empty());
    bed.ports.credentials.heal();
    // A probe that cannot run (the health store went away mid-way) is skipped too.
    bed.openai_lists(&["m"]);
    bed.openai_chat_ok();
    let done = bed.hub.retest_down(&bed.scope).await.unwrap();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].health, ProviderHealth::Ok);
}

#[tokio::test]
async fn ops_retest_uses_a_catalog_read_when_the_record_has_no_model() {
    let bed = Bed::new();
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_key(Secret::new(KEY)),
        )
        .await
        .unwrap();
    let auth = ProviderFailure::new(ReasonCode::Quota, Retry::Never);
    bed.hub
        .record_outcome(&bed.scope, &slug("openai"), Outcome::Failed(auth))
        .await
        .unwrap();
    bed.ports.clock.advance(Duration::from_secs(3600));
    bed.openai_lists(&["m"]);
    let done = bed.hub.retest_down(&bed.scope).await.unwrap();
    assert_eq!(done.len(), 1);
    assert_eq!(
        bed.ports.http.requests().last().unwrap().url,
        "https://api.openai.com/v1/models"
    );
    // A catalog pass cannot lift a quota failure recorded by a real turn: only
    // a completion can, so the provider is still down...
    assert!(
        matches!(done[0].health, ProviderHealth::Down(ReasonCode::Quota)),
        "{:?}",
        done[0].health
    );
    // ...but it is not probed again on every tick: the re-test stamped its time.
    let asked = bed.ports.http.request_count();
    assert!(bed.hub.retest_down(&bed.scope).await.unwrap().is_empty());
    bed.ports.clock.advance(Duration::from_secs(200));
    assert!(bed.hub.retest_down(&bed.scope).await.unwrap().is_empty());
    assert_eq!(
        bed.ports.http.request_count(),
        asked,
        "still inside retest_after"
    );
    bed.ports.clock.advance(Duration::from_secs(200));
    assert_eq!(
        bed.hub.retest_down(&bed.scope).await.unwrap().len(),
        1,
        "and again once it has elapsed"
    );
    // A key change forgets the stamp along with the health.
    bed.hub
        .set_key(&bed.scope, &slug("openai"), Secret::new("sk-new"))
        .await
        .unwrap();
    assert!(bed.inner_retests_empty());
}
