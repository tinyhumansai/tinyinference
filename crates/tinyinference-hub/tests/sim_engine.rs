//! Simulated end-to-end scenarios for the engine layer: ports, credential chain,
//! catalog cache, probes, health and kind drivers wired together the way the
//! `Hub` will wire them, over a scripted transport and a fake clock. No sockets,
//! no ports, no wall-clock time (08-test-plan, scenario catalogue).
#![cfg(feature = "testing")]

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::json;

use tinyinference_hub::catalog::{CatalogCache, CatalogKey, Freshness, ModelList};
use tinyinference_hub::credential::{
    CredentialChain, CredentialOrigin, StoreSource, TokenSourceAdapter,
};
use tinyinference_hub::error::{PolicyViolation, PortName};
use tinyinference_hub::health::{HealthTracker, ProviderHealth};
use tinyinference_hub::kinds::{DriverContext, DriverRegistry, Target};
use tinyinference_hub::policy::{EndpointPolicy, EndpointRefusal, HeaderPolicy};
use tinyinference_hub::ports::memory::{
    CredentialFault, MemoryCredentials, MemoryEvents, MemoryHealth,
};
use tinyinference_hub::ports::{Clock, CredentialStore, PortError, TokenSource};
use tinyinference_hub::probe::run_probe;
use tinyinference_hub::testkit::{FakeClock, Match, Script, Scripted, ScriptedHttp};
use tinyinference_hub::{
    HubError, KindId, ModelId, ProviderGroup, ReasonCode, Retry, ScopeKey, Secret, Slug, TestDepth,
};

struct Rig {
    clock: FakeClock,
    http: Arc<ScriptedHttp>,
    registry: DriverRegistry,
    cache: CatalogCache,
    tracker: HealthTracker,
    creds: Arc<MemoryCredentials>,
    events: Arc<MemoryEvents>,
    policy: EndpointPolicy,
    headers: HeaderPolicy,
    chains: HashMap<String, CredentialChain>,
}

impl Rig {
    fn new(policy: EndpointPolicy) -> Self {
        let clock = FakeClock::new();
        let events = Arc::new(MemoryEvents::new());
        Self {
            http: Arc::new(ScriptedHttp::new(clock.clone())),
            registry: DriverRegistry::with_builtin(),
            cache: CatalogCache::new(Arc::new(clock.clone())),
            tracker: HealthTracker::new(
                Arc::new(MemoryHealth::new()),
                Arc::new(clock.clone()),
                events.clone(),
            ),
            creds: Arc::new(MemoryCredentials::new()),
            events,
            policy,
            headers: HeaderPolicy::builtin(),
            chains: HashMap::new(),
            clock,
        }
    }

    fn hosted() -> Self {
        Self::new(EndpointPolicy::hosted())
    }

    /// The chain a plain provider gets: its own key.
    fn default_chain(&self) -> CredentialChain {
        CredentialChain::new().with(StoreSource::provider_key(self.creds.clone()))
    }

    /// The prototype of `Hub::list_models`: chain, driver, cache, health.
    async fn list(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        kind: &str,
        base: &str,
        refresh: bool,
    ) -> Result<ModelList, HubError> {
        let driver = self.registry.resolve(kind).expect("a driver for the kind");
        let descriptor = driver.descriptor();
        let default_chain;
        let chain = match self.chains.get(slug.as_str()) {
            Some(chain) => chain,
            None => {
                default_chain = self.default_chain();
                &default_chain
            }
        };
        let resolved = chain.resolve(scope, slug).await?;
        let key = resolved.map(|(secret, _origin)| secret);
        if descriptor.group == ProviderGroup::Managed && key.is_none() {
            self.tracker.mark_signed_out(scope, slug).await?;
            return Err(HubError::SignedOut {
                provider: slug.clone(),
            });
        }
        let credentialed = key.is_some() && descriptor.auth.needs_credential();
        let kind_id = KindId::new(kind);
        let target = Target::new(slug, &kind_id, descriptor.group, base, &descriptor.auth);
        let target = match key.as_ref() {
            Some(key) => target.with_credential(key),
            None => target,
        };
        let cx = DriverContext::new(&*self.http, &self.policy, &self.clock, &self.headers);
        let cache_key = CatalogKey::new(scope, slug, credentialed, base, descriptor.catalog);
        self.cache
            .read(cache_key, refresh, || async {
                driver.list_models(&cx, &target).await
            })
            .await
    }

    async fn store_key(&self, scope: &ScopeKey, slug: &Slug, key: &str) {
        self.creds
            .set(scope, &slug.key_slot(), Secret::new(key))
            .await
            .unwrap();
    }
}

fn scope(name: &str) -> ScopeKey {
    ScopeKey::new(name)
}

fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

fn models(ids: &[&str]) -> serde_json::Value {
    json!({"data": ids.iter().map(|i| json!({"id": i})).collect::<Vec<_>>()})
}

const OPENAI: &str = "https://api.openai.com/v1";

fn ids(list: &ModelList) -> Vec<&str> {
    list.ids()
}

#[tokio::test]
async fn sim_refresh_bypass_serves_cached_refetches_on_demand_and_falls_back_stale_on_error() {
    let rig = Rig::hosted();
    let (s, p) = (scope("company:acme"), slug("openai"));
    rig.store_key(&s, &p, "sk-not-a-real-key").await;
    rig.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &models(&["m1"])),
    );
    assert_eq!(
        rig.list(&s, &p, "openai", OPENAI, false)
            .await
            .unwrap()
            .freshness,
        Freshness::Fresh
    );
    rig.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &models(&["m2"])),
    );
    let cached = rig.list(&s, &p, "openai", OPENAI, false).await.unwrap();
    assert_eq!(
        (cached.freshness.clone(), ids(&cached)),
        (Freshness::Cached, vec!["m1"])
    );
    let refreshed = rig.list(&s, &p, "openai", OPENAI, true).await.unwrap();
    assert_eq!(
        (refreshed.freshness.clone(), ids(&refreshed)),
        (Freshness::Fresh, vec!["m2"])
    );

    // The provider goes down: the Refresh button gets the old list and a typed warning.
    rig.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(503, "service unavailable"),
    );
    let stale = rig.list(&s, &p, "openai", OPENAI, true).await.unwrap();
    assert_eq!(ids(&stale), ["m2"]);
    match &stale.freshness {
        Freshness::Stale { failure } => assert_eq!(failure.status, Some(503)),
        other => panic!("{other:?}"),
    }
    // Within the hour the entry is still fresh, so ordinary reads are cached.
    let cached_again = rig.list(&s, &p, "openai", OPENAI, false).await.unwrap();
    assert_eq!(cached_again.freshness, Freshness::Cached);
    assert_eq!(ids(&cached_again), ["m2"]);
    // Once it expires the provider is asked once; the failure is remembered for a minute.
    rig.clock.advance(Duration::from_secs(3601));
    let after_expiry = rig.list(&s, &p, "openai", OPENAI, false).await.unwrap();
    assert!(after_expiry.is_stale() && ids(&after_expiry) == ["m2"]);
    let requests_after_expiry = rig.http.request_count();
    assert!(
        rig.list(&s, &p, "openai", OPENAI, false)
            .await
            .unwrap()
            .is_stale()
    );
    assert_eq!(
        rig.http.request_count(),
        requests_after_expiry,
        "the failure memo answers for a minute"
    );
    // A minute later it tries again, and when the provider is back the list is fresh.
    rig.clock.advance(Duration::from_secs(61));
    rig.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &models(&["m3"])),
    );
    let back = rig.list(&s, &p, "openai", OPENAI, false).await.unwrap();
    assert_eq!(
        (back.freshness.clone(), ids(&back)),
        (Freshness::Fresh, vec!["m3"])
    );
}

#[tokio::test]
async fn sim_tenant_isolation_one_tenants_list_failure_and_health_are_invisible_to_another() {
    let rig = Rig::hosted();
    let (a, b, p) = (scope("company:a"), scope("company:b"), slug("openai"));
    rig.store_key(&a, &p, "sk-tenant-a").await;
    rig.store_key(&b, &p, "sk-tenant-b").await;
    rig.http.route(
        Match::get("https://api.openai.com/v1/models")
            .with_header("authorization", "Bearer sk-tenant-a"),
        Scripted::json(200, &models(&["entitled-to-a"])),
    );
    rig.http.route(
        Match::get("https://api.openai.com/v1/models")
            .with_header("authorization", "Bearer sk-tenant-b"),
        Scripted::json(200, &models(&["entitled-to-b"])),
    );
    let la = rig.list(&a, &p, "openai", OPENAI, false).await.unwrap();
    let lb = rig.list(&b, &p, "openai", OPENAI, false).await.unwrap();
    assert_eq!(
        (ids(&la), ids(&lb)),
        (vec!["entitled-to-a"], vec!["entitled-to-b"])
    );

    // A's key is revoked upstream: A sees the rejection, B never does.
    rig.http.route(
        Match::get("https://api.openai.com/v1/models")
            .with_header("authorization", "Bearer sk-tenant-a"),
        Scripted::text(401, "Incorrect API key provided"),
    );
    let error = rig.list(&a, &p, "openai", OPENAI, true).await.unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Auth);
    rig.tracker
        .record_outcome(
            &a,
            &p,
            &tinyinference_hub::health::Outcome::Failed(match error {
                HubError::Provider(f) => f,
                other => panic!("{other:?}"),
            }),
        )
        .await
        .unwrap();
    assert_eq!(
        rig.tracker.health(&a, &p).await.unwrap(),
        ProviderHealth::Down(ReasonCode::Auth)
    );
    assert_eq!(
        rig.tracker.health(&b, &p).await.unwrap(),
        ProviderHealth::Unknown
    );
    let still_b = rig.list(&b, &p, "openai", OPENAI, false).await.unwrap();
    assert_eq!(ids(&still_b), ["entitled-to-b"]);
    assert_eq!(still_b.freshness, Freshness::Cached);
}

#[tokio::test]
async fn sim_key_rotation_next_call_uses_the_new_key_once_the_scope_is_evicted() {
    let rig = Rig::hosted();
    let (s, p) = (scope("company:acme"), slug("openai"));
    rig.store_key(&s, &p, "sk-old").await;
    rig.http.route(
        Match::get("https://api.openai.com/v1/models")
            .with_header("authorization", "Bearer sk-old"),
        Scripted::json(200, &models(&["old-entitlement"])),
    );
    rig.http.route(
        Match::get("https://api.openai.com/v1/models")
            .with_header("authorization", "Bearer sk-new"),
        Scripted::json(200, &models(&["new-entitlement"])),
    );
    rig.list(&s, &p, "openai", OPENAI, false).await.unwrap();
    rig.tracker
        .record_outcome(
            &s,
            &p,
            &tinyinference_hub::health::Outcome::Ok {
                latency: Duration::ZERO,
            },
        )
        .await
        .unwrap();

    // Rotating the key without evicting is the bug the eviction exists for.
    rig.store_key(&s, &p, "sk-new").await;
    let stale = rig.list(&s, &p, "openai", OPENAI, false).await.unwrap();
    assert_eq!(
        ids(&stale),
        ["old-entitlement"],
        "without eviction the old key's list is served"
    );

    // With eviction and a health reset, the next call presents the new key.
    rig.cache.evict_scope(&s);
    rig.tracker.forget(&s, &p).await.unwrap();
    let fresh = rig.list(&s, &p, "openai", OPENAI, false).await.unwrap();
    assert_eq!(ids(&fresh), ["new-entitlement"]);
    let last = rig.http.requests().pop().unwrap();
    assert!(last.carried(&Secret::new("sk-new")) && !last.carried(&Secret::new("sk-old")));
    assert_eq!(
        rig.tracker.health(&s, &p).await.unwrap(),
        ProviderHealth::Unknown
    );
}

#[derive(Debug)]
struct MinuteToken {
    clock: FakeClock,
}

#[async_trait]
impl TokenSource for MinuteToken {
    async fn token(&self, _scope: &ScopeKey) -> Result<Option<Secret>, PortError> {
        let minute = (self.clock.wall_ms() - FakeClock::START_WALL_MS) / 60_000;
        Ok(Some(Secret::new(format!("platform-token-{minute}"))))
    }
}

#[tokio::test]
async fn sim_platform_token_rotation_no_stale_token_is_ever_sent() {
    let mut rig = Rig::hosted();
    let (s, p) = (scope("company:acme"), slug("tinyhumans"));
    let base = "https://api.example.test/agent-integrations/openrouter";
    rig.chains.insert(
        "tinyhumans".into(),
        CredentialChain::new().with(TokenSourceAdapter::new(
            Arc::new(MinuteToken {
                clock: rig.clock.clone(),
            }),
            CredentialOrigin::InstanceIdentity,
        )),
    );
    rig.http.route(
        Match::prefix("https://api.example.test/agent-integrations/openrouter/models"),
        Scripted::json(200, &json!({"success": true, "data": {"object": "list", "total": 1, "data": [{"id": "x/y"}]}})),
    );
    for minute in 0..5u64 {
        rig.list(&s, &p, "tinyhumans", base, true).await.unwrap();
        let sent = rig.http.requests().pop().unwrap();
        assert!(
            sent.carried(&Secret::new(format!("platform-token-{minute}"))),
            "minute {minute}"
        );
        assert!(
            !minute
                .checked_sub(1)
                .is_some_and(|m| sent.carried(&Secret::new(format!("platform-token-{m}"))))
        );
        rig.clock.advance(Duration::from_secs(60));
    }
}

#[tokio::test]
async fn sim_signed_out_is_typed_and_shows_as_signed_out_never_empty_or_red() {
    let mut rig = Rig::hosted();
    let (s, p) = (scope("company:acme"), slug("tinyhumans"));
    rig.chains.insert(
        "tinyhumans".into(),
        CredentialChain::new().with(StoreSource::provider_key(rig.creds.clone())),
    );
    let error = rig
        .list(
            &s,
            &p,
            "tinyhumans",
            "https://api.example.test/agent-integrations/openrouter",
            false,
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::SignedOut { .. }), "{error:?}");
    assert_eq!(
        rig.tracker.health(&s, &p).await.unwrap(),
        ProviderHealth::SignedOut
    );
    assert_eq!(
        rig.http.request_count(),
        0,
        "nothing is requested while signed out"
    );
    // Signing in recovers on the next call.
    rig.creds
        .set(&s, &p.key_slot(), Secret::new("th-signed-in"))
        .await
        .unwrap();
    rig.http.route(
        Match::prefix("https://api.example.test/agent-integrations/openrouter/models"),
        Scripted::json(
            200,
            &json!({"success": true, "data": {"total": 1, "data": [{"id": "m"}]}}),
        ),
    );
    assert!(
        rig.list(
            &s,
            &p,
            "tinyhumans",
            "https://api.example.test/agent-integrations/openrouter",
            false
        )
        .await
        .is_ok()
    );
}

#[tokio::test]
async fn sim_managed_origin_switch_reports_which_source_answered() {
    let rig = Rig::hosted();
    let (s, p) = (scope("company:acme"), slug("tinyhumans"));
    let chain = CredentialChain::new()
        .with(StoreSource::provider_key(rig.creds.clone()))
        .with(StoreSource::fixed_slot(
            rig.creds.clone(),
            "company/account-key",
            CredentialOrigin::AccountKey,
        ))
        .with(TokenSourceAdapter::new(
            Arc::new(MinuteToken {
                clock: rig.clock.clone(),
            }),
            CredentialOrigin::InstanceIdentity,
        ));
    let origin = || async { chain.resolve(&s, &p).await.unwrap().unwrap().1 };
    assert_eq!(origin().await.to_string(), "Instance identity");
    rig.creds
        .set(&s, "company/account-key", Secret::new("acct"))
        .await
        .unwrap();
    assert_eq!(origin().await.to_string(), "Using company key");
    rig.store_key(&s, &p, "pasted").await;
    assert_eq!(origin().await.to_string(), "Using the key you added");
    rig.creds.delete(&s, &p.key_slot()).await.unwrap();
    rig.creds.delete(&s, "company/account-key").await.unwrap();
    assert_eq!(origin().await.to_string(), "Instance identity");
}

#[tokio::test]
async fn sim_store_unreadable_is_typed_and_never_falls_back_to_the_managed_account() {
    let mut rig = Rig::hosted();
    let (s, p) = (scope("company:acme"), slug("tinyhumans"));
    let base = "https://api.example.test/agent-integrations/openrouter";
    rig.chains.insert(
        "tinyhumans".into(),
        CredentialChain::new()
            .with(StoreSource::provider_key(rig.creds.clone()))
            .with(TokenSourceAdapter::new(
                Arc::new(MinuteToken {
                    clock: rig.clock.clone(),
                }),
                CredentialOrigin::InstanceIdentity,
            )),
    );
    rig.creds.inject(CredentialFault::Read);
    let error = rig
        .list(&s, &p, "tinyhumans", base, false)
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
    assert_eq!(
        rig.http.request_count(),
        0,
        "the instance identity was never used"
    );
    rig.creds.heal();
}

#[tokio::test]
async fn sim_quota_vs_rate_a_spend_cap_is_never_retried_a_rate_limit_is() {
    let rig = Rig::hosted();
    let (s, anthropic, openai) = (scope("c"), slug("anthropic"), slug("openai"));
    rig.store_key(&s, &anthropic, "sk-ant-fake").await;
    rig.store_key(&s, &openai, "sk-not-a-real-key").await;
    rig.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Scripted::json(429, &json!({"type": "error", "error": {"type": "rate_limit_error",
            "message": "You have reached your specified API usage limits. You will regain access on 2026-10-01 at 00:00 UTC."}})),
    );
    rig.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(429, &json!({"error": {"message": "Rate limit reached for gpt-5 on tokens per min (TPM): Limit 30000. Please try again in 1.2s.",
            "type": "tokens", "code": "rate_limit_exceeded"}})),
    );
    let quota = rig
        .list(
            &s,
            &anthropic,
            "anthropic",
            "https://api.anthropic.com/v1",
            false,
        )
        .await
        .unwrap_err();
    match quota {
        HubError::Provider(f) => assert_eq!((f.reason, f.retry), (ReasonCode::Quota, Retry::Never)),
        other => panic!("{other:?}"),
    }
    let rate = rig
        .list(&s, &openai, "openai", OPENAI, false)
        .await
        .unwrap_err();
    match rate {
        HubError::Provider(f) => {
            assert_eq!(f.reason, ReasonCode::RateLimited);
            assert!(
                matches!(f.retry, Retry::Later(Some(d)) if d >= Duration::from_secs(1)),
                "{:?}",
                f.retry
            );
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn sim_malformed_catalogs_are_tolerated_per_row_and_refused_as_a_whole_only_when_they_must_be()
 {
    let rig = Rig::hosted();
    let (s, p) = (scope("c"), slug("custom-one"));
    let base = "https://gateway.acme.test/v1";
    // Rows missing ids, wrong types, a duplicate.
    rig.http.route(
        Match::prefix("https://gateway.acme.test/v1/models"),
        Scripted::json(200, &json!({"data": [{"id": "ok-1"}, {"id": 5}, {"name": "named-only"}, null, {"id": "ok-1"}, {"id": "ok-2", "context_length": "many"}]})),
    );
    assert_eq!(
        ids(&rig.list(&s, &p, "custom", base, false).await.unwrap()),
        ["ok-1", "named-only", "ok-2"]
    );
    // data:null on a success envelope is an empty catalog, not an error.
    rig.http.route(
        Match::prefix("https://gateway.acme.test/v1/models"),
        Scripted::text(200, r#"{"object":"list","data":null}"#),
    );
    assert!(
        rig.list(&s, &p, "custom", base, true)
            .await
            .unwrap()
            .models
            .is_empty()
    );
    // A body past the cap is refused, never truncated into invalid JSON.
    rig.http.route(
        Match::prefix("https://gateway.acme.test/v1/models"),
        Scripted::Oversize { bytes: 500_000_000 },
    );
    match rig.list(&s, &p, "custom", base, true).await {
        Err(HubError::Provider(f)) => assert!(f.truncated && f.reason == ReasonCode::Unknown),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn sim_paged_catalog_three_pages_then_an_oversize_page_then_the_twenty_page_cap() {
    let mut rig = Rig::hosted();
    let (s, p) = (scope("c"), slug("tinyhumans"));
    let base = "https://api.example.test/agent-integrations/openrouter";
    rig.chains.insert(
        "tinyhumans".into(),
        CredentialChain::new().with(tinyinference_hub::credential::StaticSource::new(
            Secret::new("th-fake"),
        )),
    );
    let page = |ids: &[&str], total: usize| {
        Scripted::json(
            200,
            &json!({"success": true, "data": {"total": total, "data": ids.iter().map(|i| json!({"id": i, "pricing": {"inputPer1M": 1.0, "outputPer1M": 2.0}})).collect::<Vec<_>>()}}),
        )
    };
    rig.http.route(
        Match::get(format!("{base}/models?limit=500&offset=0")),
        page(&["a", "b"], 5),
    );
    rig.http.route(
        Match::get(format!("{base}/models?limit=500&offset=2")),
        page(&["c", "d"], 5),
    );
    rig.http.route(
        Match::get(format!("{base}/models?limit=500&offset=4")),
        page(&["e"], 5),
    );
    let list = rig.list(&s, &p, "tinyhumans", base, false).await.unwrap();
    assert_eq!(ids(&list), ["a", "b", "c", "d", "e"]);
    assert_eq!(list.models[0].input_per_1m, Some(1.0));

    // Page 2 arrives oversize: the whole read is refused as unknown+truncated,
    // and the older list is served stale on refresh.
    rig.http.route(
        Match::get(format!("{base}/models?limit=500&offset=2")),
        Scripted::Oversize { bytes: 99_999_999 },
    );
    let stale = rig.list(&s, &p, "tinyhumans", base, true).await.unwrap();
    assert!(stale.is_stale());
    assert_eq!(ids(&stale), ["a", "b", "c", "d", "e"]);

    // A total that is never reached stops at twenty pages and says so.
    let rig2 = {
        let mut r = Rig::hosted();
        r.chains.insert(
            "tinyhumans".into(),
            CredentialChain::new().with(tinyinference_hub::credential::StaticSource::new(
                Secret::new("th-fake"),
            )),
        );
        r
    };
    let pages: Vec<_> = (0..30)
        .map(|n| page(&[format!("m{n}").as_str()], 1_000_000))
        .collect();
    rig2.http.route(
        Match::prefix(format!("{base}/models")),
        Script::Sequence(pages),
    );
    let capped = rig2.list(&s, &p, "tinyhumans", base, false).await.unwrap();
    assert!(capped.truncated && capped.models.len() == 20);
}

#[tokio::test]
async fn sim_ssrf_literal_ips_every_refused_range_as_a_base_url_and_as_a_redirect_target() {
    let rig = Rig::hosted();
    let custom = rig.registry.resolve("custom").unwrap();
    let cases: &[(&str, EndpointRefusal)] = &[
        (
            "http://169.254.169.254/latest/meta-data/",
            EndpointRefusal::LinkLocal,
        ),
        ("http://[fe80::1]/x", EndpointRefusal::LinkLocal),
        ("http://[fec0::1]/x", EndpointRefusal::LinkLocal),
        ("http://10.0.0.1/x", EndpointRefusal::PrivateNetwork),
        ("http://172.16.0.1/x", EndpointRefusal::PrivateNetwork),
        ("http://192.168.1.1/x", EndpointRefusal::PrivateNetwork),
        ("http://100.64.0.1/x", EndpointRefusal::PrivateNetwork),
        ("http://[fc00::1]/x", EndpointRefusal::PrivateNetwork),
        ("http://0.0.0.0/x", EndpointRefusal::PrivateNetwork),
        ("http://[::]/x", EndpointRefusal::PrivateNetwork),
        ("http://224.0.0.1/x", EndpointRefusal::PrivateNetwork),
        ("http://255.255.255.255/x", EndpointRefusal::PrivateNetwork),
        (
            "http://[::ffff:10.0.0.1]/x",
            EndpointRefusal::PrivateNetwork,
        ),
        (
            "http://[::ffff:169.254.169.254]/x",
            EndpointRefusal::LinkLocal,
        ),
        ("http://127.0.0.1:8080/x", EndpointRefusal::Loopback),
        ("http://localhost:11434/v1", EndpointRefusal::Loopback),
        ("http://2130706433/x", EndpointRefusal::Loopback),
        ("ftp://a.test/x", EndpointRefusal::Scheme),
    ];
    let (slug_, kind) = (slug("custom"), KindId::new("custom"));
    for (base, expected) in cases {
        let target = Target::new(
            &slug_,
            &kind,
            ProviderGroup::Custom,
            base,
            &tinyinference_hub::AuthStyle::Bearer,
        );
        let cx = DriverContext::new(&*rig.http, &rig.policy, &rig.clock, &rig.headers);
        let report = run_probe(&cx, custom.as_ref(), &target, TestDepth::Catalog)
            .await
            .unwrap();
        assert_eq!(
            report.refusal,
            Some(PolicyViolation::Endpoint(*expected)),
            "as a base url: {base}"
        );
    }
    assert_eq!(
        rig.http.request_count(),
        0,
        "not one refused address was requested"
    );

    // The same table as redirect targets from a public host.
    for (base, expected) in cases.iter().filter(|(b, _)| !b.starts_with("ftp")) {
        let rig = Rig::hosted();
        rig.http.route(
            Match::prefix("https://public.acme.test/v1/models"),
            Scripted::redirect(302, *base),
        );
        let target = Target::new(
            &slug_,
            &kind,
            ProviderGroup::Custom,
            "https://public.acme.test/v1",
            &tinyinference_hub::AuthStyle::Bearer,
        );
        let cx = DriverContext::new(&*rig.http, &rig.policy, &rig.clock, &rig.headers);
        let report = run_probe(&cx, custom.as_ref(), &target, TestDepth::Catalog)
            .await
            .unwrap();
        assert_eq!(
            report.refusal,
            Some(PolicyViolation::Endpoint(*expected)),
            "as a redirect target: {base}"
        );
        assert_eq!(
            rig.http.request_count(),
            1,
            "only the public host was requested: {base}"
        );
    }
}

#[tokio::test]
async fn sim_ssrf_redirect_chain_three_hops_are_followed_and_the_fourth_is_refused() {
    let rig = Rig::hosted();
    let custom = rig.registry.resolve("custom").unwrap();
    for n in 0..6 {
        rig.http.route(
            Match::prefix(format!("https://a.test/hop{n}")),
            Scripted::redirect(301, format!("https://a.test/hop{}", n + 1)),
        );
    }
    rig.http.route(
        Match::prefix("https://a.test/hop3"),
        Scripted::json(200, &models(&["reached"])),
    );
    let (slug_, kind) = (slug("custom"), KindId::new("custom"));
    let target = |base: &'static str| {
        Target::new(
            &slug_,
            &kind,
            ProviderGroup::Custom,
            base,
            &tinyinference_hub::AuthStyle::Bearer,
        )
    };
    let cx = DriverContext::new(&*rig.http, &rig.policy, &rig.clock, &rig.headers);
    // `/hop0/models` redirects to `/hop1` ... `/hop3` answers: three hops.
    rig.http.route(
        Match::prefix("https://a.test/hop0/models"),
        Scripted::redirect(301, "https://a.test/hop1"),
    );
    let ok = run_probe(
        &cx,
        custom.as_ref(),
        &target("https://a.test/hop0"),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert!(ok.ok(), "{:?}", ok.failure);
    // One more hop is one too many.
    rig.http.route(
        Match::prefix("https://a.test/hop3"),
        Scripted::redirect(301, "https://a.test/hop4"),
    );
    let refused = run_probe(
        &cx,
        custom.as_ref(),
        &target("https://a.test/hop0"),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert_eq!(
        refused.refusal,
        Some(PolicyViolation::TooManyRedirects { max: 3 })
    );
}

#[tokio::test]
async fn sim_cleartext_credential_only_loopback_may_receive_a_key_over_http() {
    let rig = Rig::new(EndpointPolicy::desktop());
    rig.http.route(
        Match::prefix("http://localhost:8000/v1/models"),
        Scripted::json(200, &models(&["local"])),
    );
    let (slug_, kind, key) = (
        slug("omlx"),
        KindId::new("omlx"),
        Secret::new("sk-not-a-real-key"),
    );
    let omlx = rig.registry.resolve("omlx").unwrap();
    let target = |base: &'static str| {
        Target::new(
            &slug_,
            &kind,
            ProviderGroup::Local,
            base,
            &tinyinference_hub::AuthStyle::Bearer,
        )
        .with_credential(&key)
    };
    let cx = DriverContext::new(&*rig.http, &rig.policy, &rig.clock, &rig.headers);
    let local = run_probe(
        &cx,
        omlx.as_ref(),
        &target("http://localhost:8000/v1"),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert!(
        local.ok(),
        "a key over http to this host's own loopback is fine"
    );
    for base in ["http://api.acme.test/v1", "http://sub.localhost:8000/v1"] {
        let refused = run_probe(&cx, omlx.as_ref(), &target(base), TestDepth::Catalog)
            .await
            .unwrap();
        assert!(
            matches!(
                refused.refusal,
                Some(PolicyViolation::Endpoint(EndpointRefusal::Cleartext))
            ),
            "{base}: {:?}",
            refused.refusal
        );
    }
    // On a hosted tenant loopback is refused outright.
    let hosted = Rig::hosted();
    let cx = DriverContext::new(
        &*hosted.http,
        &hosted.policy,
        &hosted.clock,
        &hosted.headers,
    );
    let refused = run_probe(
        &cx,
        omlx.as_ref(),
        &target("http://localhost:8000/v1"),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert!(matches!(
        refused.refusal,
        Some(PolicyViolation::Endpoint(EndpointRefusal::Loopback))
    ));
}

#[tokio::test]
async fn sim_offline_local_only_cloud_is_refused_and_a_local_runtime_still_works() {
    let rig = Rig::new(EndpointPolicy::local_only());
    rig.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::json(200, &models(&["llama3"])),
    );
    let cx = DriverContext::new(&*rig.http, &rig.policy, &rig.clock, &rig.headers);
    let (openai_slug, openai_kind, key) = (
        slug("openai"),
        KindId::new("openai"),
        Secret::new("sk-not-a-real-key"),
    );
    let openai = rig.registry.resolve("openai").unwrap();
    let cloud = Target::new(
        &openai_slug,
        &openai_kind,
        ProviderGroup::Cloud,
        OPENAI,
        &tinyinference_hub::AuthStyle::Bearer,
    )
    .with_credential(&key);
    let refused = run_probe(&cx, openai.as_ref(), &cloud, TestDepth::Catalog)
        .await
        .unwrap();
    assert_eq!(
        refused.refusal,
        Some(PolicyViolation::Endpoint(EndpointRefusal::NonLocal))
    );
    let (ollama_slug, ollama_kind) = (slug("ollama"), KindId::new("ollama"));
    let ollama = rig.registry.resolve("ollama").unwrap();
    let local = Target::new(
        &ollama_slug,
        &ollama_kind,
        ProviderGroup::Local,
        "http://localhost:11434/v1",
        &tinyinference_hub::AuthStyle::None,
    );
    assert!(
        run_probe(&cx, ollama.as_ref(), &local, TestDepth::Catalog)
            .await
            .unwrap()
            .ok()
    );
    assert_eq!(rig.http.request_count(), 1, "the cloud row sent nothing");
}

#[tokio::test]
async fn sim_dns_rebinding_scripted_a_host_that_turns_private_on_the_second_lookup_is_refused() {
    let rig = Rig::hosted();
    let custom = rig.registry.resolve("custom").unwrap();
    let public: std::net::IpAddr = "93.184.216.34".parse().unwrap();
    let private: std::net::IpAddr = "10.9.8.7".parse().unwrap();
    rig.http.route(
        Match::prefix("https://rebind.acme.test/v1/models"),
        Script::Sequence(vec![
            Scripted::json(200, &models(&["fine"])).resolving_to(vec![public]),
            Scripted::json(200, &models(&["internal-secrets"])).resolving_to(vec![public, private]),
        ]),
    );
    let (slug_, kind) = (slug("custom"), KindId::new("custom"));
    let target = Target::new(
        &slug_,
        &kind,
        ProviderGroup::Custom,
        "https://rebind.acme.test/v1",
        &tinyinference_hub::AuthStyle::Bearer,
    );
    let cx = DriverContext::new(&*rig.http, &rig.policy, &rig.clock, &rig.headers);
    assert!(
        run_probe(&cx, custom.as_ref(), &target, TestDepth::Catalog)
            .await
            .unwrap()
            .ok()
    );
    let second = run_probe(&cx, custom.as_ref(), &target, TestDepth::Catalog)
        .await
        .unwrap();
    assert_eq!(
        second.refusal,
        Some(PolicyViolation::Endpoint(EndpointRefusal::PrivateNetwork))
    );
    assert!(second.models.is_empty());
}

#[tokio::test]
async fn sim_partial_outage_catalog_down_completion_up_is_degraded_and_recovers() {
    let rig = Rig::hosted();
    let (s, p) = (scope("c"), slug("openai"));
    let openai = rig.registry.resolve("openai").unwrap();
    let key = Secret::new("sk-not-a-real-key");
    let model = ModelId::parse("gpt-5").unwrap();
    let kind = KindId::new("openai");
    let target = Target::new(
        &p,
        &kind,
        ProviderGroup::Cloud,
        OPENAI,
        &tinyinference_hub::AuthStyle::Bearer,
    )
    .with_credential(&key)
    .with_model(&model);
    let cx = DriverContext::new(&*rig.http, &rig.policy, &rig.clock, &rig.headers);
    rig.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(503, "upstream unavailable"),
    );
    rig.http.route(
        Match::post("https://api.openai.com/v1/chat/completions"),
        Scripted::json(200, &json!({"choices": []})),
    );
    for depth in [TestDepth::Catalog, TestDepth::Completion] {
        let report = run_probe(&cx, openai.as_ref(), &target, depth)
            .await
            .unwrap();
        rig.tracker.record_probe(&s, &p, &report).await.unwrap();
    }
    assert!(
        matches!(
            rig.tracker.health(&s, &p).await.unwrap(),
            ProviderHealth::Degraded(_)
        ),
        "not down"
    );
    // The listing comes back.
    rig.clock.advance(Duration::from_secs(30));
    rig.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &models(&["gpt-5"])),
    );
    let report = run_probe(&cx, openai.as_ref(), &target, TestDepth::Catalog)
        .await
        .unwrap();
    assert_eq!(
        rig.tracker.record_probe(&s, &p, &report).await.unwrap(),
        ProviderHealth::Ok
    );
    assert_eq!(
        rig.events.events().len(),
        2,
        "Unknown to Degraded, Degraded to Ok"
    );
}

#[tokio::test]
async fn sim_slow_stream_completion_times_out_at_the_cap_on_the_fake_clock() {
    let rig = Rig::hosted();
    let openai = rig.registry.resolve("openai").unwrap();
    rig.http.route(
        Match::post("https://api.openai.com/v1/chat/completions"),
        Scripted::SlowStream {
            chunks: vec![b"{".to_vec(); 20],
            gap: Duration::from_secs(1),
        },
    );
    let (slug_, kind, key, model) = (
        slug("openai"),
        KindId::new("openai"),
        Secret::new("sk"),
        ModelId::parse("m").unwrap(),
    );
    let target = Target::new(
        &slug_,
        &kind,
        ProviderGroup::Cloud,
        OPENAI,
        &tinyinference_hub::AuthStyle::Bearer,
    )
    .with_credential(&key)
    .with_model(&model);
    let cx = DriverContext::new(&*rig.http, &rig.policy, &rig.clock, &rig.headers);
    let before = rig.clock.now();
    let report = run_probe(&cx, openai.as_ref(), &target, TestDepth::Completion)
        .await
        .unwrap();
    assert_eq!(
        report.failure.as_ref().map(|f| f.reason),
        Some(ReasonCode::Timeout)
    );
    assert_eq!(
        rig.clock.now() - before,
        rig.policy.timeout,
        "exactly the timeout passed, and no real time"
    );
    assert!(!report.failure.unwrap().rolls_back(ProviderGroup::Cloud));
}
