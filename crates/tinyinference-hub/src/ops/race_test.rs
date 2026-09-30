//! Tests for writers that interleave: a change lands between another
//! operation's guard check and its save.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::json;

use crate::config::{HubConfig, ProviderDraft};
use crate::error::HubError;
use crate::hub::fixtures::{KEY, model, scope, slug};
use crate::hub::{Confirm, ConnectOptions};
use crate::ids::ScopeKey;
use crate::ports::memory::MemoryConfig;
use crate::ports::{ConfigStore, CredentialStore, PortError, Version};
use crate::secret::Secret;
use crate::testkit::MemoryPorts;

type Hook = Box<dyn FnOnce(&MemoryConfig) + Send>;

/// A configuration store that lets "another writer" change the document just
/// before this writer's first save, which then loses its compare-and-swap.
struct Racing {
    inner: Arc<MemoryConfig>,
    hook: Mutex<Option<Hook>>,
}

impl std::fmt::Debug for Racing {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Racing")
    }
}

#[async_trait]
impl ConfigStore for Racing {
    async fn load(&self, scope: &ScopeKey) -> Result<Option<(HubConfig, Version)>, PortError> {
        self.inner.load(scope).await
    }

    async fn save(
        &self,
        scope: &ScopeKey,
        config: &HubConfig,
        expect: Option<Version>,
    ) -> Result<Version, PortError> {
        if let Some(hook) = self.hook.lock().unwrap().take() {
            hook(&self.inner);
        }
        self.inner.save(scope, config, expect).await
    }
}

fn edit_doc(store: &MemoryConfig, scope: &ScopeKey, change: impl FnOnce(&mut serde_json::Value)) {
    let mut doc: serde_json::Value = serde_json::from_str(&store.raw(scope).unwrap()).unwrap();
    change(&mut doc);
    store.put_raw(scope, doc.to_string());
}

async fn rig(hook: Hook) -> (MemoryPorts, crate::hub::Hub, ScopeKey) {
    let ports = MemoryPorts::new();
    let racing = Racing {
        inner: ports.config.clone(),
        hook: Mutex::new(None),
    };
    let racing = Arc::new(racing);
    let hub = ports.builder().config_arc(racing.clone()).build().unwrap();
    let me = scope("company:acme");
    for (kind, key) in [("openai", KEY), ("groq", "gsk-fake")] {
        hub.add(
            &me,
            ProviderDraft::new(kind)
                .with_key(Secret::new(key))
                .with_model(model("m")),
        )
        .await
        .unwrap();
    }
    hub.clear_default(&me).await.unwrap();
    *racing.hook.lock().unwrap() = Some(hook);
    (ports, hub, me)
}

#[tokio::test]
async fn guard_g6_a_pin_added_while_a_removal_runs_is_seen_on_the_retry() {
    let me = scope("company:acme");
    let target = me.clone();
    let (ports, hub, me) = rig(Box::new(move |store| {
        edit_doc(store, &target, |doc| {
            doc["agent_pins"] = json!({"agent:late": {"provider": "groq", "model": "m"}});
        });
    }))
    .await;
    let error = hub
        .remove(&me, &slug("groq"), Confirm::no())
        .await
        .unwrap_err();
    assert!(
        matches!(&error, HubError::InUse(u) if u.agents.len() == 1),
        "the guard was re-run against the version that was actually there: {error:?}"
    );
    // Nothing was removed and the key is back.
    assert_eq!(hub.status(&me).await.unwrap().providers.len(), 2);
    let key = ports
        .credentials
        .get(&me, &slug("groq").key_slot())
        .await
        .unwrap();
    assert_eq!(key.unwrap().expose(), "gsk-fake");
}

#[tokio::test]
async fn ops_a_provider_removed_while_a_removal_runs_leaves_no_orphaned_key() {
    let me = scope("company:acme");
    let target = me.clone();
    let (ports, hub, me) = rig(Box::new(move |store| {
        edit_doc(store, &target, |doc| {
            let rows = doc["providers"].as_array_mut().unwrap();
            rows.retain(|row| row["slug"] != "groq");
        });
    }))
    .await;
    let error = hub
        .remove(&me, &slug("groq"), Confirm::no())
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::NotFound(_)), "{error:?}");
    let key = ports
        .credentials
        .get(&me, &slug("groq").key_slot())
        .await
        .unwrap();
    assert!(
        key.is_none(),
        "the other remover's deletion stands; the key is not resurrected"
    );
}

#[tokio::test]
async fn ops_a_provider_added_twice_at_once_is_created_once_and_the_loser_keeps_its_own_key_out() {
    let me = scope("company:acme");
    let target = me.clone();
    let ports = MemoryPorts::new();
    let racing = Arc::new(Racing {
        inner: ports.config.clone(),
        hook: Mutex::new(None),
    });
    let hub = ports.builder().config_arc(racing.clone()).build().unwrap();
    // Another writer adds `mistral` between this add's load and its save.
    *racing.hook.lock().unwrap() = Some(Box::new(move |store| {
        let mut config = HubConfig::new();
        let record = crate::descriptor::ProviderRecord::new(
            "prv_other",
            slug("mistral"),
            "Mistral",
            "mistral".into(),
            "https://api.mistral.ai/v1",
        );
        config.providers.push(record);
        store.put_raw(&target, serde_json::to_string(&config).unwrap());
    }));
    ports
        .credentials
        .set(&me, &slug("mistral").key_slot(), Secret::new("sk-winner"))
        .await
        .unwrap();
    let error = hub
        .add(
            &me,
            ProviderDraft::new("mistral").with_key(Secret::new("sk-loser")),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::AlreadyExists { .. }), "{error:?}");
    let key = ports
        .credentials
        .get(&me, &slug("mistral").key_slot())
        .await
        .unwrap();
    assert_eq!(
        key.unwrap().expose(),
        "sk-winner",
        "the loser's key never replaced the winner's"
    );
}

#[tokio::test]
async fn ops_a_health_store_outage_never_turns_a_committed_change_into_a_failure() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    ports.http.route(
        crate::testkit::Match::prefix("https://api.openai.com/v1/models"),
        crate::testkit::Scripted::json(200, &crate::hub::fixtures::models_body(&["m"])),
    );
    ports.health.set_unavailable(true);
    // Every operation that only *observes or forgets* health still succeeds.
    let mutation = hub
        .connect(
            &me,
            ProviderDraft::new("openai").with_key(Secret::new(KEY)),
            ConnectOptions::default(),
        )
        .await
        .expect("the add and the check worked; only the recording failed");
    assert!(mutation.probe.is_some_and(|p| p.ok()));
    hub.set_key(&me, &slug("openai"), Secret::new("sk-rotated"))
        .await
        .unwrap();
    assert!(
        hub.test(
            &me,
            &slug("openai"),
            crate::taxonomy::TestDepth::Catalog,
            None
        )
        .await
        .unwrap()
        .ok()
    );
    // The reads that need health say so, typed.
    assert!(matches!(
        hub.health(&me, &slug("openai")).await,
        Err(HubError::StoreUnreadable {
            port: crate::error::PortName::Health,
            ..
        })
    ));
    hub.clear_default(&me).await.unwrap();
    hub.remove(&me, &slug("openai"), Confirm::no())
        .await
        .unwrap();
    ports.health.set_unavailable(false);
    assert!(hub.status(&me).await.unwrap().providers.is_empty());
    assert!(ports.credentials.is_empty());
}

// ---- round 1: writers that must not touch what they do not own -------------------------------

#[tokio::test]
async fn ops_an_add_of_a_taken_slug_never_touches_the_existing_key() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("groq").with_key(Secret::new("gsk-first")),
    )
    .await
    .unwrap();
    // With every write to the credential store failing, a second add of the same
    // slug still reports AlreadyExists: it never got as far as the slot.
    ports
        .credentials
        .inject(crate::ports::memory::CredentialFault::Write);
    let error = hub
        .add(
            &me,
            ProviderDraft::new("groq").with_key(Secret::new("gsk-second")),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::AlreadyExists { .. }), "{error:?}");
    ports.credentials.heal();
    let key = ports
        .credentials
        .get(&me, &slug("groq").key_slot())
        .await
        .unwrap();
    assert_eq!(key.unwrap().expose(), "gsk-first");
}

#[tokio::test]
async fn ops_a_rolled_back_connect_restores_the_previous_default() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("gpt-x")),
    )
    .await
    .unwrap();
    let before = hub.status(&me).await.unwrap().default;
    assert!(
        matches!(&before, crate::config::DefaultChoice::Full { provider, .. } if *provider == slug("openai"))
    );
    ports.http.route(
        crate::testkit::Match::prefix("https://api.groq.com/"),
        crate::testkit::Scripted::json(
            401,
            &json!({"error": {"message": "Invalid API Key", "code": "invalid_api_key"}}),
        ),
    );
    let error = hub
        .connect(
            &me,
            ProviderDraft::new("groq")
                .with_key(Secret::new("gsk-bad"))
                .with_model(model("l")),
            ConnectOptions::default().make_default(true),
        )
        .await
        .unwrap_err();
    assert_eq!(error.reason(), crate::error::ReasonCode::Auth);
    assert_eq!(
        hub.status(&me).await.unwrap().default,
        before,
        "the operator's default survives a failed connect"
    );
    // And when nothing had been chosen, a failed connect leaves nothing chosen.
    let empty = MemoryPorts::new();
    let hub = empty.hub();
    empty.http.route(
        crate::testkit::Match::prefix("https://api.groq.com/"),
        crate::testkit::Scripted::text(401, "Invalid API Key"),
    );
    hub.connect(
        &me,
        ProviderDraft::new("groq")
            .with_key(Secret::new("gsk-bad"))
            .with_model(model("l")),
        ConnectOptions::default().make_default(true),
    )
    .await
    .unwrap_err();
    assert_eq!(
        hub.status(&me).await.unwrap().default,
        crate::config::DefaultChoice::Unset
    );
}

/// A credential store whose `set` first lets another writer remove the record.
struct RemovesOnSet {
    inner: Arc<crate::ports::memory::MemoryCredentials>,
    config: Arc<MemoryConfig>,
    scope: ScopeKey,
    armed: Arc<std::sync::atomic::AtomicBool>,
}

impl std::fmt::Debug for RemovesOnSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("RemovesOnSet")
    }
}

#[async_trait]
impl CredentialStore for RemovesOnSet {
    async fn get(&self, scope: &ScopeKey, slot: &str) -> Result<Option<Secret>, PortError> {
        self.inner.get(scope, slot).await
    }

    async fn set(&self, scope: &ScopeKey, slot: &str, value: Secret) -> Result<(), PortError> {
        if self.armed.load(std::sync::atomic::Ordering::SeqCst) {
            edit_doc(&self.config, &self.scope, |doc| {
                doc["providers"]
                    .as_array_mut()
                    .unwrap()
                    .retain(|row| row["slug"] != "groq");
            });
        }
        self.inner.set(scope, slot, value).await
    }

    async fn delete(&self, scope: &ScopeKey, slot: &str) -> Result<(), PortError> {
        self.inner.delete(scope, slot).await
    }
}

fn removing_hub(
    ports: &MemoryPorts,
    me: &ScopeKey,
) -> (crate::hub::Hub, Arc<std::sync::atomic::AtomicBool>) {
    let armed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let hub = ports
        .builder()
        .credentials(RemovesOnSet {
            inner: ports.credentials.clone(),
            config: ports.config.clone(),
            scope: me.clone(),
            armed: armed.clone(),
        })
        .build()
        .unwrap();
    (hub, armed)
}

#[tokio::test]
async fn ops_a_key_set_for_a_provider_removed_meanwhile_is_deleted_again() {
    let ports = MemoryPorts::new();
    let me = scope("company:acme");
    let (hub, armed) = removing_hub(&ports, &me);
    hub.add(&me, ProviderDraft::new("groq")).await.unwrap();
    armed.store(true, std::sync::atomic::Ordering::SeqCst);
    let error = hub
        .set_key(&me, &slug("groq"), Secret::new("gsk-late"))
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::NotFound(_)), "{error:?}");
    assert!(
        ports.credentials.is_empty(),
        "the slot was cleaned up: no orphan to answer for a later provider"
    );
}

#[tokio::test]
async fn ops_an_edit_that_loses_to_a_removal_leaves_no_key_behind() {
    let ports = MemoryPorts::new();
    let me = scope("company:acme");
    let (hub, armed) = removing_hub(&ports, &me);
    hub.add(
        &me,
        ProviderDraft::new("groq").with_key(Secret::new("gsk-first")),
    )
    .await
    .unwrap();
    armed.store(true, std::sync::atomic::Ordering::SeqCst);
    let error = hub
        .edit(
            &me,
            &slug("groq"),
            crate::hub::ProviderPatch::new().key(Secret::new("gsk-rotated")),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::NotFound(_)), "{error:?}");
    // The record went, and neither the old key nor the new one was left owned by nothing.
    let key = ports
        .credentials
        .get(&me, &slug("groq").key_slot())
        .await
        .unwrap();
    assert!(
        key.as_ref().is_none_or(|k| k.expose() != "gsk-rotated"),
        "the new key is not left behind"
    );
    assert!(
        key.is_none(),
        "and the removed provider's old key is not resurrected"
    );
}

// ---- a result measured against a credential that changed is dropped ------------------------

use std::sync::OnceLock;

use crate::ports::{Http, HttpError, HubRequest, HubResponse};

/// What runs against the hub while a request is in flight.
type HubHook = Box<dyn FnOnce(&crate::hub::Hub) + Send>;

/// An `Http` that lets something happen to the hub while a request is in flight.
struct HookHttp {
    inner: Arc<crate::testkit::ScriptedHttp>,
    hub: Arc<OnceLock<crate::hub::Hub>>,
    hook: Mutex<Option<HubHook>>,
}

impl std::fmt::Debug for HookHttp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("HookHttp")
    }
}

#[async_trait]
impl Http for HookHttp {
    async fn send(
        &self,
        request: HubRequest,
        policy: &crate::policy::EndpointPolicy,
    ) -> Result<HubResponse, HttpError> {
        let hook = self.hook.lock().unwrap().take();
        if let Some(hook) = hook
            && let Some(hub) = self.hub.get()
        {
            hook(hub);
        }
        self.inner.send(request, policy).await
    }
}

fn hooked(ports: &MemoryPorts, hook: Box<dyn FnOnce(&crate::hub::Hub) + Send>) -> crate::hub::Hub {
    let cell = Arc::new(OnceLock::new());
    let http = HookHttp {
        inner: ports.http.clone(),
        hub: cell.clone(),
        hook: Mutex::new(Some(hook)),
    };
    let hub = ports.builder().http(http).build().unwrap();
    cell.set(hub.clone()).ok();
    hub
}

#[tokio::test]
async fn health_a_probe_in_flight_when_the_key_changes_is_not_recorded_against_the_new_key() {
    let ports = MemoryPorts::new();
    let me = scope("company:acme");
    let target = me.clone();
    let hub = hooked(
        &ports,
        Box::new(move |hub| {
            // The operator rotates the key while the check with the old one runs.
            futures::executor::block_on(hub.set_key(
                &target,
                &slug("openai"),
                Secret::new("sk-new"),
            ))
            .unwrap();
        }),
    );
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("m")),
    )
    .await
    .unwrap();
    ports.http.route(
        crate::testkit::Match::post("https://api.openai.com/v1/chat/completions"),
        crate::testkit::Scripted::json(
            401,
            &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
        ),
    );
    let report = hub
        .test(
            &me,
            &slug("openai"),
            crate::taxonomy::TestDepth::Completion,
            None,
        )
        .await
        .unwrap();
    assert_eq!(
        report.failure.unwrap().reason,
        crate::error::ReasonCode::Auth,
        "the caller still hears what the old key got"
    );
    assert_eq!(
        hub.health(&me, &slug("openai")).await.unwrap().health,
        crate::health::ProviderHealth::Unknown,
        "but a working new key does not show as Down because of the old one"
    );
}

#[tokio::test]
async fn health_a_turn_in_flight_when_the_provider_is_removed_and_readded_is_not_recorded() {
    use crate::health::Outcome;
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    hub.add(
        &me,
        ProviderDraft::new("groq").with_key(Secret::new("gsk-old")),
    )
    .await
    .unwrap();
    let epoch = hub.inner.health.epoch(&me, &slug("groq"));
    hub.set_key(&me, &slug("groq"), Secret::new("gsk-new"))
        .await
        .unwrap();
    let failure = crate::error::ProviderFailure::new(
        crate::error::ReasonCode::Auth,
        crate::error::Retry::Never,
    );
    let dropped = hub
        .record_outcome_lenient(
            &me,
            &crate::route::ResolvedTurn {
                slug: slug("groq"),
                kind: "groq".into(),
                group: crate::taxonomy::ProviderGroup::Cloud,
                base_url: String::new(),
                model: None,
                protocol: crate::taxonomy::Protocol::OpenAiChat,
                auth: crate::taxonomy::AuthStyle::Bearer,
                via: crate::route::ResolvedVia::Default,
                origin: None,
                temperature: None,
                cli: None,
            },
            Outcome::Failed(failure),
            epoch,
        )
        .await;
    assert!(dropped.is_ok());
    assert_eq!(
        hub.health(&me, &slug("groq")).await.unwrap().health,
        crate::health::ProviderHealth::Unknown
    );
}

// ---- round 2 ------------------------------------------------------------------------------

/// A credential store whose `get` lets something happen to the hub after it has
/// decided what to return: the caller then holds a credential that is already old.
struct StaleGet {
    inner: Arc<crate::ports::memory::MemoryCredentials>,
    hub: Arc<OnceLock<crate::hub::Hub>>,
    hook: Mutex<Option<HubHook>>,
}

impl std::fmt::Debug for StaleGet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("StaleGet")
    }
}

#[async_trait]
impl CredentialStore for StaleGet {
    async fn get(&self, scope: &ScopeKey, slot: &str) -> Result<Option<Secret>, PortError> {
        let value = self.inner.get(scope, slot).await?;
        // Taken out first: the temporary guard would otherwise be held while the
        // hook runs, and the hook re-enters this store.
        let hook = if value.is_some() {
            self.hook.lock().unwrap().take()
        } else {
            None
        };
        if let Some(hook) = hook
            && let Some(hub) = self.hub.get()
        {
            hook(hub);
        }
        Ok(value)
    }

    async fn set(&self, scope: &ScopeKey, slot: &str, value: Secret) -> Result<(), PortError> {
        self.inner.set(scope, slot, value).await
    }

    async fn delete(&self, scope: &ScopeKey, slot: &str) -> Result<(), PortError> {
        self.inner.delete(scope, slot).await
    }
}

fn stale_get_hub(
    ports: &MemoryPorts,
) -> (
    crate::hub::Hub,
    Arc<OnceLock<crate::hub::Hub>>,
    Arc<StaleGet>,
) {
    let cell = Arc::new(OnceLock::new());
    let creds = Arc::new(StaleGet {
        inner: ports.credentials.clone(),
        hub: cell.clone(),
        hook: Mutex::new(None),
    });
    let hub = ports
        .builder()
        .credentials_arc(creds.clone())
        .build()
        .unwrap();
    cell.set(hub.clone()).ok();
    (hub, cell, creds)
}

#[tokio::test]
async fn health_a_check_whose_credential_was_read_before_a_key_change_is_not_recorded() {
    let ports = MemoryPorts::new();
    let me = scope("company:acme");
    let (hub, _cell, creds) = stale_get_hub(&ports);
    hub.add(
        &me,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("m")),
    )
    .await
    .unwrap();
    ports.http.route(
        crate::testkit::Match::prefix("https://api.openai.com/v1/models"),
        crate::testkit::Scripted::json(
            401,
            &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
        ),
    );
    let target = me.clone();
    *creds.hook.lock().unwrap() = Some(Box::new(move |hub| {
        futures::executor::block_on(hub.set_key(&target, &slug("openai"), Secret::new("sk-new")))
            .unwrap();
    }));
    // The key is read (old), then rotated, then the probe runs with the old one.
    let report = hub
        .test(
            &me,
            &slug("openai"),
            crate::taxonomy::TestDepth::Catalog,
            None,
        )
        .await
        .unwrap();
    assert!(!report.ok());
    assert_eq!(
        hub.health(&me, &slug("openai")).await.unwrap().health,
        crate::health::ProviderHealth::Unknown
    );
}

#[tokio::test]
async fn ops_a_list_read_with_an_old_key_does_not_serve_the_new_key_from_the_cache() {
    let ports = MemoryPorts::new();
    let me = scope("company:acme");
    let (hub, _cell, creds) = stale_get_hub(&ports);
    hub.add(&me, ProviderDraft::new("openai").with_key(Secret::new(KEY)))
        .await
        .unwrap();
    ports.http.route(
        crate::testkit::Match::prefix("https://api.openai.com/v1/models"),
        crate::testkit::Scripted::json(
            200,
            &crate::hub::fixtures::models_body(&["old-entitlements"]),
        ),
    );
    let target = me.clone();
    *creds.hook.lock().unwrap() = Some(Box::new(move |hub| {
        futures::executor::block_on(hub.set_key(&target, &slug("openai"), Secret::new("sk-new")))
            .unwrap();
    }));
    hub.list_models(&me, &slug("openai"), false).await.unwrap();
    ports.http.route(
        crate::testkit::Match::prefix("https://api.openai.com/v1/models"),
        crate::testkit::Scripted::json(
            200,
            &crate::hub::fixtures::models_body(&["new-entitlements"]),
        ),
    );
    let next = hub.list_models(&me, &slug("openai"), false).await.unwrap();
    assert_eq!(
        next.ids(),
        ["new-entitlements"],
        "the old key's list was not kept for the new key"
    );
    assert_eq!(next.freshness, crate::catalog::Freshness::Fresh);
}

#[tokio::test]
async fn ops_an_add_whose_record_is_removed_before_its_key_lands_leaves_no_key() {
    let ports = MemoryPorts::new();
    let me = scope("company:acme");
    let (hub, armed) = removing_hub(&ports, &me);
    armed.store(true, std::sync::atomic::Ordering::SeqCst);
    let error = hub
        .add(
            &me,
            ProviderDraft::new("groq").with_key(Secret::new("gsk-late")),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::NotFound(_)), "{error:?}");
    assert!(ports.credentials.is_empty());
    assert!(hub.status(&me).await.unwrap().providers.is_empty());
}

#[tokio::test]
async fn ops_a_rolled_back_connect_tells_event_consumers_it_was_undone() {
    use crate::ports::HubEvent;
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = scope("company:acme");
    ports
        .credentials
        .set(&me, &slug("openai").key_slot(), Secret::new("sk-previous"))
        .await
        .unwrap();
    ports.http.route(
        crate::testkit::Match::prefix("https://api.openai.com/"),
        crate::testkit::Scripted::text(401, "Incorrect API key provided"),
    );
    hub.connect(
        &me,
        ProviderDraft::new("openai").with_key(Secret::new(KEY)),
        ConnectOptions::default(),
    )
    .await
    .unwrap_err();
    let events = ports.events.events();
    let added = events
        .iter()
        .filter(|e| matches!(e, HubEvent::ProviderAdded { .. }))
        .count();
    let removed = events
        .iter()
        .filter(|e| matches!(e, HubEvent::ProviderRemoved { .. }))
        .count();
    let keys: Vec<bool> = events
        .iter()
        .filter_map(|e| match e {
            HubEvent::KeyChanged { present, .. } => Some(*present),
            _ => None,
        })
        .collect();
    assert_eq!((added, removed), (1, 1), "{events:?}");
    assert_eq!(
        keys,
        [true, true],
        "the key changed, then went back to the previous one"
    );
}

#[tokio::test]
async fn ops_an_undo_that_cannot_restore_the_key_still_reports_why_the_add_failed_and_cleans_up() {
    let ports = MemoryPorts::new();
    let me = scope("company:acme");
    let creds = ports.credentials.clone();
    let hub = hooked(
        &ports,
        Box::new(move |_| {
            // The credential store goes down while the check is in flight.
            creds.inject(crate::ports::memory::CredentialFault::Write);
        }),
    );
    ports
        .credentials
        .set(&me, &slug("openai").key_slot(), Secret::new("sk-previous"))
        .await
        .unwrap();
    ports.http.route(
        crate::testkit::Match::prefix("https://api.openai.com/"),
        crate::testkit::Scripted::text(401, "Incorrect API key provided"),
    );
    let error = hub
        .connect(
            &me,
            ProviderDraft::new("openai").with_key(Secret::new(KEY)),
            ConnectOptions::default(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.reason(),
        crate::error::ReasonCode::Auth,
        "the caller hears why, not that a restore failed: {error:?}"
    );
    ports.credentials.heal();
    assert!(
        hub.status(&me).await.unwrap().providers.is_empty(),
        "the record was taken out anyway"
    );
}

#[tokio::test]
async fn ops_an_undo_only_removes_the_record_its_own_add_made() {
    let ports = MemoryPorts::new();
    let me = scope("company:acme");
    let target = me.clone();
    let hub = hooked(
        &ports,
        Box::new(move |hub| {
            // While the check runs, somebody removes the provider and adds another
            // of the same slug with its own key.
            futures::executor::block_on(async {
                hub.clear_default(&target).await.unwrap();
                hub.remove(&target, &slug("groq"), Confirm::in_use())
                    .await
                    .unwrap();
                hub.add(
                    &target,
                    ProviderDraft::new("groq").with_key(Secret::new("gsk-theirs")),
                )
                .await
                .unwrap();
            });
        }),
    );
    ports.http.route(
        crate::testkit::Match::prefix("https://api.groq.com/"),
        crate::testkit::Scripted::text(401, "Invalid API Key"),
    );
    hub.connect(
        &me,
        ProviderDraft::new("groq").with_key(Secret::new("gsk-mine")),
        ConnectOptions::default(),
    )
    .await
    .unwrap_err();
    let key = ports
        .credentials
        .get(&me, &slug("groq").key_slot())
        .await
        .unwrap();
    assert_eq!(
        key.unwrap().expose(),
        "gsk-theirs",
        "the other writer's key was not overwritten"
    );
    assert_eq!(
        hub.status(&me).await.unwrap().providers.len(),
        1,
        "and its record was not removed"
    );
}

#[tokio::test]
async fn ops_removing_a_provider_does_not_evict_another_scopes_shared_listing() {
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let (a, b) = (scope("company:a"), scope("company:b"));
    ports.http.route(
        crate::testkit::Match::prefix("https://gw.acme.test/"),
        crate::testkit::Scripted::json(200, &crate::hub::fixtures::models_body(&["m"])),
    );
    let gateway = || {
        ProviderDraft::new("custom")
            .with_label("Gateway")
            .with_base_url("https://gw.acme.test/v1")
    };
    for s in [&a, &b] {
        hub.add(s, gateway()).await.unwrap();
        hub.list_models(s, &slug("gateway"), false).await.unwrap();
    }
    let asked = ports.http.request_count();
    hub.remove(&a, &slug("gateway"), Confirm::no())
        .await
        .unwrap();
    hub.list_models(&b, &slug("gateway"), false).await.unwrap();
    assert_eq!(
        ports.http.request_count(),
        asked,
        "one tenant's removal never forces another's refetch"
    );
}
