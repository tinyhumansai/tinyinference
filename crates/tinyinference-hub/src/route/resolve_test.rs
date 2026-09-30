//! Tests for `resolve_for_turn`: precedence, failing closed, and what the
//! resolved turn carries.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;

use crate::config::{DefaultChoice, ModelChoice, ProviderDraft};
use crate::credential::CredentialOrigin;
use crate::error::{HubError, Operation, Unresolved};
use crate::hub::fixtures::{Bed, KEY, model, slug};
use crate::hub::{Confirm, ManagedConfig};
use crate::ids::{AgentKey, ScopeKey, WorkloadKey};
use crate::ports::{ConfigStore, CredentialStore, PortError, Version};
use crate::route::{ProviderRoute, ResolvedVia, RouteTarget, Temperature, TurnQuery};
use crate::secret::Secret;
use crate::taxonomy::{CliKind, LocalRuntime, Protocol, ProviderGroup};

async fn two_providers(bed: &Bed) {
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("groq")
                .with_key(Secret::new("gsk-x"))
                .with_model(model("llama")),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn resolve_the_default_choice_and_what_the_turn_carries() {
    let bed = Bed::new();
    two_providers(&bed).await;
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(turn.slug, slug("openai"));
    assert_eq!(
        turn.kind.as_str(),
        "openai",
        "the kind is exact for telemetry"
    );
    assert_eq!(turn.base_url, "https://api.openai.com/v1");
    assert_eq!(turn.model, Some(model("gpt-x")));
    assert_eq!(turn.via, ResolvedVia::Default);
    assert_eq!(turn.group, ProviderGroup::Cloud);
    assert_eq!(turn.protocol, Protocol::OpenAiChat);
    assert_eq!(turn.origin, Some(CredentialOrigin::ProviderKey));
    assert!(!format!("{turn:?}").contains(KEY));
}

#[tokio::test]
async fn resolve_precedence_is_override_then_pin_then_workload_then_default() {
    let bed = Bed::new();
    two_providers(&bed).await;
    let (agent, workload) = (AgentKey::new("agent:a"), WorkloadKey::new("tier:chat"));
    bed.hub
        .set_workload_route(
            &bed.scope,
            &workload,
            Some(ProviderRoute::provider(slug("groq")).with_model(model("workload-model"))),
        )
        .await
        .unwrap();
    bed.hub
        .pin_agent(
            &bed.scope,
            &agent,
            Some(ModelChoice::new(slug("openai"), model("pinned"))),
        )
        .await
        .unwrap();
    let full = TurnQuery::new()
        .with_agent(agent.clone())
        .with_workload(workload.clone());

    let workload_only = TurnQuery::new().with_workload(workload.clone());
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &workload_only)
        .await
        .unwrap();
    assert_eq!(
        (turn.slug, turn.via, turn.model),
        (
            slug("groq"),
            ResolvedVia::Workload,
            Some(model("workload-model"))
        )
    );

    let turn = bed.hub.resolve_for_turn(&bed.scope, &full).await.unwrap();
    assert_eq!(
        (turn.slug, turn.via, turn.model),
        (slug("openai"), ResolvedVia::Pin, Some(model("pinned")))
    );

    let forced = full
        .clone()
        .with_override(ProviderRoute::provider(slug("groq")));
    let turn = bed.hub.resolve_for_turn(&bed.scope, &forced).await.unwrap();
    assert_eq!(
        (turn.slug, turn.via, turn.model),
        (slug("groq"), ResolvedVia::Override, Some(model("llama"))),
        "the record's model when the route names none"
    );

    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(turn.via, ResolvedVia::Default);
    // A workload route to "default" reads the default.
    bed.ports.config.put_raw(&bed.scope, {
        let mut doc = bed.stored();
        doc["workload_routes"] = serde_json::json!({"tier:x": {"target": {"kind": "default"}}});
        doc.to_string()
    });
    assert!(
        matches!(
            bed.hub
                .resolve_for_turn(&bed.scope, &TurnQuery::new().with_workload("tier:x".into()))
                .await,
            Err(HubError::Invalid(_)) | Err(HubError::StoreUnreadable { .. })
        ),
        "a stored default route is refused at load, not guessed at"
    );
}

#[tokio::test]
async fn resolve_a_pin_or_route_that_cannot_be_served_fails_closed_never_falls_through() {
    let bed = Bed::new();
    two_providers(&bed).await;
    let agent = AgentKey::new("agent:a");
    bed.hub
        .pin_agent(
            &bed.scope,
            &agent,
            Some(ModelChoice::new(slug("groq"), model("m"))),
        )
        .await
        .unwrap();
    // The pinned provider loses its key: the turn fails, it does not fall to the default.
    bed.hub
        .clear_key(&bed.scope, &slug("groq"), Confirm::in_use())
        .await
        .unwrap();
    let query = TurnQuery::new().with_agent(agent.clone());
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &query).await,
        Err(HubError::Unresolved(Unresolved::NoKey(s))) if s == slug("groq")
    ));
    bed.hub
        .set_key(&bed.scope, &slug("groq"), Secret::new("gsk-2"))
        .await
        .unwrap();
    bed.hub
        .set_enabled(&bed.scope, &slug("groq"), false, Confirm::in_use())
        .await
        .unwrap();
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &query).await,
        Err(HubError::Unresolved(Unresolved::Disabled(_)))
    ));
    bed.hub
        .remove(&bed.scope, &slug("groq"), Confirm::in_use())
        .await
        .unwrap();
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &query).await,
        Err(HubError::Unresolved(Unresolved::Missing(_)))
    ));
    // The override is checked the same way.
    let forced = TurnQuery::new().with_override(ProviderRoute::provider(slug("nope")));
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &forced).await,
        Err(HubError::Unresolved(Unresolved::Missing(_)))
    ));
}

#[tokio::test]
async fn guard_g23_a_provider_only_default_fails_closed_on_the_turn_path_only() {
    let bed = Bed::new();
    two_providers(&bed).await;
    bed.ports.config.put_raw(&bed.scope, {
        let mut doc = bed.stored();
        doc["default"] = serde_json::json!({"mode": "provider_only", "provider": "groq"});
        doc.to_string()
    });
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &TurnQuery::new()).await,
        Err(HubError::Unresolved(Unresolved::ProviderOnlyDefault(s))) if s == slug("groq")
    ));
    // Status still shows a provider for display.
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert_eq!(status.primary, Some(slug("groq")));
    bed.hub.clear_default(&bed.scope).await.unwrap();
    assert!(matches!(
        bed.hub
            .resolve_for_turn(&bed.scope, &TurnQuery::new())
            .await,
        Err(HubError::Unresolved(Unresolved::NoProvider))
    ));
    assert_eq!(
        bed.hub.status(&bed.scope).await.unwrap().default,
        DefaultChoice::Unset
    );
}

#[tokio::test]
async fn resolve_a_provider_with_no_model_anywhere_is_unresolved() {
    let bed = Bed::new();
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_key(Secret::new(KEY)),
        )
        .await
        .unwrap();
    let forced = TurnQuery::new().with_override(ProviderRoute::provider(slug("openai")));
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &forced).await,
        Err(HubError::Unresolved(Unresolved::NoModel(_)))
    ));
}

#[tokio::test]
async fn resolve_model_ids_pass_through_verbatim_and_a_hosts_reserved_word_never_reaches_the_wire()
{
    let bed =
        Bed::with(|b| b.hub_policy(crate::hub::HubPolicy::new().reserved_model_words(["chat-v1"])));
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let odd = "ft:gpt-4o:acme::A1b2-C3/v.2_x";
    let forced = TurnQuery::new()
        .with_override(ProviderRoute::provider(slug("openai")).with_model(model(odd)));
    let turn = bed.hub.resolve_for_turn(&bed.scope, &forced).await.unwrap();
    assert_eq!(turn.model.unwrap().as_str(), odd);
    let tier = TurnQuery::new()
        .with_override(ProviderRoute::provider(slug("openai")).with_model(model("chat-v1")));
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &tier).await,
        Err(HubError::Invalid(_))
    ));
}

#[tokio::test]
async fn resolve_rechecks_the_endpoint_policy_and_carries_the_temperature() {
    let bed = Bed::new();
    bed.ports.config.put_raw(
        &bed.scope,
        serde_json::json!({
            "providers": [{"id": "p", "slug": "acme", "label": "Acme", "kind": "custom",
                "base_url": "http://10.0.0.5/v1", "model": "m"}],
            "default": {"mode": "full", "provider": "acme", "model": "m"}
        })
        .to_string(),
    );
    assert!(matches!(
        bed.hub
            .resolve_for_turn(&bed.scope, &TurnQuery::new())
            .await,
        Err(HubError::Policy(_))
    ));
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let hot = TurnQuery::new().with_override(
        ProviderRoute::provider(slug("openai")).with_temperature(Temperature::new(0.3).unwrap()),
    );
    let turn = bed.hub.resolve_for_turn(&bed.scope, &hot).await.unwrap();
    assert_eq!(turn.temperature.unwrap().get(), 0.3);
}

#[tokio::test]
async fn resolve_managed_local_and_cli_targets() {
    let bed = Bed::with(|b| b.managed(ManagedConfig::new("https://api.tinyhumans.test/x")));
    let managed = TurnQuery::new()
        .with_override(ProviderRoute::new(RouteTarget::Managed).with_model(model("m")));
    assert!(
        matches!(
            bed.hub.resolve_for_turn(&bed.scope, &managed).await,
            Err(HubError::SignedOut { .. })
        ),
        "signed out is typed on the turn path too"
    );
    bed.hub
        .set_key(&bed.scope, &slug("tinyhumans"), Secret::new("th-key"))
        .await
        .unwrap();
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &managed)
        .await
        .unwrap();
    assert_eq!(
        (turn.slug.clone(), turn.group),
        (slug("tinyhumans"), ProviderGroup::Managed)
    );
    assert_eq!(turn.origin, Some(CredentialOrigin::ProviderKey));

    // Local: none configured, then a disabled one, then an enabled one.
    let local = TurnQuery::new()
        .with_override(ProviderRoute::new(RouteTarget::Local(None)).with_model(model("llama3")));
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &local).await,
        Err(HubError::Unresolved(Unresolved::NoTarget("local runtime")))
    ));
    bed.hub
        .add(&bed.scope, ProviderDraft::new("ollama"))
        .await
        .unwrap();
    bed.hub
        .add(&bed.scope, ProviderDraft::new("lmstudio"))
        .await
        .unwrap();
    bed.hub
        .set_enabled(&bed.scope, &slug("ollama"), false, Confirm::in_use())
        .await
        .unwrap();
    let turn = bed.hub.resolve_for_turn(&bed.scope, &local).await.unwrap();
    assert_eq!(
        turn.slug,
        slug("lmstudio"),
        "the first enabled local record answers"
    );
    let ollama = TurnQuery::new().with_override(
        ProviderRoute::new(RouteTarget::Local(Some(LocalRuntime::Ollama))).with_model(model("m")),
    );
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &ollama).await,
        Err(HubError::Unresolved(Unresolved::Disabled(_)))
    ));
    let vllm = TurnQuery::new().with_override(ProviderRoute::new(RouteTarget::Local(Some(
        LocalRuntime::Mlx,
    ))));
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &vllm).await,
        Err(HubError::Unresolved(Unresolved::NoTarget(_)))
    ));

    // A CLI login is a route target with no endpoint and no chat model.
    let cli = TurnQuery::new().with_override(
        ProviderRoute::new(RouteTarget::Cli(CliKind::ClaudeCode)).with_model(model("sonnet")),
    );
    let turn = bed.hub.resolve_for_turn(&bed.scope, &cli).await.unwrap();
    assert_eq!(
        (turn.cli, turn.protocol, turn.base_url.as_str()),
        (Some(CliKind::ClaudeCode), Protocol::CliStream, "")
    );
    assert!(matches!(
        bed.hub.chat_model(&bed.scope, &turn).await,
        Err(HubError::Unsupported {
            op: Operation::ChatModel,
            ..
        })
    ));
}

#[tokio::test]
async fn resolve_an_ephemeral_route_is_not_resolved_by_the_hub() {
    let bed = Bed::new();
    let query = TurnQuery::new().with_override(ProviderRoute::new(RouteTarget::Ephemeral));
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &query).await,
        Err(HubError::Unsupported {
            op: Operation::ResolveForTurn,
            ..
        })
    ));
}

#[tokio::test]
async fn resolve_a_keyless_local_runtime_needs_no_key_but_a_keyed_kind_does() {
    let bed = Bed::new();
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
    assert_eq!(
        (turn.slug, turn.origin, turn.base_url.as_str()),
        (slug("ollama"), None, "http://localhost:11434/v1")
    );
    // A custom endpoint with no key configured is served keyless (a gateway
    // that authenticates by network position).
    let custom = ProviderDraft::new("custom")
        .with_label("Gateway")
        .with_base_url("https://gw.acme.test/v1")
        .with_model(model("m"));
    bed.hub.add(&bed.scope, custom).await.unwrap();
    let forced = TurnQuery::new().with_override(ProviderRoute::provider(slug("gateway")));
    assert!(bed.hub.resolve_for_turn(&bed.scope, &forced).await.is_ok());
}

// ---- allocation-light (A16) --------------------------------------------------------------

#[derive(Debug, Default)]
struct Counts {
    config_loads: AtomicUsize,
    credential_reads: AtomicUsize,
}

#[derive(Debug)]
struct CountingConfig(Arc<crate::ports::memory::MemoryConfig>, Arc<Counts>);

#[async_trait]
impl ConfigStore for CountingConfig {
    async fn load(
        &self,
        scope: &ScopeKey,
    ) -> Result<Option<(crate::config::HubConfig, Version)>, PortError> {
        self.1.config_loads.fetch_add(1, Ordering::SeqCst);
        self.0.load(scope).await
    }

    async fn save(
        &self,
        scope: &ScopeKey,
        config: &crate::config::HubConfig,
        expect: Option<Version>,
    ) -> Result<Version, PortError> {
        self.0.save(scope, config, expect).await
    }
}

#[derive(Debug)]
struct CountingCreds(Arc<crate::ports::memory::MemoryCredentials>, Arc<Counts>);

#[async_trait]
impl CredentialStore for CountingCreds {
    async fn get(&self, scope: &ScopeKey, slot: &str) -> Result<Option<Secret>, PortError> {
        self.1.credential_reads.fetch_add(1, Ordering::SeqCst);
        self.0.get(scope, slot).await
    }

    async fn set(&self, scope: &ScopeKey, slot: &str, value: Secret) -> Result<(), PortError> {
        self.0.set(scope, slot, value).await
    }

    async fn delete(&self, scope: &ScopeKey, slot: &str) -> Result<(), PortError> {
        self.0.delete(scope, slot).await
    }
}

#[tokio::test]
async fn resolve_is_allocation_light_one_config_read_and_one_credential_read() {
    let counts = Arc::new(Counts::default());
    let ports = crate::testkit::MemoryPorts::new();
    let hub = ports
        .builder()
        .config(CountingConfig(ports.config.clone(), counts.clone()))
        .credentials(CountingCreds(ports.credentials.clone(), counts.clone()))
        .build()
        .unwrap();
    let scope = ScopeKey::new("company:acme");
    hub.add(
        &scope,
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("m")),
    )
    .await
    .unwrap();
    let (loads, reads) = (
        counts.config_loads.load(Ordering::SeqCst),
        counts.credential_reads.load(Ordering::SeqCst),
    );
    hub.resolve_for_turn(&scope, &TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(counts.config_loads.load(Ordering::SeqCst) - loads, 1);
    assert_eq!(counts.credential_reads.load(Ordering::SeqCst) - reads, 1);
}
