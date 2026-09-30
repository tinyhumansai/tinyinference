//! Tests for removing, disabling, re-keying, defaults, pins and routes.

use crate::config::{DefaultChoice, ModelChoice, ProviderDraft};
use crate::error::{HubError, Operation, ReasonCode, Unresolved, UsedBy};
use crate::health::ProviderHealth;
use crate::hub::fixtures::{Bed, KEY, model, slug};
use crate::hub::{Confirm, ConnectOptions, KeyState, MutationStatus, ProviderPatch};
use crate::ids::{AgentKey, WorkloadKey};
use crate::ports::HubEvent;
use crate::ports::memory::CredentialFault;
use crate::route::{ProviderRoute, RouteTarget};
use crate::secret::Secret;

async fn with_openai(bed: &Bed) {
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
}

// ---- remove (G6) -------------------------------------------------------------------------

#[tokio::test]
async fn guard_g6_removal_is_in_use_guarded_and_confirmed_removal_leaves_references_failing_closed()
{
    let bed = Bed::new();
    with_openai(&bed).await;
    let agent = AgentKey::new("agent:writer");
    bed.hub
        .pin_agent(
            &bed.scope,
            &agent,
            Some(ModelChoice::new(slug("openai"), model("gpt-x"))),
        )
        .await
        .unwrap();
    bed.hub
        .set_workload_route(
            &bed.scope,
            &WorkloadKey::new("tier:chat"),
            Some(ProviderRoute::provider(slug("openai"))),
        )
        .await
        .unwrap();

    let error = bed
        .hub
        .remove(&bed.scope, &slug("openai"), Confirm::no())
        .await
        .unwrap_err();
    let HubError::InUse(used) = error else {
        panic!("{error:?}")
    };
    assert!(used.default_choice, "the first provider is the default");
    assert_eq!(used.agents, std::slice::from_ref(&agent));
    assert_eq!(used.workloads, [WorkloadKey::new("tier:chat")]);
    assert_eq!(
        bed.key_of("openai").await.as_deref(),
        Some(KEY),
        "a refused removal deletes nothing"
    );

    let removed = bed
        .hub
        .remove(&bed.scope, &slug("openai"), Confirm::in_use())
        .await
        .unwrap();
    assert_eq!(removed.used_by.unwrap().count(), 3);
    assert_eq!(
        bed.key_of("openai").await,
        None,
        "the key is deleted with the row"
    );
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert!(status.providers.is_empty());
    // X14 / invariant 10: the default was never cleared, and neither was the pin.
    assert!(
        matches!(&status.default, DefaultChoice::Full { provider, .. } if *provider == slug("openai"))
    );
    let turn = crate::route::TurnQuery::new().with_agent(agent);
    assert!(matches!(
        bed.hub.resolve_for_turn(&bed.scope, &turn).await,
        Err(HubError::Unresolved(Unresolved::Missing(_)))
    ));
    assert!(matches!(
        bed.hub
            .resolve_for_turn(&bed.scope, &crate::route::TurnQuery::new())
            .await,
        Err(HubError::Unresolved(Unresolved::Missing(_)))
    ));
}

#[tokio::test]
async fn guard_g6_a_removal_that_cannot_be_saved_puts_the_key_back() {
    let bed = Bed::new();
    with_openai(&bed).await;
    bed.hub.clear_default(&bed.scope).await.unwrap();
    bed.ports.config.conflict_next(10);
    let error = bed
        .hub
        .remove(&bed.scope, &slug("openai"), Confirm::no())
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Conflict));
    assert_eq!(bed.key_of("openai").await.as_deref(), Some(KEY));
    assert_eq!(bed.hub.status(&bed.scope).await.unwrap().providers.len(), 1);
}

#[tokio::test]
async fn ops_removal_forgets_health_and_emits_one_event() {
    let bed = Bed::new();
    bed.openai_lists(&["m"]);
    bed.hub
        .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
        .await
        .unwrap();
    bed.hub.clear_default(&bed.scope).await.unwrap();
    bed.ports.events.drain();
    bed.hub
        .remove(&bed.scope, &slug("openai"), Confirm::no())
        .await
        .unwrap();
    let events = bed.ports.events.events();
    assert!(
        events.iter().any(
            |e| matches!(e, HubEvent::ProviderRemoved { slug: s, .. } if *s == slug("openai"))
        )
    );
    // Re-adding the slug starts with no memory of the old one (invariant 9).
    bed.hub
        .add(&bed.scope, ProviderDraft::new("openai"))
        .await
        .unwrap();
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("openai"))
            .await
            .unwrap()
            .health,
        ProviderHealth::Unknown
    );
}

#[tokio::test]
async fn ops_the_managed_provider_cannot_be_removed_but_a_missing_one_is_not_found() {
    let bed = Bed::with(|b| {
        b.managed(crate::hub::ManagedConfig::new(
            "https://api.tinyhumans.test/x",
        ))
    });
    assert!(matches!(
        bed.hub
            .remove(&bed.scope, &slug("tinyhumans"), Confirm::in_use())
            .await,
        Err(HubError::Unsupported {
            op: Operation::Remove,
            ..
        })
    ));
    assert!(matches!(
        bed.hub
            .remove(&bed.scope, &slug("nope"), Confirm::no())
            .await,
        Err(HubError::NotFound(_))
    ));
}

#[tokio::test]
async fn ops_removing_an_unused_provider_needs_no_confirmation() {
    let bed = Bed::new();
    with_openai(&bed).await;
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("groq").with_key(Secret::new("gsk-x")),
        )
        .await
        .unwrap();
    let removed = bed
        .hub
        .remove(&bed.scope, &slug("groq"), Confirm::no())
        .await
        .unwrap();
    assert!(removed.used_by.is_none());
    assert_eq!(removed.status, MutationStatus::Saved);
    assert_eq!(bed.key_of("groq").await, None);
    assert_eq!(bed.key_of("openai").await.as_deref(), Some(KEY));
}

#[derive(Debug)]
struct HostPins(UsedBy);

#[async_trait::async_trait]
impl crate::ports::UsageQuery for HostPins {
    async fn used_by(
        &self,
        _scope: &crate::ids::ScopeKey,
        _slug: &crate::ids::Slug,
    ) -> Result<UsedBy, crate::ports::PortError> {
        Ok(self.0.clone())
    }
}

#[tokio::test]
async fn ops_the_hosts_own_references_count_for_the_in_use_guard() {
    let mut host = UsedBy::default();
    host.other
        .push("agent Zed on the company record".to_string());
    let bed = Bed::with(|b| b.usage_query(std::sync::Arc::new(HostPins(host))));
    with_openai(&bed).await;
    bed.hub.clear_default(&bed.scope).await.unwrap();
    let error = bed
        .hub
        .remove(&bed.scope, &slug("openai"), Confirm::no())
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::InUse(u) if u.other.len() == 1));
    bed.hub
        .set_enabled(&bed.scope, &slug("openai"), true, Confirm::no())
        .await
        .expect("enabling is never guarded");
    assert!(matches!(
        bed.hub
            .set_enabled(&bed.scope, &slug("openai"), false, Confirm::no())
            .await,
        Err(HubError::InUse(_))
    ));
}

// ---- enabled (G7) ------------------------------------------------------------------------

#[tokio::test]
async fn guard_g7_disabling_is_guarded_never_scrubs_and_a_route_to_it_fails_closed() {
    let bed = Bed::new();
    with_openai(&bed).await;
    let error = bed
        .hub
        .set_enabled(&bed.scope, &slug("openai"), false, Confirm::no())
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::InUse(u) if u.default_choice));
    bed.hub
        .set_enabled(&bed.scope, &slug("openai"), false, Confirm::in_use())
        .await
        .unwrap();
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert!(
        matches!(&status.default, DefaultChoice::Full { .. }),
        "the default is untouched"
    );
    assert_eq!(status.providers[0].health, ProviderHealth::Disabled);
    assert_eq!(
        bed.key_of("openai").await.as_deref(),
        Some(KEY),
        "disabling keeps the key"
    );
    // sim_disable_fail_closed: the turn is Unresolved, never another row.
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("groq")
                .with_key(Secret::new("gsk-x"))
                .with_model(model("l")),
        )
        .await
        .unwrap();
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &crate::route::TurnQuery::new())
        .await;
    assert!(
        matches!(turn, Err(HubError::Unresolved(Unresolved::Disabled(ref s))) if *s == slug("openai")),
        "{turn:?}"
    );
    // And re-enabling is never guarded and restores service.
    bed.hub
        .set_enabled(&bed.scope, &slug("openai"), true, Confirm::no())
        .await
        .unwrap();
    assert!(
        bed.hub
            .resolve_for_turn(&bed.scope, &crate::route::TurnQuery::new())
            .await
            .is_ok()
    );
    let again = bed
        .hub
        .set_enabled(&bed.scope, &slug("openai"), true, Confirm::no())
        .await
        .unwrap();
    assert_eq!(again.status, MutationStatus::Unchanged);
}

#[tokio::test]
async fn ops_the_managed_provider_can_be_disabled_and_stays_listed_first() {
    let bed = Bed::with(|b| {
        b.managed(crate::hub::ManagedConfig::new(
            "https://api.tinyhumans.test/x",
        ))
    });
    with_openai(&bed).await;
    bed.hub
        .set_enabled(&bed.scope, &slug("tinyhumans"), false, Confirm::no())
        .await
        .unwrap();
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert_eq!(status.providers[0].view.record.slug, slug("tinyhumans"));
    assert_eq!(status.providers[0].health, ProviderHealth::Disabled);
    assert_eq!(status.providers[1].view.record.slug, slug("openai"));
    assert_eq!(
        status.primary,
        Some(slug("openai")),
        "the default's provider is the primary"
    );
}

// ---- keys (G5) ---------------------------------------------------------------------------

#[tokio::test]
async fn guard_g5_clearing_a_key_is_guarded_and_rotating_never_is() {
    let bed = Bed::new();
    with_openai(&bed).await;
    // Rotation on an in-use provider: fine.
    let rotated = bed
        .hub
        .set_key(&bed.scope, &slug("openai"), Secret::new("  sk-rotated "))
        .await
        .unwrap();
    assert_eq!(rotated.status, MutationStatus::Saved);
    assert_eq!(bed.key_of("openai").await.as_deref(), Some("sk-rotated"));
    // Clearing: guarded.
    assert!(matches!(
        bed.hub
            .clear_key(&bed.scope, &slug("openai"), Confirm::no())
            .await,
        Err(HubError::InUse(_))
    ));
    assert_eq!(bed.key_of("openai").await.as_deref(), Some("sk-rotated"));
    let cleared = bed
        .hub
        .clear_key(&bed.scope, &slug("openai"), Confirm::in_use())
        .await
        .unwrap();
    assert_eq!(cleared.record.unwrap().key, KeyState::Missing);
    assert_eq!(bed.key_of("openai").await, None);
    // Clearing again is a no-op, not an error.
    let noop = bed
        .hub
        .clear_key(&bed.scope, &slug("openai"), Confirm::in_use())
        .await
        .unwrap();
    assert_eq!(noop.status, MutationStatus::Unchanged);
    // The provider is now keyless: a turn fails closed with NoKey.
    assert!(matches!(
        bed.hub
            .resolve_for_turn(&bed.scope, &crate::route::TurnQuery::new())
            .await,
        Err(HubError::Unresolved(Unresolved::NoKey(_)))
    ));
}

#[tokio::test]
async fn ops_set_key_validates_and_reports_an_unreadable_store_as_such() {
    let bed = Bed::new();
    with_openai(&bed).await;
    assert!(matches!(
        bed.hub
            .set_key(&bed.scope, &slug("openai"), Secret::new("  "))
            .await,
        Err(HubError::Invalid(_))
    ));
    assert!(matches!(
        bed.hub
            .set_key(&bed.scope, &slug("nope"), Secret::new("k"))
            .await,
        Err(HubError::NotFound(_))
    ));
    bed.ports.credentials.inject(CredentialFault::Write);
    assert!(matches!(
        bed.hub
            .set_key(&bed.scope, &slug("openai"), Secret::new("k2"))
            .await,
        Err(HubError::StoreUnreadable { .. })
    ));
    bed.ports.credentials.heal();
    assert_eq!(bed.key_of("openai").await.as_deref(), Some(KEY));
    // A key change is announced.
    bed.ports.events.drain();
    bed.hub
        .set_key(&bed.scope, &slug("openai"), Secret::new("k3"))
        .await
        .unwrap();
    assert!(
        bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, HubEvent::KeyChanged { present: true, .. }))
    );
}

#[tokio::test]
async fn sim_key_rotation_next_call_uses_the_new_key_and_health_starts_over() {
    let bed = Bed::new();
    bed.openai_lists(&["m"]);
    bed.hub
        .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
        .await
        .unwrap();
    bed.hub
        .list_models(&bed.scope, &slug("openai"), false)
        .await
        .unwrap();
    bed.hub
        .set_key(&bed.scope, &slug("openai"), Secret::new("sk-new"))
        .await
        .unwrap();
    let list = bed
        .hub
        .list_models(&bed.scope, &slug("openai"), false)
        .await
        .unwrap();
    assert_eq!(
        list.freshness,
        crate::catalog::Freshness::Fresh,
        "the old list is not served"
    );
    let last = bed.ports.http.requests().pop().unwrap();
    assert!(last.carried(&Secret::new("sk-new")) && !last.carried(&Secret::new(KEY)));
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("openai"))
            .await
            .unwrap()
            .health,
        ProviderHealth::Unknown
    );
}

// ---- default (G8) and pins (G22) ---------------------------------------------------------

#[tokio::test]
async fn guard_g8_set_default_needs_an_enabled_provider_and_rewrites_the_rows_model() {
    let bed = Bed::new();
    with_openai(&bed).await;
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("groq").with_key(Secret::new("gsk")),
        )
        .await
        .unwrap();
    let set = bed
        .hub
        .set_default(&bed.scope, ModelChoice::new(slug("groq"), model("llama-3")))
        .await
        .unwrap();
    assert_eq!(
        set.record.unwrap().record.model,
        Some(model("llama-3")),
        "the row's model follows"
    );
    assert_eq!(
        bed.hub.status(&bed.scope).await.unwrap().default,
        DefaultChoice::Full {
            provider: slug("groq"),
            model: model("llama-3")
        }
    );
    assert!(matches!(
        bed.hub
            .set_default(&bed.scope, ModelChoice::new(slug("nope"), model("m")))
            .await,
        Err(HubError::NotFound(_))
    ));
    bed.hub
        .set_enabled(&bed.scope, &slug("openai"), false, Confirm::no())
        .await
        .unwrap();
    assert!(matches!(
        bed.hub
            .set_default(&bed.scope, ModelChoice::new(slug("openai"), model("m")))
            .await,
        Err(HubError::Unresolved(Unresolved::Disabled(_)))
    ));
    let same = bed
        .hub
        .set_default(&bed.scope, ModelChoice::new(slug("groq"), model("llama-3")))
        .await
        .unwrap();
    assert_eq!(same.status, MutationStatus::Unchanged);
    let cleared = bed.hub.clear_default(&bed.scope).await.unwrap();
    assert_eq!(cleared.status, MutationStatus::Saved);
    assert!(cleared.record.is_none());
    assert_eq!(
        bed.hub.status(&bed.scope).await.unwrap().default,
        DefaultChoice::Unset
    );
}

#[tokio::test]
async fn guard_g22_a_pin_refuses_a_missing_disabled_or_keyless_provider() {
    let bed = Bed::new();
    with_openai(&bed).await;
    bed.hub
        .add(&bed.scope, ProviderDraft::new("groq"))
        .await
        .unwrap();
    let agent = AgentKey::new("agent:a");
    let choice = |s: &str| Some(ModelChoice::new(slug(s), model("m")));
    assert!(matches!(
        bed.hub.pin_agent(&bed.scope, &agent, choice("nope")).await,
        Err(HubError::NotFound(_))
    ));
    assert!(matches!(
        bed.hub.pin_agent(&bed.scope, &agent, choice("groq")).await,
        Err(HubError::Unresolved(Unresolved::NoKey(_)))
    ));
    bed.hub
        .set_enabled(&bed.scope, &slug("openai"), false, Confirm::in_use())
        .await
        .unwrap();
    assert!(matches!(
        bed.hub
            .pin_agent(&bed.scope, &agent, choice("openai"))
            .await,
        Err(HubError::Unresolved(Unresolved::Disabled(_)))
    ));
    bed.hub
        .set_enabled(&bed.scope, &slug("openai"), true, Confirm::no())
        .await
        .unwrap();
    let pinned = bed
        .hub
        .pin_agent(&bed.scope, &agent, choice("openai"))
        .await
        .unwrap();
    assert_eq!(pinned.status, MutationStatus::Saved);
    // A keyless kind that needs none can be pinned (a local runtime).
    bed.hub
        .add(&bed.scope, ProviderDraft::new("ollama"))
        .await
        .unwrap();
    bed.hub
        .pin_agent(&bed.scope, &AgentKey::new("agent:b"), choice("ollama"))
        .await
        .unwrap();
    // Removing the pin is always allowed, and repeated removal is a no-op.
    assert_eq!(
        bed.hub
            .pin_agent(&bed.scope, &agent, None)
            .await
            .unwrap()
            .status,
        MutationStatus::Saved
    );
    assert_eq!(
        bed.hub
            .pin_agent(&bed.scope, &agent, None)
            .await
            .unwrap()
            .status,
        MutationStatus::Unchanged
    );
}

#[tokio::test]
async fn ops_a_pin_to_a_reserved_model_word_is_refused() {
    let bed =
        Bed::with(|b| b.hub_policy(crate::hub::HubPolicy::new().reserved_model_words(["chat-v1"])));
    with_openai(&bed).await;
    assert!(matches!(
        bed.hub
            .pin_agent(
                &bed.scope,
                &AgentKey::new("a"),
                Some(ModelChoice::new(slug("openai"), model("chat-v1")))
            )
            .await,
        Err(HubError::Invalid(_))
    ));
}

// ---- workload routes ---------------------------------------------------------------------

#[tokio::test]
async fn ops_workload_keys_are_opaque_and_routes_are_stored_only_when_meaningful() {
    let bed = Bed::new();
    with_openai(&bed).await;
    let w = WorkloadKey::new("anything at all: even spaces");
    let route = ProviderRoute::provider(slug("openai")).with_model(model("gpt-y"));
    let saved = bed
        .hub
        .set_workload_route(&bed.scope, &w, Some(route.clone()))
        .await
        .unwrap();
    assert_eq!(saved.status, MutationStatus::Saved);
    assert_eq!(
        bed.stored()["workload_routes"][w.as_str()]["target"]["value"],
        "openai"
    );
    for bad in [RouteTarget::Default, RouteTarget::Ephemeral] {
        assert!(matches!(
            bed.hub
                .set_workload_route(&bed.scope, &w, Some(ProviderRoute::new(bad)))
                .await,
            Err(HubError::Invalid(_))
        ));
    }
    assert!(matches!(
        bed.hub
            .set_workload_route(&bed.scope, &w, Some(ProviderRoute::provider(slug("nope"))))
            .await,
        Err(HubError::NotFound(_))
    ));
    // Non-provider targets are stored without a provider check.
    bed.hub
        .set_workload_route(
            &bed.scope,
            &WorkloadKey::new("w2"),
            Some(ProviderRoute::new(RouteTarget::Managed)),
        )
        .await
        .unwrap();
    assert_eq!(
        bed.hub
            .set_workload_route(&bed.scope, &w, None)
            .await
            .unwrap()
            .status,
        MutationStatus::Saved
    );
    assert_eq!(
        bed.hub
            .set_workload_route(&bed.scope, &w, None)
            .await
            .unwrap()
            .status,
        MutationStatus::Unchanged
    );
    let _ = ReasonCode::Auth;
}

#[tokio::test]
async fn ops_the_hosts_agents_and_workloads_are_merged_without_duplicates() {
    let mut host = UsedBy::default();
    host.agents.push(AgentKey::new("agent:writer")); // also pinned in the hub
    host.agents.push(AgentKey::new("agent:host-only"));
    host.workloads.push(WorkloadKey::new("tier:chat")); // also routed in the hub
    host.workloads.push(WorkloadKey::new("tier:host-only"));
    host.default_choice = false;
    let bed = Bed::with(|b| b.usage_query(std::sync::Arc::new(HostPins(host))));
    with_openai(&bed).await;
    bed.hub.clear_default(&bed.scope).await.unwrap();
    bed.hub
        .pin_agent(
            &bed.scope,
            &AgentKey::new("agent:writer"),
            Some(ModelChoice::new(slug("openai"), model("gpt-x"))),
        )
        .await
        .unwrap();
    bed.hub
        .set_workload_route(
            &bed.scope,
            &WorkloadKey::new("tier:chat"),
            Some(ProviderRoute::provider(slug("openai"))),
        )
        .await
        .unwrap();
    let error = bed
        .hub
        .remove(&bed.scope, &slug("openai"), Confirm::no())
        .await
        .unwrap_err();
    let HubError::InUse(used) = error else {
        panic!("{error:?}")
    };
    assert_eq!(
        (used.agents.len(), used.workloads.len()),
        (2, 2),
        "{used:?}"
    );
}

#[tokio::test]
async fn ops_a_cli_kind_in_a_stored_document_cannot_take_a_key() {
    let bed = Bed::new();
    bed.ports.config.put_raw(
        &bed.scope,
        serde_json::json!({"providers": [{"id": "p", "slug": "claude-code", "label": "Claude Code",
            "kind": "claude-code", "base_url": ""}]})
        .to_string(),
    );
    assert!(matches!(
        bed.hub
            .set_key(&bed.scope, &slug("claude-code"), Secret::new("k"))
            .await,
        Err(HubError::Unsupported {
            op: Operation::SetKey,
            ..
        })
    ));
    assert!(matches!(
        bed.hub
            .list_models(&bed.scope, &slug("claude-code"), false)
            .await,
        Err(HubError::Unsupported {
            op: Operation::ListModels,
            ..
        })
    ));
    assert!(matches!(
        bed.hub
            .edit(
                &bed.scope,
                &slug("claude-code"),
                ProviderPatch::new().key(Secret::new(" "))
            )
            .await,
        Err(HubError::Invalid(_))
    ));
    assert!(bed.ports.credentials.is_empty());
}
