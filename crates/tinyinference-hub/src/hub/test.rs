//! Tests for the builder, the change loop and the hub as a whole.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::future::join_all;

use super::fixtures::{Bed, KEY, model, models_body, slug};
use super::*;
use crate::catalog::Fetched;
use crate::catalogue::custom_descriptor;
use crate::config::{DefaultChoice, ProviderDraft};
use crate::descriptor::ProviderDescriptor;
use crate::error::{HubError, InputField, InvalidInput};
use crate::ids::ScopeKey;
use crate::kinds::{DriverContext, KindDriver, Target};
use crate::secret::Secret;
use crate::taxonomy::ProviderGroup;
use crate::testkit::{Match, MemoryPorts, Scripted};

// ---- builder -----------------------------------------------------------------------------

#[test]
fn builder_names_the_missing_required_port() {
    let ports = MemoryPorts::new();
    let cases: Vec<(HubBuilder, &str)> = vec![
        (Hub::builder(), "credential store"),
        (
            Hub::builder().credentials_arc(ports.credentials.clone()),
            "configuration store",
        ),
        (
            Hub::builder()
                .credentials_arc(ports.credentials.clone())
                .config_arc(ports.config.clone()),
            "http",
        ),
        (
            Hub::builder()
                .credentials_arc(ports.credentials.clone())
                .config_arc(ports.config.clone())
                .http_arc(ports.http.clone()),
            "clock",
        ),
    ];
    for (builder, port) in cases {
        let error = builder.build().unwrap_err();
        match &error {
            HubError::Invalid(InvalidInput::Malformed {
                field: InputField::Config,
                reason,
            }) => {
                assert!(reason.contains(port), "{reason}");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn builder_defaults_are_the_most_restrictive_and_debug_is_safe() {
    let ports = MemoryPorts::new();
    let hub = Hub::builder()
        .credentials_arc(ports.credentials.clone())
        .config_arc(ports.config.clone())
        .http_arc(ports.http.clone())
        .clock(ports.clock.clone())
        .build()
        .unwrap();
    assert_eq!(hub.policy(), &crate::policy::EndpointPolicy::hosted());
    assert!(hub.kinds().len() > 30, "every built-in kind and custom");
    assert!(format!("{hub:?}").contains("Hub"));
    assert!(format!("{:?}", Hub::builder()).contains("HubBuilder"));
    let managed = ManagedConfig::new("https://user:hunter2@api.test/x")
        .source(crate::credential::StaticSource::new(Secret::new(KEY)));
    let text = format!("{managed:?}");
    assert!(!text.contains("hunter2") && !text.contains(KEY) && text.contains("sources: 1"));
}

#[derive(Debug)]
struct AcmeKind {
    descriptor: ProviderDescriptor,
    seen: Mutex<Vec<(String, Option<String>)>>,
}

impl AcmeKind {
    fn new() -> Arc<Self> {
        let mut descriptor = custom_descriptor();
        descriptor.kind = "acme-kind".into();
        descriptor.label = "Acme kind";
        Arc::new(Self {
            descriptor,
            seen: Mutex::new(Vec::new()),
        })
    }
}

#[async_trait]
impl KindDriver for AcmeKind {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    async fn list_models(
        &self,
        _cx: &DriverContext<'_>,
        target: &Target<'_>,
    ) -> Result<Fetched, HubError> {
        self.seen.lock().unwrap().push((
            target.slug.to_string(),
            target.credential.map(|k| k.expose().to_string()),
        ));
        Ok(Fetched::new(vec![crate::catalog::ModelEntry::new(model(
            "acme-m",
        ))]))
    }
}

#[tokio::test]
async fn builder_registers_custom_kind_and_the_driver_receives_only_its_own_secret() {
    let acme = AcmeKind::new();
    let bed = Bed::with(|b| b.kind(acme.clone()));
    for (label, key) in [("First", "sk-first"), ("Second", "sk-second")] {
        bed.hub
            .add(
                &bed.scope,
                ProviderDraft::new("acme-kind")
                    .with_label(label)
                    .with_base_url("https://acme.test/v1")
                    .with_key(Secret::new(key)),
            )
            .await
            .unwrap();
    }
    // A third provider of another kind holds a key the driver must never see.
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    for name in ["first", "second"] {
        assert_eq!(
            bed.hub
                .list_models(&bed.scope, &slug(name), false)
                .await
                .unwrap()
                .ids(),
            ["acme-m"]
        );
    }
    let seen = acme.seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        [
            ("first".into(), Some("sk-first".into())),
            ("second".into(), Some("sk-second".into()))
        ]
    );
    assert!(!format!("{seen:?}").contains(KEY));
    // A kind nobody registered is a typed error, not a silent custom endpoint.
    let bare = Bed::new();
    let error = bare
        .hub
        .add(&bare.scope, ProviderDraft::new("acme-kind"))
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::NotFound(_)));
}

#[test]
fn builder_kind_registry_controls() {
    let ports = MemoryPorts::new();
    let empty = ports.builder().no_builtin_kinds().build().unwrap();
    assert_eq!(empty.kinds().len(), 0);
    let restored = ports
        .builder()
        .no_builtin_kinds()
        .kind(AcmeKind::new())
        .with_builtin_kinds()
        .build()
        .unwrap();
    assert!(
        restored.kinds().get(&"openai".into()).is_some()
            && restored.kinds().get(&"acme-kind".into()).is_some()
    );
}

// ---- the change loop ---------------------------------------------------------------------

#[tokio::test]
async fn hub_a_change_that_alters_nothing_saves_nothing() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let raw = bed.ports.config.raw(&bed.scope).unwrap();
    bed.hub
        .set_workload_route(&bed.scope, &"w".into(), None)
        .await
        .unwrap();
    bed.hub.clear_default(&bed.scope).await.unwrap();
    bed.hub.clear_default(&bed.scope).await.unwrap();
    bed.ports.config.conflict_next(50);
    // A no-op does not touch the store, so injected conflicts cannot reach it.
    let mutation = bed.hub.clear_default(&bed.scope).await.unwrap();
    assert_eq!(mutation.status, MutationStatus::Unchanged);
    assert_ne!(
        bed.ports.config.raw(&bed.scope).unwrap(),
        raw,
        "the first clear did change the default"
    );
}

#[tokio::test]
async fn hub_the_managed_endpoint_is_the_hosts_and_is_rewritten_from_the_builder() {
    let ports = MemoryPorts::new();
    let scope = ScopeKey::new("company:acme");
    let old = ports
        .builder()
        .managed(ManagedConfig::new("https://old.tinyhumans.test/x"))
        .build()
        .unwrap();
    old.set_enabled(&scope, &slug("tinyhumans"), false, Confirm::no())
        .await
        .unwrap();
    let stored = ports.config.raw(&scope).unwrap();
    assert!(
        stored.contains("https://old.tinyhumans.test/x") && stored.contains("\"enabled\":false")
    );
    let new = ports
        .builder()
        .managed(ManagedConfig::new("https://new.tinyhumans.test/x"))
        .build()
        .unwrap();
    let status = new.status(&scope).await.unwrap();
    assert_eq!(
        status.providers[0].view.record.base_url,
        "https://new.tinyhumans.test/x"
    );
    assert!(
        !status.providers[0].view.record.enabled,
        "the operator's choice survives the host's endpoint change"
    );
    // Without a managed config the row is simply not listed.
    let none = ports.builder().build().unwrap();
    assert!(
        none.status(&ScopeKey::new("other"))
            .await
            .unwrap()
            .providers
            .is_empty()
    );
}

#[tokio::test]
async fn sim_concurrent_writers_never_lose_a_provider_and_g9_holds() {
    let bed = Bed::new();
    let kinds = [
        "openai", "groq", "mistral", "deepseek", "together", "xai", "cerebras", "nvidia",
    ];
    let adds = kinds.iter().map(|kind| {
        let hub = bed.hub.clone();
        let scope = bed.scope.clone();
        async move {
            hub.add(&scope, ProviderDraft::new(*kind).with_model(model("m")))
                .await
        }
    });
    let results = join_all(adds).await;
    // The in-memory store is a single mutex, so every writer eventually wins
    // within its three attempts or reports Conflict; none may corrupt.
    let ok = results.iter().filter(|r| r.is_ok()).count();
    assert!(ok >= 1);
    for result in &results {
        assert!(
            result.is_ok() || matches!(result, Err(HubError::Conflict)),
            "{result:?}"
        );
    }
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert_eq!(
        status.providers.len(),
        ok,
        "exactly the writers that succeeded are stored"
    );
    let defaults = status
        .providers
        .iter()
        .filter(|p| p.view.is_default)
        .count();
    assert_eq!(defaults, 1, "G9: exactly one provider became the default");
    assert!(matches!(status.default, DefaultChoice::Full { .. }));
}

#[tokio::test]
async fn sim_concurrent_writers_with_injected_cas_faults_stay_consistent() {
    let bed = Bed::new();
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_model(model("m")),
        )
        .await
        .unwrap();
    bed.ports.config.conflict_next(2);
    let toggles = (0..6).map(|i| {
        let hub = bed.hub.clone();
        let scope = bed.scope.clone();
        async move {
            hub.set_enabled(&scope, &slug("openai"), i % 2 == 0, Confirm::in_use())
                .await
        }
    });
    for result in join_all(toggles).await {
        assert!(
            result.is_ok() || matches!(result, Err(HubError::Conflict)),
            "{result:?}"
        );
    }
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert_eq!(status.providers.len(), 1);
    assert!(
        matches!(status.default, DefaultChoice::Full { .. }),
        "toggling never touches the default"
    );
}

#[tokio::test]
async fn hub_a_removed_provider_cannot_be_probed_or_listed() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    bed.hub.clear_default(&bed.scope).await.unwrap();
    bed.hub
        .remove(&bed.scope, &slug("openai"), Confirm::no())
        .await
        .unwrap();
    bed.ports.http.route(
        Match::prefix("https://api.openai.com/"),
        Scripted::json(200, &models_body(&["m"])),
    );
    assert!(
        bed.hub
            .list_models(&bed.scope, &slug("openai"), false)
            .await
            .is_err()
    );
    assert_eq!(bed.ports.http.request_count(), 0);
    let _ = ProviderGroup::Cloud;
}

// ---- value types and builder corners -----------------------------------------------------

#[test]
fn hub_value_types_build_debug_and_report() {
    let patch = ProviderPatch::new()
        .label("Work")
        .base_url("https://user:hunter2@llm.acme.test/v1?api_key=abc")
        .model(model("m"))
        .key(Secret::new(KEY));
    let text = format!("{patch:?}");
    assert!(
        !text.contains("hunter2") && !text.contains(KEY) && !text.contains("abc"),
        "{text}"
    );
    assert!(text.contains("Work"));
    let options = ConnectOptions::default()
        .make_default(true)
        .add_anyway(true)
        .depth(crate::taxonomy::TestDepth::Completion);
    assert!(options.make_default && options.add_anyway);
    assert_eq!(options.depth, crate::taxonomy::TestDepth::Completion);
    let policy = HubPolicy::new()
        .one_row_per_kind(true)
        .reserved_model_words(["a", "b"])
        .retest_after(std::time::Duration::from_secs(9));
    assert!(policy.one_row_per_kind);
    assert_eq!(policy.reserved_model_words, ["a", "b"]);
    assert_eq!(policy.retest_after, std::time::Duration::from_secs(9));
    assert_eq!(
        HubPolicy::default().retest_after,
        std::time::Duration::from_secs(300)
    );
    assert_eq!(Confirm::no(), Confirm::default());
    assert!(Confirm::in_use().in_use);
    let configured = KeyState::Configured(crate::credential::CredentialOrigin::AccountKey);
    assert!(configured.is_configured() && configured.origin().is_some());
    for state in [KeyState::Missing, KeyState::Unreadable] {
        assert!(!state.is_configured() && state.origin().is_none());
    }
}

#[tokio::test]
async fn builder_an_extra_credential_source_answers_after_the_stored_key() {
    let bed = Bed::with(|b| {
        b.credential_source(
            "groq",
            crate::credential::StaticSource::new(Secret::new("gsk-static")),
        )
    });
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("groq").with_model(model("m")),
        )
        .await
        .unwrap();
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &crate::route::TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(
        turn.origin,
        Some(crate::credential::CredentialOrigin::Static)
    );
    bed.hub
        .set_key(&bed.scope, &slug("groq"), Secret::new("gsk-stored"))
        .await
        .unwrap();
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &crate::route::TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(
        turn.origin,
        Some(crate::credential::CredentialOrigin::ProviderKey)
    );
    // A source for a kind nothing registered still gets a chain (the store first).
    let bed = Bed::with(|b| {
        b.credential_source(
            "brand-new",
            crate::credential::StaticSource::new(Secret::new("k")),
        )
        .http(crate::testkit::ScriptedHttp::new(
            crate::testkit::FakeClock::new(),
        ))
    });
    assert!(bed.hub.kinds().get(&"brand-new".into()).is_none());
}

#[test]
fn hub_a_credential_has_a_cache_identity_only_when_presented_and_not_rotating() {
    use crate::credential::CredentialOrigin;
    let with = |origin| super::Credential {
        key: Some(crate::secret::Secret::new("sk-not-a-real-key")),
        origin: Some(origin),
        epoch: 0,
    };
    let stored = with(CredentialOrigin::ProviderKey);
    assert_eq!(
        stored.cache_identity(true),
        Some(crate::secret::Secret::new("sk-not-a-real-key").id())
    );
    // A key that is configured but not sent (an auth style of none) shares the
    // slot with everyone.
    assert_eq!(stored.cache_identity(false), None);
    // A token the source rotates by itself never splits the account's list.
    for rotating in [
        CredentialOrigin::InstanceIdentity,
        CredentialOrigin::SessionJwt,
        CredentialOrigin::OAuth,
    ] {
        assert_eq!(with(rotating).cache_identity(true), None);
    }
    let none = super::Credential {
        key: None,
        origin: None,
        epoch: 0,
    };
    assert_eq!(none.cache_identity(true), None);
}

#[tokio::test]
async fn hub_a_key_announcement_that_cannot_read_the_slot_makes_no_claim_about_it() {
    // A store that just failed part-way is often still down: the event must not
    // say the key is gone (or present) on the strength of a failed read.
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    bed.ports.events.drain();
    bed.ports
        .credentials
        .inject(crate::ports::memory::CredentialFault::Read);
    bed.hub
        .announce_key_state(&bed.scope, &crate::hub::fixtures::slug("openai"))
        .await;
    assert!(
        !bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, crate::ports::HubEvent::KeyChanged { .. })),
        "no KeyChanged on an unreadable slot"
    );
    bed.ports.credentials.heal();
    bed.hub
        .announce_key_state(&bed.scope, &crate::hub::fixtures::slug("openai"))
        .await;
    assert!(
        bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, crate::ports::HubEvent::KeyChanged { present: true, .. }))
    );
}
