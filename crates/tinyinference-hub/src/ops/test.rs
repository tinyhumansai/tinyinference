//! Tests for adding, connecting and editing providers, and the guards they
//! enforce (`guard_gNN_*` names follow 04-operation-matrix).

use serde_json::json;

use crate::config::{DefaultChoice, ProviderDraft};
use crate::credential::CredentialOrigin;
use crate::error::{HubError, InvalidInput, Operation, PolicyViolation, ReasonCode};
use crate::health::ProviderHealth;
use crate::hub::fixtures::{Bed, KEY, model, models_body, slug};
use crate::hub::{ConnectOptions, KeyState, MutationStatus, ProviderPatch};
use crate::ports::HealthStore;
use crate::ports::memory::CredentialFault;
use crate::secret::Secret;
use crate::testkit::{Match, Scripted};

fn bed_with_openai() -> Bed {
    let bed = Bed::new();
    bed.openai_lists(&["gpt-x", "gpt-y"]);
    bed
}

// ---- add ---------------------------------------------------------------------------------

#[tokio::test]
async fn ops_add_saves_a_record_and_the_key_lives_only_in_the_credential_store() {
    let bed = Bed::new();
    let mutation = bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    assert_eq!(mutation.status, MutationStatus::Saved);
    let view = mutation.record.clone().unwrap();
    assert_eq!(view.record.slug, slug("openai"));
    assert_eq!(view.record.base_url, "https://api.openai.com/v1");
    assert_eq!(
        view.key,
        KeyState::Configured(CredentialOrigin::ProviderKey)
    );
    assert_eq!(bed.key_of("openai").await.as_deref(), Some(KEY));
    // Invariant 1: nothing sent, nothing on the record.
    assert_eq!(bed.ports.http.request_count(), 0);
    assert!(!bed.ports.config.raw(&bed.scope).unwrap().contains(KEY));
    assert!(!format!("{mutation:?}").contains(KEY));
}

#[tokio::test]
async fn guard_g9_the_first_provider_becomes_the_default_and_the_second_does_not() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert_eq!(
        status.default,
        DefaultChoice::Full {
            provider: slug("openai"),
            model: model("gpt-x")
        }
    );
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("groq").with_model(model("llama")),
        )
        .await
        .unwrap();
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert!(
        matches!(&status.default, DefaultChoice::Full { provider, .. } if *provider == slug("openai"))
    );
}

#[tokio::test]
async fn guard_g9_a_provider_without_a_model_does_not_become_the_default() {
    let bed = Bed::new();
    bed.hub
        .add(&bed.scope, ProviderDraft::new("openai"))
        .await
        .unwrap();
    assert_eq!(
        bed.hub.status(&bed.scope).await.unwrap().default,
        DefaultChoice::Unset
    );
}

#[tokio::test]
async fn ops_make_default_needs_a_model_and_is_checked_before_anything_is_written() {
    let bed = bed_with_openai();
    let error = bed
        .hub
        .connect(
            &bed.scope,
            ProviderDraft::new("openai").with_key(Secret::new(KEY)),
            ConnectOptions::default().make_default(true),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Invalid(InvalidInput::Empty(_))));
    assert_eq!(bed.key_of("openai").await, None, "nothing was written");
    assert_eq!(bed.ports.http.request_count(), 0);
}

#[tokio::test]
async fn guard_g4_one_row_per_kind_is_a_policy_and_instances_are_the_default() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    // Default: a second account of the same kind is another instance.
    let second = bed
        .hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_label("Work OpenAI"),
        )
        .await
        .unwrap();
    assert_eq!(second.record.unwrap().record.slug, slug("work-openai"));
    // A second row that names the kind's own label is the same row.
    let again = bed.hub.add(&bed.scope, ProviderDraft::new("openai")).await;
    assert!(matches!(again, Err(HubError::AlreadyExists { slug: s }) if s == slug("openai")));

    let strict = Bed::with(|b| b.hub_policy(crate::hub::HubPolicy::new().one_row_per_kind(true)));
    strict
        .hub
        .add(&strict.scope, ProviderDraft::new("openai"))
        .await
        .unwrap();
    let error = strict
        .hub
        .add(
            &strict.scope,
            ProviderDraft::new("openai").with_label("Work OpenAI"),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::AlreadyExists { slug: s } if s == slug("openai")));
}

#[tokio::test]
async fn guard_g15_names_are_validated_taken_and_reserved() {
    let bed = Bed::new();
    let custom = |label: &str| {
        ProviderDraft::new("custom")
            .with_label(label)
            .with_base_url("https://llm.acme.test/v1")
    };
    for (label, want) in [
        ("", "empty"),
        ("   ", "empty"),
        (&"x".repeat(81), "long"),
        ("OpenAI", "reserved"),
        ("Tiny Humans", "ok-not-reserved"),
        ("managed", "reserved"),
    ] {
        let result = bed.hub.add(&bed.scope, custom(label)).await;
        match want {
            "empty" => assert!(
                matches!(result, Err(HubError::Invalid(InvalidInput::Empty(_)))),
                "{label:?}"
            ),
            "long" => assert!(matches!(
                result,
                Err(HubError::Invalid(InvalidInput::TooLong { .. }))
            )),
            "reserved" => assert!(
                matches!(
                    result,
                    Err(HubError::Invalid(InvalidInput::Reserved { .. }))
                ),
                "{label:?}"
            ),
            _ => assert!(result.is_ok(), "{label:?}: {result:?}"),
        }
    }
    bed.hub.add(&bed.scope, custom("Acme LLM")).await.unwrap();
    let taken = bed.hub.add(&bed.scope, custom("acme llm")).await;
    assert!(matches!(taken, Err(HubError::AlreadyExists { .. })));
}

#[tokio::test]
async fn guard_g2_a_typed_endpoint_for_a_cloud_preset_is_ignored() {
    let bed = Bed::new();
    let mutation = bed
        .hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_base_url("https://evil.test/v1"),
        )
        .await
        .unwrap();
    assert_eq!(
        mutation.record.unwrap().record.base_url,
        "https://api.openai.com/v1"
    );
}

#[tokio::test]
async fn guard_g14_model_ids_pass_the_hosts_reserved_words() {
    let bed =
        Bed::with(|b| b.hub_policy(crate::hub::HubPolicy::new().reserved_model_words(["chat-v1"])));
    let error = bed
        .hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_model(model("chat-v1")),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        HubError::Invalid(InvalidInput::Reserved { .. })
    ));
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_model(model("gpt-5")),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn guard_g16_and_g17_a_bad_endpoint_is_refused_at_add() {
    let bed = Bed::new();
    let custom = |url: &str| {
        ProviderDraft::new("custom")
            .with_label("Acme")
            .with_base_url(url)
    };
    for url in [
        "https://user:pass@llm.acme.test/v1",
        "https://llm.acme.test/v1?api_key=abc",
        "http://169.254.169.254/latest",
        "http://10.0.0.5/v1",
        "ftp://llm.acme.test/v1",
    ] {
        // `desktop()` allows loopback but never link-local or private ranges.
        let result = bed.hub.add(&bed.scope, custom(url)).await;
        assert!(
            matches!(result, Err(HubError::Policy(_)) | Err(HubError::Invalid(_))),
            "{url}: {result:?}"
        );
    }
    // Loopback is allowed under the desktop policy, and only there.
    bed.hub
        .add(&bed.scope, custom("http://localhost:9/v1"))
        .await
        .expect("desktop allows loopback");
    let hosted = Bed::with(|b| b.policy(crate::policy::EndpointPolicy::hosted()));
    let error = hosted
        .hub
        .add(
            &hosted.scope,
            ProviderDraft::new("ollama").with_base_url("http://localhost:11434"),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::Policy(_)),
        "a hosted tenant has no loopback"
    );
    // Nothing was written by any refusal.
    assert!(hosted.ports.config.raw(&hosted.scope).is_none());
}

#[tokio::test]
async fn ops_unsupported_and_unknown_kinds_are_typed() {
    let bed = Bed::new();
    for kind in ["claude-code", "codex", "tinyhumans"] {
        let error = bed
            .hub
            .add(&bed.scope, ProviderDraft::new(kind))
            .await
            .unwrap_err();
        assert!(
            matches!(
                error,
                HubError::Unsupported {
                    op: Operation::Add,
                    ..
                }
            ),
            "{kind}: {error:?}"
        );
    }
    let error = bed
        .hub
        .connect(
            &bed.scope,
            ProviderDraft::new("claude-code"),
            ConnectOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        HubError::Unsupported {
            op: Operation::Connect,
            ..
        }
    ));
    let unknown = bed.hub.add(&bed.scope, ProviderDraft::new("nope")).await;
    assert!(matches!(unknown, Err(HubError::NotFound(_))), "{unknown:?}");
}

#[tokio::test]
async fn ops_an_empty_key_is_refused_and_a_key_is_trimmed() {
    let bed = Bed::new();
    let error = bed
        .hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_key(Secret::new("   ")),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Invalid(InvalidInput::Empty(_))));
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("openai").with_key(Secret::new("  sk-x  \n")),
        )
        .await
        .unwrap();
    assert_eq!(bed.key_of("openai").await.as_deref(), Some("sk-x"));
}

// ---- connect -----------------------------------------------------------------------------

#[tokio::test]
async fn ops_connect_checks_the_catalog_and_records_health() {
    let bed = bed_with_openai();
    let mutation = bed
        .hub
        .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
        .await
        .unwrap();
    assert_eq!(mutation.status, MutationStatus::Saved);
    let probe = mutation.probe.unwrap();
    assert_eq!(probe.model_count(), Some(2));
    let status = bed.hub.health(&bed.scope, &slug("openai")).await.unwrap();
    assert_eq!(status.health, ProviderHealth::Ok);
    assert!(status.view.is_default, "G9 also applies to connect");
    // The request carried the key, and the recorded log shows it redacted.
    let sent = bed.ports.http.requests();
    assert!(sent[0].carried(&Secret::new(KEY)));
    assert!(!format!("{:?}", sent[0]).contains(KEY));
}

#[tokio::test]
async fn sim_connect_rollback_on_auth_leaves_no_row_and_restores_the_previous_key() {
    let bed = Bed::new();
    bed.openai_rejects_key();
    // A key already stored under the slot (an earlier, working one).
    bed.store_key("openai", "sk-previous").await;
    let error = bed
        .hub
        .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
        .await
        .unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Auth);
    assert!(error.rolls_back(crate::taxonomy::ProviderGroup::Cloud));
    let message = error.user_message(crate::error::CopyContext::undone("OpenAI"));
    assert!(
        message.contains("rejected the credential") && !message.contains("Saved"),
        "{message}"
    );
    // Row gone, previous key back, nothing remembered about the rejected one.
    let status = bed.hub.status(&bed.scope).await.unwrap();
    assert!(status.providers.is_empty());
    assert_eq!(bed.key_of("openai").await.as_deref(), Some("sk-previous"));
    assert_eq!(
        bed.hub.status(&bed.scope).await.unwrap().default,
        DefaultChoice::Unset,
        "the default this add made is undone too"
    );
    assert_eq!(
        bed.ports
            .health
            .get(&bed.scope, &slug("openai"))
            .await
            .unwrap(),
        None
    );
}

#[tokio::test]
async fn guard_g12_a_rolled_back_add_with_no_previous_key_leaves_the_slot_empty() {
    let bed = Bed::new();
    bed.openai_rejects_key();
    bed.hub
        .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
        .await
        .unwrap_err();
    assert_eq!(bed.key_of("openai").await, None);
    assert!(bed.ports.credentials.is_empty());
}

#[tokio::test]
async fn guard_g11_add_anyway_keeps_the_row_even_when_the_key_is_rejected() {
    let bed = Bed::new();
    bed.openai_rejects_key();
    let mutation = bed
        .hub
        .connect(
            &bed.scope,
            bed.openai_draft(),
            ConnectOptions::default().add_anyway(true),
        )
        .await
        .unwrap();
    assert_eq!(mutation.status, MutationStatus::SavedWithWarning);
    assert!(mutation.note.contains("rejected the credential"));
    assert_eq!(bed.key_of("openai").await.as_deref(), Some(KEY));
    let health = bed.hub.health(&bed.scope, &slug("openai")).await.unwrap();
    assert_eq!(health.health, ProviderHealth::Down(ReasonCode::Auth));
}

#[tokio::test]
async fn guard_g11_only_a_rejected_key_rolls_a_cloud_add_back() {
    for (script, reason) in [
        (
            Scripted::text(
                429,
                "You exceeded your current quota, please check your plan and billing details",
            ),
            ReasonCode::Quota,
        ),
        (
            Scripted::text(503, "service unavailable"),
            ReasonCode::Unknown,
        ),
        (Scripted::ConnectRefused, ReasonCode::Endpoint),
        (Scripted::Timeout, ReasonCode::Timeout),
    ] {
        let bed = Bed::new();
        bed.ports
            .http
            .route(Match::prefix("https://api.openai.com/v1/"), script);
        let mutation = bed
            .hub
            .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
            .await
            .unwrap_or_else(|e| panic!("{reason}: a cloud add is kept: {e:?}"));
        assert_eq!(
            mutation.status,
            MutationStatus::SavedWithWarning,
            "{reason}"
        );
        assert!(
            mutation.note.starts_with("Saved")
                || mutation.note.contains("did not answer")
                || mutation.note.contains("nothing answered"),
            "{}",
            mutation.note
        );
        assert_eq!(bed.key_of("openai").await.as_deref(), Some(KEY));
        assert_eq!(mutation.probe.unwrap().failure.unwrap().reason, reason);
    }
}

#[tokio::test]
async fn sim_local_rollback_on_timeout_and_unreachable_but_add_anyway_keeps_it() {
    for script in [Scripted::Timeout, Scripted::ConnectRefused] {
        let bed = Bed::new();
        bed.ports
            .http
            .route(Match::prefix("http://localhost:11434/"), script.clone());
        let error = bed
            .hub
            .connect(
                &bed.scope,
                ProviderDraft::new("ollama"),
                ConnectOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(
            error.reason(),
            ReasonCode::Timeout | ReasonCode::Endpoint
        ));
        assert!(
            bed.hub
                .status(&bed.scope)
                .await
                .unwrap()
                .providers
                .is_empty()
        );

        let kept = bed
            .hub
            .connect(
                &bed.scope,
                ProviderDraft::new("ollama"),
                ConnectOptions::default().add_anyway(true),
            )
            .await
            .unwrap();
        assert_eq!(kept.status, MutationStatus::SavedWithWarning);
    }
}

#[tokio::test]
async fn sim_connect_add_anyway_then_a_later_retest_recovers_the_health() {
    let bed = Bed::new();
    bed.ports.http.route(
        Match::prefix("http://localhost:11434/"),
        Scripted::ConnectRefused,
    );
    bed.hub
        .connect(
            &bed.scope,
            ProviderDraft::new("ollama"),
            ConnectOptions::default().add_anyway(true),
        )
        .await
        .unwrap();
    let health = bed.hub.health(&bed.scope, &slug("ollama")).await.unwrap();
    assert_eq!(health.health, ProviderHealth::Down(ReasonCode::Endpoint));
    bed.ports.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::json(200, &models_body(&["llama3"])),
    );
    let report = bed
        .hub
        .test(
            &bed.scope,
            &slug("ollama"),
            crate::taxonomy::TestDepth::Catalog,
            None,
        )
        .await
        .unwrap();
    assert!(report.ok());
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("ollama"))
            .await
            .unwrap()
            .health,
        ProviderHealth::Ok
    );
}

#[tokio::test]
async fn ops_a_policy_refusal_is_undone_even_with_add_anyway() {
    // The literal check passes at add (a public name), but the name resolves to
    // a private address when the request is made.
    let bed = Bed::new();
    bed.ports.http.route(
        Match::prefix("https://llm.acme.test/"),
        Scripted::json(200, &models_body(&["m"])).resolving_to(vec!["10.0.0.9".parse().unwrap()]),
    );
    let draft = ProviderDraft::new("custom")
        .with_label("Acme")
        .with_base_url("https://llm.acme.test/v1")
        .with_key(Secret::new(KEY));
    let error = bed
        .hub
        .connect(
            &bed.scope,
            draft,
            ConnectOptions::default().add_anyway(true),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::Policy(PolicyViolation::Endpoint(_))),
        "{error:?}"
    );
    assert!(
        bed.hub
            .status(&bed.scope)
            .await
            .unwrap()
            .providers
            .is_empty()
    );
    assert_eq!(bed.key_of("acme").await, None);
}

#[tokio::test]
async fn ops_connect_without_a_key_saves_unchecked_with_a_warning() {
    let bed = Bed::new();
    let mutation = bed
        .hub
        .connect(
            &bed.scope,
            ProviderDraft::new("openai"),
            ConnectOptions::default(),
        )
        .await
        .unwrap();
    assert_eq!(mutation.status, MutationStatus::SavedWithWarning);
    assert!(mutation.probe.is_none());
    assert_eq!(bed.ports.http.request_count(), 0);
    assert!(mutation.note.contains("without a key"));
}

#[tokio::test]
async fn ops_connect_depth_rules_are_checked_before_any_write() {
    let bed = Bed::new();
    // Completion needs a model.
    let error = bed
        .hub
        .connect(
            &bed.scope,
            ProviderDraft::new("openai").with_key(Secret::new(KEY)),
            ConnectOptions::default().depth(crate::taxonomy::TestDepth::Completion),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Invalid(_)));
    // OpenAI has no key-only check.
    let error = bed
        .hub
        .connect(
            &bed.scope,
            bed.openai_draft(),
            ConnectOptions::default().depth(crate::taxonomy::TestDepth::KeyOnly),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Unsupported { .. }), "{error:?}");
    assert!(bed.ports.credentials.is_empty() && bed.ports.config.raw(&bed.scope).is_none());
}

#[tokio::test]
async fn ops_connect_at_completion_depth_pings_the_model() {
    let bed = Bed::new();
    bed.openai_chat_ok();
    let mutation = bed
        .hub
        .connect(
            &bed.scope,
            bed.openai_draft(),
            ConnectOptions::default().depth(crate::taxonomy::TestDepth::Completion),
        )
        .await
        .unwrap();
    assert_eq!(mutation.status, MutationStatus::Saved);
    let sent = bed.ports.http.requests();
    assert_eq!(sent.len(), 1);
    assert!(
        sent[0]
            .body
            .as_deref()
            .unwrap()
            .contains("max_completion_tokens")
    );
}

#[tokio::test]
async fn ops_a_store_outage_during_the_key_write_changes_nothing() {
    let bed = Bed::new();
    bed.ports.credentials.inject(CredentialFault::Write);
    let error = bed
        .hub
        .add(&bed.scope, bed.openai_draft())
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::StoreUnreadable { .. }));
    assert!(
        bed.hub
            .status(&bed.scope)
            .await
            .unwrap()
            .providers
            .is_empty(),
        "the record is taken back out when its key cannot be written"
    );
    bed.ports.credentials.heal();
    // And an unreadable slot is never read as "no previous key".
    bed.ports.credentials.inject(CredentialFault::Read);
    let error = bed
        .hub
        .add(&bed.scope, bed.openai_draft())
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::StoreUnreadable { .. }));
}

#[tokio::test]
async fn ops_a_lost_config_write_puts_the_previous_key_back() {
    let bed = Bed::new();
    bed.store_key("openai", "sk-previous").await;
    // Every attempt loses the compare-and-swap.
    bed.ports.config.conflict_next(10);
    let error = bed
        .hub
        .add(&bed.scope, bed.openai_draft())
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Conflict));
    assert_eq!(bed.key_of("openai").await.as_deref(), Some("sk-previous"));
}

#[tokio::test]
async fn ops_one_lost_config_write_is_retried_transparently() {
    let bed = Bed::new();
    bed.ports.config.conflict_next(2);
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    assert_eq!(bed.hub.status(&bed.scope).await.unwrap().providers.len(), 1);
}

// ---- edit --------------------------------------------------------------------------------

#[tokio::test]
async fn ops_edit_changes_label_and_model_and_reports_unchanged_for_a_no_op() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let edited = bed
        .hub
        .edit(
            &bed.scope,
            &slug("openai"),
            ProviderPatch::new()
                .label("Company OpenAI")
                .model(model("gpt-y")),
        )
        .await
        .unwrap();
    assert_eq!(edited.status, MutationStatus::Saved);
    let record = edited.record.unwrap().record;
    assert_eq!(
        (record.label.as_str(), record.model.unwrap().as_str()),
        ("Company OpenAI", "gpt-y")
    );
    assert_eq!(record.slug, slug("openai"), "the slug never changes");
    let noop = bed
        .hub
        .edit(&bed.scope, &slug("openai"), ProviderPatch::new())
        .await
        .unwrap();
    assert_eq!(noop.status, MutationStatus::Unchanged);
}

#[tokio::test]
async fn guard_g2_editing_a_cloud_endpoint_is_ignored() {
    let bed = Bed::new();
    bed.hub.add(&bed.scope, bed.openai_draft()).await.unwrap();
    let edited = bed
        .hub
        .edit(
            &bed.scope,
            &slug("openai"),
            ProviderPatch::new().base_url("https://evil.test/v1"),
        )
        .await
        .unwrap();
    assert_eq!(edited.status, MutationStatus::Unchanged);
    assert_eq!(
        edited.record.unwrap().record.base_url,
        "https://api.openai.com/v1"
    );
}

#[tokio::test]
async fn guard_g3_a_key_does_not_follow_an_endpoint_to_another_origin() {
    let bed = Bed::new();
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("custom")
                .with_label("Acme")
                .with_base_url("https://llm.acme.test/v1")
                .with_key(Secret::new(KEY)),
        )
        .await
        .unwrap();
    // Same origin, new path: fine.
    bed.hub
        .edit(
            &bed.scope,
            &slug("acme"),
            ProviderPatch::new().base_url("https://llm.acme.test/v2"),
        )
        .await
        .unwrap();
    // Another origin with the stored key and no new key: refused.
    let error = bed
        .hub
        .edit(
            &bed.scope,
            &slug("acme"),
            ProviderPatch::new().base_url("https://other.test/v1"),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::Invalid(InvalidInput::Malformed { .. })),
        "{error:?}"
    );
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("acme"))
            .await
            .unwrap()
            .view
            .record
            .base_url,
        "https://llm.acme.test/v2"
    );
    // With a new key it goes through.
    bed.hub
        .edit(
            &bed.scope,
            &slug("acme"),
            ProviderPatch::new()
                .base_url("https://other.test/v1")
                .key(Secret::new("sk-other")),
        )
        .await
        .unwrap();
    assert_eq!(bed.key_of("acme").await.as_deref(), Some("sk-other"));
}

#[tokio::test]
async fn guard_g13_a_key_change_drops_health_and_the_scopes_cached_catalogs() {
    let bed = bed_with_openai();
    bed.hub
        .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
        .await
        .unwrap();
    bed.hub
        .list_models(&bed.scope, &slug("openai"), false)
        .await
        .unwrap();
    let before = bed.ports.http.request_count();
    bed.hub
        .list_models(&bed.scope, &slug("openai"), false)
        .await
        .unwrap();
    assert_eq!(bed.ports.http.request_count(), before, "cached");
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("openai"))
            .await
            .unwrap()
            .health,
        ProviderHealth::Ok
    );

    bed.hub
        .edit(
            &bed.scope,
            &slug("openai"),
            ProviderPatch::new().key(Secret::new("sk-rotated")),
        )
        .await
        .unwrap();
    assert_eq!(
        bed.hub
            .health(&bed.scope, &slug("openai"))
            .await
            .unwrap()
            .health,
        ProviderHealth::Unknown,
        "invariant 9: a re-keyed slug is Unknown"
    );
    bed.hub
        .list_models(&bed.scope, &slug("openai"), false)
        .await
        .unwrap();
    assert!(
        bed.ports.http.request_count() > before,
        "the rotated key re-fetches"
    );
    let last = bed.ports.http.requests().pop().unwrap();
    assert!(last.carried(&Secret::new("sk-rotated")));
}

#[tokio::test]
async fn guard_g21_entry_zero_is_read_only_through_every_operation() {
    let bed = Bed::new();
    bed.ports.config.put_raw(
        &bed.scope,
        json!({"providers": [{
            "id": "prv_entry_zero", "slug": "openrouter", "label": "openrouter",
            "kind": "openrouter", "base_url": "https://openrouter.ai/api/v1",
            "origin": "entry_zero"
        }]})
        .to_string(),
    );
    let s = slug("openrouter");
    assert!(matches!(
        bed.hub
            .edit(&bed.scope, &s, ProviderPatch::new().label("x"))
            .await,
        Err(HubError::Unsupported {
            op: Operation::Edit,
            ..
        })
    ));
    assert!(matches!(
        bed.hub
            .remove(&bed.scope, &s, crate::hub::Confirm::in_use())
            .await,
        Err(HubError::Unsupported {
            op: Operation::Remove,
            ..
        })
    ));
    assert!(matches!(
        bed.hub
            .set_enabled(&bed.scope, &s, false, crate::hub::Confirm::in_use())
            .await,
        Err(HubError::Unsupported {
            op: Operation::SetEnabled,
            ..
        })
    ));
}

#[tokio::test]
async fn ops_editing_the_managed_providers_label_or_endpoint_is_unsupported() {
    let bed = Bed::with(|b| {
        b.managed(crate::hub::ManagedConfig::new(
            "https://api.tinyhumans.test/x",
        ))
    });
    let s = slug("tinyhumans");
    assert!(matches!(
        bed.hub
            .edit(&bed.scope, &s, ProviderPatch::new().label("x"))
            .await,
        Err(HubError::Unsupported { .. })
    ));
    assert!(matches!(
        bed.hub
            .edit(
                &bed.scope,
                &s,
                ProviderPatch::new().base_url("https://x.test")
            )
            .await,
        Err(HubError::Unsupported { .. })
    ));
    bed.hub
        .edit(&bed.scope, &s, ProviderPatch::new().model(model("m")))
        .await
        .unwrap();
}

#[tokio::test]
async fn ops_edit_of_an_unknown_provider_is_not_found() {
    let bed = Bed::new();
    assert!(matches!(
        bed.hub
            .edit(&bed.scope, &slug("nope"), ProviderPatch::new())
            .await,
        Err(HubError::NotFound(_))
    ));
}
