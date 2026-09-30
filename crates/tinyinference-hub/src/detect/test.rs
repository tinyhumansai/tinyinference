//! Tests for detection: fingerprints, environment keys, and the hosted cutoff.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use super::*;
use crate::hub::fixtures::{Bed, models_body, slug};
use crate::ports::memory::MapEnv;
use crate::ports::{DetectOptions, Detector};
use crate::taxonomy::LocalRuntime;
use crate::testkit::{Match, Scripted};

fn script_machine(
    bed: &Bed,
    ollama: bool,
    lmstudio: bool,
    vllm: bool,
    llama: Option<serde_json::Value>,
) {
    let http = &bed.ports.http;
    let answer = |port: u16, path: &str, body: Option<serde_json::Value>| {
        http.route(
            Match::get(format!("http://localhost:{port}{path}")),
            body.map_or(Scripted::ConnectRefused, |b| Scripted::json(200, &b)),
        );
    };
    answer(
        11434,
        "/api/version",
        ollama.then(|| json!({"version": "0.5.7"})),
    );
    answer(
        1234,
        "/api/v0/models",
        lmstudio.then(|| json!({"data": [], "object": "list"})),
    );
    answer(8000, "/version", vllm.then(|| json!({"version": "0.6.1"})));
    answer(8080, "/props", llama);
}

#[tokio::test]
async fn sim_detect_local_fingerprint_finds_each_runtime_by_what_only_it_answers() {
    let bed = Bed::new();
    script_machine(
        &bed,
        true,
        true,
        true,
        Some(json!({"default_generation_settings": {}, "total_slots": 1})),
    );
    let drafts = bed.hub.detect(&DetectOptions::default()).await.unwrap();
    let kinds: Vec<&str> = drafts.iter().map(|d| d.kind.as_str()).collect();
    assert_eq!(
        kinds,
        ["ollama", "lmstudio", "local-openai", "local-openai"]
    );
    assert_eq!(
        drafts[0].base_url.as_deref(),
        Some("http://localhost:11434")
    );
    assert!(
        drafts.iter().all(|d| d.key.is_none()),
        "a draft never carries a key"
    );
    assert_eq!(bed.ports.http.request_count(), 4);
    // Nothing was saved.
    assert!(bed.ports.config.raw(&bed.scope).is_none() && bed.ports.credentials.is_empty());
    // A detected draft connects like any other.
    bed.ports.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::json(200, &models_body(&["llama3"])),
    );
    bed.hub
        .connect(
            &bed.scope,
            drafts[0].clone(),
            crate::hub::ConnectOptions::default(),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn sim_detect_excludes_self_a_server_on_the_hosts_own_port_is_not_llama_cpp() {
    let bed = Bed::new();
    // The host's own bind answers something on 8080, but not llama.cpp's /props.
    script_machine(&bed, false, false, false, Some(json!({"ok": true})));
    assert!(
        bed.hub
            .detect(&DetectOptions::default())
            .await
            .unwrap()
            .is_empty()
    );
    // And an explicit exclusion is never even asked.
    let genuine = json!({"total_slots": 4});
    script_machine(&bed, false, false, false, Some(genuine));
    let before = bed.ports.http.request_count();
    let options = {
        let mut options = DetectOptions::default();
        options.exclude_ports.push(8080);
        options
    };
    assert!(bed.hub.detect(&options).await.unwrap().is_empty());
    assert_eq!(
        bed.ports.http.request_count() - before,
        3,
        "port 8080 was skipped"
    );
    assert!(
        !bed.ports.http.requests()[before..]
            .iter()
            .any(|r| r.url.contains(":8080"))
    );
}

#[tokio::test]
async fn sim_detect_off_when_hosted_sends_nothing_and_reads_no_environment() {
    let ports = crate::testkit::MemoryPorts::new()
        .with_env(MapEnv::new().with("OPENAI_API_KEY", "sk-server-secret"));
    let hub = ports
        .builder()
        .policy(crate::policy::EndpointPolicy::hosted())
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

#[tokio::test]
async fn detect_env_never_persists_and_never_carries_the_key() {
    let ports = crate::testkit::MemoryPorts::new().with_env(
        MapEnv::new()
            .with("OPENAI_API_KEY", "sk-not-a-real-key")
            .with("ANTHROPIC_API_KEY", "   ")
            .with("GEMINI_API_KEY", "g1")
            .with("GOOGLE_API_KEY", "g2")
            .with("HF_TOKEN", "hf_x"),
    );
    let bed = Bed {
        hub: ports.builder().build().unwrap(),
        ports,
        scope: crate::hub::fixtures::scope("u"),
    };
    script_machine(&bed, false, false, false, None);
    let drafts = bed.hub.detect(&DetectOptions::default()).await.unwrap();
    let kinds: Vec<&str> = drafts.iter().map(|d| d.kind.as_str()).collect();
    assert_eq!(
        kinds,
        ["openai", "google", "huggingface"],
        "blank ignored, a kind with two variables reported once"
    );
    assert!(drafts.iter().all(|d| d.key.is_none()));
    assert!(!format!("{drafts:?}").contains("sk-not-a-real-key"));
    assert!(bed.ports.config.raw(&bed.scope).is_none() && bed.ports.credentials.is_empty());
}

#[tokio::test]
async fn detect_env_every_reported_variable_is_read_by_the_chain_and_blank_is_unset() {
    let ports = crate::testkit::MemoryPorts::new().with_env(
        MapEnv::new()
            .with("GOOGLE_API_KEY", "g-key\n")
            .with("OPENAI_API_KEY", " \n"),
    );
    let hub = ports.builder().env_credentials(true).build().unwrap();
    let bed = Bed {
        hub: hub.clone(),
        ports,
        scope: crate::hub::fixtures::scope("u"),
    };
    script_machine(&bed, false, false, false, None);
    let drafts = hub.detect(&DetectOptions::default()).await.unwrap();
    assert_eq!(
        drafts.iter().map(|d| d.kind.as_str()).collect::<Vec<_>>(),
        ["google"],
        "a blank value is not a key"
    );
    assert_eq!(
        env_vars_for_kind("google").collect::<Vec<_>>(),
        ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    );
    // The chain finds the second variable, trimmed, like detection did.
    let scope = crate::hub::fixtures::scope("u");
    hub.add(
        &scope,
        crate::config::ProviderDraft::new("google").with_model(crate::hub::fixtures::model("m")),
    )
    .await
    .unwrap();
    let turn = hub
        .resolve_for_turn(&scope, &crate::route::TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(turn.slug.as_str(), "google");
    // A blank OpenAI variable is no key for the chain either.
    hub.add(
        &scope,
        crate::config::ProviderDraft::new("openai").with_model(crate::hub::fixtures::model("m")),
    )
    .await
    .unwrap();
    let pinned = crate::route::TurnQuery::new()
        .with_override(crate::route::ProviderRoute::provider(slug("openai")));
    assert!(hub.resolve_for_turn(&scope, &pinned).await.is_err());
}

#[test]
fn detect_env_table_is_consistent_with_the_catalogue() {
    for (var, kind) in ENV_KEYS {
        assert!(
            crate::catalogue::descriptor(kind).is_some(),
            "{var} names {kind}"
        );
        assert!(
            var.chars()
                .all(|c| c.is_ascii_uppercase() || c == '_' || c.is_ascii_digit())
        );
    }
    assert_eq!(env_var_for_kind("openai"), Some("OPENAI_API_KEY"));
    assert_eq!(env_var_for_kind("google"), Some("GEMINI_API_KEY"));
    assert_eq!(env_var_for_kind("ollama"), None);
    let ports: Vec<u16> = fingerprints().iter().map(|f| f.port).collect();
    let mut unique = ports.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), ports.len());
}

#[tokio::test]
async fn sim_cli_oneshot_env_only_resolves_without_persistence() {
    let ports = crate::testkit::MemoryPorts::new()
        .with_env(MapEnv::new().with("OPENAI_API_KEY", "sk-env-fake"));
    let hub = ports.builder().env_credentials(true).build().unwrap();
    let scope = crate::hub::fixtures::scope("user:local");
    hub.add(
        &scope,
        crate::config::ProviderDraft::new("openai").with_model(crate::hub::fixtures::model("m")),
    )
    .await
    .unwrap();
    let turn = hub
        .resolve_for_turn(&scope, &crate::route::TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(
        turn.origin,
        Some(crate::credential::CredentialOrigin::Env(
            "OPENAI_API_KEY".into()
        ))
    );
    assert!(
        ports.credentials.is_empty(),
        "the environment key was never copied into the store"
    );
    // A stored key wins over the environment.
    hub.set_key(
        &scope,
        &slug("openai"),
        crate::secret::Secret::new("sk-stored"),
    )
    .await
    .unwrap();
    let turn = hub
        .resolve_for_turn(&scope, &crate::route::TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(
        turn.origin,
        Some(crate::credential::CredentialOrigin::ProviderKey)
    );
    // Without opting in, the environment is not a credential.
    let hub = ports.builder().build().unwrap();
    let other = crate::hub::fixtures::scope("user:other");
    hub.add(
        &other,
        crate::config::ProviderDraft::new("openai").with_model(crate::hub::fixtures::model("m")),
    )
    .await
    .unwrap();
    assert!(
        hub.resolve_for_turn(&other, &crate::route::TurnQuery::new())
            .await
            .is_err()
    );
}

#[derive(Debug)]
struct Fixed;

#[async_trait]
impl Detector for Fixed {
    async fn detect(&self, _: &DetectOptions) -> Vec<ProviderDraft> {
        vec![ProviderDraft::new("groq")]
    }
}

#[tokio::test]
async fn detect_a_host_detector_replaces_the_built_in_one_but_not_the_hosted_cutoff() {
    let ports = crate::testkit::MemoryPorts::new();
    let hub = ports.builder().detector(Arc::new(Fixed)).build().unwrap();
    assert_eq!(
        hub.detect(&DetectOptions::default()).await.unwrap().len(),
        1
    );
    assert_eq!(ports.http.request_count(), 0);
    let hosted = ports
        .builder()
        .detector(Arc::new(Fixed))
        .policy(crate::policy::EndpointPolicy::hosted())
        .build()
        .unwrap();
    assert!(
        hosted
            .detect(&DetectOptions::default())
            .await
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn detect_local_status_fingerprints_and_counts_models() {
    let bed = Bed::new();
    bed.hub
        .add(&bed.scope, crate::config::ProviderDraft::new("ollama"))
        .await
        .unwrap();
    script_machine(&bed, true, false, false, None);
    bed.ports.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::json(200, &models_body(&["a", "b"])),
    );
    let status = bed
        .hub
        .local_status(&bed.scope, &slug("ollama"))
        .await
        .unwrap();
    assert_eq!(
        status,
        LocalRuntimeStatus {
            reachable: true,
            fingerprinted: Some(LocalRuntime::Ollama),
            version: Some("0.5.7".into()),
            models: Some(2)
        }
    );
    // Down: nothing answers.
    script_machine(&bed, false, false, false, None);
    bed.ports.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::ConnectRefused,
    );
    bed.ports.http.route(
        Match::prefix("http://localhost:11434/api/tags"),
        Scripted::ConnectRefused,
    );
    let status = bed
        .hub
        .local_status(&bed.scope, &slug("ollama"))
        .await
        .unwrap();
    assert_eq!(
        (status.reachable, status.fingerprinted, status.models),
        (false, None, None)
    );
    // A cloud provider is not a local runtime.
    bed.hub
        .add(&bed.scope, crate::config::ProviderDraft::new("openai"))
        .await
        .unwrap();
    assert!(matches!(
        bed.hub.local_status(&bed.scope, &slug("openai")).await,
        Err(crate::error::HubError::Unsupported { .. })
    ));
    assert!(matches!(
        bed.hub.local_status(&bed.scope, &slug("nope")).await,
        Err(crate::error::HubError::NotFound(_))
    ));
}
