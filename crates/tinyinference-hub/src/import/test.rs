//! Tests for the OpenCompany and OpenHuman importers (06-migration-mapping).

use serde_json::json;

use super::oc::{OcSnapshot, import as import_oc};
use super::oh::{OhSnapshot, import as import_oh};
use super::*;
use crate::config::DefaultChoice;
use crate::descriptor::RecordOrigin;
use crate::error::{HubError, NotFound};
use crate::health::ProviderHealth;
use crate::ids::{ModelId, Slug, WorkloadKey};
use crate::route::{RouteTarget, legacy_oc};
use crate::taxonomy::{AuthStyle, CliKind, LocalRuntime};

fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

fn model(id: &str) -> ModelId {
    ModelId::parse(id).unwrap()
}

fn oc(value: serde_json::Value) -> OcSnapshot {
    serde_json::from_value(value).unwrap()
}

fn oh(value: serde_json::Value) -> OhSnapshot {
    serde_json::from_value(value).unwrap()
}

// ---- report ------------------------------------------------------------------------------

#[test]
fn import_a_report_is_queryable_and_extends() {
    let mut report = LossReport::new();
    assert!(report.is_empty());
    report.push("a", LossKind::Dropped, "gone");
    report.push("b", LossKind::Ambiguous, "either");
    assert!(report.has("a", LossKind::Dropped) && !report.has("a", LossKind::Ambiguous));
    assert_eq!(report.of_kind(LossKind::Ambiguous).count(), 1);
    let mut other = LossReport::new();
    other.push("c", LossKind::FailClosed, "no");
    report.extend(other);
    assert_eq!(report.entries.len(), 3);
    let text = serde_json::to_string(&report).unwrap();
    assert_eq!(serde_json::from_str::<LossReport>(&text).unwrap(), report);
}

// ---- OpenCompany -------------------------------------------------------------------------

fn realistic_oc() -> OcSnapshot {
    oc(json!({
        "providers": [
            {"id": "prv_0123456789abcdef0123456789abcdef", "slug": "openrouter", "label": "OpenRouter",
             "kind": "openrouter", "base_url": "https://openrouter.ai/api/v1",
             "models": {"chat-v1": "openai/gpt-5", "reasoning-v1": "openai/gpt-5"}, "enabled": true},
            {"id": "prv_1111111111111111111111111111111a", "slug": "acme-llm", "label": "Acme LLM",
             "kind": "openai_compatible", "base_url": "https://llm.acme.test/v1",
             "models": {"chat-v1": "acme-1", "reasoning-v1": "acme-2"}},
            {"id": "prv_2222222222222222222222222222222b", "slug": "ollama", "label": "Ollama",
             "kind": "ollama", "base_url": "", "enabled": false}
        ],
        "default": r#"{"provider":"openrouter","model":"openai/gpt-5","future":true}"#,
        "routes": {"chat-v1": "acme-llm:acme-1", "reasoning-v1": "default", "cheap-v1": "local", "code-v1": "claude-code:sonnet"},
        "entry_zero": {"provider": "managed", "models": {"chat-v1": "chat-v1"}},
        "managed_enabled": "false",
        "health": r#"{"openrouter":{"state":"ok","at":"2026-01-01T00:00:00Z"},"acme-llm":{"state":"auth","at":"2026-01-01T00:00:00Z"}}"#
    }))
}

#[test]
fn import_oc_a_realistic_company() {
    let imported = import_oc(&realistic_oc()).unwrap();
    let config = &imported.config;
    let slugs: Vec<&str> = config.providers.iter().map(|p| p.slug.as_str()).collect();
    assert_eq!(slugs, ["tinyhumans", "openrouter", "acme-llm", "ollama"]);
    let managed = &config.providers[0];
    assert!(
        !managed.enabled && managed.synthetic,
        "managed_enabled=false became a disabled managed record"
    );
    let openrouter = config.provider(&slug("openrouter")).unwrap();
    assert_eq!(
        openrouter.id, "prv_0123456789abcdef0123456789abcdef",
        "ids are kept verbatim"
    );
    assert_eq!(openrouter.model, Some(model("openai/gpt-5")));
    assert_eq!(openrouter.origin, RecordOrigin::Imported);
    let acme = config.provider(&slug("acme-llm")).unwrap();
    assert_eq!(
        acme.kind.as_str(),
        "custom",
        "openai_compatible is the custom kind"
    );
    assert_eq!(acme.model, None, "two different ids per tier: none chosen");
    assert_eq!(acme.legacy["tier_models"]["chat-v1"], "acme-1");
    let ollama = config.provider(&slug("ollama")).unwrap();
    assert_eq!(
        (ollama.enabled, ollama.base_url.as_str()),
        (false, "http://localhost:11434")
    );
    assert_eq!(
        config.default,
        DefaultChoice::Full {
            provider: slug("openrouter"),
            model: model("openai/gpt-5")
        }
    );
    // Routes: tiers are opaque keys; `default` is dropped; `local` is a category.
    assert_eq!(config.workload_routes.len(), 3);
    assert_eq!(
        config.workload_routes[&WorkloadKey::new("chat-v1")].provider_slug(),
        Some(&slug("acme-llm"))
    );
    assert_eq!(
        config.workload_routes[&WorkloadKey::new("cheap-v1")].target,
        RouteTarget::Local(None)
    );
    assert_eq!(
        config.workload_routes[&WorkloadKey::new("code-v1")].target,
        RouteTarget::Cli(CliKind::ClaudeCode)
    );
    assert_eq!(imported.health[&slug("openrouter")], ProviderHealth::Ok);
    assert_eq!(
        imported.health[&slug("acme-llm")],
        ProviderHealth::Down(crate::error::ReasonCode::Auth)
    );
    // The report says what happened, in the order it happened.
    let report = &imported.loss;
    assert!(report.has("inference/providers/acme-llm", LossKind::Normalised));
    assert!(report.has("inference/providers/acme-llm", LossKind::Ambiguous));
    assert!(report.has("inference/config", LossKind::Synthesised));
    assert!(report.has("inference/routes/reasoning-v1", LossKind::Dropped));
    assert!(report.has("inference/health", LossKind::Dropped));
    // Nothing in the configuration is a credential.
    let text = serde_json::to_string(config).unwrap();
    assert!(!text.contains("api_key") && config.validate().is_ok());
}

#[test]
fn import_oc_a_uniform_tier_map_is_the_rows_model_and_an_empty_one_is_none() {
    let snapshot = oc(json!({"providers": [
        {"id": "a", "slug": "openai", "label": "OpenAI", "kind": "openai", "models": {"a": "m", "b": " m "}},
        {"id": "b", "slug": "groq", "label": "Groq", "kind": "groq", "models": {}},
        {"id": "c", "slug": "mistral", "kind": "mistral", "models": {"a": ""}}
    ]}));
    let imported = import_oc(&snapshot).unwrap();
    let get = |s: &str| imported.config.provider(&slug(s)).unwrap();
    assert_eq!(get("openai").model, Some(model("m")));
    assert_eq!(get("groq").model, None);
    assert_eq!(get("mistral").model, None);
    assert_eq!(
        get("mistral").label,
        "mistral",
        "an empty label falls back to the slug"
    );
    assert_eq!(
        get("openai").base_url,
        "https://api.openai.com/v1",
        "an empty endpoint takes the preset"
    );
    assert!(imported.loss.is_empty(), "{:?}", imported.loss);
}

#[test]
fn guard_g27_an_unknown_stored_kind_fails_loudly() {
    let snapshot = oc(json!({"providers": [
        {"id": "a", "slug": "mystery", "label": "M", "kind": "not-a-real-kind"}
    ]}));
    assert!(matches!(
        import_oc(&snapshot),
        Err(HubError::NotFound(NotFound::Kind(k))) if k.as_str() == "not-a-real-kind"
    ));
}

#[test]
fn import_oc_cli_rows_are_dropped_and_managed_aliases_normalise() {
    let snapshot = oc(json!({"providers": [
        {"id": "a", "slug": "claude-code", "label": "Claude Code", "kind": "claude-code"},
        {"id": "b", "slug": "cloud", "label": "Cloud", "kind": "managed"}
    ]}));
    let imported = import_oc(&snapshot).unwrap();
    assert!(
        imported
            .loss
            .has("inference/providers/claude-code", LossKind::Dropped)
    );
    assert!(
        imported.config.providers.is_empty(),
        "a managed-kind row is not a second managed record"
    );
    assert!(
        imported
            .loss
            .has("inference/providers/cloud", LossKind::Normalised)
    );
}

#[test]
fn import_oc_a_disabled_managed_row_makes_exactly_one_disabled_managed_record() {
    let cases = [
        json!({"providers": [{"id": "b", "slug": "cloud", "kind": "managed", "enabled": false}]}),
        json!({"providers": [{"id": "b", "slug": "cloud", "kind": "managed", "enabled": false}], "managed_enabled": "false"}),
        json!({"providers": [{"id": "b", "slug": "cloud", "kind": "managed"}], "managed_enabled": "false"}),
        json!({"managed_enabled": "false"}),
    ];
    for case in cases {
        let imported = import_oc(&oc(case.clone())).unwrap();
        assert_eq!(imported.config.providers.len(), 1, "{case}");
        assert!(!imported.config.providers[0].enabled, "{case}");
    }
    let enabled = import_oc(&oc(
        json!({"providers": [{"id": "b", "slug": "cloud", "kind": "managed"}]}),
    ))
    .unwrap();
    assert!(enabled.config.providers.is_empty());
}

#[test]
fn import_oc_the_default_reads_a_bare_slug_json_and_nothing() {
    let cases = [
        (
            Some("openrouter"),
            DefaultChoice::ProviderOnly {
                provider: slug("openrouter"),
            },
        ),
        (
            Some(" openrouter \n"),
            DefaultChoice::ProviderOnly {
                provider: slug("openrouter"),
            },
        ),
        (
            Some(r#"{"provider":"openrouter","model":"m"}"#),
            DefaultChoice::Full {
                provider: slug("openrouter"),
                model: model("m"),
            },
        ),
        (
            Some(r#"{"provider":"openrouter"}"#),
            DefaultChoice::ProviderOnly {
                provider: slug("openrouter"),
            },
        ),
        (
            Some(r#"{"provider":"openrouter","model":"  "}"#),
            DefaultChoice::ProviderOnly {
                provider: slug("openrouter"),
            },
        ),
        (Some(""), DefaultChoice::Unset),
        (None, DefaultChoice::Unset),
    ];
    for (raw, want) in cases {
        let snapshot = OcSnapshot {
            default: raw.map(str::to_string),
            ..OcSnapshot::default()
        };
        assert_eq!(
            import_oc(&snapshot).unwrap().config.default,
            want,
            "{raw:?}"
        );
    }
    for bad in [
        r#"{"model":"m"}"#,
        "{not json",
        "Bad Slug",
        r#"{"provider":"Bad Slug"}"#,
    ] {
        let snapshot = OcSnapshot {
            default: Some(bad.to_string()),
            ..OcSnapshot::default()
        };
        assert!(import_oc(&snapshot).is_err(), "{bad:?}");
    }
}

#[test]
fn import_oc_routes_flag_unknown_providers_and_ambiguous_local() {
    let snapshot = oc(json!({
        "providers": [
            {"id": "a", "slug": "ollama", "kind": "ollama"},
            {"id": "b", "slug": "lmstudio", "kind": "lmstudio"}
        ],
        "routes": {"t1": "local:llama3", "t2": "gone:model", "t3": "managed", "t4": ""}
    }));
    let imported = import_oc(&snapshot).unwrap();
    assert!(
        imported
            .loss
            .has("inference/routes/t1", LossKind::Ambiguous)
    );
    assert!(
        imported
            .loss
            .has("inference/routes/t2", LossKind::FailClosed)
    );
    assert!(imported.loss.has("inference/routes/t4", LossKind::Dropped));
    assert_eq!(imported.config.workload_routes.len(), 3);
    assert!(
        imported
            .config
            .workload_routes
            .contains_key(&WorkloadKey::new("t2")),
        "kept: it fails closed at turn time"
    );
    let bad = oc(json!({"routes": {"t": "Bad Slug"}}));
    assert!(import_oc(&bad).is_err());
}

#[test]
fn import_oc_entry_zero_becomes_a_read_only_record_unless_it_is_managed_or_shadowed() {
    let zero = oc(
        json!({"entry_zero": {"provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "models": {"a": "m"}}}),
    );
    let imported = import_oc(&zero).unwrap();
    let record = imported.config.provider(&slug("openrouter")).unwrap();
    assert_eq!(
        (record.origin, record.id.as_str(), record.model.clone()),
        (RecordOrigin::EntryZero, "prv_entry_zero", Some(model("m")))
    );
    assert!(imported.loss.has("inference/config", LossKind::Synthesised));

    let shadowed = oc(json!({
        "providers": [{"id": "a", "slug": "openrouter", "kind": "openrouter"}],
        "entry_zero": {"provider": "openrouter"}
    }));
    let imported = import_oc(&shadowed).unwrap();
    assert_eq!(imported.config.providers.len(), 1);
    assert_eq!(imported.config.providers[0].origin, RecordOrigin::Imported);
    assert!(imported.loss.has("inference/config", LossKind::Dropped));

    let legacy = oc(
        json!({"entry_zero": {"provider": "openai_compatible", "base_url": "https://x.test/v1"}}),
    );
    let record = import_oc(&legacy).unwrap();
    assert_eq!(record.config.providers[0].kind.as_str(), "custom");
    assert!(record.loss.has("inference/config", LossKind::Normalised));
}

#[test]
fn import_oc_health_is_read_as_states_and_garbage_is_empty() {
    let snapshot = oc(
        json!({"health": r#"{"a":{"state":"quota"},"b":{"state":"timeout"},"c":{"state":"???"},"Bad Slug":{"state":"ok"},"d":{"nostate":1}}"#}),
    );
    let imported = import_oc(&snapshot).unwrap();
    assert_eq!(
        imported.health[&slug("a")],
        ProviderHealth::Down(crate::error::ReasonCode::Quota)
    );
    assert_eq!(
        imported.health[&slug("b")],
        ProviderHealth::Degraded(crate::error::ReasonCode::Timeout)
    );
    assert_eq!(
        imported.health[&slug("c")],
        ProviderHealth::Degraded(crate::error::ReasonCode::Unknown)
    );
    assert_eq!(imported.health.len(), 3);
    let garbage = oc(json!({"health": "{{not json"}));
    let imported = import_oc(&garbage).unwrap();
    assert!(imported.health.is_empty() && imported.loss.has("inference/health", LossKind::Dropped));
}

#[test]
fn import_oc_managed_enabled_only_false_disables() {
    for (raw, disabled) in [
        ("false", true),
        ("FALSE", true),
        ("true", false),
        ("", false),
        ("maybe", false),
    ] {
        let snapshot = OcSnapshot {
            managed_enabled: Some(raw.to_string()),
            ..OcSnapshot::default()
        };
        let imported = import_oc(&snapshot).unwrap();
        assert_eq!(
            imported.config.providers.len(),
            usize::from(disabled),
            "{raw:?}"
        );
    }
}

#[test]
fn import_oc_routes_written_back_as_strings_reproduce_the_input() {
    let snapshot = realistic_oc();
    let imported = import_oc(&snapshot).unwrap();
    for (tier, raw) in &snapshot.routes {
        let Some(route) = imported
            .config
            .workload_routes
            .get(&WorkloadKey::new(tier.clone()))
        else {
            assert_eq!(raw, "default", "only defaults are dropped");
            continue;
        };
        assert_eq!(
            legacy_oc::to_string(route).as_deref(),
            Some(raw.as_str()),
            "{tier}"
        );
    }
}

#[test]
fn import_oc_an_invalid_record_field_fails_loudly_and_never_carries_a_secret() {
    let snapshot = oc(json!({"providers": [
        {"id": "a", "slug": "acme", "kind": "custom", "base_url": "https://user:pass@llm.acme.test/v1"}
    ]}));
    let error = import_oc(&snapshot).unwrap_err();
    assert!(!format!("{error:?}").contains("pass@"));
    let bad_slug = oc(json!({"providers": [{"id": "a", "slug": "Bad Slug", "kind": "custom"}]}));
    assert!(import_oc(&bad_slug).is_err());
}

// ---- OpenHuman ---------------------------------------------------------------------------

fn realistic_oh() -> OhSnapshot {
    oh(json!({
        "cloud_providers": [
            {"id": "p_openai_a1b2c", "slug": "openai", "label": "OpenAI", "endpoint": "https://api.openai.com/v1", "auth_style": "bearer", "default_model": "gpt-5"},
            {"id": "p_deepseek_d3e4f", "slug": "deepseek", "label": "DeepSeek", "endpoint": "https://api.deepseek.com/v1", "auth_style": "bearer", "type": "cloud"},
            {"id": "p_acme_g5h6i", "slug": "acme", "label": "Acme", "endpoint": "https://llm.acme.test/v1", "auth_style": "openhumanjwt"},
            {"id": "p_openhuman", "slug": "openhuman", "label": "OpenHuman", "endpoint": "https://api.openhuman.ai/v1", "auth_style": "openhuman_jwt"},
            {"id": "p_cc", "slug": "claude-code", "label": "Claude Code", "endpoint": "cli://claude-code", "auth_style": "none"}
        ],
        "primary_cloud": "p_openai_a1b2c",
        "default_model": "gpt-5",
        "routes": {
            "chat_provider": "openhuman", "reasoning_provider": "openai:o3@0.2", "agentic_provider": "",
            "coding_provider": "claude-code:sonnet", "vision_provider": "hint",
            "memory_provider": "ollama:llama3", "embeddings_provider": "cloud", "heartbeat_provider": "vllm",
            "learning_provider": "acme:hint:fast", "subconscious_provider": "__byok_incomplete__",
            "burst": "openai:gpt-5", "summarization": "acme:m"
        },
        "local_ai": {"provider": "lmstudio", "base_url": "http://localhost:1234/v1", "api_key": "sk-not-a-real-key", "model_id": "qwen"},
        "byok": {"url": "https://byok.acme.test/v1", "api_key": "sk-byok-fake", "model": "b1"},
        "model_registry": [{"id": "gpt-5", "provider": "openai", "cost_per_1m_input": 1.25, "cost_per_1m_output": 10.0, "context_window": 400000, "vision": true}, {"id": "bad id"}],
        "temperature_unsupported_models": ["o3", "bad id"],
        "openai_oauth": true
    }))
}

#[test]
fn import_oh_a_realistic_config() {
    let imported = import_oh(&realistic_oh()).unwrap();
    let config = &imported.config;
    let slugs: Vec<&str> = config.providers.iter().map(|p| p.slug.as_str()).collect();
    assert_eq!(
        slugs,
        ["openai", "deepseek", "acme", "byok-inference", "lmstudio"]
    );
    let openai = config.provider(&slug("openai")).unwrap();
    assert_eq!(openai.id, "p_openai_a1b2c");
    assert_eq!(openai.model, Some(model("gpt-5")));
    let deepseek = config.provider(&slug("deepseek")).unwrap();
    assert_eq!(
        deepseek.base_url, "https://api.deepseek.com/v1",
        "a stored endpoint is kept"
    );
    assert!(
        imported
            .loss
            .has("cloud_providers/deepseek", LossKind::Normalised)
    );
    assert!(
        imported
            .loss
            .has("cloud_providers/deepseek", LossKind::Dropped),
        "the legacy row type"
    );
    let acme = config.provider(&slug("acme")).unwrap();
    assert_eq!(
        (acme.kind.as_str(), acme.auth_override.clone()),
        ("custom", Some(AuthStyle::SessionJwt))
    );
    let byok = config.provider(&slug("byok-inference")).unwrap();
    assert!(byok.synthetic);
    let lm = config.provider(&slug("lmstudio")).unwrap();
    assert_eq!(
        (lm.base_url.as_str(), lm.model.clone()),
        ("http://localhost:1234/v1", Some(model("qwen")))
    );
    // Case 10: the primary_cloud id became a slug default.
    assert_eq!(
        config.default,
        DefaultChoice::Full {
            provider: slug("openai"),
            model: model("gpt-5")
        }
    );
    // Keys are handed back, never stored on the configuration.
    assert_eq!(imported.credentials.len(), 2);
    assert!(!serde_json::to_string(config).unwrap().contains("sk-"));
    assert!(!format!("{:?}", imported.credentials).contains("sk-"));
    // Overrides: the registry row (with prices) and the temperature list.
    let gpt = imported
        .overrides
        .iter()
        .find(|o| o.model.as_str() == "gpt-5")
        .unwrap();
    assert_eq!(
        (gpt.input_per_1m, gpt.output_per_1m, gpt.context_window),
        (Some(1.25), Some(10.0), Some(400_000))
    );
    let o3 = imported
        .overrides
        .iter()
        .find(|o| o.model.as_str() == "o3")
        .unwrap();
    assert_eq!(o3.temperature, Some(crate::descriptor::Tri::No));
    assert_eq!(imported.overrides.len(), 2);
}

#[test]
fn import_oh_role_routes_and_their_reports() {
    let imported = import_oh(&realistic_oh()).unwrap();
    let route = |role: &str| imported.config.workload_routes.get(&WorkloadKey::new(role));
    assert_eq!(route("chat").unwrap().target, RouteTarget::Managed);
    assert_eq!(route("reasoning").unwrap().temperature.unwrap().get(), 0.2);
    assert_eq!(
        route("agentic").unwrap().provider_slug(),
        Some(&slug("openai")),
        "burst aliases to agentic (the explicit agentic route was the default, which is no route)"
    );
    assert_eq!(
        route("coding").unwrap().target,
        RouteTarget::Cli(CliKind::ClaudeCode)
    );
    // Case 2: `ollama:` under a local_ai of lmstudio is lmstudio.
    assert_eq!(
        route("memory").unwrap().target,
        RouteTarget::Local(Some(LocalRuntime::LmStudio))
    );
    assert!(imported.loss.has("routes/memory", LossKind::Ambiguous));
    // Case 3: bare vllm.
    assert_eq!(
        route("heartbeat").unwrap().target,
        RouteTarget::Local(Some(LocalRuntime::Vllm))
    );
    // Case 5: a hint model on a row with no default_model fails closed, on one with it is replaced.
    assert!(imported.loss.has("routes/learning", LossKind::FailClosed));
    // Bare `hint` is an unknown bare string.
    assert!(route("vision").is_none() && imported.loss.has("routes/vision", LossKind::FailClosed));
    // Case 4: empty/cloud routes are the absence of a route; the sentinel fails closed.
    assert!(route("embeddings").is_none());
    assert!(
        route("subconscious").is_none()
            && imported
                .loss
                .has("routes/subconscious", LossKind::FailClosed)
    );
    // Aliases.
    assert!(
        imported.loss.has("routes/agentic", LossKind::Normalised),
        "burst maps to agentic"
    );
    assert!(
        imported.loss.has("routes/memory", LossKind::Dropped),
        "summarization loses to memory_provider"
    );
}

#[test]
fn import_oh_lossy_case_6_the_hub_catalogue_covers_every_openhuman_builtin() {
    // The 26 shared cloud slugs plus the managed row are all recognised.
    let builtins = [
        "openai",
        "anthropic",
        "openrouter",
        "orcarouter",
        "gmi",
        "fireworks",
        "moonshot",
        "groq",
        "mistral",
        "deepseek",
        "together",
        "google",
        "cerebras",
        "xai",
        "huggingface",
        "nvidia",
        "zai",
        "minimax",
        "stepfun",
        "kilocode",
        "deepinfra",
        "novita",
        "venice",
        "vercel-ai-gateway",
        "sumopod",
        "modelscope",
    ];
    assert_eq!(builtins.len(), 26);
    for name in builtins {
        assert!(crate::catalogue::descriptor(name).is_some(), "{name}");
        let snapshot = oh(
            json!({"cloud_providers": [{"id": "p", "slug": name, "label": name, "endpoint": ""}]}),
        );
        let imported = import_oh(&snapshot).unwrap();
        let record = imported.config.provider(&slug(name)).unwrap();
        assert_ne!(
            record.kind.as_str(),
            "custom",
            "{name} maps to its built-in kind"
        );
        assert!(
            !record.base_url.is_empty(),
            "{name}: the preset fills an empty endpoint"
        );
    }
    assert!(
        crate::catalogue::descriptor("openhuman").is_some(),
        "openhuman is the managed alias"
    );
}

#[test]
fn import_oh_case_7_synthetic_routes() {
    let snapshot = oh(json!({
        "byok": {"url": "https://byok.acme.test/v1", "model": "m"},
        "routes": {"chat": "byok-inference:m", "coding": "ephemeral-route:x"}
    }));
    let imported = import_oh(&snapshot).unwrap();
    assert!(
        imported
            .config
            .provider(&slug("byok-inference"))
            .unwrap()
            .synthetic
    );
    assert_eq!(
        imported.config.workload_routes[&WorkloadKey::new("chat")].provider_slug(),
        Some(&slug("byok-inference"))
    );
    assert!(
        !imported
            .config
            .workload_routes
            .contains_key(&WorkloadKey::new("coding")),
        "ephemeral is never persisted"
    );
    assert!(imported.loss.has("routes/coding", LossKind::Dropped));
    assert!(imported.loss.has("byok-inference", LossKind::Synthesised));
}

#[test]
fn import_oh_case_8_the_claude_code_marker_row_is_dropped() {
    let snapshot = oh(json!({"cloud_providers": [
        {"id": "p", "slug": "claude-code", "label": "Claude Code", "endpoint": "cli://claude-code", "auth_style": "none"}
    ]}));
    let imported = import_oh(&snapshot).unwrap();
    assert!(imported.config.providers.is_empty());
    assert!(
        imported
            .loss
            .has("cloud_providers/claude-code", LossKind::Dropped)
    );
}

#[test]
fn import_oh_case_9_codex_has_no_row() {
    let snapshot = oh(
        json!({"cloud_providers": [{"id": "p", "slug": "openai", "endpoint": ""}], "openai_oauth": true}),
    );
    let imported = import_oh(&snapshot).unwrap();
    assert_eq!(imported.config.providers.len(), 1);
    assert!(
        !imported
            .config
            .providers
            .iter()
            .any(|p| p.slug.as_str() == "codex")
    );
    assert!(imported.loss.has("auth-profiles", LossKind::Normalised));
}

#[test]
fn import_oh_case_10_a_dangling_or_absent_primary_cloud_sets_no_default() {
    let dangling = oh(
        json!({"cloud_providers": [{"id": "p1", "slug": "openai", "endpoint": ""}], "primary_cloud": "p_gone"}),
    );
    let imported = import_oh(&dangling).unwrap();
    assert_eq!(imported.config.default, DefaultChoice::Unset);
    assert!(imported.loss.has("primary_cloud", LossKind::FailClosed));
    let absent = oh(json!({"cloud_providers": [{"id": "p1", "slug": "openai", "endpoint": ""}]}));
    assert!(
        import_oh(&absent)
            .unwrap()
            .loss
            .has("primary_cloud", LossKind::Ambiguous)
    );
    let no_model = oh(
        json!({"cloud_providers": [{"id": "p1", "slug": "openai", "endpoint": ""}], "primary_cloud": "p1"}),
    );
    let imported = import_oh(&no_model).unwrap();
    assert_eq!(
        imported.config.default,
        DefaultChoice::ProviderOnly {
            provider: slug("openai")
        }
    );
    let managed = oh(
        json!({"cloud_providers": [{"id": "pm", "slug": "openhuman", "endpoint": ""}], "primary_cloud": "pm", "default_model": "m"}),
    );
    let imported = import_oh(&managed).unwrap();
    assert_eq!(
        imported.config.default,
        DefaultChoice::Full {
            provider: slug("tinyhumans"),
            model: model("m")
        }
    );
    let cli = oh(
        json!({"cloud_providers": [{"id": "pc", "slug": "claude-code", "endpoint": "cli://claude-code"}], "primary_cloud": "pc"}),
    );
    assert!(
        import_oh(&cli)
            .unwrap()
            .loss
            .has("primary_cloud", LossKind::FailClosed)
    );
}

#[test]
fn import_oh_case_11_only_the_configured_local_runtime_gets_a_record() {
    let snapshot =
        oh(json!({"local_ai": {"provider": "llama.cpp", "base_url": "http://127.0.0.1:8080/v1"}}));
    let imported = import_oh(&snapshot).unwrap();
    assert_eq!(imported.config.providers.len(), 1);
    assert_eq!(imported.config.providers[0].kind.as_str(), "local-openai");
    assert!(imported.loss.has("local_ai", LossKind::Normalised));
    let bad = oh(json!({"local_ai": {"provider": "not-a-runtime"}}));
    assert!(import_oh(&bad).is_err());
    // A synthetic local row and local_ai for the same runtime merge into one.
    let merged = oh(json!({
        "cloud_providers": [{"id": "synthetic_local_ollama", "slug": "ollama", "endpoint": "http://localhost:11434"}],
        "local_ai": {"provider": "ollama", "base_url": "http://192.168.1.5:11434", "model_id": "llama3"}
    }));
    let imported = import_oh(&merged).unwrap();
    assert_eq!(imported.config.providers.len(), 1);
    assert_eq!(
        imported.config.providers[0].base_url,
        "http://192.168.1.5:11434"
    );
    assert_eq!(imported.config.providers[0].model, Some(model("llama3")));
    assert!(
        imported
            .loss
            .has("cloud_providers/ollama", LossKind::Normalised)
    );
}

#[test]
fn import_oh_odd_rows_are_reported_not_guessed() {
    let snapshot = oh(json!({"cloud_providers": [
        {"id": "a", "slug": "acme", "endpoint": "https://a.test/v1", "auth_style": "weird"},
        {"id": "b", "slug": "acme", "endpoint": "https://b.test/v1"},
        {"id": "c", "slug": "xapi", "endpoint": "https://c.test/v1", "auth_style": "x-api-key"}
    ]}));
    let imported = import_oh(&snapshot).unwrap();
    assert_eq!(
        imported.config.providers.len(),
        2,
        "a second row with the same slug is dropped"
    );
    assert!(
        imported
            .loss
            .has("cloud_providers/acme", LossKind::Ambiguous)
    );
    assert!(imported.loss.has("cloud_providers/acme", LossKind::Dropped));
    assert_eq!(
        imported
            .config
            .provider(&slug("xapi"))
            .unwrap()
            .auth_override,
        Some(AuthStyle::XApiKey)
    );
    let bad = oh(
        json!({"cloud_providers": [{"id": "a", "slug": "Bad Slug", "endpoint": "https://a.test"}]}),
    );
    assert!(import_oh(&bad).is_err());
    let bad_model = oh(
        json!({"cloud_providers": [{"id": "a", "slug": "acme", "endpoint": "https://a.test", "default_model": "has space"}]}),
    );
    assert!(import_oh(&bad_model).is_err());
}

#[test]
fn import_oh_inputs_never_print_a_key() {
    let snapshot = realistic_oh();
    let shown = format!("{snapshot:?}");
    assert!(
        !shown.contains("sk-not-a-real-key") && !shown.contains("sk-byok-fake"),
        "{shown}"
    );
    assert!(shown.contains("StoredKey"));
    let key = super::oh::StoredKey::new("  sk-x  ");
    assert!(!format!("{key:?}").contains("sk-x"));
    let from_string: super::oh::StoredKey = String::from("k").into();
    assert!(
        !format!("{from_string:?}").contains('k')
            || format!("{from_string:?}").contains("redacted")
    );
    // Blank keys are not credentials to move.
    let blank = oh(json!({"byok": {"url": "https://b.test/v1", "api_key": "   "}}));
    assert!(import_oh(&blank).unwrap().credentials.is_empty());
}

#[test]
fn import_inputs_deserialize_from_partial_stored_json() {
    let empty: OhSnapshot = serde_json::from_str("{}").unwrap();
    assert!(import_oh(&empty).unwrap().config.providers.is_empty());
    let empty: OcSnapshot = serde_json::from_str("{}").unwrap();
    assert!(import_oc(&empty).unwrap().config.providers.is_empty());
    // Unknown members of the stored shapes are ignored, not fatal.
    let extra: OcSnapshot = serde_json::from_str(r#"{"providers":[],"future_field":1}"#).unwrap();
    assert!(extra.providers.is_empty());
}

#[test]
fn import_oc_references_to_a_dropped_managed_alias_are_rewritten_to_the_managed_slug() {
    let imported = import_oc(&oc(json!({
        "providers": [{"id": "b", "slug": "cloud", "kind": "managed"}],
        "default": "{\"provider\":\"cloud\",\"model\":\"m\"}",
        "routes": {"chat": "cloud:m"},
        "health": "{\"cloud\": {\"state\": \"ok\"}}"
    })))
    .unwrap();
    assert!(matches!(
        &imported.config.default,
        DefaultChoice::Full { provider, .. } if provider.as_str() == "tinyhumans"
    ));
    let route = imported
        .config
        .workload_routes
        .get(&WorkloadKey::new("chat"))
        .unwrap();
    assert_eq!(route.target, RouteTarget::Provider(slug("tinyhumans")));
    assert!(imported.health.contains_key(&slug("tinyhumans")));
    assert!(!imported.health.contains_key(&slug("cloud")));
    assert!(imported.loss.has("inference/default", LossKind::Normalised));
}

#[test]
fn import_oc_a_cloud_row_on_another_origin_is_imported_disabled_and_a_path_variant_is_kept() {
    let imported = import_oc(&oc(json!({"providers": [
        {"id": "a", "slug": "openai", "kind": "openai", "base_url": "https://evil.test/v1"},
        {"id": "b", "slug": "groq", "kind": "groq", "base_url": "https://api.groq.com/openai/v2/"}
    ]})))
    .unwrap();
    let record = imported.config.provider(&slug("openai")).unwrap();
    assert!(
        !record.enabled,
        "a key is never repointed at another origin"
    );
    assert_eq!(record.base_url, "https://evil.test/v1");
    assert!(
        imported
            .loss
            .has("inference/providers/openai", LossKind::FailClosed)
    );
    let same = imported.config.provider(&slug("groq")).unwrap();
    assert!(same.enabled);
    assert!(
        !imported
            .loss
            .has("inference/providers/groq", LossKind::FailClosed)
    );
}

#[test]
fn import_oc_a_real_row_named_cloud_keeps_its_references() {
    let imported = import_oc(&oc(json!({
        "providers": [{"id": "c", "slug": "cloud", "kind": "custom", "base_url": "https://llm.acme.test/v1"}],
        "default": "{\"provider\":\"cloud\",\"model\":\"m\"}"
    })))
    .unwrap();
    assert!(imported.config.provider(&slug("cloud")).is_some());
    assert!(matches!(
        &imported.config.default,
        DefaultChoice::Full { provider, .. } if provider.as_str() == "cloud"
    ));
}
