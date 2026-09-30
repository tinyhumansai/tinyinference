//! Tests for the parsers, the entry types and the metadata merge.

use proptest::prelude::*;
use serde_json::json;

use super::*;
use crate::descriptor::{CapSource, Capabilities, Sourced, Tri};
use crate::error::{ReasonCode, Retry};
use crate::ids::{KindId, ModelId};

fn id(s: &str) -> ModelId {
    ModelId::parse(s).unwrap()
}

fn body(value: &serde_json::Value) -> Vec<u8> {
    value.to_string().into_bytes()
}

fn ids(parsed: &ParsedCatalog) -> Vec<&str> {
    parsed.entries.iter().map(|e| e.id.as_str()).collect()
}

// ---- parse_openai ----------------------------------------------------------

#[test]
fn catalog_a_standard_openai_listing_parses_in_order() {
    let parsed = parse_openai(&body(&json!({
        "object": "list",
        "data": [{"id": "gpt-5", "owned_by": "openai"}, {"id": "gpt-4o"}, {"id": "o3"}]
    })))
    .unwrap();
    assert_eq!(ids(&parsed), ["gpt-5", "gpt-4o", "o3"]);
    assert_eq!(parsed.entries[0].owned_by.as_deref(), Some("openai"));
    assert_eq!(parsed.skipped, 0);
    assert!(
        parsed
            .entries
            .iter()
            .all(|e| e.origin == EntrySource::ProviderApi && e.available)
    );
}

#[test]
fn catalog_the_models_envelope_and_the_slug_and_name_ids_are_accepted() {
    let parsed = parse_openai(&body(&json!({
        "models": [{"slug": "a"}, {"name": "b"}, {"id": "", "name": "c"}, "d"]
    })))
    .unwrap();
    assert_eq!(ids(&parsed), ["a", "b", "c", "d"]);
}

#[test]
fn catalog_a_bare_top_level_array_is_a_listing() {
    // Together's `/models` answers a bare array.
    let parsed = parse_openai(&body(&json!([{"id": "m1"}, {"id": "m2"}]))).unwrap();
    assert_eq!(ids(&parsed), ["m1", "m2"]);
}

#[test]
fn catalog_null_data_on_a_success_envelope_is_an_empty_catalog() {
    for text in [r#"{"object":"list","data":null}"#, r#"{"data":null}"#] {
        let parsed = parse_openai(text.as_bytes()).unwrap();
        assert!(parsed.entries.is_empty(), "{text}");
    }
}

#[test]
fn catalog_null_data_on_an_error_envelope_is_a_failure_not_an_empty_list() {
    let failure = parse_openai(br#"{"object":"error","data":null}"#).unwrap_err();
    assert_eq!(failure.reason, ReasonCode::Unknown);
    assert_eq!(failure.retry, Retry::Never);
}

#[test]
fn catalog_a_body_that_is_not_a_listing_is_unknown_never_auth() {
    for bad in [
        &b"<html>not json</html>"[..],
        b"",
        b"\"just a string\"",
        b"42",
        br#"{"error":"nope"}"#,
        br#"{"data":"a string"}"#,
        br#"{"data":{"nested":true}}"#,
    ] {
        let failure = parse_openai(bad).unwrap_err();
        assert_eq!(
            failure.reason,
            ReasonCode::Unknown,
            "{}",
            String::from_utf8_lossy(bad)
        );
        assert!(!failure.reason.destroys_credential());
    }
}

#[test]
fn catalog_a_bad_row_costs_that_row_and_is_counted() {
    let parsed = parse_openai(&body(&json!({"data": [
        {"id": "good-1"},
        {"id": 7},
        {"id": "has space"},
        {"nothing": true},
        {"id": "good-2", "context_length": "not a number"},
        null,
        {"id": "good-1"}
    ]})))
    .unwrap();
    assert_eq!(ids(&parsed), ["good-1", "good-2"]);
    assert_eq!(
        parsed.skipped, 5,
        "non-string id, invalid id, no id, null, duplicate"
    );
    let second = &parsed.entries[1];
    assert_eq!(
        second.capabilities.context_window.value, None,
        "a bad optional field is absent"
    );
}

#[test]
fn catalog_context_window_and_prices_are_read_and_tagged() {
    let parsed = parse_openai(&body(&json!({"data": [{
        "id": "m", "display_name": "The M", "context_length": 128000,
        "pricing": {"inputPer1M": 1.5, "outputPer1M": 6.0}
    }]})))
    .unwrap();
    let entry = &parsed.entries[0];
    assert_eq!(entry.display_name.as_deref(), Some("The M"));
    assert_eq!(
        entry.capabilities.context_window,
        Sourced::new(Some(128_000), CapSource::ProviderApi)
    );
    assert_eq!(
        (entry.input_per_1m, entry.output_per_1m),
        (Some(1.5), Some(6.0))
    );
    assert_eq!(
        entry.capabilities.tools.value,
        Tri::Unknown,
        "nothing is promoted to yes"
    );
    assert_eq!(entry.capabilities.tools.source, CapSource::Default);
}

#[test]
fn catalog_a_negative_or_missing_price_is_absent() {
    let parsed = parse_openai(&body(&json!({"data": [
        {"id": "a", "pricing": {"inputPer1M": -1.0}},
        {"id": "b", "pricing": {}},
        {"id": "c"}
    ]})))
    .unwrap();
    for entry in &parsed.entries {
        assert_eq!((entry.input_per_1m, entry.output_per_1m), (None, None));
    }
}

#[test]
fn catalog_no_id_is_filtered_by_vendor_or_name() {
    let names = [
        "claude-opus-4",
        "gpt-5",
        "x-ai/grok-9",
        "chat-v1",
        "tier-fast",
        "reasoning",
        "default",
    ];
    let rows: Vec<_> = names.iter().map(|n| json!({"id": n})).collect();
    let parsed = parse_openai(&body(&json!({"data": rows}))).unwrap();
    assert_eq!(ids(&parsed), names, "any id an endpoint returns is valid");
}

// ---- Ollama tags / LM Studio v0 --------------------------------------------

#[test]
fn catalog_ollama_tags_use_the_model_then_the_name() {
    let parsed = parse_ollama_tags(&body(&json!({"models": [
        {"name": "llama3:latest", "model": "llama3:latest", "details": {"family": "llama"}},
        {"name": "only-a-name"},
        {"size": 5}
    ]})))
    .unwrap();
    assert_eq!(ids(&parsed), ["llama3:latest", "only-a-name"]);
    assert_eq!(parsed.skipped, 1);
}

#[test]
fn catalog_ollama_with_nothing_pulled_is_an_empty_catalog_not_an_error() {
    assert!(
        parse_ollama_tags(br#"{"models":null}"#)
            .unwrap()
            .entries
            .is_empty()
    );
    assert!(
        parse_ollama_tags(br#"{"models":[]}"#)
            .unwrap()
            .entries
            .is_empty()
    );
}

#[test]
fn catalog_ollama_bad_documents_are_unknown() {
    for bad in [&b"nope"[..], br#"{"data":[]}"#, br#"{"models":"x"}"#, b"[]"] {
        assert_eq!(
            parse_ollama_tags(bad).unwrap_err().reason,
            ReasonCode::Unknown
        );
    }
}

#[test]
fn catalog_lm_studio_reports_local_probe_facts_and_skips_embeddings() {
    let parsed = parse_lmstudio_v0(&body(&json!({"data": [
        {"id": "qwen-coder", "type": "llm", "state": "loaded", "max_context_length": 32768,
         "capabilities": ["tool_use"]},
        {"id": "llava", "type": "vlm", "max_context_length": 4096},
        {"id": "nomic-embed", "type": "embeddings"},
        {"id": "plain"}
    ]})))
    .unwrap();
    assert_eq!(ids(&parsed), ["qwen-coder", "llava", "plain"]);
    assert_eq!(
        parsed.skipped, 0,
        "an embedding row is expected, not damage"
    );
    let qwen = &parsed.entries[0];
    assert_eq!(
        qwen.capabilities.context_window,
        Sourced::new(Some(32_768), CapSource::LocalProbe)
    );
    assert_eq!(
        qwen.capabilities.tools,
        Sourced::new(Tri::Yes, CapSource::LocalProbe)
    );
    assert_eq!(qwen.capabilities.vision.value, Tri::Unknown);
    assert_eq!(
        parsed.entries[1].capabilities.vision,
        Sourced::new(Tri::Yes, CapSource::LocalProbe)
    );
    assert_eq!(
        parse_lmstudio_v0(b"{}").unwrap_err().reason,
        ReasonCode::Unknown
    );
    assert_eq!(
        parse_lmstudio_v0(b"x").unwrap_err().reason,
        ReasonCode::Unknown
    );
}

// ---- types -----------------------------------------------------------------

#[test]
fn catalog_a_model_info_lifts_into_an_entry_and_an_unaddressable_id_is_refused() {
    use tinyinference_llm::catalog::ModelInfo;
    let info = ModelInfo {
        id: "m".into(),
        owned_by: Some("acme".into()),
        context_window: Some(8192),
        display_name: Some("M".into()),
        input_per_1m: Some(1.0),
        output_per_1m: Some(2.0),
    };
    let entry = ModelEntry::try_from(info).unwrap();
    assert_eq!(entry.owned_by.as_deref(), Some("acme"));
    assert_eq!(entry.capabilities.context_window.value, Some(8192));
    let bad = ModelInfo {
        id: "two words".into(),
        owned_by: None,
        context_window: None,
        display_name: None,
        input_per_1m: None,
        output_per_1m: None,
    };
    assert!(ModelEntry::try_from(bad).is_err());
}

#[test]
fn catalog_entry_builders_and_list_helpers() {
    let entry = ModelEntry::new(id("b"))
        .with_context_window(1000, CapSource::LocalProbe)
        .with_display_name("B")
        .with_prices(Some(1.0), None);
    assert_eq!(
        entry.capabilities.context_window.source,
        CapSource::LocalProbe
    );
    assert_eq!(
        (entry.display_name.as_deref(), entry.input_per_1m),
        (Some("B"), Some(1.0))
    );
    let list = ModelList {
        models: std::sync::Arc::new(vec![ModelEntry::new(id("b")), ModelEntry::new(id("a"))]),
        freshness: Freshness::Fresh,
        truncated: false,
    };
    assert_eq!(list.ids(), ["b", "a"]);
    assert!(!list.is_stale());
    assert_eq!(list.sorted().ids(), ["a", "b"]);
    let stale = ModelList {
        models: std::sync::Arc::default(),
        freshness: Freshness::Stale {
            failure: crate::ProviderFailure::new(ReasonCode::Timeout, Retry::Never),
        },
        truncated: false,
    };
    assert!(stale.is_stale());
}

#[test]
fn catalog_entries_round_trip_through_json() {
    let mut entry = ModelEntry::new(id("m")).with_prices(Some(0.5), Some(1.5));
    entry.lifecycle = Some(Lifecycle {
        status: LifecycleStatus::Deprecated,
        retires_on: Some("2026-11-01".into()),
        replacement: Some(id("m2")),
    });
    entry.alias_of = Some(id("m-2026-09"));
    entry.origin = EntrySource::User;
    let back: ModelEntry = serde_json::from_str(&serde_json::to_string(&entry).unwrap()).unwrap();
    assert_eq!(back, entry);
    assert_eq!(
        serde_json::to_value(LifecycleStatus::Retired).unwrap(),
        json!("retired")
    );
    assert_eq!(
        serde_json::to_value(EntrySource::ProviderApi).unwrap(),
        json!("provider_api")
    );
}

// ---- merge -----------------------------------------------------------------

#[derive(Debug)]
struct Registry;

impl ModelMetadataSource for Registry {
    fn lookup(&self, kind: &KindId, model: &str) -> Option<ModelMeta> {
        if kind.as_str() != "openai" {
            return None;
        }
        let mut meta = ModelMeta::default();
        match model {
            "gpt-5" => {
                meta.display_name = Some("GPT-5 (registry)".into());
                meta.capabilities = Capabilities {
                    context_window: Sourced::new(Some(400_000), CapSource::ProviderApi),
                    tools: Sourced::new(Tri::Yes, CapSource::ProviderApi),
                    vision: Sourced::new(Tri::Yes, CapSource::UserOverride),
                    reasoning: Sourced::new(Tri::Unknown, CapSource::Registry),
                    ..Capabilities::default()
                };
                meta.input_per_1m = Some(9.0);
                meta.output_per_1m = Some(90.0);
                meta.alias_of = Some(ModelId::parse("gpt-5-2026-09").unwrap());
                meta.lifecycle = Some(Lifecycle {
                    status: LifecycleStatus::Active,
                    retires_on: None,
                    replacement: None,
                });
                Some(meta)
            }
            _ => None,
        }
    }
}

#[test]
fn catalog_a_registry_fills_gaps_and_tags_everything_it_adds_as_registry() {
    let mut entries = vec![
        ModelEntry::new(id("gpt-5")),
        ModelEntry::new(id("unknown-model")),
    ];
    merge_metadata(&mut entries, &KindId::new("openai"), Some(&Registry), &[]);
    let e = &entries[0];
    assert_eq!(e.display_name.as_deref(), Some("GPT-5 (registry)"));
    assert_eq!(
        e.capabilities.context_window,
        Sourced::new(Some(400_000), CapSource::Registry)
    );
    assert_eq!(
        e.capabilities.tools.source,
        CapSource::Registry,
        "a registry cannot claim to be the provider"
    );
    assert_eq!(e.capabilities.vision.source, CapSource::Registry);
    assert_eq!(
        e.capabilities.reasoning.source,
        CapSource::Default,
        "unknown adds nothing"
    );
    assert_eq!((e.input_per_1m, e.output_per_1m), (Some(9.0), Some(90.0)));
    assert!(e.alias_of.is_some() && e.lifecycle.is_some());
    assert_eq!(entries[1], ModelEntry::new(id("unknown-model")));
}

#[test]
fn catalog_a_registry_never_overwrites_what_the_provider_said() {
    let mut provider = ModelEntry::new(id("gpt-5"))
        .with_context_window(128_000, CapSource::ProviderApi)
        .with_display_name("Provider name")
        .with_prices(Some(1.0), Some(2.0));
    provider.capabilities.tools = Sourced::new(Tri::No, CapSource::LocalProbe);
    let mut entries = vec![provider];
    merge_metadata(&mut entries, &KindId::new("openai"), Some(&Registry), &[]);
    let e = &entries[0];
    assert_eq!(
        e.capabilities.context_window,
        Sourced::new(Some(128_000), CapSource::ProviderApi)
    );
    assert_eq!(
        e.capabilities.tools,
        Sourced::new(Tri::No, CapSource::LocalProbe)
    );
    assert_eq!(e.display_name.as_deref(), Some("Provider name"));
    assert_eq!((e.input_per_1m, e.output_per_1m), (Some(1.0), Some(2.0)));
}

#[test]
fn catalog_a_registry_for_another_kind_is_not_consulted() {
    let mut entries = vec![ModelEntry::new(id("gpt-5"))];
    merge_metadata(&mut entries, &KindId::new("groq"), Some(&Registry), &[]);
    assert_eq!(entries[0], ModelEntry::new(id("gpt-5")));
}

#[test]
fn catalog_an_override_beats_the_provider_and_is_tagged_user() {
    let mut entries =
        vec![ModelEntry::new(id("m")).with_context_window(8000, CapSource::ProviderApi)];
    let mut over = ModelOverride::new(id("m"));
    over.context_window = Some(32_000);
    over.max_output = Some(4096);
    over.tools = Some(Tri::Yes);
    over.vision = Some(Tri::No);
    over.reasoning = Some(Tri::Yes);
    over.temperature = Some(Tri::No);
    over.structured_output = Some(Tri::Yes);
    over.display_name = Some("Mine".into());
    merge_metadata(&mut entries, &KindId::new("custom"), None, &[over]);
    let c = &entries[0].capabilities;
    assert_eq!(
        c.context_window,
        Sourced::new(Some(32_000), CapSource::UserOverride)
    );
    assert_eq!(
        c.max_output,
        Sourced::new(Some(4096), CapSource::UserOverride)
    );
    for (fact, value) in [
        (c.tools, Tri::Yes),
        (c.vision, Tri::No),
        (c.reasoning, Tri::Yes),
        (c.temperature, Tri::No),
        (c.structured_output, Tri::Yes),
    ] {
        assert_eq!(fact, Sourced::new(value, CapSource::UserOverride));
    }
    assert_eq!(entries[0].display_name.as_deref(), Some("Mine"));
    assert_eq!(
        entries[0].origin,
        EntrySource::ProviderApi,
        "an override of a listed model is not a user model"
    );
}

#[test]
fn catalog_an_override_for_an_unlisted_model_adds_it_as_a_user_entry() {
    let mut entries = vec![ModelEntry::new(id("listed"))];
    merge_metadata(
        &mut entries,
        &KindId::new("azure"),
        None,
        &[ModelOverride::new(id("my-deployment"))],
    );
    assert_eq!(entries.len(), 2);
    assert_eq!(entries[1].id.as_str(), "my-deployment");
    assert_eq!(entries[1].origin, EntrySource::User);
    assert_eq!(
        entries[1].capabilities,
        Capabilities::default(),
        "an empty override claims nothing"
    );
}

#[test]
fn catalog_an_override_beats_a_registry() {
    let mut entries = vec![ModelEntry::new(id("gpt-5"))];
    let mut over = ModelOverride::new(id("gpt-5"));
    over.context_window = Some(1);
    merge_metadata(
        &mut entries,
        &KindId::new("openai"),
        Some(&Registry),
        &[over],
    );
    assert_eq!(
        entries[0].capabilities.context_window,
        Sourced::new(Some(1), CapSource::UserOverride)
    );
}

// ---- properties ------------------------------------------------------------

proptest! {
    #[test]
    fn catalog_prop_no_parser_panics_on_arbitrary_bytes(bytes in proptest::collection::vec(any::<u8>(), 0..400)) {
        let _ = parse_openai(&bytes);
        let _ = parse_ollama_tags(&bytes);
        let _ = parse_lmstudio_v0(&bytes);
        if let Ok(text) = std::str::from_utf8(&bytes) {
            let _ = parse_page(text);
        }
    }

    #[test]
    fn catalog_prop_parsed_ids_are_unique_valid_and_no_more_than_the_rows(
        ids in proptest::collection::vec("[a-z0-9./:_-]{0,12}", 0..30)
    ) {
        let rows: Vec<_> = ids.iter().map(|i| json!({"id": i})).collect();
        let parsed = parse_openai(&body(&json!({"data": rows}))).unwrap();
        let mut seen = std::collections::HashSet::new();
        for entry in &parsed.entries {
            prop_assert!(seen.insert(entry.id.as_str().to_string()), "duplicate {}", entry.id);
            prop_assert!(ModelId::parse(entry.id.as_str()).is_ok());
        }
        prop_assert_eq!(parsed.entries.len() + parsed.skipped, ids.len());
    }

    #[test]
    fn catalog_prop_arbitrary_json_documents_never_panic(text in "[\\[\\]{}\":,a-z0-9 nul]{0,80}") {
        let _ = parse_openai(text.as_bytes());
        let _ = parse_ollama_tags(text.as_bytes());
        let _ = parse_lmstudio_v0(text.as_bytes());
    }
}

#[test]
fn catalog_lm_studio_counts_a_damaged_embeddings_row_once_and_a_healthy_one_not_at_all() {
    let parsed = parse_lmstudio_v0(&body(&json!({"data": [
        {"id": "chat", "type": "llm"},
        {"id": "embed-ok", "type": "embeddings"},
        {"type": "embeddings"},
        {"type": "llm"}
    ]})))
    .unwrap();
    assert_eq!(ids(&parsed), ["chat"]);
    assert_eq!(
        parsed.skipped, 1,
        "only the chat row without an id is damage"
    );
}

#[test]
fn catalog_an_override_that_adds_a_model_gets_the_registrys_facts_too() {
    let mut entries = vec![ModelEntry::new(id("listed"))];
    let mut over = ModelOverride::new(id("gpt-5"));
    over.context_window = Some(1);
    merge_metadata(
        &mut entries,
        &KindId::new("openai"),
        Some(&Registry),
        &[over],
    );
    let added = entries.iter().find(|e| e.id.as_str() == "gpt-5").unwrap();
    assert_eq!(added.origin, EntrySource::User);
    assert_eq!(
        added.capabilities.context_window,
        Sourced::new(Some(1), CapSource::UserOverride),
        "the operator wins"
    );
    assert_eq!(
        added.capabilities.tools.source,
        CapSource::Registry,
        "the registry fills what is left"
    );
    assert_eq!(added.input_per_1m, Some(9.0));
}

#[test]
fn catalog_a_listing_whose_every_row_is_unusable_is_not_a_healthy_empty_one() {
    let parsed = parse_openai(&body(
        &json!({"data": [{"model": "x"}, {"id": 5}, {"id": "has space"}]}),
    ))
    .unwrap();
    assert!(parsed.entries.is_empty() && parsed.skipped == 3);
    let failure = parsed.into_usable().unwrap_err();
    assert_eq!(failure.reason, ReasonCode::Unknown);
    assert!(!failure.reason.destroys_credential());
    // Genuinely empty (nothing pulled) is still fine, and so is a mixed list.
    assert!(
        parse_openai(br#"{"data":[]}"#)
            .unwrap()
            .into_usable()
            .unwrap()
            .is_empty()
    );
    let mixed = parse_openai(&body(&json!({"data": [{"id": "ok"}, {"id": 5}]}))).unwrap();
    assert_eq!(mixed.into_usable().unwrap().len(), 1);
}
