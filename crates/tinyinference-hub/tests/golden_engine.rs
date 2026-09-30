//! Golden tests for the engine's wire shapes: stored configuration, stored
//! health, and the catalog corpus (every listing shape a provider answers in).

use serde_json::Value;

use tinyinference_hub::catalog::{parse_lmstudio_v0, parse_ollama_tags, parse_openai, parse_page};
use tinyinference_hub::config::{DefaultChoice, HubConfig};
use tinyinference_hub::descriptor::{CapSource, Tri};
use tinyinference_hub::health::{HealthSnapshot, ProviderHealth};
use tinyinference_hub::{AuthStyle, ReasonCode};

const CONFIG: &str = include_str!("golden/hub_config_v1.json");
const HEALTH: &str = include_str!("golden/health_snapshot.json");
const OPENAI: &str = include_str!("golden/catalogs/openai.json");
const TOGETHER: &str = include_str!("golden/catalogs/together_bare_array.json");
const OPENROUTER: &str = include_str!("golden/catalogs/openrouter.json");
const OLLAMA_TAGS: &str = include_str!("golden/catalogs/ollama_tags.json");
const OLLAMA_NULL: &str = include_str!("golden/catalogs/ollama_null.json");
const LMSTUDIO: &str = include_str!("golden/catalogs/lmstudio_v0.json");
const ANTHROPIC: &str = include_str!("golden/catalogs/anthropic.json");
const MANAGED: &str = include_str!("golden/catalogs/managed_page.json");

fn ids(parsed: &tinyinference_hub::catalog::ParsedCatalog) -> Vec<&str> {
    parsed.entries.iter().map(|e| e.id.as_str()).collect()
}

#[test]
fn golden_a_stored_configuration_loads_and_saves_back_unchanged() {
    let config: HubConfig = serde_json::from_str(CONFIG).unwrap();
    assert_eq!(config.providers.len(), 3);
    assert!(
        matches!(&config.default, DefaultChoice::Full { provider, .. } if provider.as_str() == "openai")
    );
    assert_eq!(config.agent_pins.len(), 1);
    let home = &config.providers[2];
    assert_eq!(
        home.auth_override,
        Some(AuthStyle::Custom("x-home-key".into()))
    );
    assert!(
        home.legacy.contains_key("tiers"),
        "an adapter's own fields ride along"
    );
    assert!(
        config.extra.contains_key("future_field"),
        "so do a newer hub's"
    );
    let saved = serde_json::to_value(&config).unwrap();
    let original: Value = serde_json::from_str(CONFIG).unwrap();
    assert_eq!(saved, original, "load then save is the identity");
}

#[test]
fn golden_stored_health_loads_and_saves_back_unchanged() {
    let snapshot: HealthSnapshot = serde_json::from_str(HEALTH).unwrap();
    assert_eq!(
        snapshot.health,
        ProviderHealth::Degraded(ReasonCode::Timeout)
    );
    assert_eq!(snapshot.consecutive_failures, 1);
    assert_eq!(
        serde_json::to_value(&snapshot).unwrap(),
        serde_json::from_str::<Value>(HEALTH).unwrap()
    );
}

#[test]
fn golden_the_openai_shaped_corpus() {
    assert_eq!(
        ids(&parse_openai(OPENAI.as_bytes()).unwrap()),
        ["gpt-5", "gpt-5-mini", "o3"]
    );

    let together = parse_openai(TOGETHER.as_bytes()).unwrap();
    assert_eq!(
        ids(&together),
        [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo"
        ]
    );
    assert_eq!(
        together.entries[0].capabilities.context_window.value,
        Some(131_072)
    );
    assert_eq!(together.entries[0].input_per_1m, Some(0.88));

    let openrouter = parse_openai(OPENROUTER.as_bytes()).unwrap();
    assert_eq!(
        ids(&openrouter),
        [
            "anthropic/claude-sonnet-4",
            "openai/gpt-5",
            "x-ai/grok-4:free"
        ]
    );
    assert_eq!(
        openrouter.entries[0].display_name.as_deref(),
        Some("Anthropic: Claude Sonnet 4")
    );

    let anthropic = parse_openai(ANTHROPIC.as_bytes()).unwrap();
    assert_eq!(
        ids(&anthropic),
        ["claude-opus-4-1-20250805", "claude-sonnet-4-20250514"]
    );
    assert_eq!(
        anthropic.entries[1].display_name.as_deref(),
        Some("Claude Sonnet 4")
    );

    assert!(
        parse_openai(OLLAMA_NULL.as_bytes())
            .unwrap()
            .entries
            .is_empty()
    );
}

#[test]
fn golden_the_native_local_listings() {
    let tags = parse_ollama_tags(OLLAMA_TAGS.as_bytes()).unwrap();
    assert_eq!(ids(&tags), ["llama3.2:latest", "qwen2.5-coder:7b"]);

    let lm = parse_lmstudio_v0(LMSTUDIO.as_bytes()).unwrap();
    assert_eq!(
        ids(&lm),
        ["qwen2.5-coder-7b-instruct", "llava-1.5-7b"],
        "the embedding model is skipped"
    );
    let qwen = &lm.entries[0].capabilities;
    assert_eq!(
        (qwen.tools.value, qwen.tools.source),
        (Tri::Yes, CapSource::LocalProbe)
    );
    assert_eq!(qwen.context_window.value, Some(32_768));
    assert_eq!(lm.entries[1].capabilities.vision.value, Tri::Yes);
}

#[test]
fn golden_the_managed_page_carries_charged_prices() {
    let page = parse_page(MANAGED).unwrap();
    assert_eq!((page.raw_len, page.total), (3, Some(3)));
    let names: Vec<_> = page.entries.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(
        names,
        [
            "anthropic/claude-sonnet-4",
            "openai/gpt-5",
            "deepseek/deepseek-chat-v3"
        ]
    );
    assert_eq!(page.entries[0].input_per_1m, Some(3.3));
    assert_eq!(page.entries[1].output_per_1m, Some(11.0));
    assert_eq!(
        page.entries[1].display_name.as_deref(),
        Some("GPT-5"),
        "name is the fallback"
    );
    assert_eq!(page.entries[2].input_per_1m, None);
}

#[test]
fn golden_no_corpus_body_contains_a_credential_shaped_string() {
    for (name, body) in [
        ("openai", OPENAI),
        ("config", CONFIG),
        ("health", HEALTH),
        ("managed", MANAGED),
    ] {
        for needle in ["sk-", "Bearer ", "api_key", "secret"] {
            assert!(!body.contains(needle), "{name} contains {needle}");
        }
    }
}
