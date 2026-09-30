//! Tests for `HubConfig`, `DefaultChoice` and `ProviderDraft`.

use serde_json::json;

use super::*;
use crate::descriptor::ProviderRecord;
use crate::error::InvalidInput;
use crate::ids::{AgentKey, KindId, ModelId, Slug};
use crate::secret::Secret;

fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

fn record(name: &str) -> ProviderRecord {
    ProviderRecord::new(
        format!("prv_{name}"),
        slug(name),
        name,
        KindId::new("custom"),
        "https://api.acme.test/v1",
    )
}

#[test]
fn config_an_empty_document_is_the_current_schema_with_nothing_in_it() {
    let config: HubConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(config, HubConfig::new());
    assert_eq!(config.schema_version, CONFIG_SCHEMA_VERSION);
    assert!(config.providers.is_empty());
    assert_eq!(config.default, DefaultChoice::Unset);
}

#[test]
fn config_round_trips_with_providers_default_and_pins() {
    let mut config = HubConfig::new();
    config.providers.push(record("acme"));
    config.providers.push(record("beta"));
    config.default = DefaultChoice::Full {
        provider: slug("acme"),
        model: ModelId::parse("gpt-5").unwrap(),
    };
    config.agent_pins.insert(
        AgentKey::new("agent:writer"),
        ModelChoice::new(slug("beta"), ModelId::parse("m1").unwrap()),
    );
    let text = serde_json::to_string(&config).unwrap();
    let back: HubConfig = serde_json::from_str(&text).unwrap();
    assert_eq!(back, config);
}

#[test]
fn config_wire_shape_of_the_default_choice_is_tagged() {
    let unset = serde_json::to_value(DefaultChoice::Unset).unwrap();
    assert_eq!(unset, json!({"mode": "unset"}));
    let only = serde_json::to_value(DefaultChoice::ProviderOnly {
        provider: slug("acme"),
    })
    .unwrap();
    assert_eq!(only, json!({"mode": "provider_only", "provider": "acme"}));
    let full = serde_json::to_value(DefaultChoice::Full {
        provider: slug("acme"),
        model: ModelId::parse("m").unwrap(),
    })
    .unwrap();
    assert_eq!(
        full,
        json!({"mode": "full", "provider": "acme", "model": "m"})
    );
}

#[test]
fn config_fields_a_newer_hub_added_survive_a_load_and_save() {
    let text = r#"{"schema_version":1,"providers":[],"future_feature":{"a":[1,2,3]}}"#;
    let config: HubConfig = serde_json::from_str(text).unwrap();
    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(value["future_feature"], json!({"a": [1, 2, 3]}));
    assert_eq!(value["schema_version"], json!(1));
}

#[test]
fn config_a_credential_shaped_top_level_field_will_not_load() {
    for name in [
        "api_key",
        "apiKey",
        "authorization",
        "client_secret",
        "refresh_token",
    ] {
        let text = format!(r#"{{"providers":[],"{name}":"sk-not-a-real-key"}}"#);
        let error = serde_json::from_str::<HubConfig>(&text)
            .unwrap_err()
            .to_string();
        assert!(error.contains("looks like a credential"), "{name}: {error}");
        assert!(!error.contains("sk-not-a-real-key"), "{error}");
    }
}

#[test]
fn config_ordinary_extra_fields_are_not_credentials() {
    let text = r#"{"providers":[],"max_tokens":5,"page_token":"x","tokenizer":"y"}"#;
    assert!(serde_json::from_str::<HubConfig>(text).is_ok());
}

#[test]
fn config_a_record_carrying_a_credential_will_not_load() {
    let text = json!({"providers": [{
        "id": "p", "slug": "acme", "label": "Acme", "kind": "custom",
        "base_url": "https://a.test/v1", "api_key": "sk-not-a-real-key"
    }]});
    assert!(serde_json::from_value::<HubConfig>(text).is_err());
}

#[test]
fn config_two_providers_with_one_slug_are_refused() {
    let mut config = HubConfig::new();
    config.providers.push(record("acme"));
    config.providers.push(record("acme"));
    assert!(matches!(
        config.validate(),
        Err(InvalidInput::Malformed { .. })
    ));
    let text = serde_json::to_string(&config).unwrap();
    assert!(serde_json::from_str::<HubConfig>(&text).is_err());
}

#[test]
fn config_provider_lookup_reads_and_mutates_by_slug() {
    let mut config = HubConfig::new();
    config.providers.push(record("acme"));
    assert!(config.contains(&slug("acme")) && !config.contains(&slug("nope")));
    assert_eq!(config.provider(&slug("acme")).unwrap().label, "acme");
    config.provider_mut(&slug("acme")).unwrap().enabled = false;
    assert!(!config.provider(&slug("acme")).unwrap().enabled);
    assert!(config.provider_mut(&slug("nope")).is_none());
}

#[test]
fn config_pins_are_keyed_by_the_hosts_agent_key() {
    let text = r#"{"agent_pins":{"agent:a":{"provider":"acme","model":"m"}}}"#;
    let config: HubConfig = serde_json::from_str(text).unwrap();
    let pin = config.agent_pins.get(&AgentKey::new("agent:a")).unwrap();
    assert_eq!(pin.provider, slug("acme"));
    assert_eq!(pin.model.as_str(), "m");
}

#[test]
fn config_a_pin_with_an_invalid_model_will_not_load() {
    let text = r#"{"agent_pins":{"a":{"provider":"acme","model":"has space"}}}"#;
    assert!(serde_json::from_str::<HubConfig>(text).is_err());
}

#[test]
fn draft_builders_set_fields_and_the_debug_never_prints_the_key() {
    let draft = ProviderDraft::new("OpenAI")
        .with_base_url("https://api.openai.com/v1")
        .with_key(Secret::new("sk-not-a-real-key"))
        .with_model(ModelId::parse("gpt-5").unwrap())
        .with_label("Work OpenAI");
    assert_eq!(draft.kind.as_str(), "openai", "kind ids normalise");
    assert_eq!(draft.base_url.as_deref(), Some("https://api.openai.com/v1"));
    assert_eq!(draft.model.as_ref().unwrap().as_str(), "gpt-5");
    assert_eq!(draft.label.as_deref(), Some("Work OpenAI"));
    assert!(!format!("{draft:?}").contains("sk-not-a-real-key"));
    let bare = ProviderDraft::new("ollama");
    assert!(bare.key.is_none() && bare.base_url.is_none() && bare.model.is_none());
}

mod config_props {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// A configuration always survives a save and a load, whatever slugs and
        /// models it holds, and what is written never contains a credential-shaped name.
        #[test]
        fn config_prop_round_trips_and_never_grows_a_credential_field(
            slugs in proptest::collection::btree_set("[a-z0-9][a-z0-9_-]{0,20}", 0..6),
            model in "[a-zA-Z0-9./:_-]{1,30}",
            pick in 0usize..7,
            pins in proptest::collection::vec("[a-z:]{1,12}", 0..4),
        ) {
            let mut config = HubConfig::new();
            for s in &slugs {
                config.providers.push(record(s));
            }
            let all: Vec<_> = slugs.iter().collect();
            if let Some(chosen) = all.get(pick % all.len().max(1)) {
                config.default = DefaultChoice::Full { provider: slug(chosen), model: ModelId::parse(&model).unwrap() };
                for pin in &pins {
                    config.agent_pins.insert(AgentKey::new(pin.clone()), ModelChoice::new(slug(chosen), ModelId::parse(&model).unwrap()));
                }
            }
            let text = serde_json::to_string(&config).unwrap();
            let back: HubConfig = serde_json::from_str(&text).unwrap();
            prop_assert_eq!(&back, &config);
            let value: serde_json::Value = serde_json::from_str(&text).unwrap();
            for key in value.as_object().unwrap().keys() {
                prop_assert!(!crate::secret::is_credential_name(key), "{key}");
            }
        }
    }
}

#[test]
fn config_an_extra_field_can_never_shadow_a_known_one_on_the_way_out() {
    let mut config = HubConfig::new();
    config.schema_version = 1;
    config.extra.insert("providers".into(), json!("shadow"));
    config.extra.insert("schema_version".into(), json!(99));
    config.extra.insert("kept".into(), json!(1));
    let value = serde_json::to_value(&config).unwrap();
    assert_eq!(value["providers"], json!([]));
    assert_eq!(value["schema_version"], json!(1));
    assert_eq!(value["kept"], json!(1));
}

#[test]
fn config_a_document_from_a_newer_hub_is_refused_not_rewritten() {
    let text = format!(
        r#"{{"schema_version":{},"providers":[]}}"#,
        CONFIG_SCHEMA_VERSION + 1
    );
    let error = serde_json::from_str::<HubConfig>(&text)
        .unwrap_err()
        .to_string();
    assert!(error.contains("newer version"), "{error}");
    let mut config = HubConfig::new();
    config.schema_version = CONFIG_SCHEMA_VERSION + 1;
    assert!(config.validate().is_err());
    // The current version and older ones load.
    assert!(serde_json::from_str::<HubConfig>(r#"{"schema_version":0}"#).is_ok());
}

#[test]
fn config_a_credential_nested_under_an_unknown_field_is_refused_too() {
    for text in [
        r#"{"telemetry":{"api_key":"sk-not-a-real-key"}}"#,
        r#"{"a":[{"b":{"client_secret":"x"}}]}"#,
        r#"{"a":{"b":{"c":{"password":"x"}}}}"#,
    ] {
        let error = serde_json::from_str::<HubConfig>(text)
            .unwrap_err()
            .to_string();
        assert!(error.contains("looks like a credential"), "{text}: {error}");
        assert!(!error.contains("sk-not-a-real-key"));
    }
    assert!(
        serde_json::from_str::<HubConfig>(r#"{"telemetry":{"page_token":"x","max_tokens":5}}"#)
            .is_ok()
    );
}
