//! Descriptor, record and capability tests.

use serde_json::json;

use super::*;
use crate::ids::{KindId, ModelId, Slug};
use crate::taxonomy::AuthStyle;

fn record() -> ProviderRecord {
    ProviderRecord::new(
        "prv_0123456789abcdef0123456789abcdef",
        Slug::parse("acme").unwrap(),
        "Acme",
        KindId::new("custom"),
        "https://api.acme.test/v1",
    )
}

#[test]
fn a_new_record_is_enabled_indexed_and_carries_no_credential() {
    let r = record();
    assert!(r.enabled && !r.synthetic);
    assert_eq!(r.origin, RecordOrigin::Indexed);
    assert!(r.model.is_none() && r.auth_override.is_none() && r.legacy.is_empty());
    let value = serde_json::to_value(&r).unwrap();
    let object = value.as_object().unwrap();
    // Invariant 1: no credential-shaped field, ever.
    for key in object.keys() {
        let lower = key.to_ascii_lowercase();
        for forbidden in ["key", "secret", "token", "credential", "password"] {
            assert!(!lower.contains(forbidden), "record has a `{key}` field");
        }
    }
    assert_eq!(value["slug"], json!("acme"));
    assert_eq!(value["kind"], json!("custom"));
    assert_eq!(value["origin"], json!("indexed"));
}

#[test]
fn a_record_round_trips_including_optional_fields() {
    let mut r = record();
    r.model = Some(ModelId::parse("gpt-5").unwrap());
    r.auth_override = Some(AuthStyle::Custom("api-key".into()));
    r.synthetic = true;
    r.origin = RecordOrigin::Imported;
    r.enabled = false;
    let back: ProviderRecord = serde_json::from_value(serde_json::to_value(&r).unwrap()).unwrap();
    assert_eq!(back, r);
}

#[test]
fn a_minimal_stored_record_loads_with_defaults() {
    let r: ProviderRecord = serde_json::from_value(json!({
        "id": "p_1", "slug": "acme", "label": "Acme", "kind": "Custom", "base_url": "https://a.test/v1"
    }))
    .unwrap();
    assert!(r.enabled, "enabled defaults to true");
    assert!(!r.synthetic);
    assert_eq!(r.origin, RecordOrigin::Indexed);
    assert_eq!(r.kind.as_str(), "custom", "kind ids normalise on load");
}

#[test]
fn unknown_fields_are_preserved_verbatim_through_a_round_trip() {
    // OpenCompany's tier map and any future field survive a load and a save.
    let stored = json!({
        "id": "p_1", "slug": "acme", "label": "Acme", "kind": "custom", "base_url": "https://a.test/v1",
        "tiers": {"chat-v1": "gpt-5"}, "future_flag": true
    });
    let r: ProviderRecord = serde_json::from_value(stored.clone()).unwrap();
    assert_eq!(r.legacy.len(), 2);
    assert_eq!(r.legacy["future_flag"], json!(true));
    let saved = serde_json::to_value(&r).unwrap();
    assert_eq!(saved["tiers"], stored["tiers"]);
    assert_eq!(saved["future_flag"], json!(true));
}

#[test]
fn a_record_with_a_bad_slug_or_model_id_does_not_load() {
    let bad_slug =
        json!({"id":"1","slug":"Bad Slug","label":"x","kind":"custom","base_url":"https://a.test"});
    assert!(serde_json::from_value::<ProviderRecord>(bad_slug).is_err());
    let bad_model = json!({"id":"1","slug":"ok","label":"x","kind":"custom","base_url":"https://a.test","model":"has space"});
    assert!(serde_json::from_value::<ProviderRecord>(bad_model).is_err());
}

#[test]
fn record_origins_serialise_snake_case() {
    for (origin, wire) in [
        (RecordOrigin::Indexed, "indexed"),
        (RecordOrigin::EntryZero, "entry_zero"),
        (RecordOrigin::Imported, "imported"),
    ] {
        assert_eq!(serde_json::to_value(origin).unwrap(), json!(wire));
    }
    assert_eq!(RecordOrigin::default(), RecordOrigin::Indexed);
}

#[test]
fn capabilities_default_to_unknown_never_yes() {
    let caps = Capabilities::default();
    for tri in [
        caps.tools,
        caps.vision,
        caps.reasoning,
        caps.temperature,
        caps.structured_output,
    ] {
        assert_eq!(tri.value, Tri::Unknown);
        assert_eq!(tri.source, CapSource::Default);
        assert!(!tri.value.is_yes());
    }
    assert_eq!(caps.context_window.value, None);
    assert_eq!(caps.max_output.source, CapSource::Default);
    assert_eq!(Tri::default(), Tri::Unknown);
    assert!(Tri::Yes.is_yes() && !Tri::No.is_yes());
}

#[test]
fn sourced_values_remember_where_they_came_from() {
    let ctx = Sourced::new(Some(128_000u64), CapSource::ProviderApi);
    assert_eq!(ctx.value, Some(128_000));
    assert_eq!(ctx.source, CapSource::ProviderApi);
    let json = serde_json::to_value(ctx).unwrap();
    assert_eq!(json, json!({"value": 128000, "source": "provider_api"}));
    let back: Sourced<Option<u64>> = serde_json::from_value(json).unwrap();
    assert_eq!(back, ctx);
    for (source, wire) in [
        (CapSource::LocalProbe, "local_probe"),
        (CapSource::Registry, "registry"),
        (CapSource::UserOverride, "user_override"),
        (CapSource::Default, "default"),
    ] {
        assert_eq!(serde_json::to_value(source).unwrap(), json!(wire));
    }
    let caps = Capabilities::default();
    let back: Capabilities = serde_json::from_value(serde_json::to_value(caps).unwrap()).unwrap();
    assert_eq!(back, caps);
}

#[test]
fn a_descriptor_serialises_for_a_ui_without_leaking_anything_secret() {
    let d = crate::catalogue::descriptor("anthropic").unwrap();
    let value = serde_json::to_value(d).unwrap();
    assert_eq!(value["kind"], json!("anthropic"));
    assert_eq!(value["auth"], json!("anthropic"));
    assert_eq!(value["protocol"], json!("anthropic_messages"));
    assert_eq!(
        value["default_endpoint"],
        json!("https://api.anthropic.com/v1")
    );
    assert_eq!(value["endpoint_editable"], json!(false));
    assert_eq!(d.slug(), "anthropic");
}

#[test]
fn quirks_serialise_snake_case() {
    assert_eq!(
        serde_json::to_value(Quirk::CatalogUnauthenticated).unwrap(),
        json!("catalog_unauthenticated")
    );
    assert_eq!(
        serde_json::to_value(Quirk::ResponsesApi).unwrap(),
        json!("responses_api")
    );
    assert_eq!(
        serde_json::to_value(Quirk::MaxCompletionTokens).unwrap(),
        json!("max_completion_tokens")
    );
}

proptest::proptest! {
    #[test]
    fn a_record_round_trips_through_json(
        id in "[a-z0-9_]{1,20}",
        slug in "[a-z0-9][a-z0-9_-]{0,20}",
        label in "\\PC{1,30}",
        kind in "[A-Za-z-]{1,15}",
        url in "https://[a-z]{3,10}\\.test(/[a-z0-9]{1,6}){0,2}",
        enabled in proptest::bool::ANY,
        synthetic in proptest::bool::ANY,
        extra_key in "meta[0-9]{1,4}",
        extra_val in 0i64..1000,
    ) {
        let mut r = ProviderRecord::new(id, Slug::parse(&slug).unwrap(), label, KindId::new(kind), url);
        r.enabled = enabled;
        r.synthetic = synthetic;
        r.legacy.insert(extra_key, json!(extra_val));
        let back: ProviderRecord = serde_json::from_value(serde_json::to_value(&r).unwrap()).unwrap();
        proptest::prop_assert_eq!(back, r);
    }
}

// ---- the record enforces its own invariant (review finding) ---------------------------

fn stored(extra: serde_json::Value) -> serde_json::Value {
    let mut base = json!({
        "id": "p_1", "slug": "acme", "label": "Acme", "kind": "custom", "base_url": "https://a.test/v1"
    });
    base.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    base
}

#[test]
fn a_stored_record_with_a_credential_shaped_field_does_not_load() {
    // Regression: `#[serde(flatten)] legacy` used to carry an inline key
    // through every load and save, so "no credential field, ever" was only a
    // comment.
    for name in [
        "api_key",
        "apiKey",
        "API-KEY",
        "openai_api_key",
        "secret",
        "client_secret",
        "password",
        "authorization",
        "bearer_token",
        "key",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "credential",
        "credentials",
    ] {
        let loaded =
            serde_json::from_value::<ProviderRecord>(stored(json!({ name: "sk-not-a-real-key" })));
        let error = loaded.expect_err(name).to_string();
        assert!(error.contains("credential"), "{name}: {error}");
        assert!(
            !error.contains("sk-not-a-real-key"),
            "the value is never echoed: {error}"
        );
    }
}

#[test]
fn ordinary_legacy_fields_are_not_mistaken_for_credentials() {
    for name in [
        "tiers",
        "max_tokens",
        "keywords",
        "models",
        "monkey",
        "tokenizer",
        "note",
        "display_key_hint_count",
    ] {
        let r: ProviderRecord = serde_json::from_value(stored(json!({ name: 1 }))).unwrap();
        assert!(r.legacy.contains_key(name), "{name}");
    }
}

#[test]
fn a_stored_record_whose_endpoint_carries_userinfo_does_not_load() {
    let loaded = serde_json::from_value::<ProviderRecord>(stored(
        json!({"base_url": "https://u:pw@host.test/v1"}),
    ));
    let error = loaded.unwrap_err().to_string();
    assert!(error.contains("username or password"), "{error}");
    assert!(!error.contains("pw@"), "{error}");
}

#[test]
fn validate_checks_a_hand_built_record_too() {
    let mut r = record();
    assert_eq!(r.validate(), Ok(()));
    r.legacy.insert("apikey".into(), json!("sk-not-a-real-key"));
    assert_eq!(
        r.validate(),
        Err(crate::error::InvalidInput::CredentialField {
            name: "apikey".into()
        })
    );
    r.legacy.clear();
    r.base_url = "http://alice:hunter2@localhost:8080/v1".into();
    assert!(matches!(
        r.validate(),
        Err(crate::error::InvalidInput::Malformed { .. })
    ));
}

#[test]
fn credentials_nested_or_spelled_differently_are_refused_too() {
    // Regression (review round 2): only top-level, exact-ish names were checked.
    for extra in [
        json!({"tiers": {"api_key": "sk-not-a-real-key"}}),
        json!({"headers": {"Authorization": "Bearer sk-not-a-real-key"}}),
        json!({"nested": [{"deeper": {"client_secret": "x"}}]}),
        json!({"auth_token": "x"}),
        json!({"session_token": "x"}),
        json!({"access_key": "x"}),
        json!({"private_key": "x"}),
        json!({"passphrase": "x"}),
        json!({"db_passwd": "x"}),
    ] {
        let name = extra.as_object().unwrap().keys().next().unwrap().clone();
        let loaded = serde_json::from_value::<ProviderRecord>(stored(extra));
        let error = loaded.expect_err(&name).to_string();
        assert!(
            error.contains("credential") && !error.contains("sk-not-a-real-key"),
            "{name}: {error}"
        );
    }
    // A too-deep structure is not walked without bound (and is refused), and ordinary nested data loads.
    let mut deep = json!({"leaf": 1});
    for _ in 0..20 {
        deep = json!({ "n": deep });
    }
    // Fail closed: nesting beyond the bound is refused, not assumed clean.
    assert!(serde_json::from_value::<ProviderRecord>(stored(json!({"tiers": deep}))).is_err());
    let ok =
        json!({"tiers": {"chat-v1": "gpt-5", "list": [1, {"max_tokens": 5, "tokenizer": "x"}]}});
    assert!(serde_json::from_value::<ProviderRecord>(stored(ok)).is_ok());
}

// ---- round 3 review regressions --------------------------------------------------------

#[test]
fn a_credential_hidden_below_the_depth_bound_is_refused_not_assumed_clean() {
    let mut nested = json!({"password": "x"});
    for _ in 0..9 {
        nested = json!({ "wrap": nested });
    }
    let error =
        serde_json::from_value::<ProviderRecord>(stored(json!({"tiers": nested}))).unwrap_err();
    assert!(error.to_string().contains("credential"), "{error}");
}

#[test]
fn camel_case_credential_names_are_recognised() {
    for name in [
        "accessToken",
        "authToken",
        "refreshToken",
        "sessionToken",
        "accessKey",
        "privateKey",
        "apiKey",
        "clientSecret",
        "auth",
        "Signature",
        "APIKEY",
    ] {
        assert!(
            serde_json::from_value::<ProviderRecord>(stored(json!({ name: "x" }))).is_err(),
            "{name}"
        );
    }
    for name in [
        "maxTokens",
        "tokenizer",
        "keywords",
        "displayName",
        "authors",
        "oauthRedirect",
    ] {
        assert!(
            serde_json::from_value::<ProviderRecord>(stored(json!({ name: "x" }))).is_ok(),
            "{name}"
        );
    }
}

#[test]
fn a_credential_in_the_endpoint_query_string_is_refused() {
    for url in [
        "http://h.test/v1?api_key=sk-not-a-real-key",
        "https://h.test/v1?key=AIzaFAKE",
        "https://h.test/v1?x=1&access_token=abc",
        "https://h.test/v1?Signature=abc",
    ] {
        let error = serde_json::from_value::<ProviderRecord>(stored(json!({"base_url": url})))
            .unwrap_err()
            .to_string();
        assert!(error.contains("query string"), "{url}: {error}");
        assert!(
            !error.contains("sk-not-a-real-key") && !error.contains("AIza"),
            "{error}"
        );
    }
    assert!(
        serde_json::from_value::<ProviderRecord>(stored(
            json!({"base_url": "https://h.test/v1?version=2&limit=10"})
        ))
        .is_ok()
    );
}

#[test]
fn extract_credentials_is_the_migration_path_for_an_old_record() {
    // Regression (review round 3): a stored record from before the rule could
    // neither load nor be migrated.
    let old = stored(json!({
        "api_key": "sk-not-a-real-key", "accessToken": "tok", "tiers": {"chat-v1": "gpt-5"}, "count": 3
    }));
    assert!(serde_json::from_value::<ProviderRecord>(old.clone()).is_err());
    let (clean, extracted) = ProviderRecord::extract_credentials(old);
    let mut fields: Vec<_> = extracted.iter().map(|e| e.field.as_str()).collect();
    fields.sort_unstable();
    assert_eq!(fields, vec!["accessToken", "api_key"]);
    let key = extracted.iter().find(|e| e.field == "api_key").unwrap();
    assert_eq!(key.value.expose(), "sk-not-a-real-key");
    assert!(
        !format!("{extracted:?}").contains("sk-not-a-real-key"),
        "Debug redacts"
    );
    let record: ProviderRecord = serde_json::from_value(clean).unwrap();
    assert_eq!(record.legacy["tiers"], json!({"chat-v1": "gpt-5"}));
    assert!(!record.legacy.contains_key("api_key"));
    // Nested credentials and non-string values are not extracted: the cleaned
    // value still fails loudly rather than persisting them.
    let (still_bad, none) =
        ProviderRecord::extract_credentials(stored(json!({"tiers": {"api_key": "x"}, "token": 5})));
    assert!(none.is_empty());
    assert!(serde_json::from_value::<ProviderRecord>(still_bad).is_err());
    // A non-object value passes through untouched.
    let (same, none) = ProviderRecord::extract_credentials(json!(7));
    assert_eq!((same, none.len()), (json!(7), 0));
}

#[test]
fn a_record_endpoint_must_be_empty_or_an_http_url_with_a_host() {
    // Regression (review round 4): `file:///etc/passwd` loaded.
    for bad in [
        "file:///etc/passwd",
        "ftp://h.test/x",
        "javascript:alert(1)",
        "not a url",
        "http://",
    ] {
        let error = serde_json::from_value::<ProviderRecord>(stored(json!({"base_url": bad})))
            .unwrap_err()
            .to_string();
        assert!(error.contains("http or https"), "{bad}: {error}");
    }
    // A subprocess kind has no endpoint.
    assert!(serde_json::from_value::<ProviderRecord>(stored(json!({"base_url": ""}))).is_ok());
    assert!(
        serde_json::from_value::<ProviderRecord>(stored(
            json!({"base_url": "http://[::1]:11434/v1"})
        ))
        .is_ok()
    );
}

#[test]
fn extract_credentials_drops_null_and_empty_fields_instead_of_extracting_nothing() {
    // Regression (review round 4): a null left the record unloadable and an
    // empty string became a credential that could overwrite a real key.
    let old = stored(
        json!({"api_key": null, "token": "", "access_token": "   ", "auth": "real", "note": 1}),
    );
    let (clean, extracted) = ProviderRecord::extract_credentials(old);
    assert_eq!(extracted.len(), 1);
    assert_eq!(extracted[0].field, "auth");
    let record: ProviderRecord = serde_json::from_value(clean).unwrap();
    assert_eq!(record.legacy.len(), 1);
    assert!(record.legacy.contains_key("note"));
}

#[test]
fn a_legacy_field_named_like_one_of_the_records_own_is_refused_before_it_can_write_a_duplicate_key()
{
    for name in [
        "id",
        "slug",
        "label",
        "kind",
        "base_url",
        "model",
        "enabled",
        "auth_override",
        "synthetic",
        "origin",
    ] {
        let mut r = record();
        r.legacy.insert(name.to_string(), json!("shadow"));
        let error = r.validate().unwrap_err();
        assert!(
            matches!(error, crate::InvalidInput::Reserved { .. }),
            "{name}: {error:?}"
        );
    }
    let mut ok = record();
    ok.legacy.insert("tiers".into(), json!({"chat-v1": "m"}));
    assert!(ok.validate().is_ok());
}

#[test]
fn extract_credentials_leaves_a_non_string_credential_in_place_so_the_load_still_fails() {
    let (cleaned, extracted) = ProviderRecord::extract_credentials(json!({
        "id": "p", "slug": "acme", "label": "A", "kind": "custom", "base_url": "https://a.test/v1",
        "api_key": {"v": "sk-not-a-real-key"}
    }));
    assert!(extracted.is_empty());
    assert!(serde_json::from_value::<ProviderRecord>(cleaned).is_err());
}

#[test]
fn every_serialised_field_of_a_record_is_in_the_reserved_list() {
    // Ties the hand-kept list to the struct: a field added to `ProviderRecord`
    // and forgotten in the list would let a legacy entry write a duplicate key.
    let mut r = record();
    r.model = Some(ModelId::parse("m").unwrap());
    r.auth_override = Some(AuthStyle::Bearer);
    r.legacy.insert("extra".into(), json!(1));
    let value = serde_json::to_value(&r).unwrap();
    for key in value.as_object().unwrap().keys() {
        if key == "extra" {
            continue;
        }
        let mut clash = record();
        clash.legacy.insert(key.clone(), json!("x"));
        assert!(
            clash.validate().is_err(),
            "`{key}` is serialised by the record but not reserved"
        );
    }
}
