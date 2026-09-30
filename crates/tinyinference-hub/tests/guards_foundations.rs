//! Operation-matrix guards (plan 04 section 2) that the foundations slice can
//! already prove through the public API. Later slices add the guards that need
//! operations (G2..G10, G12, G13, G20..G25); each is named `guard_gNN_*`.

use std::time::Duration;

use serde_json::json;
use tinyinference_hub::endpoint::{endpoint_has_credentials, redact_endpoint};
use tinyinference_hub::ids::{check_model_id, check_provider_name, check_slug, slugify};
use tinyinference_hub::policy::{check_address, check_endpoint};
use tinyinference_hub::{
    EndpointPolicy, EndpointRefusal, HeaderPolicy, HubError, KindId, ProviderGroup, ProviderRecord,
    ReasonCode, Retry, Secret, Slug, catalogue, classify,
};

#[test]
fn guard_g01_a_credential_is_never_on_the_record_and_never_printed() {
    let record = ProviderRecord::new(
        "prv_1",
        Slug::parse("acme").unwrap(),
        "Acme",
        KindId::new("custom"),
        "https://api.acme.test/v1",
    );
    let value = serde_json::to_value(&record).unwrap();
    let keys: Vec<&String> = value.as_object().unwrap().keys().collect();
    assert!(
        keys.iter()
            .all(|k| !k.contains("key") && !k.contains("secret")),
        "{keys:?}"
    );
    assert_eq!(Slug::parse("acme").unwrap().key_slot(), "provider/acme/key");
    let secret = Secret::new("sk-not-a-real-key");
    assert!(!format!("{secret:?} {secret}").contains("sk-not"));
}

#[test]
fn guard_g11_only_a_rejected_credential_rolls_back_and_a_local_runtime_also_when_unreachable() {
    let auth = classify(401, &[], "");
    let unreachable = classify_text_free("connection refused");
    for group in [
        ProviderGroup::Cloud,
        ProviderGroup::Custom,
        ProviderGroup::Local,
    ] {
        assert!(auth.rolls_back(group));
    }
    assert!(unreachable.rolls_back(ProviderGroup::Local));
    assert!(!unreachable.rolls_back(ProviderGroup::Cloud));
    assert!(!classify(429, &[], "").rolls_back(ProviderGroup::Local));
}

fn classify_text_free(text: &str) -> tinyinference_hub::ProviderFailure {
    // A transport-level failure has no status; the hub maps llm's text errors.
    match HubError::from(tinyinference_llm::Error::Model(text.into())) {
        HubError::Provider(failure) => failure,
        other => panic!("{other:?}"),
    }
}

#[test]
fn guard_g14_model_ids_are_bounded_and_free_of_whitespace_and_reserved_words() {
    assert!(check_model_id("gpt-5", &[]).is_ok());
    assert!(check_model_id("has space", &[]).is_err());
    assert!(check_model_id(&"m".repeat(257), &[]).is_err());
    assert!(check_model_id("chat-v1", &["chat-v1"]).is_err());
}

#[test]
fn guard_g15_slugs_and_names_are_bounded_unique_and_unreserved() {
    assert!(check_provider_name(&"n".repeat(81)).is_err());
    assert!(check_slug(["taken"], "taken", catalogue::is_reserved_slug).is_err());
    assert!(check_slug(Vec::<&str>::new(), "groq", catalogue::is_reserved_slug).is_err());
    assert!(
        check_slug(
            Vec::<&str>::new(),
            &slugify("Acme Gateway"),
            catalogue::is_reserved_slug
        )
        .is_ok()
    );
}

#[test]
fn guard_g16_userinfo_is_refused_and_redacted_wherever_an_endpoint_is_handled() {
    let endpoint = "https://alice:hunter2@api.acme.test/v1";
    assert!(endpoint_has_credentials(endpoint));
    assert!(!redact_endpoint(endpoint).contains("hunter2"));
    assert_eq!(
        check_endpoint(endpoint, &EndpointPolicy::hosted()),
        Err(EndpointRefusal::CredentialInUrl)
    );
}

#[test]
fn guard_g17_link_local_private_and_cleartext_credentialed_endpoints_are_refused() {
    let hosted = EndpointPolicy::hosted();
    assert_eq!(
        check_endpoint("http://169.254.169.254/", &hosted),
        Err(EndpointRefusal::LinkLocal)
    );
    assert_eq!(
        check_endpoint("http://10.0.0.5/", &hosted),
        Err(EndpointRefusal::PrivateNetwork)
    );
    assert_eq!(
        check_endpoint("http://127.0.0.1/", &hosted),
        Err(EndpointRefusal::Loopback)
    );
    assert_eq!(
        tinyinference_hub::policy::check_endpoint_with_credential(
            "http://gw.acme.test/v1",
            &hosted,
            true
        ),
        Err(EndpointRefusal::Cleartext)
    );
    assert_eq!(hosted.max_redirects, 3);
    assert!(check_address("::ffff:169.254.169.254".parse().unwrap(), &hosted).is_err());
}

#[test]
fn guard_g18_the_documented_caps_are_the_defaults() {
    let p = EndpointPolicy::hosted();
    assert_eq!(p.timeout, Duration::from_secs(10));
    assert_eq!(p.fail_body_cap, 64 * 1024);
    assert_eq!(p.catalog_cap, 16 * 1024 * 1024);
    assert_eq!(p.page_cap, 4 * 1024 * 1024);
}

#[test]
fn guard_g19_the_classifier_never_reads_our_own_url() {
    let f = classify(
        200,
        &[],
        "error sending request for url (https://x.test/v1/models)",
    );
    assert_eq!(f.reason, ReasonCode::Unknown);
    assert_eq!(f.retry, Retry::Never);
}

#[test]
fn guard_g26_the_product_header_goes_only_to_our_own_hosts() {
    let policy = HeaderPolicy::builtin();
    assert!(policy.allows_product_header_to("https://api.tinyhumans.ai/v1"));
    assert!(!policy.allows_product_header_to("https://openrouter.ai/api/v1"));
    assert!(!policy.allows_product_header_to("http://localhost:11434/v1"));
}

#[test]
fn guard_g27_the_legacy_managed_spellings_alias_to_one_kind_and_an_unknown_kind_is_not_a_row() {
    for legacy in ["managed", "openhuman", "cloud", "tinyhumans"] {
        assert_eq!(
            catalogue::resolve_kind(legacy),
            Some(KindId::new("tinyhumans")),
            "{legacy}"
        );
    }
    assert_eq!(catalogue::resolve_kind("no-such-kind"), None);
    // A stored record with an unknown kind still loads as data (the config
    // layer fails it loudly in a later slice), and never gains a preset.
    let record: ProviderRecord = serde_json::from_value(json!({
        "id": "1", "slug": "x", "label": "X", "kind": "no-such-kind", "base_url": "https://x.test"
    }))
    .unwrap();
    assert!(catalogue::descriptor(record.kind.as_str()).is_none());
}
