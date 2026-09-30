use std::marker::PhantomData;

use proptest::prelude::*;
use serde::Serialize;

use super::*;

/// Auto-ref probe: resolves to `true` only when `T: Serialize`.
struct Probe<T>(PhantomData<T>);
trait Fallback {
    fn implements_serialize(&self) -> bool {
        false
    }
}
impl<T> Fallback for Probe<T> {}
impl<T: Serialize> Probe<T> {
    fn implements_serialize(&self) -> bool {
        true
    }
}

#[test]
fn secret_debug_and_display_hide_the_value() {
    let secret = Secret::new("sk-not-a-real-key");
    assert_eq!(format!("{secret:?}"), "Secret(<redacted>)");
    assert_eq!(format!("{secret}"), "<redacted>");
    assert_eq!(format!("{secret:#?}"), "Secret(<redacted>)");
}

#[test]
fn secret_exposes_only_through_expose() {
    let secret = Secret::from("test-token");
    assert_eq!(secret.expose(), "test-token");
    assert_eq!(secret.len(), 10);
    assert!(!secret.is_empty());
    assert_eq!(Secret::from(String::from("test-token")), secret);
}

#[test]
fn secret_blank_is_empty() {
    assert!(Secret::new("").is_empty());
    assert!(Secret::new("   \n").is_empty());
}

#[test]
fn secret_inside_a_derived_debug_struct_stays_redacted() {
    #[derive(Debug)]
    struct Holder {
        key: Secret,
    }
    let holder = Holder {
        key: Secret::new("sk-not-a-real-key"),
    };
    assert!(!format!("{holder:?}").contains("sk-not"));
    assert_eq!(holder.key.len(), 17);
}

#[test]
fn secret_does_not_implement_serialize() {
    assert!(!Probe::<Secret>(PhantomData).implements_serialize());
    // The probe itself works: a type that is Serialize reports true.
    assert!(Probe::<String>(PhantomData).implements_serialize());
}

#[test]
fn log_only_redacts_in_debug_and_display() {
    let raw = LogOnly::new(String::from("Authorization: Bearer sk-not-a-real-key"));
    assert_eq!(format!("{raw}"), "<redacted>");
    assert_eq!(format!("{raw:?}"), "LogOnly(<redacted>)");
    assert!(raw.expose().contains("Bearer"));
    assert!(raw.into_inner().contains("sk-not"));
}

#[test]
fn log_only_default_is_empty() {
    let raw: LogOnly<String> = LogOnly::default();
    assert!(raw.expose().is_empty());
}

proptest! {
    #[test]
    fn secret_debug_never_contains_the_value(value in "[A-Za-z0-9_-]{8,40}") {
        let secret = Secret::new(value.clone());
        let secret_text = format!("{secret:?}{secret}");
        prop_assert!(!secret_text.contains(&value));
        let raw = LogOnly::new(value.clone());
        let raw_text = format!("{raw:?}{raw}");
        prop_assert!(!raw_text.contains(&value));
    }
}

#[test]
fn credential_names_are_recognised_across_spellings_and_ordinary_names_are_not() {
    for name in [
        "api_key",
        "apiKey",
        "API-KEY",
        "x-api-key",
        "openai_api_key",
        "secret",
        "clientSecret",
        "password",
        "Authorization",
        "bearer_token",
        "key",
        "token",
        "access_token",
        "accessToken",
        "refreshToken",
        "auth",
        "credentials",
        "sig",
        "Signature",
        "private_key",
        "passphrase",
        "db_passwd",
        "APIKEY",
    ] {
        assert!(is_credential_name(name), "{name}");
    }
    for name in [
        "max_tokens",
        "maxTokens",
        "tokenizer",
        "keywords",
        "monkey",
        "tiers",
        "models",
        "display_name",
        "authors",
        "author",
        "signal",
        "design",
        "",
    ] {
        assert!(!is_credential_name(name), "{name}");
    }
}

#[test]
fn credential_names_are_judged_by_their_last_word() {
    // Regression (review round 4): the substring rule flagged `secretary` and
    // missed Azure-style names.
    for name in [
        "subscription-key",
        "Ocp-Apim-Subscription-Key",
        "x-functions-key",
        "app_key",
        "cookie",
        "Set-Cookie",
        "X-Amz-Signature",
        "pwd",
        "client_secret",
        "clientSecret",
        "db_passwd",
    ] {
        assert!(is_credential_name(name), "{name}");
    }
    for name in [
        "secretary",
        "keyword",
        "token_limit",
        "signature_algorithm",
        "cookies_enabled",
        "pwdx",
    ] {
        assert!(!is_credential_name(name), "{name}");
    }
}

#[test]
fn run_together_names_are_credentials_and_benign_key_and_token_names_are_not() {
    // Regression (review round 5): the last-word rule dropped the substring
    // cases and made every `*_key` / `*_token` a credential.
    for name in [
        "openaiapikey",
        "OPENAIAPIKEY",
        "dbpassword",
        "clientsecret",
        "APITOKEN",
        "myapikey",
        "key",
        "token",
    ] {
        assert!(is_credential_name(name), "{name}");
    }
    for name in [
        "public_key",
        "ssh_public_key",
        "cache_key",
        "sort_key",
        "partition_key",
        "idempotency_key",
        "page_token",
        "next_page_token",
        "continuation_token",
        "publicKey",
        "pageToken",
    ] {
        assert!(!is_credential_name(name), "{name}");
    }
    // A benign word does not excuse a name that is a credential on other grounds.
    assert!(is_credential_name("public_api_key"));
    assert!(is_credential_name("cache_secret"));
    assert!(is_credential_name("page_password"));
}

#[test]
fn concatenated_and_prefixed_credential_names_are_caught_and_benign_words_do_not_veto_them() {
    // Regression (review round 6): `secretkey`, `accesstoken`, `secret_value`
    // leaked, and a benign word anywhere vetoed a real credential
    // (`primaryKey`, `page_access_token`).
    for name in [
        "secretkey",
        "accesstoken",
        "authtoken",
        "refreshtoken",
        "sessiontoken",
        "privatekey",
        "clientsecretkey",
        "secret_value",
        "password_value",
        "primaryKey",
        "primary_master_key",
        "page_access_token",
        "group_access_token",
        "next_secret_key",
        "public_access_token",
        "ordertoken_secret",
        "idToken",
        "subscriptionKey",
        "masterkey",
    ] {
        assert!(is_credential_name(name), "{name}");
    }
    for name in [
        "secretary",
        "secretariat_id",
        "signature_algorithm",
        "cookies_enabled",
        "keywords",
        "monkey",
    ] {
        assert!(!is_credential_name(name), "{name}");
    }
}

#[test]
fn a_compound_spanning_a_word_boundary_is_not_a_credential() {
    // Regression (review round 7): `valid_tokens` collapsed to contain `idtoken`.
    for name in ["valid_tokens", "paid_tokens", "no_key_words"] {
        assert!(!is_credential_name(name), "{name}");
    }
    // Adjacent whole words that join into a compound still count.
    for name in [
        "access_token",
        "api_key",
        "refresh_token",
        "id_token",
        "master_key",
        "subscription_key",
        "api-key",
        "accessToken",
    ] {
        assert!(is_credential_name(name), "{name}");
    }
}

#[test]
fn a_plural_or_numbered_compound_is_still_a_credential() {
    // Regression (review round 8): `api_keys` and `api_key2` were flagged before
    // the word-boundary change and not after it.
    for name in [
        "api_keys",
        "apiKeys",
        "API_KEYS",
        "api_key2",
        "access_tokens",
        "session_tokens",
        "auth_tokens",
        "id_tokens",
        "api_token2",
        "master_key2",
        "accessKey2",
        "apikeys",
        "password2",
    ] {
        assert!(is_credential_name(name), "{name}");
    }
    for name in ["valid_tokens", "paid_tokens", "keys_total", "tokens_used"] {
        assert!(!is_credential_name(name), "{name}");
    }
}
