//! Wire-shape goldens: the spellings hosts persist and send. A change here is
//! a compatibility change and must be deliberate.
//!
//! `tests/golden/descriptors.json` is the snapshot of every built-in
//! descriptor. To update it after an intentional catalogue change, replace its
//! contents with the JSON printed by the failing assertion (pretty-printed by
//! `serde_json::to_string_pretty`) and review the diff like any other change.

use serde_json::{Value, json};
use tinyinference_hub::{
    AuthStyle, CatalogShape, LocalRuntime, ProviderGroup, ReasonCode, Retry, TestDepth, catalogue,
    classify,
};

const DESCRIPTORS: &str = include_str!("golden/descriptors.json");

#[test]
fn golden_descriptors_snapshot_matches_the_catalogue() {
    let actual = serde_json::to_value(catalogue::descriptors()).unwrap();
    let expected: Value = serde_json::from_str(DESCRIPTORS).unwrap();
    assert_eq!(
        actual,
        expected,
        "descriptor snapshot changed; new snapshot:\n{}",
        serde_json::to_string_pretty(&actual).unwrap()
    );
}

#[test]
fn golden_reason_codes() {
    let all: Vec<Value> = ReasonCode::ALL.iter().map(|c| json!(c)).collect();
    assert_eq!(
        Value::Array(all),
        json!([
            "auth",
            "model",
            "quota",
            "rate_limited",
            "endpoint",
            "timeout",
            "signed_out",
            "unsupported",
            "unknown",
            "policy",
            "invalid",
            "not_found",
            "already_exists",
            "in_use",
            "conflict",
            "store_unreadable",
            "unresolved"
        ])
    );
}

#[test]
fn golden_taxonomy_spellings() {
    assert_eq!(
        json!([
            ProviderGroup::Managed,
            ProviderGroup::Cloud,
            ProviderGroup::Local,
            ProviderGroup::Cli,
            ProviderGroup::Custom,
            ProviderGroup::OAuthBacked
        ]),
        json!(["managed", "cloud", "local", "cli", "custom", "oauth_backed"])
    );
    assert_eq!(
        json!([
            TestDepth::KeyOnly,
            TestDepth::Catalog,
            TestDepth::Completion
        ]),
        json!(["key_only", "catalog", "completion"])
    );
    assert_eq!(
        json!([
            CatalogShape::OpenAi,
            CatalogShape::PagedEnvelope,
            CatalogShape::OllamaTags,
            CatalogShape::LmStudioV0,
            CatalogShape::None
        ]),
        json!([
            "open_ai",
            "paged_envelope",
            "ollama_tags",
            "lm_studio_v0",
            "none"
        ])
    );
    assert_eq!(
        json!(LocalRuntime::ALL),
        json!([
            "ollama",
            "lm_studio",
            "llama_cpp",
            "vllm",
            "mlx",
            "omlx",
            "openai_compatible"
        ])
    );
}

#[test]
fn golden_auth_styles_and_legacy_spellings() {
    assert_eq!(
        json!([
            AuthStyle::Bearer,
            AuthStyle::XApiKey,
            AuthStyle::Anthropic,
            AuthStyle::SessionJwt,
            AuthStyle::None
        ]),
        json!(["bearer", "x_api_key", "anthropic", "session_jwt", "none"])
    );
    for legacy in ["openhuman_jwt", "openhumanjwt", "session_jwt"] {
        let parsed: AuthStyle = serde_json::from_value(json!(legacy)).unwrap();
        assert_eq!(parsed, AuthStyle::SessionJwt, "{legacy}");
    }
}

#[test]
fn golden_classification_of_real_vendor_bodies() {
    // (status, body, reason, retry-is-never)
    let cases: &[(u16, &str, ReasonCode, bool)] = &[
        (
            401,
            r#"{"error":{"message":"Incorrect API key provided: sk-not-a-real-key","type":"invalid_request_error","code":"invalid_api_key"}}"#,
            ReasonCode::Auth,
            true,
        ),
        (
            429,
            r#"{"error":{"message":"You exceeded your current quota","type":"insufficient_quota","code":"insufficient_quota"}}"#,
            ReasonCode::Quota,
            true,
        ),
        (
            429,
            r#"{"error":{"message":"Rate limit reached for requests","type":"requests","code":"rate_limit_exceeded"}}"#,
            ReasonCode::RateLimited,
            false,
        ),
        (
            404,
            r#"{"error":{"message":"The model `gpt-9` does not exist","type":"invalid_request_error","code":"model_not_found"}}"#,
            ReasonCode::Model,
            true,
        ),
        (
            400,
            r#"{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API."}}"#,
            ReasonCode::Quota,
            true,
        ),
        (
            503,
            "upstream connect error or disconnect/reset before headers",
            ReasonCode::Unknown,
            false,
        ),
        (
            407,
            "Proxy Authentication Required",
            ReasonCode::Unknown,
            true,
        ),
    ];
    for (status, body, reason, never) in cases {
        let failure = classify(*status, &[], body);
        assert_eq!(failure.reason, *reason, "{status} {body}");
        assert_eq!(failure.retry == Retry::Never, *never, "{status} {body}");
    }
}
