//! Tests for the driver context, auth application and the OpenAI-compatible
//! driver.

use std::time::Duration;

use serde_json::json;

use super::context::apply_auth;
use super::*;
use crate::catalogue::{custom_descriptor, descriptor};
use crate::error::{HubError, Operation, ReasonCode, Retry};
use crate::ids::{KindId, ModelId, Slug};
use crate::policy::{EndpointPolicy, HeaderPolicy};
use crate::ports::HubRequest;
use crate::secret::Secret;
use crate::taxonomy::{AuthStyle, ProviderGroup, TestDepth};
use crate::testkit::{FakeClock, Match, Script, Scripted, ScriptedHttp};

struct Bed {
    http: ScriptedHttp,
    clock: FakeClock,
    policy: EndpointPolicy,
    headers: HeaderPolicy,
}

impl Bed {
    fn new() -> Self {
        let clock = FakeClock::new();
        Self {
            http: ScriptedHttp::new(clock.clone()),
            clock,
            policy: EndpointPolicy::hosted(),
            headers: HeaderPolicy::builtin(),
        }
    }
    fn cx(&self) -> DriverContext<'_> {
        DriverContext::new(&self.http, &self.policy, &self.clock, &self.headers)
    }
}

fn target<'a>(
    slug: &'a Slug,
    kind: &'a KindId,
    base: &'a str,
    auth: &'a AuthStyle,
    key: Option<&'a Secret>,
) -> Target<'a> {
    Target {
        slug,
        kind,
        group: ProviderGroup::Cloud,
        base_url: base,
        auth,
        credential: key,
        model: None,
    }
}

fn driver(kind: &str) -> OpenAiCompatDriver {
    OpenAiCompatDriver::for_descriptor(descriptor(kind).unwrap().clone())
}

// ---- apply_auth ------------------------------------------------------------

fn auth_headers(auth: &AuthStyle, key: Option<&str>) -> (bool, Vec<(String, String)>) {
    let mut headers = Vec::new();
    let credentialed = apply_auth(&mut headers, auth, key);
    (credentialed, headers)
}

#[test]
fn kinds_each_auth_style_presents_the_key_the_way_its_provider_expects() {
    let (c, h) = auth_headers(&AuthStyle::Bearer, Some("sk-x"));
    assert!(c);
    assert_eq!(
        h,
        [("authorization".to_string(), "Bearer sk-x".to_string())]
    );
    let (_, h) = auth_headers(&AuthStyle::SessionJwt, Some("jwt"));
    assert_eq!(h[0].1, "Bearer jwt");
    let (_, h) = auth_headers(&AuthStyle::XApiKey, Some("k"));
    assert_eq!(h, [("x-api-key".to_string(), "k".to_string())]);
    let (_, h) = auth_headers(&AuthStyle::Anthropic, Some("k"));
    assert_eq!(
        h,
        [
            ("x-api-key".to_string(), "k".to_string()),
            ("anthropic-version".to_string(), "2023-06-01".to_string())
        ]
    );
    let (_, h) = auth_headers(&AuthStyle::Custom("API-Key".into()), Some("k"));
    assert_eq!(h, [("api-key".to_string(), "k".to_string())]);
}

#[test]
fn kinds_no_key_none_style_and_blank_values_attach_nothing() {
    for (auth, key) in [
        (AuthStyle::Bearer, None),
        (AuthStyle::Bearer, Some("")),
        (AuthStyle::Bearer, Some("   ")),
        (AuthStyle::None, Some("sk-x")),
        (AuthStyle::Custom("  ".into()), Some("sk-x")),
    ] {
        let (credentialed, headers) = auth_headers(&auth, key);
        assert!(!credentialed && headers.is_empty(), "{auth:?} {key:?}");
    }
}

#[test]
fn kinds_the_key_is_trimmed_before_it_goes_on_the_wire() {
    let (_, h) = auth_headers(&AuthStyle::Bearer, Some("  sk-x \n"));
    assert_eq!(h[0].1, "Bearer sk-x");
}

// ---- request building ------------------------------------------------------

#[test]
fn kinds_a_request_carries_the_credential_the_extra_headers_and_the_credentialed_flag() {
    let bed = Bed::new();
    let d = driver("openrouter");
    let (slug, kind, key) = (
        Slug::parse("openrouter").unwrap(),
        KindId::new("openrouter"),
        Secret::new("sk-or-fake"),
    );
    let t = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    let request = bed.cx().request(
        d.descriptor(),
        &t,
        HubRequest::get("https://openrouter.ai/api/v1/models"),
    );
    assert!(request.credentialed);
    assert_eq!(request.header("authorization"), Some("Bearer sk-or-fake"));
    assert!(
        request.header("http-referer").is_some(),
        "OpenRouter attribution headers"
    );
    let keyless = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        None,
    );
    let request = bed.cx().request(
        d.descriptor(),
        &keyless,
        HubRequest::get("https://openrouter.ai/api/v1/models"),
    );
    assert!(!request.credentialed);
    assert!(request.header("authorization").is_none());
}

#[test]
fn kinds_the_product_header_goes_only_to_first_party_hosts() {
    let bed = Bed::new();
    let product = ("x-sdk-name".to_string(), "hub-test".to_string());
    let cx = bed.cx().with_product(&product);
    let d = driver("openai");
    let (slug, kind) = (Slug::parse("x").unwrap(), KindId::new("openai"));
    let t = target(
        &slug,
        &kind,
        "https://api.openai.com/v1",
        &AuthStyle::Bearer,
        None,
    );
    let third = cx.request(
        d.descriptor(),
        &t,
        HubRequest::get("https://api.openai.com/v1/models"),
    );
    assert!(
        third.header("x-sdk-name").is_none(),
        "never to a third party"
    );
    let first = cx.request(
        d.descriptor(),
        &t,
        HubRequest::get("https://api.tinyhumans.ai/x/models"),
    );
    assert_eq!(first.header("x-sdk-name"), Some("hub-test"));
    let lookalike = cx.request(
        d.descriptor(),
        &t,
        HubRequest::get("https://tinyhumans.ai.evil.test/models"),
    );
    assert!(lookalike.header("x-sdk-name").is_none());
    let debug = format!("{cx:?}");
    assert!(debug.contains("x-sdk-name") && !debug.contains("hub-test"));
}

#[test]
fn kinds_target_debug_never_prints_the_key_or_a_url_credential() {
    let (slug, kind, key) = (
        Slug::parse("x").unwrap(),
        KindId::new("custom"),
        Secret::new("sk-not-a-real-key"),
    );
    let t = target(
        &slug,
        &kind,
        "https://u:hunter2@x.test/v1?key=abc",
        &AuthStyle::Bearer,
        Some(&key),
    );
    let debug = format!("{t:?}");
    for leak in ["sk-not-a-real-key", "hunter2", "abc"] {
        assert!(!debug.contains(leak), "{debug}");
    }
    assert_eq!(t.key(), Some("sk-not-a-real-key"));
    assert_eq!(
        target(&slug, &kind, "https://x.test/v1/", &AuthStyle::None, None).base(),
        "https://x.test/v1"
    );
}

// ---- the OpenAI-compatible driver ------------------------------------------

#[tokio::test]
async fn kinds_it_lists_models_with_the_bearer_key_and_the_hosts_query() {
    let bed = Bed::new();
    bed.http.route(
        Match::get("https://api.groq.com/openai/v1/models")
            .with_header("authorization", "Bearer gsk-fake"),
        Scripted::json(
            200,
            &json!({"data": [{"id": "llama-3.3-70b"}, {"id": "gemma"}]}),
        ),
    );
    let d = driver("groq");
    let (slug, kind, key) = (
        Slug::parse("groq").unwrap(),
        KindId::new("groq"),
        Secret::new("gsk-fake"),
    );
    let t = target(
        &slug,
        &kind,
        "https://api.groq.com/openai/v1/",
        &AuthStyle::Bearer,
        Some(&key),
    );
    let fetched = d.list_models(&bed.cx(), &t).await.unwrap();
    let ids: Vec<_> = fetched.models.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(ids, ["llama-3.3-70b", "gemma"]);
    assert!(!fetched.truncated);
}

#[tokio::test]
async fn kinds_a_rejected_key_is_an_auth_failure_with_the_provider_code() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(
            401,
            &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
        ),
    );
    let d = driver("openai");
    let (slug, kind, key) = (
        Slug::parse("openai").unwrap(),
        KindId::new("openai"),
        Secret::new("sk-bad"),
    );
    let t = target(
        &slug,
        &kind,
        "https://api.openai.com/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    match d.list_models(&bed.cx(), &t).await.unwrap_err() {
        HubError::Provider(f) => {
            assert_eq!((f.reason, f.status), (ReasonCode::Auth, Some(401)));
            assert!(f.rolls_back(ProviderGroup::Cloud));
            assert!(
                !format!("{f} {f:?}").contains("Incorrect API key"),
                "raw text is log-only"
            );
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn kinds_a_body_past_the_cap_is_refused_as_unknown_and_truncated() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.acme.test/v1/models"),
        Scripted::Oversize { bytes: 100_000_000 },
    );
    let d = OpenAiCompatDriver::custom();
    let (slug, kind) = (Slug::parse("acme").unwrap(), KindId::new("custom"));
    let t = target(
        &slug,
        &kind,
        "https://api.acme.test/v1",
        &AuthStyle::Bearer,
        None,
    );
    match d.list_models(&bed.cx(), &t).await.unwrap_err() {
        HubError::Provider(f) => {
            assert_eq!(f.reason, ReasonCode::Unknown);
            assert!(f.truncated && !f.reason.destroys_credential());
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn kinds_a_listing_that_is_not_json_is_unknown_never_auth() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.acme.test/v1/models"),
        Scripted::text(200, "<html>welcome</html>"),
    );
    let d = OpenAiCompatDriver::custom();
    let (slug, kind) = (Slug::parse("acme").unwrap(), KindId::new("custom"));
    let t = target(
        &slug,
        &kind,
        "https://api.acme.test/v1",
        &AuthStyle::Bearer,
        None,
    );
    match d.list_models(&bed.cx(), &t).await.unwrap_err() {
        HubError::Provider(f) => assert_eq!(f.reason, ReasonCode::Unknown),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn kinds_openrouter_uses_the_account_listing_with_a_key_and_the_public_one_without() {
    let bed = Bed::new();
    let scoped = bed.http.route(
        Match::get("https://openrouter.ai/api/v1/models/user?limit=1000&output_modalities=all"),
        Scripted::json(200, &json!({"data": [{"id": "mine/1"}]})),
    );
    let public = bed.http.route(
        Match::get("https://openrouter.ai/api/v1/models?limit=1000&output_modalities=all"),
        Scripted::json(200, &json!({"data": [{"id": "pub/1"}, {"id": "pub/2"}]})),
    );
    let d = driver("openrouter");
    let (slug, kind, key) = (
        Slug::parse("openrouter").unwrap(),
        KindId::new("openrouter"),
        Secret::new("sk-or-fake"),
    );
    let authed = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    assert_eq!(
        d.list_models(&bed.cx(), &authed)
            .await
            .unwrap()
            .models
            .len(),
        1
    );
    let keyless = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        None,
    );
    assert_eq!(
        d.list_models(&bed.cx(), &keyless)
            .await
            .unwrap()
            .models
            .len(),
        2
    );
    assert_eq!((bed.http.hits(scoped), bed.http.hits(public)), (1, 1));
}

#[tokio::test]
async fn kinds_an_account_listing_that_404s_falls_back_to_the_public_one() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://openrouter.ai/api/v1/models/user"),
        Scripted::text(404, "not found"),
    );
    bed.http.route(
        Match::prefix("https://openrouter.ai/api/v1/models?"),
        Scripted::json(200, &json!({"data": [{"id": "pub/1"}]})),
    );
    let d = driver("openrouter");
    let (slug, kind, key) = (
        Slug::parse("openrouter").unwrap(),
        KindId::new("openrouter"),
        Secret::new("sk-or-fake"),
    );
    let t = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    assert_eq!(
        d.list_models(&bed.cx(), &t).await.unwrap().models[0]
            .id
            .as_str(),
        "pub/1"
    );
}

#[tokio::test]
async fn kinds_an_account_listing_that_fails_otherwise_does_not_fall_back() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://openrouter.ai/api/v1/models/user"),
        Scripted::text(401, "no"),
    );
    let d = driver("openrouter");
    let (slug, kind, key) = (
        Slug::parse("openrouter").unwrap(),
        KindId::new("openrouter"),
        Secret::new("sk-or-bad"),
    );
    let t = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    assert!(
        matches!(d.list_models(&bed.cx(), &t).await, Err(HubError::Provider(f)) if f.status == Some(401))
    );
    assert_eq!(
        bed.http.request_count(),
        1,
        "a 401 is the key's answer, not a reason to ask the public list"
    );
}

#[tokio::test]
async fn kinds_only_openrouter_has_a_key_only_check_and_it_is_get_key() {
    let bed = Bed::new();
    bed.http.route(
        Match::get("https://openrouter.ai/api/v1/key")
            .with_header("authorization", "Bearer sk-or-fake"),
        Scripted::json(200, &json!({"data": {"label": "x"}})),
    );
    let (slug, kind, key) = (
        Slug::parse("openrouter").unwrap(),
        KindId::new("openrouter"),
        Secret::new("sk-or-fake"),
    );
    let t = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    driver("openrouter").key_check(&bed.cx(), &t).await.unwrap();
    let (slug2, kind2) = (Slug::parse("groq").unwrap(), KindId::new("groq"));
    let t2 = target(
        &slug2,
        &kind2,
        "https://api.groq.com/openai/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    match driver("groq").key_check(&bed.cx(), &t2).await.unwrap_err() {
        HubError::Unsupported { op, kind } => {
            assert_eq!(op, Operation::Test(TestDepth::KeyOnly));
            assert_eq!(kind.as_str(), "groq");
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn kinds_a_completion_ping_posts_one_small_chat_completion() {
    let bed = Bed::new();
    bed.http.route(
        Match::post("https://api.groq.com/openai/v1/chat/completions"),
        Scripted::json(200, &json!({"choices": [{"message": {"content": "pong"}}]})),
    );
    let d = driver("groq");
    let (slug, kind, key) = (
        Slug::parse("groq").unwrap(),
        KindId::new("groq"),
        Secret::new("gsk-fake"),
    );
    let t = target(
        &slug,
        &kind,
        "https://api.groq.com/openai/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    d.completion_ping(&bed.cx(), &t, &ModelId::parse("llama-3.3-70b").unwrap())
        .await
        .unwrap();
    let sent = &bed.http.requests()[0];
    let body: serde_json::Value = serde_json::from_str(sent.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["model"], "llama-3.3-70b");
    assert_eq!(body["max_tokens"], 16);
    assert!(body.get("max_completion_tokens").is_none());
    assert_eq!(sent.header("content-type"), Some("application/json"));
}

#[tokio::test]
async fn kinds_a_429_on_a_ping_is_rate_limited_with_the_providers_delay() {
    let bed = Bed::new();
    bed.http.route(
        Match::post("https://api.groq.com/openai/v1/chat/completions"),
        Scripted::with_headers(
            429,
            &[("retry-after", "7")],
            r#"{"error":{"message":"slow down"}}"#,
        ),
    );
    let d = driver("groq");
    let (slug, kind) = (Slug::parse("groq").unwrap(), KindId::new("groq"));
    let t = target(
        &slug,
        &kind,
        "https://api.groq.com/openai/v1",
        &AuthStyle::Bearer,
        None,
    );
    match d
        .completion_ping(&bed.cx(), &t, &ModelId::parse("m").unwrap())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => {
            assert_eq!(f.reason, ReasonCode::RateLimited);
            assert_eq!(f.retry, Retry::Later(Some(Duration::from_secs(7))));
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn kinds_a_transport_failure_is_classified_by_condition() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://a.test/v1/models"),
        Script::Sequence(vec![Scripted::Timeout, Scripted::ConnectRefused]),
    );
    let d = OpenAiCompatDriver::custom();
    let (slug, kind) = (Slug::parse("a").unwrap(), KindId::new("custom"));
    let t = target(&slug, &kind, "https://a.test/v1", &AuthStyle::Bearer, None);
    assert_eq!(
        d.list_models(&bed.cx(), &t).await.unwrap_err().reason(),
        ReasonCode::Timeout
    );
    assert_eq!(
        d.list_models(&bed.cx(), &t).await.unwrap_err().reason(),
        ReasonCode::Endpoint
    );
}

#[tokio::test]
async fn kinds_a_redirect_the_transport_did_not_follow_is_an_endpoint_failure() {
    let bed = Bed::new();
    // No `Location`: the transport hands the 302 back.
    bed.http.route(
        Match::prefix("https://a.test/v1/models"),
        Scripted::text(302, ""),
    );
    let d = OpenAiCompatDriver::custom();
    let (slug, kind) = (Slug::parse("a").unwrap(), KindId::new("custom"));
    let t = target(&slug, &kind, "https://a.test/v1", &AuthStyle::Bearer, None);
    match d.list_models(&bed.cx(), &t).await.unwrap_err() {
        HubError::Provider(f) => {
            assert_eq!((f.reason, f.status), (ReasonCode::Endpoint, Some(302)))
        }
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn kinds_a_failure_body_is_cut_to_the_failure_cap_before_it_is_classified() {
    let bed = Bed::new();
    let huge = format!("{}invalid api key", "x ".repeat(200_000));
    bed.http.route(
        Match::prefix("https://a.test/v1/models"),
        Scripted::text(500, huge),
    );
    let d = OpenAiCompatDriver::custom();
    let (slug, kind) = (Slug::parse("a").unwrap(), KindId::new("custom"));
    let t = target(&slug, &kind, "https://a.test/v1", &AuthStyle::Bearer, None);
    match d.list_models(&bed.cx(), &t).await.unwrap_err() {
        HubError::Provider(f) => {
            assert!(f.truncated);
            assert_ne!(
                f.reason,
                ReasonCode::Auth,
                "the phrase past the cap was never read"
            );
            assert!(f.raw.expose().len() <= 70_000);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn kinds_the_custom_descriptor_is_an_editable_bearer_endpoint_outside_the_catalogue() {
    let d = custom_descriptor();
    assert_eq!(d.kind.as_str(), "custom");
    assert!(d.endpoint_editable && !d.needs_key);
    assert_eq!(d.group, ProviderGroup::Custom);
    assert!(
        descriptor("custom").is_none(),
        "custom is a group, not a catalogue row"
    );
    assert!(d.supports_depth(TestDepth::Catalog) && !d.supports_depth(TestDepth::KeyOnly));
}

#[tokio::test]
async fn kinds_openai_pings_with_max_completion_tokens_because_its_reasoning_models_reject_max_tokens()
 {
    let bed = Bed::new();
    bed.http.route(
        Match::post("https://api.openai.com/v1/chat/completions"),
        Scripted::json(200, &json!({"choices": []})),
    );
    let (slug, kind, key) = (
        Slug::parse("openai").unwrap(),
        KindId::new("openai"),
        Secret::new("sk-fake"),
    );
    let t = target(
        &slug,
        &kind,
        "https://api.openai.com/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    driver("openai")
        .completion_ping(&bed.cx(), &t, &ModelId::parse("gpt-5").unwrap())
        .await
        .unwrap();
    let sent = &bed.http.requests()[0];
    let body: serde_json::Value = serde_json::from_str(sent.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["max_completion_tokens"], 16);
    assert!(body.get("max_tokens").is_none());
}

#[tokio::test]
async fn kinds_the_ping_asks_for_max_completion_tokens_wherever_the_endpoint_is_openai_or_azure() {
    for (base, expect) in [
        ("https://api.openai.com/v1", "max_completion_tokens"),
        (
            "https://my-resource.openai.azure.com/openai/v1",
            "max_completion_tokens",
        ),
        ("https://gateway.acme.test/v1", "max_tokens"),
    ] {
        let bed = Bed::new();
        bed.http
            .route(Match::prefix(base), Scripted::json(200, &json!({})));
        let (slug, kind) = (Slug::parse("acme").unwrap(), KindId::new("custom"));
        let t = target(&slug, &kind, base, &AuthStyle::Bearer, None);
        OpenAiCompatDriver::custom()
            .completion_ping(&bed.cx(), &t, &ModelId::parse("m").unwrap())
            .await
            .unwrap();
        let sent = &bed.http.requests()[0];
        let body: serde_json::Value = serde_json::from_str(sent.body.as_deref().unwrap()).unwrap();
        assert_eq!(body[expect], 16, "{base}");
    }
}

#[tokio::test]
async fn kinds_anthropic_protocol_pings_always_use_max_tokens_even_on_an_azure_host() {
    let bed = Bed::new();
    let base = "https://my-resource.openai.azure.com/anthropic/v1";
    bed.http.route(
        Match::post(format!("{base}/messages")),
        Scripted::json(200, &json!({})),
    );
    let (slug, kind, key) = (
        Slug::parse("claude-on-azure").unwrap(),
        KindId::new("anthropic"),
        Secret::new("k"),
    );
    let t = target(&slug, &kind, base, &AuthStyle::Anthropic, Some(&key));
    driver("anthropic")
        .completion_ping(&bed.cx(), &t, &ModelId::parse("claude-x").unwrap())
        .await
        .unwrap();
    let body: serde_json::Value =
        serde_json::from_str(bed.http.requests()[0].body.as_deref().unwrap()).unwrap();
    assert_eq!(body["max_tokens"], 16);
    assert!(body.get("max_completion_tokens").is_none());
}

// ---- joining a path onto an endpoint that has its own query -----------------

#[test]
fn kinds_a_path_is_joined_before_the_endpoints_own_query() {
    let (slug, kind) = (Slug::parse("azure").unwrap(), KindId::new("custom"));
    let join =
        |base: &str, path: &str| target(&slug, &kind, base, &AuthStyle::Bearer, None).join(path);
    assert_eq!(
        join("https://x.test/v1", "/models"),
        "https://x.test/v1/models"
    );
    assert_eq!(
        join("https://x.test/v1/", "/models"),
        "https://x.test/v1/models"
    );
    assert_eq!(
        join("https://x.test/v1?api-version=preview", "/models"),
        "https://x.test/v1/models?api-version=preview"
    );
    assert_eq!(
        join("https://x.test/v1?api-version=preview", "/models?limit=5"),
        "https://x.test/v1/models?limit=5&api-version=preview"
    );
    assert_eq!(
        join("https://x.test/v1?", "/models"),
        "https://x.test/v1/models"
    );
    assert_eq!(
        join("https://x.test/v1#frag", "/models"),
        "https://x.test/v1/models",
        "a fragment is never sent"
    );
    assert_eq!(
        join("https://x.test/v1/?a=b#frag", "/key"),
        "https://x.test/v1/key?a=b"
    );
    let t = target(
        &slug,
        &kind,
        " https://x.test/v1/?a=b ",
        &AuthStyle::Bearer,
        None,
    );
    assert_eq!(t.base(), "https://x.test/v1");
}

#[tokio::test]
async fn kinds_an_endpoint_with_a_query_lists_and_pings_at_the_right_url() {
    let bed = Bed::new();
    let base = "https://res.openai.azure.com/openai/v1?api-version=preview";
    bed.http.route(
        Match::get("https://res.openai.azure.com/openai/v1/models?api-version=preview"),
        Scripted::json(200, &json!({"data": [{"id": "dep-1"}]})),
    );
    bed.http.route(
        Match::post("https://res.openai.azure.com/openai/v1/chat/completions?api-version=preview"),
        Scripted::json(200, &json!({})),
    );
    let (slug, kind) = (Slug::parse("azure").unwrap(), KindId::new("custom"));
    let auth = AuthStyle::Custom("api-key".into());
    let t = target(&slug, &kind, base, &auth, None);
    let d = OpenAiCompatDriver::custom();
    assert_eq!(d.list_models(&bed.cx(), &t).await.unwrap().models.len(), 1);
    d.completion_ping(&bed.cx(), &t, &ModelId::parse("dep-1").unwrap())
        .await
        .unwrap();
}

#[tokio::test]
async fn kinds_openrouters_fallback_to_the_public_list_is_reported_and_does_not_prove_the_key() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://openrouter.ai/api/v1/models/user"),
        Scripted::text(404, "gone"),
    );
    bed.http.route(
        Match::prefix("https://openrouter.ai/api/v1/models?"),
        Scripted::json(200, &json!({"data": [{"id": "pub/1"}]})),
    );
    let (slug, kind, key) = (
        Slug::parse("openrouter").unwrap(),
        KindId::new("openrouter"),
        Secret::new("sk-or-revoked"),
    );
    let t = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    let fetched = driver("openrouter")
        .list_models(&bed.cx(), &t)
        .await
        .unwrap();
    assert!(fetched.public_fallback);
    let report = crate::probe::run_probe(&bed.cx(), &driver("openrouter"), &t, TestDepth::Catalog)
        .await
        .unwrap();
    assert!(report.ok(), "the list is real");
    assert!(
        !report.proves_key,
        "but a public list says nothing about a key"
    );
    assert!(
        report
            .notes
            .contains(&crate::probe::ProbeNote::CatalogDoesNotProveKey)
    );
}

#[tokio::test]
async fn kinds_a_2xx_that_is_not_a_json_answer_is_not_a_passing_ping_or_key_check() {
    let cases = [
        (
            "html landing page",
            Scripted::text(200, "<html>welcome to the hotel wifi</html>"),
        ),
        ("empty body", Scripted::text(200, "")),
        ("a bare array", Scripted::text(200, "[1,2]")),
        (
            "a wrapped upstream error",
            Scripted::json(200, &json!({"error": {"message": "upstream exploded"}})),
        ),
        (
            "a body cut at the cap",
            Scripted::Oversize { bytes: 999_999_999 },
        ),
    ];
    for (what, answer) in cases {
        let bed = Bed::new();
        bed.http
            .route(Match::prefix("https://a.test/v1/"), answer.clone());
        let (slug, kind, key) = (
            Slug::parse("acme").unwrap(),
            KindId::new("custom"),
            Secret::new("k"),
        );
        let t = target(
            &slug,
            &kind,
            "https://a.test/v1",
            &AuthStyle::Bearer,
            Some(&key),
        );
        match OpenAiCompatDriver::custom()
            .completion_ping(&bed.cx(), &t, &ModelId::parse("m").unwrap())
            .await
        {
            Err(HubError::Provider(f)) => assert!(
                f.reason == ReasonCode::Unknown && !f.rolls_back(ProviderGroup::Custom),
                "{what}: {f:?}"
            ),
            other => panic!("{what}: {other:?}"),
        }
        // OpenRouter's key check reads the same way.
        bed.http
            .route(Match::prefix("https://openrouter.ai/api/v1/key"), answer);
        let (slug, kind) = (
            Slug::parse("openrouter").unwrap(),
            KindId::new("openrouter"),
        );
        let t = target(
            &slug,
            &kind,
            "https://openrouter.ai/api/v1",
            &AuthStyle::Bearer,
            Some(&key),
        );
        assert!(
            driver("openrouter").key_check(&bed.cx(), &t).await.is_err(),
            "{what}"
        );
    }
    // A JSON object with a null error member is an answer.
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://a.test/v1/"),
        Scripted::json(200, &json!({"error": null, "choices": []})),
    );
    let (slug, kind) = (Slug::parse("acme").unwrap(), KindId::new("custom"));
    let t = target(&slug, &kind, "https://a.test/v1", &AuthStyle::Bearer, None);
    assert!(
        OpenAiCompatDriver::custom()
            .completion_ping(&bed.cx(), &t, &ModelId::parse("m").unwrap())
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn kinds_small_answers_have_their_own_cap_apart_from_the_failure_cap() {
    let mut bed = Bed::new();
    bed.policy.fail_body_cap = 100;
    bed.http.route(
        Match::post("https://a.test/v1/chat/completions"),
        Scripted::json(200, &json!({"id": "x".repeat(5_000), "choices": []})),
    );
    let (slug, kind, key) = (
        Slug::parse("acme").unwrap(),
        KindId::new("custom"),
        Secret::new("k"),
    );
    let t = target(
        &slug,
        &kind,
        "https://a.test/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    // A healthy 5 KB answer is not "cut off" because failure bodies are capped tiny.
    OpenAiCompatDriver::custom()
        .completion_ping(&bed.cx(), &t, &ModelId::parse("m").unwrap())
        .await
        .unwrap();
    assert_eq!(bed.policy.answer_cap, 256 * 1024);
    // A request built for a listing keeps the cap it asked for.
    let d = OpenAiCompatDriver::custom();
    let listing = bed.cx().request(
        d.descriptor(),
        &t,
        HubRequest::get("https://a.test/v1/models").with_body_cap(64 * 1024),
    );
    assert_eq!(listing.body_cap, 64 * 1024);
    assert_eq!(listing.timeout, bed.policy.timeout);
}

#[tokio::test]
async fn kinds_a_chain_of_fallbacks_shares_one_list_deadline() {
    // OpenRouter's account listing takes 8 s and 404s; the public one would take
    // 8 s more. With a 10 s deadline the second request gets 2 s and times out.
    let mut bed = Bed::new();
    bed.policy = EndpointPolicy::hosted().with_list_deadline(Duration::from_secs(10));
    bed.http.route(
        Match::prefix("https://openrouter.ai/api/v1/models/user"),
        Scripted::text(404, "gone").after(Duration::from_secs(8)),
    );
    bed.http.route(
        Match::prefix("https://openrouter.ai/api/v1/models?"),
        Scripted::json(200, &json!({"data": [{"id": "pub/1"}]})).after(Duration::from_secs(8)),
    );
    let (slug, kind, key) = (
        Slug::parse("openrouter").unwrap(),
        KindId::new("openrouter"),
        Secret::new("sk-or-fake"),
    );
    let t = target(
        &slug,
        &kind,
        "https://openrouter.ai/api/v1",
        &AuthStyle::Bearer,
        Some(&key),
    );
    let started = crate::ports::Clock::now(&bed.clock);
    let error = driver("openrouter")
        .list_models(&bed.cx(), &t)
        .await
        .unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Timeout);
    assert_eq!(
        crate::ports::Clock::now(&bed.clock) - started,
        Duration::from_secs(10),
        "the whole read stopped at the deadline"
    );
}

#[tokio::test]
async fn kinds_azure_resources_on_an_older_api_version_keep_max_tokens() {
    for (base, expect) in [
        (
            "https://r.openai.azure.com/openai/deployments/d?api-version=2024-06-01",
            "max_tokens",
        ),
        (
            "https://r.openai.azure.com/openai/deployments/d?api-version=2024-08-01-preview",
            "max_tokens",
        ),
        (
            "https://r.openai.azure.com/openai/deployments/d?api-version=2024-09-01-preview",
            "max_completion_tokens",
        ),
        (
            "https://r.openai.azure.com/openai/deployments/d?api-version=2025-03-01",
            "max_completion_tokens",
        ),
        (
            "https://r.openai.azure.com/openai/v1?api-version=preview",
            "max_completion_tokens",
        ),
        (
            "https://r.openai.azure.com/openai/v1",
            "max_completion_tokens",
        ),
    ] {
        let bed = Bed::new();
        bed.http.route(
            Match::prefix("https://r.openai.azure.com/"),
            Scripted::json(200, &json!({})),
        );
        let (slug, kind) = (Slug::parse("azure").unwrap(), KindId::new("custom"));
        let t = target(&slug, &kind, base, &AuthStyle::Bearer, None);
        OpenAiCompatDriver::custom()
            .completion_ping(&bed.cx(), &t, &ModelId::parse("d").unwrap())
            .await
            .unwrap();
        let body: serde_json::Value =
            serde_json::from_str(bed.http.requests()[0].body.as_deref().unwrap()).unwrap();
        assert_eq!(body[expect], 16, "{base}");
    }
}
