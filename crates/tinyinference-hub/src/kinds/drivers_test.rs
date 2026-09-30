//! Tests for the Anthropic, managed, local and CLI drivers and the registry.

use serde_json::json;

use super::*;
use crate::catalogue::{descriptor, descriptors};
use crate::error::{HubError, Operation, ReasonCode};
use crate::ids::{KindId, ModelId, Slug};
use crate::policy::{EndpointPolicy, HeaderPolicy};
use crate::secret::Secret;
use crate::taxonomy::{AuthStyle, CatalogShape, ProviderGroup};
use crate::testkit::{FakeClock, Match, Script, Scripted, ScriptedHttp};

struct Bed {
    http: ScriptedHttp,
    clock: FakeClock,
    policy: EndpointPolicy,
    headers: HeaderPolicy,
}

impl Bed {
    fn new(policy: EndpointPolicy) -> Self {
        let clock = FakeClock::new();
        Self {
            http: ScriptedHttp::new(clock.clone()),
            clock,
            policy,
            headers: HeaderPolicy::builtin(),
        }
    }
    fn hosted() -> Self {
        Self::new(EndpointPolicy::hosted())
    }
    fn desktop() -> Self {
        Self::new(EndpointPolicy::desktop())
    }
    fn cx(&self) -> DriverContext<'_> {
        DriverContext::new(&self.http, &self.policy, &self.clock, &self.headers)
    }
}

struct Subject {
    slug: Slug,
    kind: KindId,
    base: String,
    auth: AuthStyle,
    key: Option<Secret>,
    group: ProviderGroup,
}

impl Subject {
    fn new(
        kind: &str,
        base: &str,
        auth: AuthStyle,
        key: Option<&str>,
        group: ProviderGroup,
    ) -> Self {
        Self {
            slug: Slug::parse(kind).unwrap(),
            kind: KindId::new(kind),
            base: base.to_string(),
            auth,
            key: key.map(Secret::new),
            group,
        }
    }
    fn target(&self) -> Target<'_> {
        Target {
            slug: &self.slug,
            kind: &self.kind,
            group: self.group,
            base_url: &self.base,
            auth: &self.auth,
            credential: self.key.as_ref(),
            model: None,
        }
    }
}

fn ids(fetched: &crate::catalog::Fetched) -> Vec<&str> {
    fetched.models.iter().map(|m| m.id.as_str()).collect()
}

// ---- Anthropic -------------------------------------------------------------

const ANTHROPIC: &str = "https://api.anthropic.com/v1";

fn anthropic() -> AnthropicDriver {
    AnthropicDriver::for_descriptor(descriptor("anthropic").unwrap().clone())
}

fn anthropic_subject() -> Subject {
    Subject::new(
        "anthropic",
        ANTHROPIC,
        AuthStyle::Anthropic,
        Some("sk-ant-fake"),
        ProviderGroup::Cloud,
    )
}

#[tokio::test]
async fn drivers_anthropic_asks_for_the_maximum_page_and_the_versioned_key_headers() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::get("https://api.anthropic.com/v1/models?limit=1000")
            .with_header("x-api-key", "sk-ant-fake")
            .with_header("anthropic-version", "2023-06-01"),
        Scripted::json(
            200,
            &json!({"data": [{"id": "claude-x", "display_name": "Claude X"}], "has_more": false}),
        ),
    );
    let s = anthropic_subject();
    let fetched = anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert_eq!(ids(&fetched), ["claude-x"]);
    assert_eq!(fetched.models[0].display_name.as_deref(), Some("Claude X"));
    assert!(
        bed.http.requests()[0].header("authorization").is_none(),
        "never as a bearer"
    );
}

#[tokio::test]
async fn drivers_anthropic_follows_has_more_with_the_last_id_cursor() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::get("https://api.anthropic.com/v1/models?limit=1000"),
        Scripted::json(
            200,
            &json!({"data": [{"id": "a"}, {"id": "b"}], "has_more": true, "last_id": "b"}),
        ),
    );
    bed.http.route(
        Match::get("https://api.anthropic.com/v1/models?limit=1000&after_id=b"),
        Scripted::json(
            200,
            &json!({"data": [{"id": "b"}, {"id": "c"}], "has_more": false, "last_id": "c"}),
        ),
    );
    let s = anthropic_subject();
    let fetched = anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert_eq!(ids(&fetched), ["a", "b", "c"], "deduplicated across pages");
    assert!(!fetched.truncated);
}

#[tokio::test]
async fn drivers_anthropic_encodes_the_cursor_and_stops_at_the_page_cap() {
    let bed = Bed::hosted();
    // Every page claims there is more and hands back a fresh cursor with a
    // character that needs encoding.
    let responses: Vec<_> = (0..20)
        .map(|n| Scripted::json(200, &json!({"data": [{"id": format!("m{n}")}], "has_more": true, "last_id": format!("id/{n}")})))
        .collect();
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Script::Sequence(responses),
    );
    let s = anthropic_subject();
    let fetched = anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert!(fetched.truncated, "a has_more that never clears is capped");
    assert_eq!(fetched.models.len(), 10);
    assert_eq!(bed.http.request_count(), 10);
    assert!(
        bed.http.requests()[1].url.ends_with("after_id=id%2F0"),
        "{}",
        bed.http.requests()[1].url
    );
}

#[tokio::test]
async fn drivers_anthropic_has_more_without_a_cursor_stops_instead_of_looping() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Scripted::json(200, &json!({"data": [{"id": "a"}], "has_more": true})),
    );
    let s = anthropic_subject();
    let fetched = anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert_eq!(ids(&fetched), ["a"]);
    assert!(
        fetched.truncated,
        "a catalog that says there is more but cannot be followed is a prefix"
    );
    assert_eq!(bed.http.request_count(), 1);
}

#[tokio::test]
async fn drivers_anthropic_a_400_for_the_missing_version_header_is_not_read_as_a_bad_key() {
    // The native API answers a bearer-authenticated request with a 400.
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Scripted::json(400, &json!({"type": "error", "error": {"type": "invalid_request_error", "message": "anthropic-version: header is required"}})),
    );
    let s = anthropic_subject();
    match anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert!(!f.reason.destroys_credential(), "{f:?}"),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn drivers_anthropic_a_failure_or_a_bad_page_stops_the_read() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Script::Sequence(vec![
            Scripted::json(
                200,
                &json!({"data": [{"id": "a"}], "has_more": true, "last_id": "a"}),
            ),
            Scripted::text(500, "boom"),
        ]),
    );
    let s = anthropic_subject();
    assert!(
        anthropic()
            .list_models(&bed.cx(), &s.target())
            .await
            .is_err(),
        "a page failing fails the read; a half list is never returned as whole"
    );
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Scripted::Oversize { bytes: 99_999_999 },
    );
    match anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert!(f.truncated),
        other => panic!("{other:?}"),
    }
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Scripted::text(200, "not json"),
    );
    assert_eq!(
        anthropic()
            .list_models(&bed.cx(), &s.target())
            .await
            .unwrap_err()
            .reason(),
        ReasonCode::Unknown
    );
}

#[tokio::test]
async fn drivers_anthropic_pings_the_native_messages_endpoint() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::post("https://api.anthropic.com/v1/messages"),
        Scripted::json(200, &json!({"content": []})),
    );
    let s = anthropic_subject();
    anthropic()
        .completion_ping(&bed.cx(), &s.target(), &ModelId::parse("claude-x").unwrap())
        .await
        .unwrap();
    let sent = &bed.http.requests()[0];
    let body: serde_json::Value = serde_json::from_str(sent.body.as_deref().unwrap()).unwrap();
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(sent.header("anthropic-version"), Some("2023-06-01"));
}

// ---- managed ---------------------------------------------------------------

const MANAGED: &str = "https://api.example.test/agent-integrations/openrouter";

fn managed() -> ManagedDriver {
    ManagedDriver::paged(descriptor("tinyhumans").unwrap().clone())
}

fn managed_subject(key: Option<&str>) -> Subject {
    Subject::new(
        "tinyhumans",
        MANAGED,
        AuthStyle::Bearer,
        key,
        ProviderGroup::Managed,
    )
}

fn page(ids: &[&str], total: usize) -> Scripted {
    let rows: Vec<_> = ids
        .iter()
        .map(|i| json!({"id": i, "pricing": {"inputPer1M": 1.0, "outputPer1M": 4.0}}))
        .collect();
    Scripted::json(
        200,
        &json!({"success": true, "data": {"object": "list", "total": total, "data": rows}}),
    )
}

#[tokio::test]
async fn drivers_managed_reads_every_page_to_total_with_prices() {
    let bed = Bed::hosted();
    let a = bed.http.route(
        Match::get(format!("{MANAGED}/models?limit=500&offset=0")),
        page(&["a", "b"], 3),
    );
    let b = bed.http.route(
        Match::get(format!("{MANAGED}/models?limit=500&offset=2")),
        page(&["c"], 3),
    );
    let s = managed_subject(Some("th-fake"));
    let fetched = managed().list_models(&bed.cx(), &s.target()).await.unwrap();
    assert_eq!(ids(&fetched), ["a", "b", "c"]);
    assert_eq!(fetched.models[0].input_per_1m, Some(1.0));
    assert_eq!(fetched.models[0].output_per_1m, Some(4.0));
    assert!(!fetched.truncated);
    assert_eq!((bed.http.hits(a), bed.http.hits(b)), (1, 1));
}

#[tokio::test]
async fn drivers_managed_a_page_over_the_cap_is_unknown_and_truncated() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix(format!("{MANAGED}/models?limit=500&offset=0")),
        page(&["a"], 2),
    );
    bed.http.route(
        Match::prefix(format!("{MANAGED}/models?limit=500&offset=1")),
        Scripted::Oversize { bytes: 99_999_999 },
    );
    let s = managed_subject(Some("th-fake"));
    match managed()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert!(
            f.truncated && f.reason == ReasonCode::Unknown && !f.reason.destroys_credential()
        ),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn drivers_managed_a_total_never_reached_stops_at_twenty_pages_and_says_so() {
    let bed = Bed::hosted();
    let pages: Vec<_> = (0..25)
        .map(|n| page(&[format!("m{n}").as_str()], 1_000_000))
        .collect();
    bed.http.route(
        Match::prefix(format!("{MANAGED}/models")),
        Script::Sequence(pages),
    );
    let s = managed_subject(Some("th-fake"));
    let fetched = managed().list_models(&bed.cx(), &s.target()).await.unwrap();
    assert!(fetched.truncated);
    assert_eq!(fetched.models.len(), 20);
    assert_eq!(bed.http.request_count(), 20);
}

#[tokio::test]
async fn drivers_managed_a_non_utf8_or_odd_page_is_unknown_never_auth() {
    for body in [
        Scripted::Malformed(vec![0xff, 0xfe]),
        Scripted::text(200, r#"{"data":[{"id":"a"}]}"#),
        Scripted::json(200, &json!({"success": false, "error": "nope"})),
    ] {
        let bed = Bed::hosted();
        bed.http
            .route(Match::prefix(format!("{MANAGED}/models")), body);
        let s = managed_subject(Some("th-fake"));
        match managed()
            .list_models(&bed.cx(), &s.target())
            .await
            .unwrap_err()
        {
            HubError::Provider(f) => assert_eq!(f.reason, ReasonCode::Unknown),
            other => panic!("{other:?}"),
        }
    }
}

#[tokio::test]
async fn drivers_managed_signed_out_is_typed_and_sends_nothing() {
    let bed = Bed::hosted();
    let s = managed_subject(None);
    let error = managed()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err();
    assert!(
        matches!(&error, HubError::SignedOut { provider } if provider.as_str() == "tinyhumans")
    );
    assert_eq!(error.reason(), ReasonCode::SignedOut);
    let blank = managed_subject(Some("   "));
    assert!(
        matches!(
            managed().list_models(&bed.cx(), &blank.target()).await,
            Err(HubError::SignedOut { .. })
        ),
        "a blank credential is no credential"
    );
    assert_eq!(bed.http.request_count(), 0);
}

#[tokio::test]
async fn drivers_managed_the_openai_shaped_backend_reads_prices_with_the_hosts_query() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::get("https://api.example.test/openai/v1/models?catalog=openrouter"),
        Scripted::json(
            200,
            &json!({"data": [{"id": "x/y", "pricing": {"inputPer1M": 2.5, "outputPer1M": 10.0}}]}),
        ),
    );
    let d = ManagedDriver::openai_shaped(
        descriptor("tinyhumans").unwrap().clone(),
        "?catalog=openrouter",
    );
    assert_eq!(d.shape(), CatalogShape::OpenAi);
    let s = Subject::new(
        "tinyhumans",
        "https://api.example.test/openai/v1",
        AuthStyle::SessionJwt,
        Some("jwt-fake"),
        ProviderGroup::Managed,
    );
    let fetched = d.list_models(&bed.cx(), &s.target()).await.unwrap();
    assert_eq!(fetched.models[0].output_per_1m, Some(10.0));
    assert_eq!(
        bed.http.requests()[0].header("authorization"),
        Some("<redacted>")
    );
    assert!(bed.http.requests()[0].carried(&Secret::new("jwt-fake")));
}

#[test]
fn drivers_managed_reads_the_shape_it_was_built_for() {
    assert_eq!(managed().shape(), CatalogShape::PagedEnvelope);
}

#[tokio::test]
async fn drivers_managed_a_rejected_platform_token_is_an_auth_failure_the_caller_can_invalidate() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix(format!("{MANAGED}/models")),
        Scripted::text(401, "token expired"),
    );
    let s = managed_subject(Some("th-stale"));
    match managed()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert_eq!((f.reason, f.status), (ReasonCode::Auth, Some(401))),
        other => panic!("{other:?}"),
    }
}

// ---- local -----------------------------------------------------------------

fn local(kind: &str) -> LocalDriver {
    LocalDriver::for_descriptor(descriptor(kind).unwrap().clone())
}

fn local_subject(kind: &str, base: &str, key: Option<&str>) -> Subject {
    let auth = descriptor(kind).unwrap().auth.clone();
    Subject::new(kind, base, auth, key, ProviderGroup::Local)
}

#[tokio::test]
async fn drivers_local_ollama_reads_the_openai_listing_first_and_without_a_key_header() {
    let bed = Bed::desktop();
    bed.http.route(
        Match::get("http://localhost:11434/v1/models").without_header("authorization"),
        Scripted::json(
            200,
            &json!({"object": "list", "data": [{"id": "llama3:latest"}]}),
        ),
    );
    let s = local_subject("ollama", "http://localhost:11434/v1", Some("ignored-key"));
    let fetched = local("ollama")
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert_eq!(ids(&fetched), ["llama3:latest"]);
    assert!(
        !bed.http.requests()[0].credentialed,
        "Ollama answers spurious 401s to a key, so none is sent"
    );
}

#[tokio::test]
async fn drivers_local_ollama_with_nothing_pulled_is_healthy_and_empty() {
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::text(200, r#"{"object":"list","data":null}"#),
    );
    let s = local_subject("ollama", "http://localhost:11434/v1", None);
    let fetched = local("ollama")
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert!(fetched.models.is_empty());
}

#[tokio::test]
async fn drivers_local_ollama_falls_back_to_api_tags_when_v1_models_is_missing() {
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::text(404, "404 page not found"),
    );
    bed.http.route(
        Match::get("http://localhost:11434/api/tags"),
        Scripted::json(
            200,
            &json!({"models": [{"name": "old-model:latest", "model": "old-model:latest"}]}),
        ),
    );
    let s = local_subject("ollama", "http://localhost:11434/v1", None);
    let fetched = local("ollama")
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert_eq!(ids(&fetched), ["old-model:latest"]);
}

#[tokio::test]
async fn drivers_local_a_runtime_that_is_down_is_not_asked_twice() {
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:11434/"),
        Scripted::ConnectRefused,
    );
    let s = local_subject("ollama", "http://localhost:11434/v1", None);
    let error = local("ollama")
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Endpoint);
    assert_eq!(
        bed.http.request_count(),
        1,
        "no fallback for a refused connection"
    );
    let bed = Bed::desktop();
    bed.http
        .route(Match::prefix("http://localhost:11434/"), Scripted::Timeout);
    assert_eq!(
        local("ollama")
            .list_models(&bed.cx(), &s.target())
            .await
            .unwrap_err()
            .reason(),
        ReasonCode::Timeout
    );
    assert_eq!(bed.http.request_count(), 1);
}

#[tokio::test]
async fn drivers_local_ollama_tags_that_are_oversize_or_unreadable_report_the_first_failure() {
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::text(404, "no"),
    );
    bed.http.route(
        Match::prefix("http://localhost:11434/api/tags"),
        Scripted::text(200, "garbage"),
    );
    let s = local_subject("ollama", "http://localhost:11434/v1", None);
    assert_eq!(
        local("ollama")
            .list_models(&bed.cx(), &s.target())
            .await
            .unwrap_err()
            .reason(),
        ReasonCode::Unknown
    );

    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::Oversize { bytes: 99_999_999 },
    );
    bed.http.route(
        Match::prefix("http://localhost:11434/api/tags"),
        Scripted::Oversize { bytes: 99_999_999 },
    );
    match local("ollama")
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert!(f.truncated),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn drivers_local_lm_studio_prefers_its_native_listing_for_the_richer_facts() {
    let bed = Bed::desktop();
    bed.http.route(
        Match::get("http://localhost:1234/api/v0/models"),
        Scripted::json(200, &json!({"data": [{"id": "qwen", "type": "llm", "max_context_length": 32768, "capabilities": ["tool_use"]}]})),
    );
    let s = local_subject("lmstudio", "http://localhost:1234/v1", None);
    let fetched = local("lmstudio")
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert_eq!(
        fetched.models[0].capabilities.context_window.value,
        Some(32_768)
    );
    assert_eq!(
        bed.http.request_count(),
        1,
        "the native listing answered; no second request"
    );
}

#[tokio::test]
async fn drivers_local_lm_studio_falls_back_to_the_openai_listing_when_the_native_one_is_absent_or_unreadable()
 {
    for native in [
        Scripted::text(404, "not found"),
        Scripted::text(200, "not json"),
        Scripted::Oversize { bytes: 99_999_999 },
    ] {
        let bed = Bed::desktop();
        bed.http
            .route(Match::prefix("http://localhost:1234/api/v0/models"), native);
        bed.http.route(
            Match::prefix("http://localhost:1234/v1/models"),
            Scripted::json(200, &json!({"data": [{"id": "plain"}]})),
        );
        let s = local_subject("lmstudio", "http://localhost:1234/v1", None);
        let fetched = local("lmstudio")
            .list_models(&bed.cx(), &s.target())
            .await
            .unwrap();
        assert_eq!(ids(&fetched), ["plain"]);
    }
    // A runtime that is down is not asked twice.
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:1234/"),
        Scripted::ConnectRefused,
    );
    let s = local_subject("lmstudio", "http://localhost:1234/v1", None);
    assert!(
        local("lmstudio")
            .list_models(&bed.cx(), &s.target())
            .await
            .is_err()
    );
    assert_eq!(bed.http.request_count(), 1);
    // A rejected key on the native listing is the answer, not a reason to retry.
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:1234/api/v0/models"),
        Scripted::text(401, "no"),
    );
    assert!(
        local("lmstudio")
            .list_models(&bed.cx(), &s.target())
            .await
            .is_err()
    );
    assert_eq!(bed.http.request_count(), 1);
}

#[tokio::test]
async fn drivers_local_a_key_is_sent_only_to_runtimes_that_take_one() {
    let bed = Bed::desktop();
    bed.http.route(
        Match::get("http://127.0.0.1:8000/v1/models")
            .with_header("authorization", "Bearer local-key"),
        Scripted::json(200, &json!({"data": [{"id": "m"}]})),
    );
    let s = local_subject("omlx", "http://127.0.0.1:8000/v1", Some("local-key"));
    assert!(
        local("omlx")
            .list_models(&bed.cx(), &s.target())
            .await
            .is_ok()
    );
    let keyless = local_subject("omlx", "http://127.0.0.1:8000/v1", None);
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://127.0.0.1:8000/v1/models"),
        Scripted::json(200, &json!({"data": []})),
    );
    local("omlx")
        .list_models(&bed.cx(), &keyless.target())
        .await
        .unwrap();
    assert!(!bed.http.requests()[0].credentialed);
}

#[tokio::test]
async fn drivers_local_a_hosted_policy_refuses_loopback_runtimes() {
    let bed = Bed::hosted();
    let s = local_subject("ollama", "http://localhost:11434/v1", None);
    let report = crate::probe::run_probe(
        &bed.cx(),
        &local("ollama"),
        &s.target(),
        crate::taxonomy::TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert!(
        report.refusal.is_some(),
        "local kinds are not offered on a hosted tenant"
    );
    assert_eq!(bed.http.request_count(), 0);
}

// ---- CLI -------------------------------------------------------------------

#[tokio::test]
async fn drivers_cli_kinds_never_touch_http() {
    let bed = Bed::hosted();
    for kind in ["claude-code", "codex"] {
        let d = CliDriver::for_descriptor(descriptor(kind).unwrap().clone());
        let s = Subject::new(kind, "", AuthStyle::None, None, ProviderGroup::Cli);
        match d.list_models(&bed.cx(), &s.target()).await.unwrap_err() {
            HubError::Unsupported { op, kind: k } => {
                assert_eq!((op, k.as_str()), (Operation::ListModels, kind))
            }
            other => panic!("{other:?}"),
        }
        assert!(
            d.completion_ping(&bed.cx(), &s.target(), &ModelId::parse("m").unwrap())
                .await
                .is_err()
        );
    }
    assert_eq!(bed.http.request_count(), 0);
}

// ---- registry --------------------------------------------------------------

#[test]
fn registry_the_builtin_registry_has_a_driver_for_every_catalogue_kind_and_custom() {
    let registry = DriverRegistry::with_builtin();
    for d in descriptors() {
        let driver = registry
            .get(&d.kind)
            .unwrap_or_else(|| panic!("no driver for {}", d.kind));
        assert_eq!(driver.descriptor().kind, d.kind);
        assert_eq!(driver.descriptor().group, d.group);
    }
    assert_eq!(registry.len(), 35);
    assert!(!registry.is_empty() && DriverRegistry::new().is_empty());
    assert!(registry.get(&KindId::new("custom")).is_some());
    assert_eq!(registry.kinds().len(), 35);
    assert!(format!("{registry:?}").contains("openai"));
}

#[test]
fn registry_aliases_resolve_and_unknown_kinds_do_not() {
    let registry = DriverRegistry::with_builtin();
    assert_eq!(
        registry
            .resolve("openhuman")
            .unwrap()
            .descriptor()
            .kind
            .as_str(),
        "tinyhumans"
    );
    assert_eq!(
        registry
            .resolve("LM-Studio")
            .unwrap()
            .descriptor()
            .kind
            .as_str(),
        "lmstudio"
    );
    assert_eq!(
        registry.resolve("vllm").unwrap().descriptor().kind.as_str(),
        "local-openai"
    );
    assert_eq!(
        registry.resolve("openai").unwrap().descriptor().group,
        ProviderGroup::Cloud,
        "a bare openai is the hosted row"
    );
    assert!(
        registry.resolve("not-a-kind").is_none(),
        "an unknown kind is never guessed"
    );
    assert!(registry.resolve("").is_none());
}

#[test]
fn registry_the_right_driver_serves_each_group() {
    let registry = DriverRegistry::with_builtin();
    let name = |kind: &str| format!("{:?}", registry.get(&KindId::new(kind)).unwrap());
    assert!(name("anthropic").starts_with("AnthropicDriver"));
    assert!(name("tinyhumans").starts_with("ManagedDriver"));
    assert!(name("ollama").starts_with("LocalDriver"));
    assert!(name("codex").starts_with("CliDriver"));
    assert!(name("groq").starts_with("OpenAiCompatDriver"));
    assert!(name("custom").starts_with("OpenAiCompatDriver"));
}

#[test]
fn registry_registering_again_replaces_and_returns_the_old_driver() {
    let mut registry = DriverRegistry::with_builtin();
    let openai_shaped = ManagedDriver::openai_shaped(
        descriptor("tinyhumans").unwrap().clone(),
        "?catalog=openrouter",
    );
    let old = registry
        .register(std::sync::Arc::new(openai_shaped))
        .expect("replaced the paged driver");
    assert!(format!("{old:?}").contains("PagedEnvelope"));
    assert!(format!("{:?}", registry.resolve("tinyhumans").unwrap()).contains("OpenAi"));
    assert_eq!(registry.len(), 35, "replacing does not grow it");
}

#[tokio::test]
async fn drivers_local_ollama_falls_back_to_tags_when_v1_models_answers_with_something_that_is_not_a_listing()
 {
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::text(200, "<html>proxy landing page</html>"),
    );
    bed.http.route(
        Match::get("http://localhost:11434/api/tags"),
        Scripted::json(200, &json!({"models": [{"name": "pulled:latest"}]})),
    );
    let s = local_subject("ollama", "http://localhost:11434/v1", None);
    let fetched = local("ollama")
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert_eq!(ids(&fetched), ["pulled:latest"]);
}

#[tokio::test]
async fn drivers_local_a_transport_failure_that_looks_unreadable_is_never_a_reason_to_ask_again() {
    // `HttpError::Failed` classifies as unknown with no status, exactly like a
    // body that did not parse; only the latter falls back.
    struct Failing;
    #[async_trait::async_trait]
    impl crate::ports::Http for Failing {
        async fn send(
            &self,
            _r: crate::ports::HubRequest,
            _p: &EndpointPolicy,
        ) -> Result<crate::ports::HubResponse, crate::ports::HttpError> {
            Err(crate::ports::HttpError::Failed(crate::LogOnly::new(
                "tls handshake eof".into(),
            )))
        }
    }
    impl std::fmt::Debug for Failing {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("Failing")
        }
    }
    let bed = Bed::desktop();
    let failing = Failing;
    let cx = DriverContext::new(&failing, &bed.policy, &bed.clock, &bed.headers);
    let s = local_subject("ollama", "http://localhost:11434/v1", None);
    let error = local("ollama")
        .list_models(&cx, &s.target())
        .await
        .unwrap_err();
    assert_eq!(error.reason(), ReasonCode::Unknown);
}

#[tokio::test]
async fn drivers_local_an_oversize_tags_list_is_reported_as_too_large_not_as_the_earlier_404() {
    let bed = Bed::desktop();
    bed.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::text(404, "no"),
    );
    bed.http.route(
        Match::prefix("http://localhost:11434/api/tags"),
        Scripted::Oversize { bytes: 99_999_999 },
    );
    let s = local_subject("ollama", "http://localhost:11434/v1", None);
    match local("ollama")
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert!(
            f.truncated && f.status.is_none() && f.reason == ReasonCode::Unknown,
            "{f:?}"
        ),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn drivers_a_paged_read_is_bounded_by_the_list_deadline_across_pages() {
    let mut bed = Bed::hosted();
    bed.policy = EndpointPolicy::hosted().with_list_deadline(std::time::Duration::from_secs(60));
    // Every page takes nine seconds and there is always another.
    let pages: Vec<_> = (0..30)
        .map(|n| {
            page(&[format!("m{n}").as_str()], 1_000_000).after(std::time::Duration::from_secs(9))
        })
        .collect();
    bed.http.route(
        Match::prefix(format!("{MANAGED}/models")),
        Script::Sequence(pages),
    );
    let s = managed_subject(Some("th-fake"));
    let started = crate::ports::Clock::now(&bed.clock);
    match managed()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert_eq!(f.reason, ReasonCode::Timeout),
        other => panic!("{other:?}"),
    }
    let spent = crate::ports::Clock::now(&bed.clock) - started;
    assert!(
        spent <= std::time::Duration::from_secs(63),
        "{spent:?}: the read stopped near its deadline, not after 20 pages"
    );
    assert!(bed.http.request_count() <= 7);
}

#[tokio::test]
async fn drivers_anthropic_stops_when_the_cursor_does_not_move() {
    let bed = Bed::hosted();
    // A proxy that ignores `after_id` and always answers the same page.
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Scripted::json(
            200,
            &json!({"data": [{"id": "a"}], "has_more": true, "last_id": "a"}),
        ),
    );
    let s = anthropic_subject();
    let fetched = anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert!(fetched.truncated);
    assert_eq!(
        bed.http.request_count(),
        2,
        "the second answer repeated the cursor, so it stopped"
    );
}

#[test]
fn drivers_managed_openai_shaped_normalises_the_query_and_the_descriptor_shape() {
    for (given, expected) in [
        ("catalog=openrouter", "?catalog=openrouter"),
        ("?catalog=openrouter", "?catalog=openrouter"),
        ("  ?x=1 ", "?x=1"),
        ("", ""),
        ("?", ""),
    ] {
        let d = ManagedDriver::openai_shaped(descriptor("tinyhumans").unwrap().clone(), given);
        assert!(
            format!("{d:?}").contains(&format!("query: {expected:?}")),
            "{given:?} -> {d:?}"
        );
        assert_eq!(
            d.descriptor().catalog,
            CatalogShape::OpenAi,
            "the descriptor a cache key is built from says how it is read"
        );
    }
}

#[tokio::test]
async fn drivers_a_200_listing_with_no_usable_rows_fails_the_read_instead_of_passing_empty() {
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Scripted::json(
            200,
            &json!({"data": [{"model": "no-id"}], "has_more": false}),
        ),
    );
    let s = anthropic_subject();
    match anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert_eq!(f.reason, ReasonCode::Unknown),
        other => panic!("{other:?}"),
    }
    // The generic driver and a probe read it the same way: not proven, not healthy.
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix("https://a.test/v1/models"),
        Scripted::json(200, &json!({"data": [{"x": 1}]})),
    );
    let (slug, kind) = (Slug::parse("acme").unwrap(), KindId::new("custom"));
    let auth = AuthStyle::Bearer;
    let key = Secret::new("k");
    let t = Target::new(
        &slug,
        &kind,
        ProviderGroup::Custom,
        "https://a.test/v1",
        &auth,
    )
    .with_credential(&key);
    let report = crate::probe::run_probe(
        &bed.cx(),
        &OpenAiCompatDriver::custom(),
        &t,
        crate::taxonomy::TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert!(!report.ok() && !report.proves_key);
}

#[tokio::test]
async fn drivers_one_bad_late_page_does_not_discard_the_good_pages_before_it() {
    // Managed: page 1 good, page 2 has only rows with no usable id.
    let bed = Bed::hosted();
    bed.http.route(
        Match::get(format!("{MANAGED}/models?limit=500&offset=0")),
        page(&["a", "b"], 4),
    );
    bed.http.route(
        Match::get(format!("{MANAGED}/models?limit=500&offset=2")),
        Scripted::json(
            200,
            &json!({"success": true, "data": {"total": 4, "data": [{"nope": 1}, {"id": 5}]}}),
        ),
    );
    let s = managed_subject(Some("th-fake"));
    let fetched = managed().list_models(&bed.cx(), &s.target()).await.unwrap();
    assert_eq!(ids(&fetched), ["a", "b"]);
    // Every page bad: an error, not a passing empty list.
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix(format!("{MANAGED}/models")),
        Scripted::json(
            200,
            &json!({"success": true, "data": {"total": 2, "data": [{"nope": 1}, {"id": 5}]}}),
        ),
    );
    assert!(managed().list_models(&bed.cx(), &s.target()).await.is_err());

    // Anthropic: the same shape.
    let bed = Bed::hosted();
    bed.http.route(
        Match::prefix("https://api.anthropic.com/v1/models"),
        Script::Sequence(vec![
            Scripted::json(
                200,
                &json!({"data": [{"id": "keep"}], "has_more": true, "last_id": "keep"}),
            ),
            Scripted::json(
                200,
                &json!({"data": [{"model": "no-id"}], "has_more": false}),
            ),
        ]),
    );
    let s = anthropic_subject();
    let fetched = anthropic()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap();
    assert_eq!(ids(&fetched), ["keep"]);
}

#[tokio::test]
async fn drivers_a_spent_list_deadline_stops_a_read_before_it_sends_anything() {
    let mut bed = Bed::hosted();
    bed.policy = EndpointPolicy::hosted().with_list_deadline(std::time::Duration::ZERO);
    let s = managed_subject(Some("th-fake"));
    match managed()
        .list_models(&bed.cx(), &s.target())
        .await
        .unwrap_err()
    {
        HubError::Provider(f) => assert_eq!(f.reason, ReasonCode::Timeout),
        other => panic!("{other:?}"),
    }
    let a = anthropic_subject();
    assert_eq!(
        anthropic()
            .list_models(&bed.cx(), &a.target())
            .await
            .unwrap_err()
            .reason(),
        ReasonCode::Timeout
    );
    assert_eq!(bed.http.request_count(), 0);
}
