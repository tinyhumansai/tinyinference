//! Tests for the probe runner.

use std::time::Duration;

use serde_json::json;

use super::*;
use crate::catalogue::{custom_descriptor, descriptor};
use crate::error::{HubError, InputField, InvalidInput, Operation, PolicyViolation, ReasonCode};
use crate::ids::{KindId, ModelId, Slug};
use crate::kinds::{DriverContext, OpenAiCompatDriver, Target};
use crate::policy::{EndpointPolicy, EndpointRefusal, HeaderPolicy};
use crate::secret::Secret;
use crate::taxonomy::{AuthStyle, ProviderGroup, TestDepth};
use crate::testkit::{FakeClock, Match, Scripted, ScriptedHttp};

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

fn driver(kind: &str) -> OpenAiCompatDriver {
    OpenAiCompatDriver::for_descriptor(descriptor(kind).unwrap().clone())
}

struct Subject {
    slug: Slug,
    kind: KindId,
    key: Option<Secret>,
    model: Option<ModelId>,
    base: String,
    group: ProviderGroup,
    auth: AuthStyle,
}

impl Subject {
    fn cloud(kind: &str, base: &str, key: Option<&str>) -> Self {
        Self {
            slug: Slug::parse(kind).unwrap(),
            kind: KindId::new(kind),
            key: key.map(Secret::new),
            model: None,
            base: base.to_string(),
            group: ProviderGroup::Cloud,
            auth: AuthStyle::Bearer,
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
            model: self.model.as_ref(),
        }
    }
}

const OPENAI: &str = "https://api.openai.com/v1";

fn listing(bed: &Bed, url: &str, ids: &[&str]) {
    let rows: Vec<_> = ids.iter().map(|i| json!({"id": i})).collect();
    bed.http.route(
        Match::prefix(url),
        Scripted::json(200, &json!({"data": rows})),
    );
}

#[tokio::test]
async fn probe_a_passing_catalog_reports_the_models_latency_and_proves_the_key() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::json(200, &json!({"data": [{"id": "gpt-5"}, {"id": "o3"}]}))
            .after(Duration::from_millis(250)),
    );
    let s = Subject::cloud("openai", OPENAI, Some("sk-not-a-real-key"));
    let report = run_probe(
        &bed.cx(),
        &driver("openai"),
        &s.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert!(report.ok() && report.proves_key && report.notes.is_empty());
    assert_eq!(report.model_count(), Some(2));
    assert_eq!(
        report.latency,
        Duration::from_millis(250),
        "latency is the hub clock's, not the wall's"
    );
    assert_eq!(report.depth, TestDepth::Catalog);
    assert!(report.into_result().is_ok());
}

#[tokio::test]
async fn probe_a_rejected_key_is_an_auth_failure_in_the_report_that_rolls_back() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(401, "Incorrect API key provided"),
    );
    let s = Subject::cloud("openai", OPENAI, Some("sk-bad"));
    let report = run_probe(
        &bed.cx(),
        &driver("openai"),
        &s.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    let failure = report.failure.clone().unwrap();
    assert_eq!(failure.reason, ReasonCode::Auth);
    assert!(failure.rolls_back(ProviderGroup::Cloud));
    assert!(!report.ok() && report.model_count().is_none() && !report.proves_key);
    assert!(matches!(report.into_result(), Err(HubError::Provider(_))));
}

#[tokio::test]
async fn probe_only_the_supported_depths_run_and_the_rest_are_typed_unsupported() {
    let bed = Bed::new();
    let s = Subject::cloud("groq", "https://api.groq.com/openai/v1", Some("gsk"));
    match run_probe(&bed.cx(), &driver("groq"), &s.target(), TestDepth::KeyOnly)
        .await
        .unwrap_err()
    {
        HubError::Unsupported { op, kind } => {
            assert_eq!(op, Operation::Test(TestDepth::KeyOnly));
            assert_eq!(kind.as_str(), "groq");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(
        bed.http.request_count(),
        0,
        "nothing is sent for an unsupported depth"
    );
}

#[tokio::test]
async fn probe_openrouter_key_only_hits_get_key_and_proves_the_key() {
    let bed = Bed::new();
    let key = bed.http.route(
        Match::get("https://openrouter.ai/api/v1/key"),
        Scripted::json(200, &json!({"data": {}})),
    );
    let s = Subject::cloud(
        "openrouter",
        "https://openrouter.ai/api/v1",
        Some("sk-or-fake"),
    );
    let report = run_probe(
        &bed.cx(),
        &driver("openrouter"),
        &s.target(),
        TestDepth::KeyOnly,
    )
    .await
    .unwrap();
    assert!(report.ok() && report.proves_key);
    assert!(report.models.is_empty() && report.model_count().is_none());
    assert_eq!(bed.http.hits(key), 1);
}

#[tokio::test]
async fn probe_a_public_catalog_does_not_prove_the_key_so_only_a_completion_can_fail_a_bad_one() {
    // Hugging Face / Venice list models without a key.
    let bed = Bed::new();
    listing(&bed, "https://router.huggingface.co/v1/models", &["m"]);
    bed.http.route(
        Match::post("https://router.huggingface.co/v1/chat/completions"),
        Scripted::text(401, "Invalid credentials"),
    );
    let hf = descriptor("huggingface").map(|d| d.slug().to_string());
    let kind = hf.expect("the catalogue ships a Hugging Face row");
    let base = descriptor(&kind).unwrap().default_endpoint.unwrap();
    let mut s = Subject::cloud(&kind, base, Some("hf_bad"));
    s.model = Some(ModelId::parse("m").unwrap());
    let d = driver(&kind);
    let catalog = run_probe(&bed.cx(), &d, &s.target(), TestDepth::Catalog)
        .await
        .unwrap();
    assert!(catalog.ok(), "the listing answers whatever the key");
    assert!(!catalog.proves_key);
    assert!(catalog.notes.contains(&ProbeNote::CatalogDoesNotProveKey));
    let completion = run_probe(&bed.cx(), &d, &s.target(), TestDepth::Completion)
        .await
        .unwrap();
    assert_eq!(
        completion.failure.as_ref().unwrap().reason,
        ReasonCode::Auth,
        "only the completion fails the bad key"
    );
}

#[tokio::test]
async fn probe_an_empty_account_scoped_catalog_is_noted_not_failed() {
    // Fireworks: a fresh key sees no public models.
    let bed = Bed::new();
    let kind = "fireworks";
    let base = descriptor(kind).unwrap().default_endpoint.unwrap();
    listing(&bed, &format!("{base}/models"), &[]);
    let s = Subject::cloud(kind, base, Some("fw-fresh"));
    let report = run_probe(&bed.cx(), &driver(kind), &s.target(), TestDepth::Catalog)
        .await
        .unwrap();
    assert!(report.ok());
    assert!(
        report
            .notes
            .contains(&ProbeNote::AccountScopedCatalogIsEmpty)
    );
    assert_eq!(report.model_count(), Some(0));
}

#[tokio::test]
async fn probe_a_completion_needs_a_model_and_a_key_where_the_kind_needs_one() {
    let bed = Bed::new();
    let s = Subject::cloud("openai", OPENAI, Some("sk"));
    let error = run_probe(
        &bed.cx(),
        &driver("openai"),
        &s.target(),
        TestDepth::Completion,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        HubError::Invalid(InvalidInput::Malformed {
            field: InputField::ModelId,
            ..
        })
    ));
    let mut keyless = Subject::cloud("openai", OPENAI, None);
    keyless.model = Some(ModelId::parse("gpt-5").unwrap());
    let error = run_probe(
        &bed.cx(),
        &driver("openai"),
        &keyless.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap_err();
    assert!(matches!(
        error,
        HubError::Invalid(InvalidInput::Empty(InputField::Key))
    ));
    assert_eq!(bed.http.request_count(), 0);
}

#[tokio::test]
async fn probe_a_keyless_local_runtime_probes_without_one() {
    let bed = Bed::new();
    let policy = EndpointPolicy::desktop();
    let cx = DriverContext::new(&bed.http, &policy, &bed.clock, &bed.headers);
    listing(&bed, "http://localhost:11434/v1/models", &["llama3"]);
    let d = OpenAiCompatDriver::for_descriptor(descriptor("ollama").unwrap().clone());
    let mut s = Subject::cloud("ollama", "http://localhost:11434/v1", None);
    s.group = ProviderGroup::Local;
    s.auth = AuthStyle::None;
    let report = run_probe(&cx, &d, &s.target(), TestDepth::Catalog)
        .await
        .unwrap();
    assert!(report.ok());
    assert!(
        !report.proves_key,
        "no key was presented, so none was proven"
    );
    assert!(!bed.http.requests()[0].credentialed);
}

#[tokio::test]
async fn probe_a_managed_provider_with_no_credential_is_signed_out_not_a_missing_key() {
    let bed = Bed::new();
    let d = OpenAiCompatDriver::for_descriptor(descriptor("tinyhumans").unwrap().clone());
    let mut s = Subject::cloud(
        "tinyhumans",
        "https://api.example.test/agent-integrations/openrouter",
        None,
    );
    s.group = ProviderGroup::Managed;
    let error = run_probe(&bed.cx(), &d, &s.target(), TestDepth::Catalog)
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::SignedOut { provider } if provider.as_str() == "tinyhumans"));
    assert_eq!(bed.http.request_count(), 0);
}

#[tokio::test]
async fn probe_a_refused_endpoint_is_an_endpoint_failure_that_sends_nothing() {
    let bed = Bed::new();
    let d = OpenAiCompatDriver::custom();
    for (base, refusal) in [
        ("http://169.254.169.254/latest", EndpointRefusal::LinkLocal),
        ("http://10.0.0.5/v1", EndpointRefusal::PrivateNetwork),
        ("http://127.0.0.1:8080/v1", EndpointRefusal::Loopback),
        (
            "http://[::ffff:169.254.169.254]/v1",
            EndpointRefusal::LinkLocal,
        ),
        ("http://api.acme.test/v1", EndpointRefusal::Cleartext),
    ] {
        let s = Subject {
            group: ProviderGroup::Custom,
            ..Subject::cloud("custom", base, Some("sk-not-a-real-key"))
        };
        let report = run_probe(&bed.cx(), &d, &s.target(), TestDepth::Catalog)
            .await
            .unwrap();
        let failure = report.failure.clone().unwrap();
        assert_eq!(failure.reason, ReasonCode::Endpoint, "{base}");
        assert!(
            !failure.rolls_back(ProviderGroup::Custom),
            "a refusal keeps the key: {base}"
        );
        assert_eq!(
            report.refusal,
            Some(PolicyViolation::Endpoint(refusal)),
            "{base}"
        );
        assert!(matches!(report.into_result(), Err(HubError::Policy(_))));
    }
    assert_eq!(
        bed.http.request_count(),
        0,
        "no request left for a refused endpoint"
    );
}

#[tokio::test]
async fn probe_a_redirect_into_the_metadata_service_is_refused_mid_flight() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.acme.test/v1/models"),
        Scripted::redirect(302, "http://169.254.169.254/latest/meta-data/"),
    );
    let s = Subject {
        group: ProviderGroup::Custom,
        ..Subject::cloud("custom", "https://api.acme.test/v1", None)
    };
    let report = run_probe(
        &bed.cx(),
        &OpenAiCompatDriver::custom(),
        &s.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert_eq!(
        report.refusal,
        Some(PolicyViolation::Endpoint(EndpointRefusal::LinkLocal))
    );
    assert_eq!(
        bed.http.request_count(),
        1,
        "the metadata address was never requested"
    );
}

#[tokio::test]
async fn probe_a_redirect_chain_over_three_hops_and_a_cross_origin_hop_with_a_key_are_refused() {
    let bed = Bed::new();
    let d = OpenAiCompatDriver::custom();
    let s = Subject {
        group: ProviderGroup::Custom,
        ..Subject::cloud(
            "custom",
            "https://api.acme.test/v1",
            Some("sk-not-a-real-key"),
        )
    };
    bed.http.route(
        Match::prefix("https://api.acme.test/v1/models"),
        Scripted::redirect(302, "https://other.test/v1/models"),
    );
    let report = run_probe(&bed.cx(), &d, &s.target(), TestDepth::Catalog)
        .await
        .unwrap();
    assert_eq!(report.refusal, Some(PolicyViolation::CrossOriginRedirect));
    assert_eq!(
        bed.http.request_count(),
        1,
        "the key never reaches the other origin"
    );

    let bed = Bed::new();
    for n in 0..5 {
        bed.http.route(
            Match::prefix(format!("https://api.acme.test/hop{n}")),
            Scripted::redirect(302, format!("https://api.acme.test/hop{}", n + 1)),
        );
    }
    bed.http.route(
        Match::prefix("https://api.acme.test/v1/models"),
        Scripted::redirect(302, "https://api.acme.test/hop0"),
    );
    let keyless = Subject { key: None, ..s };
    let report = run_probe(&bed.cx(), &d, &keyless.target(), TestDepth::Catalog)
        .await
        .unwrap();
    assert_eq!(
        report.refusal,
        Some(PolicyViolation::TooManyRedirects { max: 3 })
    );
}

#[tokio::test]
async fn probe_dns_rebinding_to_a_private_address_is_refused() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://rebind.test/v1/models"),
        Scripted::json(200, &json!({"data": [{"id": "leak"}]})).resolving_to(vec![
            "93.184.216.34".parse().unwrap(),
            "10.1.2.3".parse().unwrap(),
        ]),
    );
    let s = Subject {
        group: ProviderGroup::Custom,
        ..Subject::cloud("custom", "https://rebind.test/v1", None)
    };
    let report = run_probe(
        &bed.cx(),
        &OpenAiCompatDriver::custom(),
        &s.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert_eq!(
        report.refusal,
        Some(PolicyViolation::Endpoint(EndpointRefusal::PrivateNetwork))
    );
    assert!(report.models.is_empty());
}

#[tokio::test]
async fn probe_a_timeout_is_a_timeout_failure_and_advances_the_hub_clock() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::Timeout,
    );
    let s = Subject::cloud("openai", OPENAI, Some("sk"));
    let report = run_probe(
        &bed.cx(),
        &driver("openai"),
        &s.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    assert_eq!(report.failure.as_ref().unwrap().reason, ReasonCode::Timeout);
    assert_eq!(report.latency, bed.policy.timeout);
    assert!(
        !report.failure.unwrap().rolls_back(ProviderGroup::Cloud),
        "a slow gateway must not delete a good key"
    );
}

#[tokio::test]
async fn probe_a_local_runtime_that_is_down_rolls_back_but_a_cloud_one_does_not() {
    let bed = Bed::new();
    let policy = EndpointPolicy::desktop();
    let cx = DriverContext::new(&bed.http, &policy, &bed.clock, &bed.headers);
    bed.http.route(
        Match::prefix("http://localhost:11434/v1/models"),
        Scripted::ConnectRefused,
    );
    let d = OpenAiCompatDriver::for_descriptor(descriptor("ollama").unwrap().clone());
    let mut s = Subject::cloud("ollama", "http://localhost:11434/v1", None);
    s.group = ProviderGroup::Local;
    s.auth = AuthStyle::None;
    let report = run_probe(&cx, &d, &s.target(), TestDepth::Catalog)
        .await
        .unwrap();
    let failure = report.failure.unwrap();
    assert_eq!(failure.reason, ReasonCode::Endpoint);
    assert!(failure.rolls_back(ProviderGroup::Local));
    assert!(!failure.rolls_back(ProviderGroup::Cloud));
}

#[tokio::test]
async fn probe_the_partial_outage_catalog_fails_while_the_completion_works() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(503, "service unavailable"),
    );
    bed.http.route(
        Match::post("https://api.openai.com/v1/chat/completions"),
        Scripted::json(200, &json!({"choices": []})),
    );
    let mut s = Subject::cloud("openai", OPENAI, Some("sk"));
    s.model = Some(ModelId::parse("gpt-5").unwrap());
    let d = driver("openai");
    let catalog = run_probe(&bed.cx(), &d, &s.target(), TestDepth::Catalog)
        .await
        .unwrap();
    let completion = run_probe(&bed.cx(), &d, &s.target(), TestDepth::Completion)
        .await
        .unwrap();
    assert!(!catalog.ok() && completion.ok() && completion.proves_key);
    // Fed to the health fold, that is degraded, not down.
    let mut snapshot = crate::health::HealthSnapshot::default();
    for report in [&catalog, &completion] {
        let failure = report.failure.as_ref().map(|f| (f.reason, f.status));
        snapshot.record_probe(report.depth, failure, None, report.proves_key, 1);
    }
    assert!(
        matches!(snapshot.health, crate::health::ProviderHealth::Degraded(_)),
        "{:?}",
        snapshot.health
    );
}

#[tokio::test]
async fn probe_a_catalog_too_large_to_read_is_unknown_and_noted() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://a.test/v1/models"),
        Scripted::Oversize { bytes: 999_999_999 },
    );
    let s = Subject {
        group: ProviderGroup::Custom,
        ..Subject::cloud("custom", "https://a.test/v1", None)
    };
    let report = run_probe(
        &bed.cx(),
        &OpenAiCompatDriver::custom(),
        &s.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    let failure = report.failure.clone().unwrap();
    assert_eq!(failure.reason, ReasonCode::Unknown);
    assert!(failure.truncated && !failure.rolls_back(ProviderGroup::Custom));
    assert!(
        !report.notes.contains(&ProbeNote::CatalogTruncated),
        "no models were read, so there is no prefix to warn about; the failure says it"
    );
}

#[tokio::test]
async fn probe_the_report_and_its_target_never_print_the_key() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://api.openai.com/v1/models"),
        Scripted::text(401, "bad key sk-not-a-real-key"),
    );
    let s = Subject::cloud("openai", OPENAI, Some("sk-not-a-real-key"));
    let report = run_probe(
        &bed.cx(),
        &driver("openai"),
        &s.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    let all = format!("{report:?} {:?} {:?}", s.target(), bed.http.requests());
    assert!(!all.contains("sk-not-a-real-key"), "{all}");
    assert!(
        !bed.http.requests()[0]
            .headers
            .iter()
            .any(|(_, v)| v.contains("sk-not"))
    );
    let _ = custom_descriptor();
}

/// A driver whose listing fails with an error that is not a provider's.
#[derive(Debug)]
struct Broken(crate::ProviderDescriptor);

#[async_trait::async_trait]
impl crate::kinds::KindDriver for Broken {
    fn descriptor(&self) -> &crate::ProviderDescriptor {
        &self.0
    }
    async fn list_models(
        &self,
        _cx: &DriverContext<'_>,
        _target: &Target<'_>,
    ) -> Result<crate::catalog::Fetched, HubError> {
        Err(HubError::Conflict)
    }
}

#[tokio::test]
async fn probe_an_error_that_is_not_the_providers_stops_the_probe_instead_of_reporting_a_failure() {
    let bed = Bed::new();
    let s = Subject {
        group: ProviderGroup::Custom,
        ..Subject::cloud("custom", "https://a.test/v1", None)
    };
    let driver = Broken(custom_descriptor());
    let error = run_probe(&bed.cx(), &driver, &s.target(), TestDepth::Catalog)
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Conflict), "{error:?}");
}

#[tokio::test]
async fn probe_an_oversize_error_page_is_not_a_truncated_catalog() {
    let bed = Bed::new();
    bed.http.route(
        Match::prefix("https://a.test/v1/models"),
        Scripted::text(502, "<html>bad gateway</html>".repeat(20_000)),
    );
    let s = Subject {
        group: ProviderGroup::Custom,
        ..Subject::cloud("custom", "https://a.test/v1", None)
    };
    let report = run_probe(
        &bed.cx(),
        &OpenAiCompatDriver::custom(),
        &s.target(),
        TestDepth::Catalog,
    )
    .await
    .unwrap();
    let failure = report.failure.clone().unwrap();
    assert!(
        failure.truncated,
        "the failure body was cut for classification"
    );
    assert_eq!(failure.status, Some(502));
    assert!(
        !report.notes.contains(&ProbeNote::CatalogTruncated),
        "no catalog was read, so none was truncated"
    );
}
