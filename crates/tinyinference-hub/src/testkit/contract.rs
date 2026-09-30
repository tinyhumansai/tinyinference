//! The kind contract suite: one set of checks every [`KindDriver`] must pass,
//! run through the same trait against a scripted transport.
//!
//! A host that adds a kind with `DriverRegistry::register` runs
//! [`run_contract`] against it in its own tests. The checks assert the
//! behaviours the hub's operations depend on and that OpenCompany learned the
//! hard way: a rejected key is `auth` and nothing else is; a listing that is
//! not a listing is `unknown`, never `auth`; a body past its cap is refused, not
//! truncated into invalid JSON; the credential is presented in the provider's
//! style, never in a URL, and never in a log; a signed-out managed provider is
//! typed, not an empty list.
//!
//! The checks panic on the first violation, naming the kind, so they read as
//! ordinary test failures.

use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};

use crate::error::{HubError, Operation, ReasonCode, Retry};
use crate::ids::{KindId, ModelId, Slug};
use crate::kinds::{DriverContext, KindDriver, Target};
use crate::policy::{EndpointPolicy, HeaderPolicy};
use crate::probe::run_probe;
use crate::secret::Secret;
use crate::taxonomy::{AuthStyle, Protocol, ProviderGroup, TestDepth, Transport};

use super::{FakeClock, Match, Scripted, ScriptedHttp};

/// Builds a successful listing body for the given ids.
pub type ListingBody = Arc<dyn Fn(&[&str]) -> Value + Send + Sync>;

/// How a kind's server behaves, so the suite can script it.
#[non_exhaustive]
pub struct ContractFixture {
    /// The provider's slug.
    pub slug: Slug,
    /// The endpoint the target uses.
    pub base_url: String,
    /// The policy the deployment runs under.
    pub policy: EndpointPolicy,
    /// The group.
    pub group: ProviderGroup,
    /// How the credential is presented.
    pub auth: AuthStyle,
    /// The credential the suite presents, if the kind takes one.
    pub key: Option<Secret>,
    /// A model id for completion pings.
    pub model: ModelId,
    /// URL prefixes the listing is read from. The **first** answers the happy
    /// path; every one receives the failure scripts (a driver may fall back
    /// from a native listing to a compatible one).
    pub listing_urls: Vec<String>,
    /// Builds a successful listing body for the given ids.
    pub listing_body: ListingBody,
    /// Where the key-only check goes, for a kind that supports that depth.
    pub key_check_url: Option<String>,
}

impl ContractFixture {
    /// The fixture for a built-in catalogue kind: where its listing lives, the
    /// shape it answers in, the policy it runs under.
    ///
    /// The managed kind is placed at OpenCompany's proxy path with the paged
    /// envelope (the hub hardcodes no backend URL); local runtimes run under the
    /// desktop policy, with the native listings their drivers fall back to.
    ///
    /// # Panics
    ///
    /// Never for a catalogue descriptor: its slug is valid.
    pub fn for_builtin(descriptor: &crate::descriptor::ProviderDescriptor) -> Self {
        let slug = descriptor.kind.as_str();
        match descriptor.group {
            ProviderGroup::Managed => {
                let base = "https://api.example.test/agent-integrations/openrouter";
                let mut f = Self::openai_shaped(slug, base);
                f.group = ProviderGroup::Managed;
                f.listing_body = Arc::new(|ids| {
                    json!({"success": true, "data": {"object": "list", "total": ids.len(),
                        "data": ids.iter().map(|i| json!({"id": i})).collect::<Vec<_>>()}})
                });
                f
            }
            ProviderGroup::Local => {
                let base = match descriptor.default_endpoint {
                    Some(preset) => crate::endpoint::normalize_local_endpoint(preset)
                        .unwrap_or_else(|| preset.to_string()),
                    None => "http://127.0.0.1:8000/v1".to_string(),
                };
                let mut f = Self::openai_shaped(slug, &base);
                f.group = ProviderGroup::Local;
                f.policy = EndpointPolicy::desktop();
                f.auth = descriptor.auth.clone();
                f.key = descriptor
                    .auth
                    .needs_credential()
                    .then(|| Secret::new("sk-not-a-real-key"));
                let origin = base.trim_end_matches("/v1").to_string();
                match descriptor.local_runtime {
                    Some(crate::taxonomy::LocalRuntime::LmStudio) => {
                        f.listing_urls =
                            vec![format!("{origin}/api/v0/models"), format!("{base}/models")];
                        f.listing_body = Arc::new(|ids| {
                            json!({"object": "list", "data": ids.iter()
                                .map(|i| json!({"id": i, "type": "llm", "state": "loaded"})).collect::<Vec<_>>()})
                        });
                    }
                    Some(crate::taxonomy::LocalRuntime::Ollama) => {
                        f.listing_urls.push(format!("{origin}/api/tags"));
                    }
                    _ => {}
                }
                f
            }
            ProviderGroup::Cli => {
                let mut f = Self::openai_shaped(slug, "https://cli.invalid");
                f.group = ProviderGroup::Cli;
                f.auth = AuthStyle::None;
                f.key = None;
                f
            }
            _ => {
                let base = descriptor
                    .default_endpoint
                    .unwrap_or("https://api.acme.test/v1");
                let mut f = Self::openai_shaped(slug, base);
                f.auth = descriptor.auth.clone();
                if descriptor.slug() == "anthropic" {
                    f.listing_urls =
                        vec![format!("{}/models?limit=1000", base.trim_end_matches('/'))];
                }
                if descriptor.has_quirk(crate::descriptor::Quirk::KeyCheckEndpoint) {
                    f.key_check_url = Some(format!("{}/key", base.trim_end_matches('/')));
                }
                f
            }
        }
    }

    /// A fixture for an OpenAI-shaped listing at `{base}/models`.
    ///
    /// # Panics
    ///
    /// When `slug` is not a valid [`Slug`]; the message names it.
    pub fn openai_shaped(slug: &str, base_url: &str) -> Self {
        Self {
            slug: Slug::parse(slug)
                .unwrap_or_else(|e| panic!("`{slug}` is not a valid provider slug: {e}")),
            base_url: base_url.to_string(),
            policy: EndpointPolicy::hosted(),
            group: ProviderGroup::Cloud,
            auth: AuthStyle::Bearer,
            key: Some(Secret::new("sk-not-a-real-key")),
            model: ModelId::parse("contract-model").unwrap_or_else(|e| panic!("{e}")),
            listing_urls: vec![format!("{}/models", base_url.trim_end_matches('/'))],
            listing_body: Arc::new(
                |ids| json!({"object": "list", "data": ids.iter().map(|i| json!({"id": i})).collect::<Vec<_>>()}),
            ),
            key_check_url: None,
        }
    }
}

impl std::fmt::Debug for ContractFixture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ContractFixture")
            .field("slug", &self.slug)
            .field("base_url", &self.base_url)
            .field("group", &self.group)
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

struct Bed {
    http: ScriptedHttp,
    clock: FakeClock,
    headers: HeaderPolicy,
}

impl Bed {
    fn new() -> Self {
        let clock = FakeClock::new();
        Self {
            http: ScriptedHttp::new(clock.clone()),
            clock,
            headers: HeaderPolicy::builtin(),
        }
    }
}

fn target<'a>(f: &'a ContractFixture, kind: &'a KindId, model: bool) -> Target<'a> {
    Target {
        slug: &f.slug,
        kind,
        group: f.group,
        base_url: &f.base_url,
        auth: &f.auth,
        credential: f.key.as_ref(),
        model: model.then_some(&f.model),
    }
}

fn script_all(bed: &Bed, f: &ContractFixture, scripted: &Scripted) {
    for url in &f.listing_urls {
        bed.http.route(Match::prefix(url.clone()), scripted.clone());
    }
}

fn provider_failure(
    kind: &str,
    what: &str,
    result: Result<impl std::fmt::Debug, HubError>,
) -> crate::ProviderFailure {
    match result {
        Err(HubError::Provider(failure)) => failure,
        other => panic!("[{kind}] {what}: expected a provider failure, got {other:?}"),
    }
}

/// Whether an operation came back as the typed `Unsupported`.
fn unsupported<T>(result: &Result<T, HubError>) -> bool {
    matches!(result, Err(HubError::Unsupported { .. }))
}

/// Whether an operation came back as the typed `SignedOut`.
fn signed_out<T>(result: &Result<T, HubError>) -> bool {
    matches!(result, Err(HubError::SignedOut { .. }))
}

fn ping_path(protocol: Protocol) -> &'static str {
    match protocol {
        Protocol::AnthropicMessages => "/messages",
        _ => "/chat/completions",
    }
}

/// Runs the contract for `driver` against `fixture`.
///
/// # Panics
///
/// On the first violated check, with a message naming the kind and the check.
pub async fn run_contract(driver: &dyn KindDriver, f: &ContractFixture) {
    let descriptor = driver.descriptor();
    let name = descriptor.kind.as_str().to_string();
    let kind = descriptor.kind.clone();
    let (n, kind_ref) = (name.as_str(), &kind);

    // A CLI kind has no HTTP surface: every HTTP-shaped operation is typed.
    if descriptor.transport == Transport::Subprocess {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        let t = target(f, kind_ref, true);
        for depth in [
            TestDepth::KeyOnly,
            TestDepth::Catalog,
            TestDepth::Completion,
        ] {
            assert!(
                !descriptor.supports_depth(depth),
                "[{n}] a CLI kind supports no test depth"
            );
        }
        let listed = driver.list_models(&cx, &t).await;
        assert!(
            unsupported(&listed)
                && matches!(
                    &listed,
                    Err(HubError::Unsupported {
                        op: Operation::ListModels,
                        ..
                    })
                ),
            "[{n}] list_models is Unsupported"
        );
        assert!(
            unsupported(&driver.key_check(&cx, &t).await),
            "[{n}] key_check is Unsupported"
        );
        assert!(
            unsupported(&driver.completion_ping(&cx, &t, &f.model).await),
            "[{n}] completion_ping is Unsupported"
        );
        assert_eq!(
            bed.http.request_count(),
            0,
            "[{n}] a CLI kind never touches HTTP"
        );
        return;
    }

    let credentialed = f.key.is_some() && f.auth.needs_credential();

    // C1: a listing reads in the provider's order, presents the key in the
    // provider's style, and never puts it in a URL or a log.
    {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        let body = (f.listing_body)(&["model-b", "model-a", "model-c"]);
        bed.http.route(
            Match::prefix(f.listing_urls[0].clone()),
            Scripted::json(200, &body),
        );
        let fetched = driver
            .list_models(&cx, &target(f, kind_ref, false))
            .await
            .unwrap_or_else(|e| panic!("[{n}] list_models failed: {e:?}"));
        let ids: Vec<&str> = fetched.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(
            ids,
            ["model-b", "model-a", "model-c"],
            "[{n}] provider order is kept"
        );
        assert!(!fetched.truncated, "[{n}] a small listing is not truncated");
        let log = bed.http.requests();
        assert!(!log.is_empty(), "[{n}] the listing made a request");
        if let Some(key) = &f.key {
            for entry in &log {
                assert!(
                    !entry.url.contains(key.expose()),
                    "[{n}] the key is never in a URL"
                );
                assert_eq!(
                    entry.credentialed, credentialed,
                    "[{n}] the credentialed flag follows the auth style"
                );
                assert_eq!(
                    entry.carried(key),
                    credentialed,
                    "[{n}] the key is presented exactly when the style sends one"
                );
                assert!(
                    entry.headers.iter().all(|(_, v)| !v.contains(key.expose())),
                    "[{n}] the log never shows the key"
                );
            }
        }
    }

    // C2: an empty listing is success, not an error (a runtime with nothing
    // pulled is healthy).
    {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        bed.http.route(
            Match::prefix(f.listing_urls[0].clone()),
            Scripted::json(200, &(f.listing_body)(&[])),
        );
        let fetched = driver.list_models(&cx, &target(f, kind_ref, false)).await;
        assert!(
            fetched.is_ok_and(|x| x.models.is_empty()),
            "[{n}] an empty listing is an empty list"
        );
    }

    // C3: failure classes.
    for (what, status, headers, body, echo, check) in [
        (
            "401",
            401u16,
            vec![],
            "Incorrect API key provided",
            "Incorrect API key",
            ReasonCode::Auth,
        ),
        (
            "429",
            429,
            vec![("retry-after", "3")],
            r#"{"error":{"message":"slow down"}}"#,
            "slow down",
            ReasonCode::RateLimited,
        ),
    ] {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        script_all(&bed, f, &Scripted::with_headers(status, &headers, body));
        let failure = provider_failure(
            n,
            what,
            driver.list_models(&cx, &target(f, kind_ref, false)).await,
        );
        assert_eq!(failure.reason, check, "[{n}] {what}");
        assert_eq!(
            failure.status,
            Some(status),
            "[{n}] {what} keeps its status"
        );
        if check == ReasonCode::RateLimited {
            assert_eq!(
                failure.retry,
                Retry::Later(Some(Duration::from_secs(3))),
                "[{n}] the provider's delay is kept"
            );
        }
        assert_eq!(
            failure.reason.destroys_credential(),
            check == ReasonCode::Auth,
            "[{n}] only auth destroys a credential"
        );
        let shown = format!("{failure} {failure:?}");
        assert!(
            !shown.contains(echo),
            "[{n}] raw text never reaches Display/Debug"
        );
    }
    {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        script_all(&bed, f, &Scripted::text(500, "internal server error"));
        let failure = provider_failure(
            n,
            "500",
            driver.list_models(&cx, &target(f, kind_ref, false)).await,
        );
        assert_ne!(
            failure.reason,
            ReasonCode::Auth,
            "[{n}] a 500 is never a bad key"
        );
        assert!(
            matches!(failure.retry, Retry::Later(_)),
            "[{n}] a 500 may be retried later"
        );
    }
    {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        script_all(&bed, f, &Scripted::text(404, "not found"));
        let failure = provider_failure(
            n,
            "404",
            driver.list_models(&cx, &target(f, kind_ref, false)).await,
        );
        assert_eq!(failure.status, Some(404), "[{n}] 404 keeps its status");
        assert!(
            !failure.reason.destroys_credential(),
            "[{n}] a 404 is not a bad key"
        );
    }

    // C4: transport failures are classified by condition.
    for (what, scripted, expect) in [
        ("timeout", Scripted::Timeout, ReasonCode::Timeout),
        ("refused", Scripted::ConnectRefused, ReasonCode::Endpoint),
    ] {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        script_all(&bed, f, &scripted);
        let failure = provider_failure(
            n,
            what,
            driver.list_models(&cx, &target(f, kind_ref, false)).await,
        );
        assert_eq!(failure.reason, expect, "[{n}] {what}");
        assert!(
            !failure.reason.destroys_credential(),
            "[{n}] {what} must not delete a good key"
        );
    }

    // C5: a body that is not a listing is unknown, never auth; a body past the
    // cap is refused as truncated.
    {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        script_all(&bed, f, &Scripted::text(200, "<html>welcome</html>"));
        let failure = provider_failure(
            n,
            "html",
            driver.list_models(&cx, &target(f, kind_ref, false)).await,
        );
        assert_eq!(failure.reason, ReasonCode::Unknown, "[{n}] html is unknown");
    }
    {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        script_all(&bed, f, &Scripted::Oversize { bytes: 200_000_000 });
        let failure = provider_failure(
            n,
            "oversize",
            driver.list_models(&cx, &target(f, kind_ref, false)).await,
        );
        assert!(
            failure.truncated && failure.reason == ReasonCode::Unknown,
            "[{n}] oversize is unknown+truncated"
        );
    }

    // C6: a managed provider with no credential is signed out, not empty.
    if f.group == ProviderGroup::Managed {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        let mut anonymous = target(f, kind_ref, true);
        anonymous.credential = None;
        let listed = driver.list_models(&cx, &anonymous).await;
        assert!(signed_out(&listed), "[{n}] signed out is typed");
        let pinged = driver.completion_ping(&cx, &anonymous, &f.model).await;
        assert!(signed_out(&pinged), "[{n}] a ping is signed out too");
        assert_eq!(
            bed.http.request_count(),
            0,
            "[{n}] nothing is sent signed out"
        );
    }

    // C7: depths. What is declared is what works; what is not is typed.
    {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        let t = target(f, kind_ref, true);
        if descriptor.supports_depth(TestDepth::KeyOnly) {
            let url = f.key_check_url.clone().unwrap_or_else(|| {
                panic!("[{n}] declares KeyOnly, so the fixture needs key_check_url")
            });
            bed.http
                .route(Match::get(url.clone()), Scripted::json(200, &json!({})));
            driver
                .key_check(&cx, &t)
                .await
                .unwrap_or_else(|e| panic!("[{n}] key_check failed: {e:?}"));
            bed.http
                .route(Match::get(url), Scripted::text(401, "bad key"));
            let failure = provider_failure(n, "key_check 401", driver.key_check(&cx, &t).await);
            assert_eq!(
                failure.reason,
                ReasonCode::Auth,
                "[{n}] a rejected key fails the key check"
            );
        } else {
            let probed = run_probe(&cx, driver, &t, TestDepth::KeyOnly).await;
            assert!(
                matches!(
                    &probed,
                    Err(HubError::Unsupported {
                        op: Operation::Test(TestDepth::KeyOnly),
                        ..
                    })
                ),
                "[{n}] an undeclared depth is typed Unsupported"
            );
            assert!(
                unsupported(&driver.key_check(&cx, &t).await),
                "[{n}] key_check is Unsupported"
            );
        }
    }

    // C8: the probe end to end, at every declared depth, and a refused endpoint
    // sends nothing.
    let declared = [TestDepth::Catalog, TestDepth::Completion]
        .into_iter()
        .filter(|depth| descriptor.supports_depth(*depth));
    for depth in declared {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        bed.http.route(
            Match::prefix(f.listing_urls[0].clone()),
            Scripted::json(200, &(f.listing_body)(&["m"])),
        );
        let ping = format!(
            "{}{}",
            f.base_url.trim_end_matches('/'),
            ping_path(descriptor.protocol)
        );
        bed.http.route(
            Match::post(ping),
            Scripted::json(200, &json!({"choices": [], "content": []})),
        );
        let report = run_probe(&cx, driver, &target(f, kind_ref, true), depth)
            .await
            .unwrap_or_else(|e| panic!("[{n}] {depth} probe could not run: {e:?}"));
        assert!(
            report.ok(),
            "[{n}] {depth} probe passes: {:?}",
            report.failure
        );
        if depth == TestDepth::Catalog {
            assert_eq!(
                report.model_count(),
                Some(1),
                "[{n}] the catalog probe read the model"
            );
        }
    }
    {
        let bed = Bed::new();
        let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
        let mut refused = target(f, kind_ref, true);
        refused.base_url = "http://169.254.169.254/latest/meta-data";
        let report = run_probe(&cx, driver, &refused, TestDepth::Catalog)
            .await
            .unwrap_or_else(|e| panic!("[{n}] the probe could not run: {e:?}"));
        assert_eq!(
            report.failure.as_ref().map(|x| x.reason),
            Some(ReasonCode::Endpoint),
            "[{n}] the metadata address is refused"
        );
        assert!(report.refusal.is_some());
        assert_eq!(
            bed.http.request_count(),
            0,
            "[{n}] the metadata address is never requested"
        );
    }

    // C9: completion pings, where declared.
    if descriptor.supports_depth(TestDepth::Completion) {
        let ping = format!(
            "{}{}",
            f.base_url.trim_end_matches('/'),
            ping_path(descriptor.protocol)
        );
        for (what, scripted, expect) in [
            (
                "401",
                Scripted::text(401, "invalid api key"),
                Some(ReasonCode::Auth),
            ),
            (
                "model",
                Scripted::text(
                    404,
                    r#"{"error":{"message":"The model `contract-model` does not exist"}}"#,
                ),
                Some(ReasonCode::Model),
            ),
            (
                "429",
                Scripted::text(429, "rate limit exceeded"),
                Some(ReasonCode::RateLimited),
            ),
            ("ok", Scripted::json(200, &json!({})), None),
        ] {
            let bed = Bed::new();
            let cx = DriverContext::new(&bed.http, &f.policy, &bed.clock, &bed.headers);
            bed.http.route(Match::post(ping.clone()), scripted);
            let result = driver
                .completion_ping(&cx, &target(f, kind_ref, true), &f.model)
                .await;
            match expect {
                None => assert!(result.is_ok(), "[{n}] ping ok: {result:?}"),
                Some(reason) => {
                    let failure = provider_failure(n, what, result);
                    assert_eq!(failure.reason, reason, "[{n}] ping {what}");
                }
            }
            if what == "ok" {
                let sent = &bed.http.requests()[0];
                let body: Value = serde_json::from_str(sent.body.as_deref().unwrap_or("{}"))
                    .unwrap_or(Value::Null);
                assert_eq!(
                    body["model"], "contract-model",
                    "[{n}] the ping names the model"
                );
                let (field, other) =
                    if descriptor.has_quirk(crate::descriptor::Quirk::MaxCompletionTokens) {
                        ("max_completion_tokens", "max_tokens")
                    } else {
                        ("max_tokens", "max_completion_tokens")
                    };
                assert!(
                    body[field].as_u64().is_some_and(|t| t <= 16),
                    "[{n}] a ping is one small completion in `{field}`"
                );
                assert!(body.get(other).is_none(), "[{n}] and never in `{other}`");
            }
        }
    }

    // C10: nothing prints the key.
    if let Some(key) = &f.key {
        let debug = format!("{driver:?} {:?}", target(f, kind_ref, true));
        assert!(
            !debug.contains(key.expose()),
            "[{n}] Debug never prints the key"
        );
    }
}
