//! Tests for the simulation kit itself: a test double that lies is worse than
//! none.

use std::net::{IpAddr, Ipv4Addr};
use std::time::Duration;

use super::*;
use crate::error::PolicyViolation;
use crate::policy::{EndpointPolicy, EndpointRefusal};
use crate::ports::{Clock, Http, HttpError, HubRequest, Method};
use crate::secret::Secret;

fn http() -> (ScriptedHttp, FakeClock) {
    let clock = FakeClock::new();
    (ScriptedHttp::new(clock.clone()), clock)
}

fn hosted() -> EndpointPolicy {
    EndpointPolicy::hosted()
}

#[test]
fn testkit_the_fake_clock_moves_only_when_told_and_clones_share_time() {
    let clock = FakeClock::new();
    let twin = clock.clone();
    let (t0, w0) = (clock.now(), clock.wall_ms());
    assert_eq!(w0, FakeClock::START_WALL_MS);
    assert_eq!(clock.now(), t0, "time does not pass by itself");
    twin.advance(Duration::from_secs(90));
    assert_eq!(clock.now() - t0, Duration::from_secs(90));
    assert_eq!(clock.wall_ms() - w0, 90_000);
    assert_eq!(clock.elapsed(), Duration::from_secs(90));
    assert!(format!("{clock:?}").contains("90"));
    assert_eq!(FakeClock::default().wall_ms(), FakeClock::START_WALL_MS);
}

#[tokio::test]
async fn testkit_the_newest_rule_wins_so_a_provider_can_be_flipped() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/models"),
        Scripted::text(200, "old"),
    );
    http.route(
        Match::get("https://a.test/models"),
        Scripted::text(503, "down"),
    );
    let response = http
        .send(HubRequest::get("https://a.test/models"), &hosted())
        .await
        .unwrap();
    assert_eq!((response.status, response.text().as_str()), (503, "down"));
}

#[tokio::test]
async fn testkit_a_sequence_answers_in_order_then_runs_dry() {
    let (http, _) = http();
    let id = http.route(
        Match::get("https://a.test/x"),
        Script::Sequence(vec![Scripted::text(500, "a"), Scripted::text(200, "b")]),
    );
    let policy = hosted();
    let send = || http.send(HubRequest::get("https://a.test/x"), &policy);
    assert_eq!(send().await.unwrap().status, 500);
    assert_eq!(send().await.unwrap().status, 200);
    assert_eq!(http.hits(id), 2);
    assert_eq!(http.hits(9999), 0);
}

#[tokio::test]
#[should_panic(expected = "the script for GET https://a.test/x is exhausted")]
async fn testkit_an_exhausted_sequence_panics_rather_than_repeating() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/x"),
        Script::Sequence(vec![Scripted::text(200, "once")]),
    );
    let policy = hosted();
    let send = || http.send(HubRequest::get("https://a.test/x"), &policy);
    send().await.unwrap();
    let _ = send().await;
}

#[tokio::test]
#[should_panic(expected = "no rule answers POST https://a.test/other")]
async fn testkit_an_unscripted_request_panics_and_names_it_redacted() {
    let (http, _) = http();
    let request = HubRequest::post_json("https://a.test/other", &serde_json::json!({}));
    let _ = http.send(request, &hosted()).await;
}

#[tokio::test]
async fn testkit_matching_by_method_prefix_and_header() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/models").with_header("authorization", "Bearer good"),
        Scripted::text(200, "ok"),
    );
    http.route(
        Match::get("https://a.test/models").with_header("authorization", "Bearer bad"),
        Scripted::text(401, "no"),
    );
    http.route(
        Match::get("https://a.test/models").without_header("authorization"),
        Scripted::text(200, "public"),
    );
    http.route(
        Match::prefix("https://a.test/pfx").with_header_present("x-token"),
        Scripted::text(202, "prefixed"),
    );
    let policy = hosted();
    let send = |request: HubRequest| http.send(request, &policy);
    let get = |auth: Option<&str>| {
        let request = HubRequest::get("https://a.test/models");
        match auth {
            Some(value) => request
                .with_header("authorization", value)
                .with_credentialed(true),
            None => request,
        }
    };
    assert_eq!(send(get(Some("Bearer good"))).await.unwrap().text(), "ok");
    assert_eq!(send(get(Some("Bearer bad"))).await.unwrap().status, 401);
    assert_eq!(send(get(None)).await.unwrap().text(), "public");
    let prefixed = HubRequest::get("https://a.test/pfx/anything?x=1").with_header("X-Token", "t");
    assert_eq!(send(prefixed).await.unwrap().status, 202);
}

#[tokio::test]
#[should_panic(expected = "no rule answers")]
async fn testkit_a_method_mismatch_does_not_match() {
    let (http, _) = http();
    http.route(Match::post("https://a.test/x"), Scripted::text(200, ""));
    let _ = http
        .send(HubRequest::get("https://a.test/x"), &hosted())
        .await;
}

#[tokio::test]
async fn testkit_the_request_log_redacts_credentials_but_can_say_which_key_was_sent() {
    let (http, clock) = http();
    http.ok_json("https://a.test/models", &serde_json::json!({"data": []}));
    let key = Secret::new("sk-not-a-real-key");
    let other = Secret::new("sk-another-fake");
    let request = HubRequest::get("https://a.test/models")
        .with_header("authorization", format!("Bearer {}", key.expose()))
        .with_header("x-acme-key", "custom-fake-value")
        .with_header("accept", "application/json")
        .with_credentialed(true);
    http.send(request, &hosted()).await.unwrap();
    clock.advance(Duration::from_secs(5));
    let log = http.requests();
    assert_eq!(log.len(), 1);
    let entry = &log[0];
    assert_eq!(entry.header("authorization"), Some("<redacted>"));
    assert_eq!(
        entry.header("x-acme-key"),
        Some("<redacted>"),
        "custom key headers too"
    );
    assert_eq!(entry.header("accept"), Some("application/json"));
    assert!(entry.credentialed && entry.carried(&key) && !entry.carried(&other));
    assert!(!entry.carried(&Secret::new("")));
    assert_eq!(entry.at_wall_ms, FakeClock::START_WALL_MS);
    assert_eq!(entry.method, Method::Get);
    let debug = format!("{entry:?}{http:?}");
    assert!(!debug.contains("sk-not-a-real-key") && !debug.contains("custom-fake-value"));
}

#[tokio::test]
async fn testkit_the_log_scrubs_a_request_body_and_a_credential_query() {
    let (http, _) = http();
    http.route(Match::prefix("https://a.test/"), Scripted::text(200, "{}"));
    let body = serde_json::json!({"note": "call https://u:hunter2@x.test/ now"});
    http.send(
        HubRequest::post_json("https://a.test/chat", &body),
        &hosted(),
    )
    .await
    .unwrap();
    let entry = &http.requests()[0];
    let logged = entry.body.clone().unwrap();
    assert!(!logged.contains("hunter2"), "{logged}");
    assert_eq!(http.request_count(), 1);
}

#[tokio::test]
async fn testkit_latency_advances_fake_time_and_a_slow_answer_times_out() {
    let (http, clock) = http();
    http.route(
        Match::get("https://a.test/fast"),
        Scripted::text(200, "ok").after(Duration::from_secs(2)),
    );
    http.route(
        Match::get("https://a.test/slow"),
        Scripted::text(200, "late").after(Duration::from_secs(30)),
    );
    let start = clock.now();
    let response = http
        .send(HubRequest::get("https://a.test/fast"), &hosted())
        .await
        .unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(clock.now() - start, Duration::from_secs(2));
    let slow = HubRequest::get("https://a.test/slow").with_timeout(Duration::from_secs(5));
    assert_eq!(
        http.send(slow, &hosted()).await.unwrap_err(),
        HttpError::Timeout
    );
    assert_eq!(clock.now() - start, Duration::from_secs(7));
}

#[tokio::test]
async fn testkit_timeout_refused_and_dns_failures() {
    let (http, clock) = http();
    http.route(Match::get("https://a.test/t"), Scripted::Timeout);
    http.route(Match::get("https://a.test/r"), Scripted::ConnectRefused);
    http.route(Match::get("https://a.test/d"), Scripted::DnsFail);
    let start = clock.now();
    let t = http
        .send(HubRequest::get("https://a.test/t"), &hosted())
        .await;
    assert_eq!(t.unwrap_err(), HttpError::Timeout);
    assert_eq!(
        clock.now() - start,
        Duration::from_secs(10),
        "the whole timeout passes"
    );
    for path in ["r", "d"] {
        let url = format!("https://a.test/{path}");
        let e = http.send(HubRequest::get(url), &hosted()).await;
        assert_eq!(e.unwrap_err(), HttpError::ConnectFailed);
    }
}

#[tokio::test]
async fn testkit_a_body_past_the_cap_is_cut_and_marked_truncated() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/big"),
        Scripted::Oversize { bytes: 1_000_000 },
    );
    http.route(
        Match::get("https://a.test/small"),
        Scripted::Oversize { bytes: 10 },
    );
    http.route(
        Match::get("https://a.test/status"),
        Scripted::text(500, "x".repeat(100)),
    );
    let big = http
        .send(
            HubRequest::get("https://a.test/big").with_body_cap(64),
            &hosted(),
        )
        .await
        .unwrap();
    assert!(big.truncated && big.body.len() == 64);
    let small = http
        .send(
            HubRequest::get("https://a.test/small").with_body_cap(64),
            &hosted(),
        )
        .await
        .unwrap();
    assert!(!small.truncated && small.body.len() == 10);
    let cut = http
        .send(
            HubRequest::get("https://a.test/status").with_body_cap(8),
            &hosted(),
        )
        .await
        .unwrap();
    assert!(cut.truncated && cut.body.len() == 8 && cut.status == 500);
}

#[tokio::test]
async fn testkit_malformed_bytes_are_a_200_with_that_body() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/m"),
        Scripted::Malformed(vec![0xff, b'{']),
    );
    let response = http
        .send(HubRequest::get("https://a.test/m"), &hosted())
        .await
        .unwrap();
    assert_eq!(
        (response.status, response.body.clone()),
        (200, vec![0xff, b'{'])
    );
}

#[tokio::test]
async fn testkit_a_slow_stream_delivers_or_times_out_by_its_total_time() {
    let (http, clock) = http();
    let chunks = vec![b"ab".to_vec(), b"cd".to_vec(), b"ef".to_vec()];
    http.route(
        Match::get("https://a.test/s"),
        Scripted::SlowStream {
            chunks,
            gap: Duration::from_secs(2),
        },
    );
    let start = clock.now();
    let ok = http
        .send(
            HubRequest::get("https://a.test/s").with_timeout(Duration::from_secs(10)),
            &hosted(),
        )
        .await
        .unwrap();
    assert_eq!(ok.text(), "abcdef");
    assert_eq!(clock.now() - start, Duration::from_secs(6));
    let late = http
        .send(
            HubRequest::get("https://a.test/s").with_timeout(Duration::from_secs(5)),
            &hosted(),
        )
        .await;
    assert_eq!(late.unwrap_err(), HttpError::Timeout);
}

#[tokio::test]
async fn testkit_the_headers_of_a_status_answer_come_through() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/h"),
        Scripted::with_headers(429, &[("retry-after", "30")], "slow down"),
    );
    let response = http
        .send(HubRequest::get("https://a.test/h"), &hosted())
        .await
        .unwrap();
    assert_eq!(response.header("Retry-After"), Some("30"));
}

#[tokio::test]
async fn testkit_dns_rebinding_is_refused_when_a_resolved_address_is_private() {
    let (http, _) = http();
    let private = IpAddr::V4(Ipv4Addr::new(10, 0, 0, 5));
    let public = IpAddr::V4(Ipv4Addr::new(93, 184, 216, 34));
    http.route(
        Match::get("https://rebind.test/models"),
        Scripted::text(200, "secret").resolving_to(vec![public, private]),
    );
    http.route(
        Match::get("https://fine.test/models"),
        Scripted::text(200, "ok").resolving_to(vec![public]),
    );
    let refused = http
        .send(HubRequest::get("https://rebind.test/models"), &hosted())
        .await;
    assert_eq!(
        refused.unwrap_err(),
        HttpError::Policy(PolicyViolation::Endpoint(EndpointRefusal::PrivateNetwork))
    );
    assert_eq!(http.refused().len(), 1);
    assert!(
        http.send(HubRequest::get("https://fine.test/models"), &hosted())
            .await
            .is_ok()
    );
    // A LAN policy accepts the same answer.
    let lan = hosted().with_private(true);
    assert!(
        http.send(HubRequest::get("https://rebind.test/models"), &lan)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn testkit_a_host_that_resolves_to_nothing_is_a_connect_failure() {
    let (http, _) = http();
    http.route(
        Match::get("https://void.test/"),
        Scripted::text(200, "").resolving_to(Vec::new()),
    );
    let e = http
        .send(HubRequest::get("https://void.test/"), &hosted())
        .await;
    assert_eq!(e.unwrap_err(), HttpError::ConnectFailed);
}

#[tokio::test]
async fn testkit_redirects_follow_the_policy_and_every_hop_is_recorded() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/start"),
        Scripted::redirect(302, "/mid"),
    );
    http.route(
        Match::get("https://a.test/mid"),
        Scripted::redirect(301, "https://b.test/end"),
    );
    http.route(
        Match::get("https://b.test/end"),
        Scripted::text(200, "landed"),
    );
    let response = http
        .send(HubRequest::get("https://a.test/start"), &hosted())
        .await
        .unwrap();
    assert_eq!(
        (response.status, response.url.as_str()),
        (200, "https://b.test/end")
    );
    let urls: Vec<String> = http.requests().into_iter().map(|r| r.url).collect();
    assert_eq!(
        urls,
        [
            "https://a.test/start",
            "https://a.test/mid",
            "https://b.test/end"
        ]
    );
}

#[tokio::test]
async fn testkit_a_redirect_into_the_metadata_service_is_refused_and_recorded_as_refused() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/start"),
        Scripted::redirect(302, "http://169.254.169.254/latest/meta-data/"),
    );
    let refused = http
        .send(HubRequest::get("https://a.test/start"), &hosted())
        .await;
    assert_eq!(
        refused.unwrap_err(),
        HttpError::Policy(PolicyViolation::Endpoint(EndpointRefusal::LinkLocal))
    );
    assert_eq!(
        http.request_count(),
        1,
        "the metadata URL was never requested"
    );
    assert_eq!(http.refused().len(), 1);
}

#[tokio::test]
async fn testkit_clear_rules_keeps_the_log() {
    let (http, _) = http();
    http.ok_json("https://a.test/x", &serde_json::json!({}));
    http.send(HubRequest::get("https://a.test/x"), &hosted())
        .await
        .unwrap();
    http.clear_rules();
    assert_eq!(http.request_count(), 1);
}

#[tokio::test]
#[should_panic(expected = "no rule answers")]
async fn testkit_a_rule_that_requires_a_header_does_not_answer_a_request_without_it() {
    let (http, _) = http();
    http.route(
        Match::get("https://a.test/x").with_header_present("x-token"),
        Scripted::text(200, "ok"),
    );
    http.route(
        Match::get("https://a.test/y").with_header("x-token", "expected"),
        Scripted::text(200, "ok"),
    );
    let policy = hosted();
    // The wrong value does not match either.
    let wrong = HubRequest::get("https://a.test/y").with_header("x-token", "other");
    let unmatched = std::panic::AssertUnwindSafe(http.send(wrong, &policy));
    let caught = futures::FutureExt::catch_unwind(unmatched).await;
    assert!(caught.is_err(), "a different value must not match");
    // And a missing header does not match: this one panics the test.
    let _ = http
        .send(HubRequest::get("https://a.test/x"), &policy)
        .await;
}

// ---- the contract suite must fail a driver that breaks the contract --------

mod contract_mutations {
    use async_trait::async_trait;

    use crate::catalog::Fetched;
    use crate::catalogue::descriptor;
    use crate::error::{HubError, ProviderFailure};
    use crate::kinds::{DriverContext, KindDriver, OpenAiCompatDriver, Target};
    use crate::testkit::{ContractFixture, run_contract};

    #[derive(Clone, Copy, Debug)]
    enum Fault {
        LeakKeyInUrl,
        EmptyList,
        Reordered,
        EverythingIsAuth,
        Truncates,
        NoKeyCheckDeclared,
        SwallowSignedOut,
    }

    /// A driver that delegates to the OpenAI-compatible one and breaks one rule.
    #[derive(Debug)]
    struct Faulty {
        inner: Box<dyn KindDriver>,
        fault: Fault,
    }

    fn faulty(kind: &str, fault: Fault) -> Faulty {
        let d = descriptor(kind).unwrap().clone();
        let inner: Box<dyn KindDriver> = if kind == "tinyhumans" {
            Box::new(crate::kinds::ManagedDriver::paged(d))
        } else {
            Box::new(OpenAiCompatDriver::for_descriptor(d))
        };
        Faulty { inner, fault }
    }

    #[async_trait]
    impl KindDriver for Faulty {
        fn descriptor(&self) -> &crate::ProviderDescriptor {
            self.inner.descriptor()
        }

        fn classify(&self, status: u16, headers: &[(&str, &str)], body: &str) -> ProviderFailure {
            match self.fault {
                Fault::EverythingIsAuth => {
                    ProviderFailure::new(crate::ReasonCode::Auth, crate::Retry::Never)
                        .with_status(status)
                }
                _ => self.inner.classify(status, headers, body),
            }
        }

        async fn list_models(
            &self,
            cx: &DriverContext<'_>,
            target: &Target<'_>,
        ) -> Result<Fetched, HubError> {
            match self.fault {
                Fault::LeakKeyInUrl => {
                    // Puts the key in the query string, where logs will find it.
                    let leaky = format!("{}?key={}", target.base(), target.key().unwrap_or(""));
                    let t = Target {
                        base_url: &leaky,
                        ..*target
                    };
                    self.inner.list_models(cx, &t).await
                }
                Fault::EmptyList => Ok(Fetched::new(Vec::new())),
                Fault::Reordered => {
                    let mut fetched = self.inner.list_models(cx, target).await?;
                    fetched.models.sort_by(|a, b| a.id.cmp(&b.id));
                    Ok(fetched)
                }
                Fault::Truncates => {
                    let mut fetched = self.inner.list_models(cx, target).await?;
                    fetched.truncated = true;
                    Ok(fetched)
                }
                Fault::SwallowSignedOut if target.key().is_none() => Ok(Fetched::new(Vec::new())),
                _ => self.inner.list_models(cx, target).await,
            }
        }

        async fn key_check(
            &self,
            cx: &DriverContext<'_>,
            target: &Target<'_>,
        ) -> Result<(), HubError> {
            self.inner.key_check(cx, target).await
        }
    }

    async fn run(kind: &str, fault: Fault, fixture: ContractFixture) {
        run_contract(&faulty(kind, fault), &fixture).await;
    }

    #[tokio::test]
    #[should_panic(expected = "list_models failed: Policy(CredentialInEndpoint)")]
    async fn testkit_a_driver_that_puts_the_key_in_a_url_is_stopped_by_the_transports_own_policy() {
        // Defence in depth: the driver leaks, and the endpoint policy every
        // transport applies refuses the URL before anything is sent.
        let f = ContractFixture::openai_shaped("groq", "https://api.groq.com/openai/v1");
        run("groq", Fault::LeakKeyInUrl, f).await;
    }

    #[tokio::test]
    #[should_panic(expected = "provider order is kept")]
    async fn testkit_the_contract_fails_a_driver_that_reorders_the_listing() {
        let f = ContractFixture::openai_shaped("groq", "https://api.groq.com/openai/v1");
        run("groq", Fault::Reordered, f).await;
    }

    #[tokio::test]
    #[should_panic(expected = "provider order is kept")]
    async fn testkit_the_contract_fails_a_driver_that_loses_the_models() {
        let f = ContractFixture::openai_shaped("groq", "https://api.groq.com/openai/v1");
        run("groq", Fault::EmptyList, f).await;
    }

    #[tokio::test]
    #[should_panic(expected = "a small listing is not truncated")]
    async fn testkit_the_contract_fails_a_driver_that_flags_a_small_listing_truncated() {
        let f = ContractFixture::openai_shaped("groq", "https://api.groq.com/openai/v1");
        run("groq", Fault::Truncates, f).await;
    }

    #[tokio::test]
    #[should_panic(expected = "ping model")]
    async fn testkit_the_contract_fails_a_driver_that_reads_everything_as_a_bad_key() {
        let f = ContractFixture::openai_shaped("groq", "https://api.groq.com/openai/v1");
        run("groq", Fault::EverythingIsAuth, f).await;
    }

    #[tokio::test]
    #[should_panic(expected = "declares KeyOnly, so the fixture needs key_check_url")]
    async fn testkit_the_contract_asks_a_key_only_kind_for_its_check_url() {
        let f = ContractFixture::openai_shaped("openrouter", "https://openrouter.ai/api/v1");
        run("openrouter", Fault::NoKeyCheckDeclared, f).await;
    }

    #[tokio::test]
    #[should_panic(expected = "signed out is typed")]
    async fn testkit_the_contract_fails_a_managed_driver_that_answers_an_empty_list_when_signed_out()
     {
        let d = descriptor("tinyhumans").unwrap();
        let f = ContractFixture::for_builtin(d);
        run("tinyhumans", Fault::SwallowSignedOut, f).await;
    }

    #[tokio::test]
    #[should_panic(expected = "[groq] list_models failed: Conflict")]
    async fn testkit_the_contract_names_the_kind_and_check_when_a_driver_fails_for_its_own_reasons()
    {
        // A driver whose 401 is not a provider failure at all.
        #[derive(Debug)]
        struct Odd(OpenAiCompatDriver);
        #[async_trait]
        impl KindDriver for Odd {
            fn descriptor(&self) -> &crate::ProviderDescriptor {
                self.0.descriptor()
            }
            async fn list_models(
                &self,
                _cx: &DriverContext<'_>,
                _t: &Target<'_>,
            ) -> Result<Fetched, HubError> {
                Err(HubError::Conflict)
            }
        }
        let f = ContractFixture::openai_shaped("groq", "https://api.groq.com/openai/v1");
        let driver = Odd(OpenAiCompatDriver::for_descriptor(
            descriptor("groq").unwrap().clone(),
        ));
        run_contract(&driver, &f).await;
    }

    #[test]
    fn testkit_a_fixture_prints_without_the_key_or_the_body_builder() {
        let f = ContractFixture::openai_shaped("groq", "https://api.groq.com/openai/v1");
        let debug = format!("{f:?}");
        assert!(
            debug.contains("groq") && !debug.contains("sk-not-a-real-key"),
            "{debug}"
        );
    }
}

#[test]
#[should_panic(expected = "`Not A Slug` is not a valid provider slug")]
fn testkit_a_fixture_with_a_bad_slug_says_which() {
    let _ = crate::testkit::ContractFixture::openai_shaped("Not A Slug", "https://a.test/v1");
}

#[test]
fn testkit_sub_millisecond_advances_accumulate_instead_of_vanishing() {
    let clock = FakeClock::new();
    let start = clock.now();
    for _ in 0..1000 {
        clock.advance(Duration::from_micros(400));
    }
    assert_eq!(clock.now() - start, Duration::from_millis(400));
    assert_eq!(clock.elapsed(), Duration::from_millis(400));
    assert_eq!(clock.wall_ms() - FakeClock::START_WALL_MS, 400);
}

#[tokio::test]
#[should_panic(expected = "is exhausted")]
async fn testkit_an_exhausted_sequence_does_not_fall_through_to_an_older_broad_rule() {
    let (http, _) = http();
    http.route(
        Match::prefix("https://a.test/"),
        Scripted::text(200, "broad"),
    );
    http.route(
        Match::get("https://a.test/x"),
        Script::Sequence(vec![Scripted::text(503, "once")]),
    );
    let policy = hosted();
    assert_eq!(
        http.send(HubRequest::get("https://a.test/x"), &policy)
            .await
            .unwrap()
            .status,
        503
    );
    // The third request must not be answered by the broad 200.
    let _ = http
        .send(HubRequest::get("https://a.test/x"), &policy)
        .await;
}

#[tokio::test]
async fn testkit_a_request_refused_after_resolving_is_not_logged_as_sent_and_names_its_own_hop() {
    let (http, _) = http();
    let key = Secret::new("sk-not-a-real-key");
    http.route(
        Match::get("https://a.test/start"),
        Scripted::redirect(302, "https://rebind.test/next"),
    );
    http.route(
        Match::get("https://rebind.test/next"),
        Scripted::text(200, "internal").resolving_to(vec![IpAddr::V4(Ipv4Addr::new(10, 0, 0, 9))]),
    );
    let request = HubRequest::get("https://a.test/start")
        .with_header("x-note", "n")
        .with_credentialed(false);
    let refused = http.send(request, &hosted()).await;
    assert!(matches!(refused, Err(HttpError::Policy(_))));
    assert_eq!(http.request_count(), 1, "only the first hop was sent");
    assert!(http.requests().iter().all(|r| !r.carried(&key)));
    let refusals = http.refused();
    assert_eq!(refusals.len(), 1);
    assert_eq!(
        refusals[0].0, "https://rebind.test/next",
        "the refusal names the hop that hit it"
    );
}
