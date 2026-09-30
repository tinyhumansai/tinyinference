//! Tests for the ports, their in-memory defaults, and the redirect loop.

use std::sync::Mutex;

use super::memory::*;
use super::*;
use crate::config::HubConfig;
use crate::descriptor::ProviderRecord;
use crate::error::{PolicyViolation, PortName, ReasonCode};
use crate::ids::{KindId, ScopeKey, Slug};
use crate::policy::{EndpointPolicy, EndpointRefusal, HeaderPolicy};
use crate::secret::Secret;

fn scope(name: &str) -> ScopeKey {
    ScopeKey::new(name)
}

fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

// ---- CredentialStore -------------------------------------------------------

#[tokio::test]
async fn ports_credentials_round_trip_and_delete() {
    let store = MemoryCredentials::new();
    let s = scope("company:acme");
    assert_eq!(store.get(&s, "provider/openai/key").await.unwrap(), None);
    store
        .set(&s, "provider/openai/key", Secret::new("sk-not-a-real-key"))
        .await
        .unwrap();
    let got = store.get(&s, "provider/openai/key").await.unwrap().unwrap();
    assert_eq!(got.expose(), "sk-not-a-real-key");
    store.delete(&s, "provider/openai/key").await.unwrap();
    assert_eq!(store.get(&s, "provider/openai/key").await.unwrap(), None);
    assert!(store.is_empty());
}

#[tokio::test]
async fn ports_deleting_an_absent_slot_succeeds() {
    let store = MemoryCredentials::new();
    store.delete(&scope("a"), "nothing").await.unwrap();
}

#[tokio::test]
async fn ports_credentials_are_partitioned_by_scope() {
    let store = MemoryCredentials::new();
    store
        .set(&scope("a"), "k", Secret::new("one"))
        .await
        .unwrap();
    assert_eq!(store.get(&scope("b"), "k").await.unwrap(), None);
    assert_eq!(store.slots_in(&scope("a")), vec!["k".to_string()]);
    assert!(store.slots_in(&scope("b")).is_empty());
}

#[tokio::test]
async fn ports_an_unreadable_store_is_an_error_and_never_no_key() {
    let store = MemoryCredentials::new();
    let s = scope("a");
    store.set(&s, "k", Secret::new("v")).await.unwrap();
    store.inject(CredentialFault::Read);
    let read = store.get(&s, "k").await;
    assert!(matches!(read, Err(PortError::Unavailable(_))), "{read:?}");
    // Writes still work under a read-only fault, and the reverse.
    store.set(&s, "k2", Secret::new("v")).await.unwrap();
    store.inject(CredentialFault::Write);
    assert!(store.get(&s, "k").await.unwrap().is_some());
    assert!(store.set(&s, "k3", Secret::new("v")).await.is_err());
    assert!(store.delete(&s, "k").await.is_err());
    store.inject(CredentialFault::All);
    assert!(store.get(&s, "k").await.is_err());
    store.heal();
    assert!(store.get(&s, "k").await.unwrap().is_some());
}

#[test]
fn ports_the_credential_store_debug_never_prints_a_value() {
    let store = MemoryCredentials::new();
    futures::executor::block_on(store.set(&scope("a"), "k", Secret::new("sk-not-a-real-key")))
        .unwrap();
    let debug = format!("{store:?}");
    assert!(debug.contains("slots"), "{debug}");
    assert!(!debug.contains("sk-not-a-real-key"), "{debug}");
}

// ---- ConfigStore -----------------------------------------------------------

fn config_with(slug_name: &str) -> HubConfig {
    let mut config = HubConfig::new();
    config.providers.push(ProviderRecord::new(
        "prv_1",
        slug(slug_name),
        "Acme",
        KindId::new("custom"),
        "https://api.acme.test/v1",
    ));
    config
}

#[tokio::test]
async fn ports_config_versions_are_real_compare_and_swap() {
    let store = MemoryConfig::new();
    let s = scope("a");
    assert_eq!(store.load(&s).await.unwrap(), None);
    let v1 = store.save(&s, &HubConfig::new(), None).await.unwrap();
    assert_eq!(v1, Version::new(1));
    let v2 = store
        .save(&s, &config_with("acme"), Some(v1))
        .await
        .unwrap();
    assert_eq!(v2.get(), 2);
    // The loser of a race holds the old version.
    let lost = store.save(&s, &config_with("other"), Some(v1)).await;
    assert_eq!(lost, Err(PortError::Conflict));
    let (loaded, version) = store.load(&s).await.unwrap().unwrap();
    assert_eq!(version, v2);
    assert!(loaded.contains(&slug("acme")) && !loaded.contains(&slug("other")));
}

#[tokio::test]
async fn ports_a_first_save_expecting_a_version_conflicts() {
    let store = MemoryConfig::new();
    let lost = store
        .save(&scope("a"), &HubConfig::new(), Some(Version::new(3)))
        .await;
    assert_eq!(lost, Err(PortError::Conflict));
}

#[tokio::test]
async fn ports_saving_over_an_existing_document_expecting_none_conflicts() {
    let store = MemoryConfig::new();
    let s = scope("a");
    store.save(&s, &HubConfig::new(), None).await.unwrap();
    assert_eq!(
        store.save(&s, &HubConfig::new(), None).await,
        Err(PortError::Conflict)
    );
}

#[tokio::test]
async fn ports_an_injected_conflict_loses_the_save_and_changes_nothing() {
    let store = MemoryConfig::new();
    let s = scope("a");
    let v1 = store.save(&s, &HubConfig::new(), None).await.unwrap();
    store.conflict_next(2);
    for _ in 0..2 {
        let lost = store.save(&s, &config_with("acme"), Some(v1)).await;
        assert_eq!(lost, Err(PortError::Conflict));
    }
    // Two conflicts spent; the third attempt with the right version wins.
    assert!(store.save(&s, &config_with("acme"), Some(v1)).await.is_ok());
}

#[tokio::test]
async fn ports_config_scopes_are_independent() {
    let store = MemoryConfig::new();
    store
        .save(&scope("a"), &config_with("acme"), None)
        .await
        .unwrap();
    assert_eq!(store.load(&scope("b")).await.unwrap(), None);
    assert!(
        store
            .save(&scope("b"), &HubConfig::new(), None)
            .await
            .is_ok()
    );
}

#[tokio::test]
async fn ports_an_outage_is_an_error_not_an_empty_configuration() {
    let store = MemoryConfig::new();
    let s = scope("a");
    store.save(&s, &config_with("acme"), None).await.unwrap();
    store.set_unavailable(true);
    assert!(matches!(
        store.load(&s).await,
        Err(PortError::Unavailable(_))
    ));
    assert!(matches!(
        store.save(&s, &HubConfig::new(), None).await,
        Err(PortError::Unavailable(_))
    ));
    store.set_unavailable(false);
    assert!(store.load(&s).await.unwrap().is_some());
}

#[tokio::test]
async fn ports_a_corrupt_document_is_unreadable_and_never_empty() {
    let store = MemoryConfig::new();
    let s = scope("a");
    store.put_raw(&s, "{not json");
    assert!(matches!(
        store.load(&s).await,
        Err(PortError::Unavailable(_))
    ));
}

#[tokio::test]
async fn ports_a_document_with_a_credential_field_will_not_load() {
    let store = MemoryConfig::new();
    let s = scope("a");
    store.put_raw(&s, r#"{"providers":[],"api_key":"sk-not-a-real-key"}"#);
    let error = store.load(&s).await.unwrap_err();
    assert!(!format!("{error:?}").contains("sk-not-a-real-key"));
}

#[tokio::test]
async fn ports_what_is_stored_never_contains_a_credential() {
    let store = MemoryConfig::new();
    let s = scope("a");
    store.save(&s, &config_with("acme"), None).await.unwrap();
    let raw = store.raw(&s).unwrap();
    assert!(raw.contains("acme"));
    for forbidden in ["sk-", "secret", "password", "bearer"] {
        assert!(!raw.to_ascii_lowercase().contains(forbidden), "{raw}");
    }
}

// ---- HealthStore, events, env, clock ---------------------------------------

#[tokio::test]
async fn ports_health_round_trips_and_forgets() {
    use crate::health::{HealthSnapshot, ProviderHealth};
    let store = MemoryHealth::new();
    let (s, p) = (scope("a"), slug("openai"));
    assert_eq!(store.get(&s, &p).await.unwrap(), None);
    let snapshot = HealthSnapshot {
        health: ProviderHealth::Ok,
        ..HealthSnapshot::default()
    };
    store.put(&s, &p, snapshot.clone()).await.unwrap();
    assert_eq!(store.get(&s, &p).await.unwrap(), Some(snapshot));
    store.forget(&s, &p).await.unwrap();
    assert_eq!(store.get(&s, &p).await.unwrap(), None);
    store.set_unavailable(true);
    assert!(store.get(&s, &p).await.is_err());
    assert!(store.forget(&s, &p).await.is_err());
    assert!(format!("{store:?}").contains("MemoryHealth"));
}

#[test]
fn ports_events_are_delivered_in_order_and_drained() {
    use crate::health::ProviderHealth;
    let sink = MemoryEvents::new();
    let event = |to| HubEvent::HealthChanged {
        scope: scope("a"),
        slug: slug("openai"),
        from: ProviderHealth::Unknown,
        to,
    };
    sink.emit(event(ProviderHealth::Ok));
    sink.emit(event(ProviderHealth::Down(ReasonCode::Auth)));
    assert_eq!(sink.events().len(), 2);
    let drained = sink.drain();
    assert_eq!(drained.len(), 2);
    assert!(sink.events().is_empty());
    NoopEvents.emit(event(ProviderHealth::Ok));
    assert!(format!("{sink:?}").contains("events"));
}

#[test]
fn ports_the_map_env_reads_only_what_it_was_given_and_prints_names_only() {
    let env = MapEnv::new().with("OPENAI_API_KEY", "sk-not-a-real-key");
    assert_eq!(
        env.var("OPENAI_API_KEY").as_deref(),
        Some("sk-not-a-real-key")
    );
    assert_eq!(env.var("ANTHROPIC_API_KEY"), None);
    let debug = format!("{env:?}");
    assert!(debug.contains("OPENAI_API_KEY") && !debug.contains("sk-not-a-real-key"));
}

#[test]
fn ports_the_system_clock_moves_forward() {
    let clock = SystemClock;
    let a = clock.now();
    let b = clock.now();
    assert!(b >= a);
    assert!(
        clock.wall_ms() > 1_700_000_000_000,
        "wall time is past 2023"
    );
}

#[test]
fn ports_a_port_error_becomes_the_right_hub_error() {
    assert!(matches!(
        PortError::Conflict.into_hub(PortName::Config),
        crate::HubError::Conflict
    ));
    let error =
        PortError::unavailable("connection string user:pw@host").into_hub(PortName::Credentials);
    match &error {
        crate::HubError::StoreUnreadable { port, detail } => {
            assert_eq!(*port, PortName::Credentials);
            assert!(detail.expose().contains("host"));
        }
        other => panic!("{other:?}"),
    }
    // The store's own text is log-only.
    assert!(!error.to_string().contains("pw@"));
    assert!(!format!("{error:?}").contains("pw@"));
}

// ---- HubRequest / HubResponse / HttpError ----------------------------------

#[test]
fn ports_a_request_debug_never_prints_a_header_value_or_the_body() {
    let request = HubRequest::post_json(
        "https://api.acme.test/v1/chat/completions?key=abc123",
        &serde_json::json!({"secret": "body-secret"}),
    )
    .with_header("authorization", "Bearer sk-not-a-real-key")
    .with_header("x-acme-key", "sk-other")
    .with_credentialed(true);
    let debug = format!("{request:?}");
    for leak in ["sk-not-a-real-key", "sk-other", "body-secret", "abc123"] {
        assert!(!debug.contains(leak), "{debug}");
    }
    assert!(debug.contains("authorization"), "names are fine: {debug}");
    assert_eq!(
        request.header("Authorization"),
        Some("Bearer sk-not-a-real-key")
    );
    assert_eq!(request.header("missing"), None);
}

#[test]
fn ports_request_builders_set_their_fields() {
    let request = HubRequest::get("https://a.test/")
        .with_timeout(std::time::Duration::from_secs(3))
        .with_body_cap(10)
        .with_credentialed(true);
    assert_eq!(request.method, Method::Get);
    assert_eq!(request.timeout.as_secs(), 3);
    assert_eq!(request.body_cap, 10);
    assert!(request.credentialed && request.body.is_none());
    let post = HubRequest::post_json("https://a.test/", &serde_json::json!({"a": 1}));
    assert_eq!(post.method, Method::Post);
    assert_eq!(post.header("content-type"), Some("application/json"));
    assert_eq!(post.body.as_deref(), Some(br#"{"a":1}"#.as_slice()));
    assert_eq!(Method::Get.to_string(), "GET");
    assert_eq!(Method::Post.to_string(), "POST");
}

#[test]
fn ports_response_helpers() {
    let mut response = HubResponse::new(200, "hello", "https://a.test/");
    response.headers.push(("Retry-After".into(), "7".into()));
    assert!(response.is_success());
    assert_eq!(response.header("retry-after"), Some("7"));
    assert_eq!(response.header_pairs(), vec![("Retry-After", "7")]);
    assert_eq!(response.text(), "hello");
    assert!(!HubResponse::new(404, "", "u").is_success());
    assert!(!HubResponse::new(199, "", "u").is_success());
    assert!(!HubResponse::new(300, "", "u").is_success());
    let debug = format!("{response:?}");
    assert!(
        !debug.contains("hello"),
        "the body is never printed: {debug}"
    );
    assert_eq!(
        HubResponse::new(200, vec![0xff, 0xfe], "u").text(),
        "\u{fffd}\u{fffd}"
    );
}

#[test]
fn ports_a_transport_failure_is_classified_from_its_condition() {
    let reason = |e: HttpError| e.into_hub().reason();
    assert_eq!(reason(HttpError::Timeout), ReasonCode::Timeout);
    assert_eq!(reason(HttpError::ConnectFailed), ReasonCode::Endpoint);
    assert_eq!(
        reason(HttpError::Failed(crate::LogOnly::new("boom".into()))),
        ReasonCode::Unknown
    );
    let policy = HttpError::Policy(PolicyViolation::CrossOriginRedirect);
    assert_eq!(policy.into_hub().reason(), ReasonCode::Policy);
    // A transport's text is log-only.
    let failed = HttpError::Failed(crate::LogOnly::new("Bearer sk-not-a-real-key".into()));
    assert!(!failed.to_string().contains("sk-not"));
    assert!(!format!("{failed:?}").contains("sk-not"));
}

// ---- follow_redirects ------------------------------------------------------

type Answer = (u16, Option<&'static str>);

/// Drives `follow_redirects` over a scripted list of answers, returning what it
/// did and the requests it sent.
async fn drive(
    request: HubRequest,
    policy: &EndpointPolicy,
    answers: &[Answer],
) -> (Result<HubResponse, HttpError>, Vec<HubRequest>) {
    let sent = Mutex::new(Vec::<HubRequest>::new());
    let cursor = Mutex::new(0usize);
    let result = follow_redirects(
        request,
        policy,
        &HeaderPolicy::builtin(),
        &SystemClock,
        |hop| {
            sent.lock().unwrap().push(hop.clone());
            let mut i = cursor.lock().unwrap();
            let (status, location) = answers[(*i).min(answers.len() - 1)];
            *i += 1;
            let mut response = HubResponse::new(status, "", hop.url.clone());
            if let Some(location) = location {
                response.headers.push(("location".into(), location.into()));
            }
            async move { Ok(response) }
        },
    )
    .await;
    let sent = sent.into_inner().unwrap();
    (result, sent)
}

#[tokio::test]
async fn ports_a_plain_response_is_returned_with_its_url() {
    let (result, sent) = drive(
        HubRequest::get("https://api.acme.test/v1/models"),
        &EndpointPolicy::hosted(),
        &[(200, None)],
    )
    .await;
    let response = result.unwrap();
    assert_eq!(response.status, 200);
    assert_eq!(response.url, "https://api.acme.test/v1/models");
    assert_eq!(sent.len(), 1);
}

#[tokio::test]
async fn ports_three_redirects_are_followed_and_the_fourth_is_refused() {
    let policy = EndpointPolicy::hosted();
    let hop = |n: u32| {
        Some(Box::leak(format!("https://api.acme.test/hop{n}").into_boxed_str()) as &'static str)
    };
    let (ok, sent) = drive(
        HubRequest::get("https://api.acme.test/start"),
        &policy,
        &[(302, hop(1)), (302, hop(2)), (302, hop(3)), (200, None)],
    )
    .await;
    assert_eq!(ok.unwrap().url, "https://api.acme.test/hop3");
    assert_eq!(sent.len(), 4);

    let (refused, sent) = drive(
        HubRequest::get("https://api.acme.test/start"),
        &policy,
        &[
            (302, hop(1)),
            (302, hop(2)),
            (302, hop(3)),
            (302, hop(4)),
            (200, None),
        ],
    )
    .await;
    assert_eq!(
        refused.unwrap_err(),
        HttpError::Policy(PolicyViolation::TooManyRedirects { max: 3 })
    );
    assert_eq!(sent.len(), 4, "the fourth hop is never sent");
}

#[tokio::test]
async fn ports_a_redirect_to_a_metadata_address_is_refused_and_never_sent() {
    let (result, sent) = drive(
        HubRequest::get("https://api.acme.test/models"),
        &EndpointPolicy::hosted(),
        &[
            (302, Some("http://169.254.169.254/latest/meta-data/")),
            (200, None),
        ],
    )
    .await;
    assert_eq!(
        result.unwrap_err(),
        HttpError::Policy(PolicyViolation::Endpoint(EndpointRefusal::LinkLocal))
    );
    assert_eq!(sent.len(), 1);
}

#[tokio::test]
async fn ports_a_redirect_to_loopback_is_refused_when_the_policy_has_none() {
    let target = Some("http://127.0.0.1:8080/x");
    let (hosted, _) = drive(
        HubRequest::get("https://api.acme.test/models"),
        &EndpointPolicy::hosted(),
        &[(301, target), (200, None)],
    )
    .await;
    assert!(matches!(hosted, Err(HttpError::Policy(_))));
    let (desktop, sent) = drive(
        HubRequest::get("https://api.acme.test/models"),
        &EndpointPolicy::desktop(),
        &[(301, target), (200, None)],
    )
    .await;
    assert!(desktop.is_ok());
    assert_eq!(sent.len(), 2);
}

#[tokio::test]
async fn ports_a_credentialed_request_never_follows_a_redirect_off_its_origin() {
    let request = HubRequest::get("https://api.acme.test/models")
        .with_header("authorization", "Bearer sk-not-a-real-key")
        .with_credentialed(true);
    let (result, sent) = drive(
        request,
        &EndpointPolicy::hosted(),
        &[(302, Some("https://other.test/models")), (200, None)],
    )
    .await;
    assert_eq!(
        result.unwrap_err(),
        HttpError::Policy(PolicyViolation::CrossOriginRedirect)
    );
    assert_eq!(sent.len(), 1, "the second origin never receives the key");
}

#[tokio::test]
async fn ports_a_credentialed_same_origin_redirect_keeps_the_key() {
    let request = HubRequest::get("https://api.acme.test/models")
        .with_header("authorization", "Bearer sk-not-a-real-key")
        .with_credentialed(true);
    let (result, sent) = drive(
        request,
        &EndpointPolicy::hosted(),
        &[(308, Some("/v2/models")), (200, None)],
    )
    .await;
    assert_eq!(result.unwrap().url, "https://api.acme.test/v2/models");
    assert_eq!(
        sent[1].header("authorization"),
        Some("Bearer sk-not-a-real-key")
    );
}

#[tokio::test]
async fn ports_an_uncredentialed_cross_origin_hop_drops_credential_shaped_headers() {
    let request = HubRequest::get("https://api.acme.test/models")
        .with_header("x-api-key", "leftover")
        .with_header("x-sdk-name", "opencompany")
        .with_header("accept", "application/json");
    let (result, sent) = drive(
        request,
        &EndpointPolicy::hosted(),
        &[(302, Some("https://cdn.other.test/models")), (200, None)],
    )
    .await;
    assert!(result.is_ok());
    assert_eq!(sent[1].header("x-api-key"), None);
    assert_eq!(sent[1].header("x-sdk-name"), None);
    assert_eq!(sent[1].header("accept"), Some("application/json"));
}

#[tokio::test]
async fn ports_302_turns_a_post_into_a_get_and_307_keeps_it() {
    let post = || HubRequest::post_json("https://api.acme.test/chat", &serde_json::json!({"a": 1}));
    let (_, sent) = drive(
        post(),
        &EndpointPolicy::hosted(),
        &[(302, Some("/elsewhere")), (200, None)],
    )
    .await;
    assert_eq!(sent[1].method, Method::Get);
    assert!(sent[1].body.is_none());
    assert_eq!(sent[1].header("content-type"), None);
    let (_, sent) = drive(
        post(),
        &EndpointPolicy::hosted(),
        &[(307, Some("/elsewhere")), (200, None)],
    )
    .await;
    assert_eq!(sent[1].method, Method::Post);
    assert!(sent[1].body.is_some());
}

#[tokio::test]
async fn ports_a_redirect_without_a_location_is_returned_as_the_response_it_is() {
    let (result, sent) = drive(
        HubRequest::get("https://api.acme.test/models"),
        &EndpointPolicy::hosted(),
        &[(302, None)],
    )
    .await;
    assert_eq!(result.unwrap().status, 302);
    assert_eq!(sent.len(), 1);
}

#[tokio::test]
async fn ports_a_credential_over_cleartext_is_refused_before_anything_is_sent() {
    let request = HubRequest::get("http://api.acme.test/models").with_credentialed(true);
    let (result, sent) = drive(request, &EndpointPolicy::hosted(), &[(200, None)]).await;
    assert_eq!(
        result.unwrap_err(),
        HttpError::Policy(PolicyViolation::Endpoint(EndpointRefusal::Cleartext))
    );
    assert!(sent.is_empty());
}

#[tokio::test]
async fn ports_a_send_error_from_the_transport_is_passed_through() {
    let result = follow_redirects(
        HubRequest::get("https://api.acme.test/models"),
        &EndpointPolicy::hosted(),
        &HeaderPolicy::builtin(),
        &SystemClock,
        |_| async { Err(HttpError::Timeout) },
    )
    .await;
    assert_eq!(result.unwrap_err(), HttpError::Timeout);
}

#[tokio::test]
async fn ports_a_configuration_that_cannot_be_read_back_is_not_stored() {
    let store = MemoryConfig::new();
    let mut config = HubConfig::new();
    config.extra.insert(
        "api_key".to_string(),
        serde_json::json!("sk-not-a-real-key"),
    );
    let saved = store.save(&scope("a"), &config, None).await;
    assert!(matches!(saved, Err(PortError::Unavailable(_))), "{saved:?}");
    assert_eq!(store.raw(&scope("a")), None, "nothing was written");
    assert!(format!("{store:?}").contains("MemoryConfig"));
}

mod redirect_props {
    use proptest::prelude::*;

    use super::*;
    use crate::policy::check_endpoint;

    /// Targets a hostile or careless server might name, public and not.
    const TARGETS: &[&str] = &[
        "https://a.test/x",
        "https://b.test/y",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/x",
        "http://127.0.0.1:8080/x",
        "http://[::1]/x",
        "http://[::ffff:169.254.169.254]/x",
        "/relative",
        "ftp://a.test/x",
        "https://user:pw@a.test/x",
    ];

    fn policy_for(pick: u8) -> EndpointPolicy {
        match pick % 3 {
            0 => EndpointPolicy::hosted(),
            1 => EndpointPolicy::desktop(),
            _ => EndpointPolicy::local_only(),
        }
    }

    proptest! {
        /// Invariant 5 of the test plan: every request that leaves the hub, at
        /// every hop, passed the endpoint policy; a chain never runs past the
        /// redirect limit; and a credentialed request never leaves its origin.
        #[test]
        fn ports_prop_every_hop_that_is_sent_passed_the_policy(
            picks in proptest::collection::vec(0..TARGETS.len(), 0..8),
            credentialed in any::<bool>(),
            policy_pick in any::<u8>(),
            max_redirects in 0usize..5,
        ) {
            let policy = policy_for(policy_pick).with_max_redirects(max_redirects);
            let mut request = HubRequest::get("https://a.test/start");
            if credentialed {
                request = request
                    .with_header("authorization", "Bearer sk-not-a-real-key")
                    .with_credentialed(true);
            }
            let sent = Mutex::new(Vec::<HubRequest>::new());
            let cursor = Mutex::new(0usize);
            let result = futures::executor::block_on(follow_redirects(
                request,
                &policy,
                &HeaderPolicy::builtin(),
                &SystemClock,
                |hop| {
                    sent.lock().unwrap().push(hop.clone());
                    let mut i = cursor.lock().unwrap();
                    let response = match picks.get(*i) {
                        Some(pick) => {
                            let mut r = HubResponse::new(302, "", hop.url.clone());
                            r.headers.push(("location".into(), TARGETS[*pick].into()));
                            r
                        }
                        None => HubResponse::new(200, "", hop.url.clone()),
                    };
                    *i += 1;
                    async move { Ok(response) }
                },
            ));
            let sent = sent.into_inner().unwrap();
            prop_assert!(sent.len() <= max_redirects + 1, "{} requests for a limit of {max_redirects}", sent.len());
            for hop in &sent {
                prop_assert!(
                    check_endpoint(&hop.url, &policy).is_ok(),
                    "a request was sent to {} which the policy refuses",
                    hop.url
                );
                if credentialed {
                    prop_assert_eq!(
                        hop.url.split('/').nth(2),
                        Some("a.test"),
                        "a credentialed request left its origin: {}", hop.url
                    );
                }
            }
            if let Ok(response) = result {
                prop_assert!(response.status == 200 || sent.len() <= max_redirects + 1);
            }
        }
    }
}

#[tokio::test]
async fn ports_the_timeout_is_a_total_for_the_whole_redirect_chain() {
    use std::time::Duration;

    use crate::testkit::FakeClock;

    let clock = FakeClock::new();
    let policy = EndpointPolicy::hosted();
    let request = HubRequest::get("https://a.test/0").with_timeout(Duration::from_secs(10));
    let seen = Mutex::new(Vec::<Duration>::new());
    let hops = Mutex::new(0u32);
    let result = follow_redirects(request, &policy, &HeaderPolicy::builtin(), &clock, |hop| {
        seen.lock().unwrap().push(hop.timeout);
        // Every hop takes four fake seconds and redirects again.
        clock.advance(Duration::from_secs(4));
        let mut n = hops.lock().unwrap();
        *n += 1;
        let mut response = HubResponse::new(302, "", hop.url.clone());
        response
            .headers
            .push(("location".into(), format!("https://a.test/{n}")));
        async move { Ok(response) }
    })
    .await;
    assert_eq!(
        result.unwrap_err(),
        HttpError::Timeout,
        "the third hop found nothing left"
    );
    let seen = seen.into_inner().unwrap();
    assert_eq!(
        seen,
        [
            Duration::from_secs(10),
            Duration::from_secs(6),
            Duration::from_secs(2)
        ],
        "each hop gets what the chain has left"
    );
}
