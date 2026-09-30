//! Tests for `ReqwestHttp` through its two seams. No socket is opened, no name
//! is resolved and no port is bound: the resolver and the executor are fakes.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use super::*;
use crate::error::PolicyViolation;

fn canned(status: u16, headers: &[(&str, &str)], body: &[u8]) -> reqwest::Response {
    let mut builder = http::Response::builder().status(status);
    for (name, value) in headers {
        builder = builder.header(*name, *value);
    }
    reqwest::Response::from(builder.body(body.to_vec()).unwrap())
}

#[derive(Debug, Default)]
struct FakeResolver {
    answers: Mutex<HashMap<String, VecDeque<Vec<IpAddr>>>>,
    asked: Mutex<Vec<(String, u16)>>,
}

impl FakeResolver {
    fn answer(&self, host: &str, ips: &[&str]) {
        self.answers
            .lock()
            .unwrap()
            .entry(host.to_string())
            .or_default()
            .push_back(ips.iter().map(|ip| ip.parse().unwrap()).collect());
    }
}

#[async_trait]
impl Resolver for FakeResolver {
    async fn resolve(
        &self,
        host: &str,
        port: u16,
        _timeout: Duration,
    ) -> Result<Vec<IpAddr>, HttpError> {
        self.asked.lock().unwrap().push((host.to_string(), port));
        let mut answers = self.answers.lock().unwrap();
        let queue = answers.get_mut(host).ok_or(HttpError::ConnectFailed)?;
        // The last answer repeats, so a name that never changes needs one entry.
        if queue.len() > 1 {
            Ok(queue.pop_front().unwrap())
        } else {
            queue.front().cloned().ok_or(HttpError::ConnectFailed)
        }
    }
}

type Seen = (Pin, String, String, Vec<(String, String)>, Option<Vec<u8>>);

/// A resolver that takes fake time to answer.
#[derive(Debug)]
struct SlowResolver {
    clock: crate::testkit::FakeClock,
    cost: Duration,
}

#[async_trait]
impl Resolver for SlowResolver {
    async fn resolve(&self, _: &str, _: u16, _: Duration) -> Result<Vec<IpAddr>, HttpError> {
        self.clock.advance(self.cost);
        Ok(vec!["93.184.216.34".parse().unwrap()])
    }
}

#[derive(Default)]
struct FakeExecutor {
    script: Mutex<VecDeque<Result<reqwest::Response, HttpError>>>,
    seen: Mutex<Vec<Seen>>,
    timeouts: Mutex<Vec<Duration>>,
}

impl Debug for FakeExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("FakeExecutor")
    }
}

impl FakeExecutor {
    fn push(&self, response: Result<reqwest::Response, HttpError>) {
        self.script.lock().unwrap().push_back(response);
    }
}

#[async_trait]
impl Executor for FakeExecutor {
    async fn execute(
        &self,
        pin: &Pin,
        request: reqwest::Request,
        timeout: Duration,
    ) -> Result<reqwest::Response, HttpError> {
        self.timeouts.lock().unwrap().push(timeout);
        self.seen.lock().unwrap().push((
            pin.clone(),
            request.method().to_string(),
            request.url().to_string(),
            request
                .headers()
                .iter()
                .map(|(n, v)| (n.as_str().to_string(), v.to_str().unwrap().to_string()))
                .collect(),
            request
                .body()
                .and_then(|b| b.as_bytes())
                .map(<[u8]>::to_vec),
        ));
        self.script
            .lock()
            .unwrap()
            .pop_front()
            .expect("an unscripted request is a test bug")
    }
}

fn transport() -> (ReqwestHttp, Arc<FakeResolver>, Arc<FakeExecutor>) {
    let resolver = Arc::new(FakeResolver::default());
    let executor = Arc::new(FakeExecutor::default());
    (
        ReqwestHttp::with_parts(resolver.clone(), executor.clone()),
        resolver,
        executor,
    )
}

#[tokio::test]
async fn reqwest_a_request_is_pinned_to_the_addresses_that_were_checked() {
    let (http, resolver, executor) = transport();
    resolver.answer("api.example.test", &["93.184.216.34", "2606:2800:220:1::1"]);
    executor.push(Ok(canned(
        200,
        &[("x-request-id", "r1"), ("content-type", "application/json")],
        b"{\"ok\":true}",
    )));
    let request = HubRequest::get("https://api.example.test/v1/models")
        .with_header("authorization", "Bearer sk-not-a-real-key")
        .with_credentialed(true);
    let response = http.send(request, &EndpointPolicy::hosted()).await.unwrap();
    assert_eq!(
        (response.status, response.text().as_str()),
        (200, "{\"ok\":true}")
    );
    assert_eq!(response.header("x-request-id"), Some("r1"));
    assert_eq!(response.url, "https://api.example.test/v1/models");
    assert!(!response.truncated);
    let seen = executor.seen.lock().unwrap();
    let (pin, method, url, headers, body) = &seen[0];
    assert_eq!(pin.domain.as_deref(), Some("api.example.test"));
    assert_eq!(
        pin.addrs,
        [
            "93.184.216.34:443".parse::<SocketAddr>().unwrap(),
            "[2606:2800:220:1::1]:443".parse().unwrap()
        ]
    );
    assert_eq!(
        (method.as_str(), url.as_str(), body.as_deref()),
        ("GET", "https://api.example.test/v1/models", None)
    );
    assert!(
        headers
            .iter()
            .any(|(n, v)| n == "authorization" && v.starts_with("Bearer "))
    );
    assert_eq!(
        resolver.asked.lock().unwrap().as_slice(),
        [("api.example.test".to_string(), 443)]
    );
}

#[tokio::test]
async fn reqwest_a_name_that_resolves_to_a_private_address_is_refused_before_anything_is_sent() {
    for bad in [
        "10.0.0.5",
        "192.168.1.1",
        "169.254.169.254",
        "127.0.0.1",
        "::1",
        "fe80::1",
        "100.64.0.1",
        "0.0.0.0",
    ] {
        let (http, resolver, executor) = transport();
        resolver.answer("evil.example.test", &[bad]);
        let error = http
            .send(
                HubRequest::get("https://evil.example.test/v1/models"),
                &EndpointPolicy::hosted(),
            )
            .await
            .unwrap_err();
        assert!(
            matches!(error, HttpError::Policy(PolicyViolation::Endpoint(_))),
            "{bad}: {error:?}"
        );
        assert!(
            executor.seen.lock().unwrap().is_empty(),
            "{bad}: nothing was sent"
        );
    }
    // A mixed answer is refused whole: the public half is not connected to.
    let (http, resolver, executor) = transport();
    resolver.answer("mixed.example.test", &["93.184.216.34", "10.0.0.1"]);
    assert!(
        http.send(
            HubRequest::get("https://mixed.example.test/"),
            &EndpointPolicy::hosted()
        )
        .await
        .is_err()
    );
    assert!(executor.seen.lock().unwrap().is_empty());
}

#[tokio::test]
async fn sim_dns_rebinding_the_second_answer_is_checked_too() {
    // Hop 1 answers public. The redirect target's name resolves private at the
    // moment of connection: refused, though its literal text looks harmless.
    let (http, resolver, executor) = transport();
    resolver.answer("a.example.test", &["93.184.216.34"]);
    resolver.answer("b.example.test", &["10.9.9.9"]);
    executor.push(Ok(canned(
        302,
        &[("location", "https://b.example.test/x")],
        b"",
    )));
    let error = http
        .send(
            HubRequest::get("https://a.example.test/"),
            &EndpointPolicy::hosted(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HttpError::Policy(_)), "{error:?}");
    assert_eq!(
        executor.seen.lock().unwrap().len(),
        1,
        "only the first hop was sent"
    );
    // The same name asked twice can answer differently: each hop resolves afresh.
    let (http, resolver, executor) = transport();
    resolver.answer("c.example.test", &["93.184.216.34"]);
    resolver.answer("c.example.test", &["10.0.0.1"]);
    executor.push(Ok(canned(200, &[], b"first")));
    executor.push(Ok(canned(200, &[], b"second")));
    assert!(
        http.send(
            HubRequest::get("https://c.example.test/"),
            &EndpointPolicy::hosted()
        )
        .await
        .is_ok()
    );
    assert!(
        http.send(
            HubRequest::get("https://c.example.test/"),
            &EndpointPolicy::hosted()
        )
        .await
        .is_err()
    );
}

#[tokio::test]
async fn reqwest_ip_literals_need_no_resolver_and_the_policy_still_applies() {
    let (http, resolver, executor) = transport();
    executor.push(Ok(canned(200, &[], b"ok")));
    http.send(
        HubRequest::get("http://127.0.0.1:11434/api/version"),
        &EndpointPolicy::desktop(),
    )
    .await
    .unwrap();
    assert!(resolver.asked.lock().unwrap().is_empty());
    {
        let seen = executor.seen.lock().unwrap();
        assert_eq!(seen[0].0.domain, None);
        assert_eq!(
            seen[0].0.addrs,
            ["127.0.0.1:11434".parse::<SocketAddr>().unwrap()]
        );
    }
    let error = http
        .send(
            HubRequest::get("http://[::1]:8080/"),
            &EndpointPolicy::hosted(),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, HttpError::Policy(_)));
    let v6 = {
        executor.push(Ok(canned(200, &[], b"v6")));
        http.send(
            HubRequest::get("http://[::1]:8080/"),
            &EndpointPolicy::desktop(),
        )
        .await
    };
    assert!(v6.is_ok());
}

#[tokio::test]
async fn reqwest_redirects_are_followed_with_the_policy_on_every_hop() {
    let (http, resolver, executor) = transport();
    resolver.answer("h0.example.test", &["93.184.216.34"]);
    for n in 1..=4 {
        resolver.answer(&format!("h{n}.example.test"), &["93.184.216.35"]);
    }
    for n in 1..=4 {
        executor.push(Ok(canned(
            302,
            &[("location", &format!("https://h{n}.example.test/"))],
            b"",
        )));
    }
    let error = http
        .send(
            HubRequest::get("https://h0.example.test/"),
            &EndpointPolicy::hosted(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(
            error,
            HttpError::Policy(PolicyViolation::TooManyRedirects { max: 3 })
        ),
        "{error:?}"
    );
    // Three redirects then an answer is fine.
    let (http, resolver, executor) = transport();
    resolver.answer("h0.example.test", &["93.184.216.34"]);
    for n in 1..=3 {
        resolver.answer(&format!("h{n}.example.test"), &["93.184.216.35"]);
    }
    for n in 1..=3 {
        executor.push(Ok(canned(
            307,
            &[("location", &format!("https://h{n}.example.test/"))],
            b"",
        )));
    }
    executor.push(Ok(canned(200, &[], b"done")));
    let response = http
        .send(
            HubRequest::get("https://h0.example.test/"),
            &EndpointPolicy::hosted(),
        )
        .await
        .unwrap();
    assert_eq!(response.text(), "done");
    assert_eq!(response.url, "https://h3.example.test/");
}

#[tokio::test]
async fn reqwest_a_credential_never_crosses_an_origin_and_an_uncredentialed_read_is_stripped() {
    let (http, resolver, executor) = transport();
    resolver.answer("a.example.test", &["93.184.216.34"]);
    resolver.answer("b.example.test", &["93.184.216.35"]);
    executor.push(Ok(canned(
        302,
        &[("location", "https://b.example.test/")],
        b"",
    )));
    let credentialed = HubRequest::get("https://a.example.test/")
        .with_header("x-api-key", "sk-not-a-real-key")
        .with_credentialed(true);
    let error = http
        .send(credentialed, &EndpointPolicy::hosted())
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        HttpError::Policy(PolicyViolation::CrossOriginRedirect)
    ));
    // Not credentialed, but carrying a stray credential header: stripped on the hop.
    executor.push(Ok(canned(
        301,
        &[("location", "https://b.example.test/")],
        b"",
    )));
    executor.push(Ok(canned(200, &[], b"ok")));
    let stray = HubRequest::get("https://a.example.test/").with_header("x-api-key", "sk-stray");
    http.send(stray, &EndpointPolicy::hosted()).await.unwrap();
    let seen = executor.seen.lock().unwrap();
    let last = seen.last().unwrap();
    assert!(!last.3.iter().any(|(n, _)| n == "x-api-key"));
}

#[tokio::test]
async fn reqwest_a_302_turns_a_post_into_a_get_and_a_307_keeps_the_body() {
    let (http, resolver, executor) = transport();
    resolver.answer("a.example.test", &["93.184.216.34"]);
    executor.push(Ok(canned(302, &[("location", "/next")], b"")));
    executor.push(Ok(canned(200, &[], b"ok")));
    let post = HubRequest::post_json("https://a.example.test/chat", &serde_json::json!({"a": 1}));
    http.send(post.clone(), &EndpointPolicy::hosted())
        .await
        .unwrap();
    {
        let seen = executor.seen.lock().unwrap();
        assert_eq!((seen[0].1.as_str(), seen[0].4.is_some()), ("POST", true));
        assert_eq!((seen[1].1.as_str(), seen[1].4.is_some()), ("GET", false));
        assert_eq!(seen[1].2, "https://a.example.test/next");
    }
    executor.push(Ok(canned(307, &[("location", "/again")], b"")));
    executor.push(Ok(canned(200, &[], b"ok")));
    http.send(post, &EndpointPolicy::hosted()).await.unwrap();
    let seen = executor.seen.lock().unwrap();
    assert_eq!(
        (seen[3].1.as_str(), seen[3].4.as_deref()),
        ("POST", Some(&b"{\"a\":1}"[..]))
    );
}

#[tokio::test]
async fn reqwest_the_body_is_capped_and_says_so() {
    let (http, resolver, executor) = transport();
    resolver.answer("a.example.test", &["93.184.216.34"]);
    executor.push(Ok(canned(200, &[], &[b'x'; 100])));
    let response = http
        .send(
            HubRequest::get("https://a.example.test/").with_body_cap(10),
            &EndpointPolicy::hosted(),
        )
        .await
        .unwrap();
    assert_eq!((response.body.len(), response.truncated), (10, true));
    executor.push(Ok(canned(200, &[], &[b'x'; 10])));
    let exact = http
        .send(
            HubRequest::get("https://a.example.test/").with_body_cap(10),
            &EndpointPolicy::hosted(),
        )
        .await
        .unwrap();
    assert_eq!(
        (exact.body.len(), exact.truncated),
        (10, false),
        "a body exactly at the cap is whole"
    );
}

#[tokio::test]
async fn reqwest_a_chunked_body_is_capped_across_chunks_and_a_broken_stream_is_failed() {
    let chunks = vec![
        Ok::<_, std::io::Error>(vec![b'a'; 6]),
        Ok(vec![b'b'; 6]),
        Ok(vec![b'c'; 6]),
    ];
    let body = reqwest::Body::wrap_stream(futures::stream::iter(chunks));
    let response = reqwest::Response::from(http::Response::new(body));
    let read = read_response(response, 10, "https://x.test/")
        .await
        .unwrap();
    assert_eq!(read.body, b"aaaaaabbbb");
    assert!(read.truncated);
    let broken = vec![
        Ok::<_, std::io::Error>(vec![b'a'; 3]),
        Err(std::io::Error::other("reset by peer with sk-secret-value")),
    ];
    let body = reqwest::Body::wrap_stream(futures::stream::iter(broken));
    let response = reqwest::Response::from(http::Response::new(body));
    let error = read_response(response, 100, "https://x.test/")
        .await
        .unwrap_err();
    assert!(matches!(error, HttpError::Failed(_)));
    assert!(
        !format!("{error:?}").contains("sk-secret-value"),
        "the detail is log-only"
    );
    // A header that is not text is skipped, not fatal.
    let mut raw = http::Response::new(Vec::<u8>::new());
    raw.headers_mut().insert(
        "x-bin",
        http::HeaderValue::from_bytes(&[0xff, 0xfe]).unwrap(),
    );
    raw.headers_mut()
        .insert("x-ok", http::HeaderValue::from_static("1"));
    let read = read_response(reqwest::Response::from(raw), 10, "https://x.test/")
        .await
        .unwrap();
    assert_eq!(
        (read.header("x-ok"), read.header("x-bin")),
        (Some("1"), None)
    );
}

#[test]
fn reqwest_error_mapping_and_pin_rules() {
    assert_eq!(
        map_error_parts(true, true, "x"),
        HttpError::Timeout,
        "a timeout wins"
    );
    assert_eq!(map_error_parts(false, true, "x"), HttpError::ConnectFailed);
    let other = map_error_parts(false, false, "GET https://u:pw@host/?key=abc failed");
    let HttpError::Failed(detail) = other else {
        panic!()
    };
    assert!(
        !detail.expose().contains("pw@") && !detail.expose().contains("key=abc"),
        "{}",
        detail.expose()
    );
    assert_eq!(
        pin_addresses(&[], 443, &EndpointPolicy::hosted()),
        Err(HttpError::ConnectFailed)
    );
    let addrs = pin_addresses(
        &["93.184.216.34".parse().unwrap()],
        8443,
        &EndpointPolicy::hosted(),
    )
    .unwrap();
    assert_eq!(addrs, ["93.184.216.34:8443".parse::<SocketAddr>().unwrap()]);
}

#[test]
fn reqwest_requests_and_clients_build_without_a_network() {
    let request = HubRequest::post_json("https://a.test/x", &serde_json::json!({"k": "v"}))
        .with_header("x-a", "1");
    let built = build_request(&request, Url::parse(&request.url).unwrap()).unwrap();
    assert_eq!(built.method(), reqwest::Method::POST);
    assert_eq!(built.headers().get("x-a").unwrap(), "1");
    assert!(built.body().is_some());
    let bad_name = HubRequest::get("https://a.test/").with_header("bad name", "v");
    assert!(build_request(&bad_name, Url::parse("https://a.test/").unwrap()).is_err());
    let bad_value = HubRequest::get("https://a.test/").with_header("x-a", "line\nbreak");
    assert!(build_request(&bad_value, Url::parse("https://a.test/").unwrap()).is_err());
    let pinned = Pin {
        domain: Some("a.test".into()),
        addrs: vec!["93.184.216.34:443".parse().unwrap()],
    };
    assert!(build_client(&pinned, Duration::from_secs(1)).is_ok());
    assert!(
        build_client(
            &Pin {
                domain: None,
                addrs: vec![]
            },
            Duration::from_secs(1)
        )
        .is_ok()
    );
}

#[tokio::test]
async fn reqwest_transport_failures_pass_through_typed() {
    let (http, resolver, executor) = transport();
    resolver.answer("a.example.test", &["93.184.216.34"]);
    executor.push(Err(HttpError::Timeout));
    executor.push(Err(HttpError::ConnectFailed));
    for want in [HttpError::Timeout, HttpError::ConnectFailed] {
        let got = http
            .send(
                HubRequest::get("https://a.example.test/"),
                &EndpointPolicy::hosted(),
            )
            .await
            .unwrap_err();
        assert_eq!(got, want);
    }
    // An unresolvable name is a connect failure, and a hostless URL cannot be sent.
    let error = http
        .send(
            HubRequest::get("https://nx.example.test/"),
            &EndpointPolicy::hosted(),
        )
        .await
        .unwrap_err();
    assert_eq!(error, HttpError::ConnectFailed);
    assert!(
        http.send(
            HubRequest::get("https:///nohost"),
            &EndpointPolicy::hosted()
        )
        .await
        .is_err()
    );
    assert!(format!("{http:?}").contains("ReqwestHttp"));
    let _ = ReqwestHttp::default().with_clock(Arc::new(SystemClock));
}

#[tokio::test]
async fn reqwest_resolving_a_name_spends_the_hops_timeout_budget() {
    let clock = crate::testkit::FakeClock::new();
    let executor = Arc::new(FakeExecutor::default());
    let http = ReqwestHttp::with_parts(
        Arc::new(SlowResolver {
            clock: clock.clone(),
            cost: Duration::from_secs(4),
        }),
        executor.clone(),
    )
    .with_clock(Arc::new(clock.clone()));
    executor.push(Ok(canned(200, &[], b"ok")));
    http.send(
        HubRequest::get("https://a.example.test/").with_timeout(Duration::from_secs(10)),
        &EndpointPolicy::hosted(),
    )
    .await
    .unwrap();
    assert_eq!(
        executor.timeouts.lock().unwrap().as_slice(),
        [Duration::from_secs(6)],
        "the connection gets what the lookup left, not a fresh budget"
    );
    // A lookup that used it all leaves nothing to connect with.
    let clock = crate::testkit::FakeClock::new();
    let executor = Arc::new(FakeExecutor::default());
    let http = ReqwestHttp::with_parts(
        Arc::new(SlowResolver {
            clock: clock.clone(),
            cost: Duration::from_secs(10),
        }),
        executor.clone(),
    )
    .with_clock(Arc::new(clock));
    let error = http
        .send(
            HubRequest::get("https://a.example.test/").with_timeout(Duration::from_secs(10)),
            &EndpointPolicy::hosted(),
        )
        .await
        .unwrap_err();
    assert_eq!(error, HttpError::Timeout);
    assert!(executor.seen.lock().unwrap().is_empty(), "nothing was sent");
}
