//! Client transport, retry, measurement, and secret-handling tests.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::Arc,
    time::{Duration, SystemTime},
};

use reqwest::{Method, StatusCode, header::HeaderMap};
use serde_json::json;
use tokio::sync::Mutex;

use super::*;
use crate::{Choice, Question};

fn request() -> EvaluationRequest {
    EvaluationRequest::jev(
        json!({"message": "review this"}),
        BTreeMap::from([(
            "route".to_owned(),
            Question::Choice(Choice {
                instructions: json!("Who should answer?"),
                criteria: BTreeMap::from([("alice".to_owned(), None), ("bob".to_owned(), None)]),
            }),
        )]),
    )
}

fn success() -> String {
    json!({
        "model": "jev-latest",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "bob",
                "probabilities": {"alice": 0.2, "bob": 0.8},
                "confidence": 0.6
            }
        },
        "usage": {"input_tokens": 12, "output_tokens": 2}
    })
    .to_string()
}

fn response(status: u16, body: &str, extra_headers: &str) -> MockReply {
    let request_id = header_value(extra_headers, "x-request-id").map(str::to_owned);
    let retry_after = header_value(extra_headers, "retry-after")
        .and_then(|value| reqwest::header::HeaderValue::from_str(value).ok())
        .and_then(|value| parse_retry_after(Some(&value)));
    MockReply::Response(TransportResponse {
        status: StatusCode::from_u16(status).unwrap(),
        request_id,
        retry_after,
        body: body.as_bytes().to_vec(),
    })
}

fn header_value<'a>(headers: &'a str, name: &str) -> Option<&'a str> {
    headers.lines().find_map(|header| {
        let (header_name, value) = header.split_once(':')?;
        header_name
            .eq_ignore_ascii_case(name)
            .then_some(value.trim())
    })
}

enum MockReply {
    Response(TransportResponse),
    Failure(Failure),
}

#[derive(Clone, Debug)]
struct RecordedRequest {
    method: Method,
    url: String,
    headers: HeaderMap,
    body: String,
}

struct MockTransport {
    responses: Mutex<VecDeque<MockReply>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
}

#[async_trait::async_trait]
impl HttpTransport for MockTransport {
    async fn send(
        &self,
        request: reqwest::Request,
        _response_limit: usize,
    ) -> std::result::Result<TransportResponse, Failure> {
        let body = request
            .body()
            .and_then(reqwest::Body::as_bytes)
            .map(|body| String::from_utf8_lossy(body).into_owned())
            .unwrap_or_default();
        self.requests.lock().await.push(RecordedRequest {
            method: request.method().clone(),
            url: request.url().to_string(),
            headers: request.headers().clone(),
            body,
        });
        match self.responses.lock().await.pop_front().unwrap() {
            MockReply::Response(response) => Ok(response),
            MockReply::Failure(failure) => Err(failure),
        }
    }
}

fn mock_client(
    config: ClientConfig,
    responses: Vec<MockReply>,
) -> (Client, Arc<Mutex<Vec<RecordedRequest>>>) {
    let requests = Arc::new(Mutex::new(Vec::new()));
    let transport = Arc::new(MockTransport {
        responses: Mutex::new(responses.into()),
        requests: Arc::clone(&requests),
    });
    let client = Client::with_transport(config, transport).unwrap();
    (client, requests)
}

fn evaluation_failure(result: crate::Result<EvaluationResult>) -> EvaluationFailure {
    match result {
        Err(Error::EvaluationFailure(failure)) => failure,
        Err(error) => panic!("expected an evaluation failure, got {error:?}"),
        Ok(_) => panic!("expected evaluation to fail"),
    }
}

fn config(base_url: String) -> ClientConfig {
    let mut config = ClientConfig::new("secret-test-key");
    config.base_url = base_url;
    config.timeout = Duration::from_secs(1);
    config.retry = RetryPolicy {
        max_retries: 0,
        initial_backoff: Duration::from_millis(1),
        max_backoff: Duration::from_millis(5),
    };
    config
}

#[tokio::test]
async fn sends_the_documented_endpoint_and_bearer_header() {
    let (client, requests) = mock_client(
        config("http://127.0.0.1:1".into()),
        vec![response(200, &success(), "x-request-id: request-7\r\n")],
    );
    let result = client.evaluate(&request()).await.unwrap();
    assert_eq!(result.attempts, 1);
    assert_eq!(result.request_id.as_deref(), Some("request-7"));
    assert_eq!(result.response.usage.input_tokens, Some(12));
    let requests = requests.lock().await;
    assert_eq!(requests[0].method, Method::POST);
    assert_eq!(
        reqwest::Url::parse(&requests[0].url).unwrap().path(),
        "/v1/systemone"
    );
    assert_eq!(
        requests[0].headers.get("authorization").unwrap(),
        "Bearer secret-test-key"
    );
    assert!(requests[0].body.contains("\"model\":\"jev-latest\""));
}

#[tokio::test]
async fn an_exact_endpoint_override_is_not_extended_with_a_provider_path() {
    let endpoint = "https://example.com/api/alpha/decisions";
    let config = ClientConfig::openrouter("secret-test-key").with_endpoint_url(endpoint);
    let (client, requests) = mock_client(
        config,
        vec![response(
            200,
            &success().replace("jev-latest", "typesafe/jev-1.13-20260917"),
            "",
        )],
    );

    client.evaluate(&request()).await.unwrap();

    assert_eq!(requests.lock().await[0].url, endpoint);
}

#[tokio::test]
async fn openrouter_uses_system_one_and_accepts_a_resolved_jev_model() {
    let mut config = ClientConfig::openrouter("secret-test-key");
    config.retry.max_retries = 0;
    let (client, requests) = mock_client(
        config,
        vec![response(
            200,
            &success().replace("jev-latest", "typesafe/jev-1.13-20260917"),
            "",
        )],
    );
    let result = client.evaluate(&request()).await.unwrap();
    assert_eq!(result.response.model, "typesafe/jev-1.13-20260917");
    assert_eq!(
        reqwest::Url::parse(&requests.lock().await[0].url)
            .unwrap()
            .path(),
        "/api/v1/systemone"
    );
}

#[tokio::test]
async fn openjev_uses_system_one_endpoint_and_default_model() {
    let mut config = ClientConfig::openjev("secret-test-key");
    config.retry.max_retries = 0;
    let mut openjev_request = request();
    openjev_request.model = "openjev".into();
    let (client, requests) = mock_client(
        config,
        vec![response(
            200,
            &success().replace("jev-latest", "openjev"),
            "",
        )],
    );

    let result = client.evaluate(&openjev_request).await.unwrap();

    assert_eq!(result.response.model, "openjev");
    let requests = requests.lock().await;
    let url = reqwest::Url::parse(&requests[0].url).unwrap();
    assert_eq!(url.host_str(), Some("api.openjev.sh"));
    assert_eq!(url.path(), "/v1/systemone");
    assert_eq!(
        requests[0].headers.get("authorization").unwrap(),
        "Bearer secret-test-key"
    );
    assert!(requests[0].body.contains("\"model\":\"openjev\""));
}

#[tokio::test]
async fn tinyhumans_proxy_uses_the_direct_system_one_path() {
    let mut config = ClientConfig::tinyhumans_openrouter("secret-test-key");
    config.retry.max_retries = 0;
    let (client, requests) = mock_client(
        config,
        vec![response(
            200,
            &success().replace("jev-latest", "typesafe/jev-1.13-20260917"),
            "",
        )],
    );
    client.evaluate(&request()).await.unwrap();
    assert_eq!(
        reqwest::Url::parse(&requests.lock().await[0].url)
            .unwrap()
            .path(),
        "/agent-integrations/openrouter/systemone"
    );
}

#[test]
fn sdk_name_is_sanitized_and_sent_only_to_the_exact_tinyhumans_proxy() {
    let config = ClientConfig::tinyhumans_openrouter("key")
        .with_sdk_name(" OpenCompany\r\nInjected: yes / test ");
    let client = Client::new(config.clone()).unwrap();
    let outgoing = client.evaluation_request(&request()).build().unwrap();
    assert_eq!(
        outgoing.headers().get("x-sdk-name").unwrap(),
        "opencompanyinjectedyestest"
    );

    let direct = Client::new(ClientConfig::openrouter("key").with_sdk_name("openhuman")).unwrap();
    assert!(
        direct
            .evaluation_request(&request())
            .build()
            .unwrap()
            .headers()
            .get("x-sdk-name")
            .is_none()
    );

    for endpoint in [
        "https://api.tinyhumans.ai.evil.example/agent-integrations/openrouter/systemone",
        "https://api.tinyhumans.ai:444/agent-integrations/openrouter/systemone",
        "https://example.com/agent-integrations/openrouter/systemone",
    ] {
        let other = Client::new(config.clone().with_endpoint_url(endpoint)).unwrap();
        assert!(
            other
                .evaluation_request(&request())
                .build()
                .unwrap()
                .headers()
                .get("x-sdk-name")
                .is_none(),
            "must not attribute {endpoint}"
        );
    }
    assert!(
        Client::new(config.with_endpoint_url(
            "https://api.tinyhumans.ai/agent-integrations/openrouter/systemone?redirect=1"
        ))
        .is_err()
    );
}

#[tokio::test]
async fn retries_rate_limits_and_reports_attempts() {
    let mut config = config("http://127.0.0.1:1".into());
    config.retry.max_retries = 1;
    let (client, requests) = mock_client(
        config,
        vec![
            response(429, "{}", "Retry-After: 0\r\n"),
            response(200, &success(), ""),
        ],
    );
    let result = client.evaluate(&request()).await.unwrap();
    assert_eq!(result.attempts, 2);
    assert_eq!(requests.lock().await.len(), 2);
}

#[tokio::test]
async fn retries_retryable_transport_failures() {
    let mut config = config("https://example.com".into());
    config.retry.max_retries = 1;
    let (client, requests) = mock_client(
        config,
        vec![
            MockReply::Failure(Failure::Retryable {
                error: Error::Timeout,
                retry_after: Some(Duration::ZERO),
            }),
            response(200, &success(), ""),
        ],
    );

    let result = client.evaluate(&request()).await.unwrap();

    assert_eq!(result.attempts, 2);
    assert_eq!(requests.lock().await.len(), 2);
}

#[tokio::test]
async fn authentication_is_terminal() {
    let (client, requests) = mock_client(
        config("http://127.0.0.1:1".into()),
        vec![response(401, "{}", "")],
    );
    let error = evaluation_failure(client.evaluate(&request()).await);
    assert!(matches!(error.error.as_ref(), Error::Authentication));
    assert_eq!(error.attempts, 1);
    assert_eq!(requests.lock().await.len(), 1);
}

#[tokio::test]
async fn malformed_success_body_is_a_decode_failure() {
    let (client, _) = mock_client(
        config("http://127.0.0.1:1".into()),
        vec![response(200, "not-json", "")],
    );
    let error = evaluation_failure(client.evaluate(&request()).await);
    assert!(matches!(error.error.as_ref(), Error::Decode { .. }));
}

#[tokio::test]
async fn debug_output_redacts_the_api_key() {
    let mut config = config("http://127.0.0.1:1/private-base-token".to_owned());
    config.endpoint_url = Some("https://example.com/private-token".into());
    let rendered = format!("{config:?}");
    assert!(rendered.contains("[REDACTED]"));
    assert!(!rendered.contains("secret-test-key"));
    assert!(!rendered.contains("private-base-token"));
    assert!(!rendered.contains("private-token"));
    let client = Client::new(config).unwrap();
    let rendered = format!("{client:?}");
    assert!(rendered.contains("[REDACTED]"));
    assert!(!rendered.contains("secret-test-key"));
}

#[test]
fn response_chunks_cannot_exceed_the_configured_body_limit() {
    let mut body = Vec::new();
    append_response_chunk(&mut body, b"123", 4).unwrap();
    let error = append_response_chunk(&mut body, b"45", 4).unwrap_err();
    assert!(matches!(error, Error::ResponseTooLarge { limit: 4 }));
    assert_eq!(body, b"123");
}

#[tokio::test]
async fn rejects_response_content_length_over_the_body_limit() {
    assert!(matches!(
        validate_content_length(Some(MAX_RESPONSE_BYTES as u64 + 1), MAX_RESPONSE_BYTES),
        Err(Error::ResponseTooLarge {
            limit: MAX_RESPONSE_BYTES
        })
    ));
}

#[test]
fn rejects_invalid_configuration_before_transport() {
    let mut empty = ClientConfig::new("");
    empty.base_url = "not a URL".to_owned();
    assert!(matches!(
        Client::new(empty),
        Err(Error::InvalidConfig { .. })
    ));
    let mut invalid_url = ClientConfig::new("key");
    invalid_url.base_url = "not a URL".into();
    assert!(matches!(
        Client::new(invalid_url),
        Err(Error::InvalidConfig { .. })
    ));
}

#[test]
fn validates_every_configuration_bound_and_redacted_key_replacement() {
    let replaced = ClientConfig::new("old").with_api_key("new-secret");
    let rendered = format!("{replaced:?}");
    assert!(!rendered.contains("new-secret"));
    let endpoint =
        ClientConfig::new("key").with_endpoint_url("https://example.com/custom/decisions");
    assert_eq!(
        endpoint.endpoint_url(),
        Some("https://example.com/custom/decisions")
    );
    for endpoint_url in [
        "not a URL",
        "file:///tmp/decisions",
        "http://example.com/decisions",
        "http://localhost:8080/decisions",
        "https://user:password@example.com/decisions",
        "https://example.com/decisions?tenant=x",
        "https://example.com/decisions#fragment",
    ] {
        let config = ClientConfig::new("key").with_endpoint_url(endpoint_url);
        assert!(matches!(
            Client::new(config),
            Err(Error::InvalidConfig { .. })
        ));
    }

    let mut scheme = ClientConfig::new("key");
    scheme.base_url = "file:///tmp/socket".into();
    assert!(matches!(
        Client::new(scheme),
        Err(Error::InvalidConfig { .. })
    ));

    for base_url in ["http://example.com", "http://localhost:8080"] {
        let mut cleartext = ClientConfig::new("key");
        cleartext.base_url = base_url.into();
        assert!(matches!(
            Client::new(cleartext),
            Err(Error::InvalidConfig { .. })
        ));
    }
    let mut secure = ClientConfig::new("key");
    secure.base_url = "https://example.com".into();
    assert!(Client::new(secure).is_ok());
    let openrouter = ClientConfig::openrouter("key");
    assert_eq!(openrouter.base_url, "https://openrouter.ai/api");
    assert_eq!(openrouter.provider, Provider::OpenRouter);
    let tinyhumans = ClientConfig::tinyhumans_openrouter("key");
    assert_eq!(tinyhumans.base_url, "https://api.tinyhumans.ai");
    assert_eq!(
        tinyhumans.system_one_path,
        "agent-integrations/openrouter/systemone"
    );
    assert_eq!(tinyhumans.provider, Provider::OpenRouter);
    let mut ipv6_loopback = ClientConfig::new("key");
    ipv6_loopback.base_url = "http://[::1]:8080".into();
    assert!(Client::new(ipv6_loopback).is_ok());
    let mut userinfo = ClientConfig::new("key");
    userinfo.base_url = "https://user:password@example.com".into();
    assert!(matches!(
        Client::new(userinfo),
        Err(Error::InvalidConfig { .. })
    ));
    for base_url in [
        "https://example.com?tenant=x",
        "https://example.com#fragment",
    ] {
        let mut component = ClientConfig::new("key");
        component.base_url = base_url.into();
        assert!(matches!(
            Client::new(component),
            Err(Error::InvalidConfig { .. })
        ));
    }

    let mut timeout = ClientConfig::new("key");
    timeout.timeout = Duration::ZERO;
    assert!(matches!(
        Client::new(timeout),
        Err(Error::InvalidConfig { .. })
    ));

    let mut retry = ClientConfig::new("key");
    retry.retry.initial_backoff = Duration::ZERO;
    assert!(matches!(
        Client::new(retry),
        Err(Error::InvalidConfig { .. })
    ));
    let mut unbounded = ClientConfig::new("key");
    unbounded.retry.max_retries = u32::MAX;
    assert!(matches!(
        Client::new(unbounded),
        Err(Error::InvalidConfig { .. })
    ));
}

#[test]
fn retry_delay_is_exponential_and_bounded() {
    let policy = RetryPolicy {
        max_retries: 5,
        initial_backoff: Duration::from_millis(10),
        max_backoff: Duration::from_millis(25),
    };
    assert_eq!(policy.delay(1), Duration::from_millis(10));
    assert_eq!(policy.delay(2), Duration::from_millis(20));
    assert_eq!(policy.delay(30), Duration::from_millis(25));
}

#[test]
fn status_classification_covers_terminal_and_retryable_classes() {
    assert!(matches!(
        classify_status(StatusCode::BAD_REQUEST, None),
        Failure::Terminal(Error::Unprocessable)
    ));
    assert!(matches!(
        classify_status(StatusCode::NOT_FOUND, None),
        Failure::Terminal(Error::HttpStatus { status: 404 })
    ));
    assert!(matches!(
        classify_status(StatusCode::INTERNAL_SERVER_ERROR, None),
        Failure::Retryable {
            error: Error::HttpStatus { status: 500 },
            ..
        }
    ));
    assert!(matches!(
        classify_status(StatusCode::REQUEST_TIMEOUT, None),
        Failure::Retryable {
            error: Error::Timeout,
            ..
        }
    ));
    assert!(matches!(
        classify_status(StatusCode::from_u16(529).unwrap(), None),
        Failure::Retryable {
            error: Error::Overloaded,
            ..
        }
    ));
    assert_eq!(
        parse_retry_after(Some(&reqwest::header::HeaderValue::from_static("3"))),
        Some(Duration::from_secs(3))
    );
    assert_eq!(
        parse_retry_after(Some(&reqwest::header::HeaderValue::from_static("date"))),
        None
    );
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000);
    let future = now + Duration::from_secs(30);
    let date = httpdate::fmt_http_date(future);
    let header = reqwest::header::HeaderValue::from_str(&date).unwrap();
    assert_eq!(
        parse_retry_after_at(Some(&header), now),
        Some(Duration::from_secs(30))
    );
}

#[tokio::test]
async fn request_timeout_is_retryable_but_respects_the_attempt_bound() {
    let mut config = config("http://127.0.0.1:1".into());
    config.retry.max_retries = 0;
    let (client, _) = mock_client(config, vec![response(408, "{}", "")]);
    let error = evaluation_failure(client.evaluate(&request()).await);
    assert!(matches!(error.error.as_ref(), Error::Timeout));
    assert_eq!(error.attempts, 1);
}

#[tokio::test]
async fn exhausted_rate_limit_returns_the_classified_error() {
    let (client, _) = mock_client(
        config("http://127.0.0.1:1".into()),
        vec![response(429, "{}", "")],
    );
    let error = evaluation_failure(client.evaluate(&request()).await);
    assert!(matches!(error.error.as_ref(), Error::RateLimited));
    assert_eq!(error.attempts, 1);
}

#[tokio::test]
async fn local_validation_and_response_validation_report_failure_metadata() {
    let client = Client::new(config("http://127.0.0.1:1".into())).unwrap();
    let invalid = EvaluationRequest::jev("state", BTreeMap::new());
    let failure = evaluation_failure(client.evaluate(&invalid).await);
    assert!(matches!(
        failure.error.as_ref(),
        Error::InvalidRequest { .. }
    ));
    assert_eq!(failure.attempts, 0);

    let body = json!({
        "model": "jev-latest",
        "answers": {},
        "usage": {"input_tokens": 1, "output_tokens": 1}
    })
    .to_string();
    let (client, _) = mock_client(
        config("http://127.0.0.1:1".into()),
        vec![response(200, &body, "")],
    );
    let failure = evaluation_failure(client.evaluate(&request()).await);
    assert!(matches!(
        failure.error.as_ref(),
        Error::InvalidResponse { .. }
    ));
    assert_eq!(failure.attempts, 1);
}

#[tokio::test]
async fn redirect_is_not_followed() {
    let (client, requests) = mock_client(
        config("http://127.0.0.1:1".into()),
        vec![response(
            307,
            "{}",
            "Location: http://example.com/downgrade\r\n",
        )],
    );
    let failure = evaluation_failure(client.evaluate(&request()).await);
    assert!(matches!(
        failure.error.as_ref(),
        Error::HttpStatus { status: 307 }
    ));
    assert_eq!(requests.lock().await.len(), 1);
}

#[tokio::test]
async fn self_hosted_posts_to_the_declared_endpoint_and_requires_the_model_echo() {
    let endpoint = "https://decisions.internal.example/v1/decide";
    let mut config = ClientConfig::self_hosted(endpoint, "secret-test-key");
    config.retry.max_retries = 0;
    let mut self_hosted_request = request();
    self_hosted_request.model = "surogate-decisions-v1".into();
    let (client, requests) = mock_client(
        config.clone(),
        vec![response(
            200,
            &success().replace("jev-latest", "surogate-decisions-v1"),
            "",
        )],
    );

    let result = client.evaluate(&self_hosted_request).await.unwrap();

    assert_eq!(result.response.model, "surogate-decisions-v1");
    let requests = requests.lock().await;
    assert_eq!(requests[0].url, endpoint);
    assert_eq!(
        requests[0].headers.get("authorization").unwrap(),
        "Bearer secret-test-key"
    );
    assert!(requests[0].headers.get("x-sdk-name").is_none());
    assert!(
        requests[0]
            .body
            .contains("\"model\":\"surogate-decisions-v1\"")
    );
    drop(requests);

    let (client, _) = mock_client(config, vec![response(200, &success(), "")]);
    let failure = evaluation_failure(client.evaluate(&self_hosted_request).await);
    assert!(matches!(*failure.error, Error::InvalidResponse { .. }));
}

#[tokio::test]
async fn self_hosted_without_a_key_sends_no_authorization_header() {
    let mut config = ClientConfig::self_hosted("http://127.0.0.1:8080/v1/systemone", "");
    config.retry.max_retries = 0;
    let (client, requests) = mock_client(config, vec![response(200, &success(), "")]);

    client.evaluate(&request()).await.unwrap();

    assert!(requests.lock().await[0].headers.get("authorization").is_none());
}

#[test]
fn self_hosted_configuration_is_validated() {
    assert_eq!(
        ClientConfig::self_hosted("https://example.com/decide", "").provider,
        Provider::SelfHosted
    );
    for endpoint in [
        "not a URL",
        "http://decisions.example/v1/systemone",
        "https://user:pass@example.com/decide",
        "https://example.com/decide?key=1",
    ] {
        assert!(
            matches!(
                Client::new(ClientConfig::self_hosted(endpoint, "key")),
                Err(Error::InvalidConfig { .. })
            ),
            "{endpoint} should be rejected"
        );
    }
    let mut missing_endpoint = ClientConfig::new("key");
    missing_endpoint.provider = Provider::SelfHosted;
    assert!(matches!(
        Client::new(missing_endpoint),
        Err(Error::InvalidConfig { reason }) if reason.contains("endpoint URL")
    ));
    assert!(matches!(
        Client::new(ClientConfig::new(" ")),
        Err(Error::InvalidConfig { .. })
    ));
    let rendered = format!(
        "{:?}",
        ClientConfig::self_hosted("https://example.com/private/path", "secret-test-key")
    );
    assert!(!rendered.contains("secret-test-key"));
    assert!(!rendered.contains("private/path"));
}
