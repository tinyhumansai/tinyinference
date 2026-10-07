//! Async HTTP client, retry policy, and measured evaluation result.

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;

mod types;

pub use types::{Client, ClientConfig, EvaluationFailure, EvaluationResult, Provider, RetryPolicy};

use std::time::Duration;

use reqwest::{StatusCode, header::RETRY_AFTER};

use crate::{Error, EvaluationRequest, EvaluationResponse, Result};

const MAX_RETRIES: u32 = 100;
const MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

impl Client {
    /// Construct a client from an explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidConfig`] for an empty key (except for
    /// [`Provider::SelfHosted`]), invalid base or endpoint URL, a self-hosted
    /// configuration without an endpoint URL, zero timeout, or invalid retry
    /// policy.
    pub fn new(config: ClientConfig) -> Result<Self> {
        config.validate()?;
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|source| Error::Transport { source })?;
        let transport = std::sync::Arc::new(ReqwestTransport { http: http.clone() });
        Ok(Self {
            config,
            http,
            transport,
        })
    }

    #[cfg(test)]
    pub(super) fn with_transport(
        config: ClientConfig,
        transport: std::sync::Arc<dyn HttpTransport>,
    ) -> Result<Self> {
        config.validate()?;
        let http = reqwest::Client::builder()
            .timeout(config.timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|source| Error::Transport { source })?;
        Ok(Self {
            config,
            http,
            transport,
        })
    }

    /// Construct a client using `TYPESAFE_API_KEY` and production defaults.
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingApiKey`] when the variable is absent, or the
    /// same configuration errors as [`Self::new`].
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("TYPESAFE_API_KEY").map_err(|_| Error::MissingApiKey)?;
        Self::new(ClientConfig::new(api_key))
    }

    /// Evaluate typed questions against shared state.
    ///
    /// The returned latency includes retry delays and all attempts. Request and
    /// response bodies are never logged by this crate.
    ///
    /// # Errors
    ///
    /// Returns [`Error::EvaluationFailure`] wrapping request validation,
    /// transport, HTTP, decoding, or response contract errors while preserving
    /// attempt count and elapsed-time metadata. Timeouts, connection
    /// establishment failures, and response body-transfer failures are retried
    /// within the explicit attempt cap; other transport errors are terminal.
    /// HTTP request validation and authentication failures remain terminal.
    pub async fn evaluate(&self, request: &EvaluationRequest) -> Result<EvaluationResult> {
        let started = tokio::time::Instant::now();
        request
            .validate()
            .map_err(|error| evaluation_failure(error, 0, started.elapsed()))?;
        let mut attempts = 0_u32;
        loop {
            attempts = attempts.saturating_add(1);
            match self.send_once(request).await {
                Ok((response, request_id)) => {
                    self.validate_response(&response, request)
                        .map_err(|error| evaluation_failure(error, attempts, started.elapsed()))?;
                    return Ok(EvaluationResult {
                        response,
                        request_id,
                        attempts,
                        latency: started.elapsed(),
                    });
                }
                Err(Failure::Terminal(error)) => {
                    return Err(evaluation_failure(error, attempts, started.elapsed()));
                }
                Err(Failure::Retryable { error, retry_after }) => {
                    if attempts > self.config.retry.max_retries {
                        return Err(evaluation_failure(error, attempts, started.elapsed()));
                    }
                    let delay = retry_after.unwrap_or_else(|| self.config.retry.delay(attempts));
                    tokio::time::sleep(delay.min(self.config.retry.max_backoff)).await;
                }
            }
        }
    }

    async fn send_once(
        &self,
        request: &EvaluationRequest,
    ) -> std::result::Result<(EvaluationResponse, Option<String>), Failure> {
        let outgoing = self
            .evaluation_request(request)
            .build()
            .map_err(classify_transport)?;
        let response = self.transport.send(outgoing, MAX_RESPONSE_BYTES).await?;
        let status = response.status;
        if !status.is_success() {
            return Err(classify_status(status, response.retry_after));
        }
        let decoded = serde_json::from_slice(&response.body)
            .map_err(|source| Failure::Terminal(Error::Decode { source }))?;
        Ok((decoded, response.request_id))
    }

    fn evaluation_request(&self, request: &EvaluationRequest) -> reqwest::RequestBuilder {
        let url = self.config.endpoint_url.clone().unwrap_or_else(|| {
            format!(
                "{}/{}",
                self.config.base_url.trim_end_matches('/'),
                self.config.system_one_path
            )
        });
        let mut builder = self.http.post(&url).json(request);
        let api_key = self.config.api_key.expose();
        if !(self.config.provider == Provider::SelfHosted && api_key.trim().is_empty()) {
            builder = builder.bearer_auth(api_key);
        }
        if let Some(name) = self.config.sdk_name.as_deref()
            && is_tinyhumans_proxy_endpoint(&url)
        {
            builder = builder.header("x-sdk-name", name);
        }
        builder
    }

    fn validate_response(
        &self,
        response: &EvaluationResponse,
        request: &EvaluationRequest,
    ) -> Result<()> {
        match self.config.provider {
            Provider::TypeSafe | Provider::OpenJev | Provider::SelfHosted => {
                response.validate_for(request)
            }
            Provider::OpenRouter => response.validate_for_openrouter(request),
        }
    }
}

pub(super) fn append_response_chunk(body: &mut Vec<u8>, chunk: &[u8], limit: usize) -> Result<()> {
    if body.len().saturating_add(chunk.len()) > limit {
        return Err(Error::ResponseTooLarge { limit });
    }
    body.extend_from_slice(chunk);
    Ok(())
}

pub(super) struct TransportResponse {
    pub(super) status: StatusCode,
    pub(super) request_id: Option<String>,
    pub(super) retry_after: Option<Duration>,
    pub(super) body: Vec<u8>,
}

#[async_trait::async_trait]
pub(super) trait HttpTransport: Send + Sync {
    async fn send(
        &self,
        request: reqwest::Request,
        response_limit: usize,
    ) -> std::result::Result<TransportResponse, Failure>;
}

struct ReqwestTransport {
    http: reqwest::Client,
}

#[async_trait::async_trait]
impl HttpTransport for ReqwestTransport {
    async fn send(
        &self,
        request: reqwest::Request,
        response_limit: usize,
    ) -> std::result::Result<TransportResponse, Failure> {
        let response = self
            .http
            .execute(request)
            .await
            .map_err(classify_transport)?;
        let request_id = response
            .headers()
            .get("x-request-id")
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let retry_after = parse_retry_after(response.headers().get(RETRY_AFTER));
        let status = response.status();
        if !status.is_success() {
            return Ok(TransportResponse {
                status,
                request_id,
                retry_after,
                body: Vec::new(),
            });
        }
        validate_content_length(response.content_length(), response_limit)
            .map_err(Failure::Terminal)?;
        let mut response = response;
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(classify_transport)? {
            append_response_chunk(&mut body, &chunk, response_limit).map_err(Failure::Terminal)?;
        }
        Ok(TransportResponse {
            status,
            request_id,
            retry_after,
            body,
        })
    }
}

fn validate_content_length(content_length: Option<u64>, limit: usize) -> Result<()> {
    if content_length.is_some_and(|length| length > u64::try_from(limit).unwrap_or(u64::MAX)) {
        return Err(Error::ResponseTooLarge { limit });
    }
    Ok(())
}

fn evaluation_failure(error: Error, attempts: u32, latency: Duration) -> Error {
    EvaluationFailure {
        error: Box::new(error),
        attempts,
        latency,
    }
    .into()
}

fn is_tinyhumans_proxy_endpoint(raw: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(raw) else {
        return false;
    };
    url.scheme() == "https"
        && url.host_str() == Some("api.tinyhumans.ai")
        && url.port_or_known_default() == Some(443)
        && url.username().is_empty()
        && url.password().is_none()
        && url.path() == "/agent-integrations/openrouter/systemone"
        && url.query().is_none()
        && url.fragment().is_none()
}

impl ClientConfig {
    fn validate(&self) -> Result<()> {
        if self.provider != Provider::SelfHosted && self.api_key.expose().trim().is_empty() {
            return Err(Error::InvalidConfig {
                reason: "api key must not be empty".to_owned(),
            });
        }
        if self.provider == Provider::SelfHosted && self.endpoint_url.is_none() {
            return Err(Error::InvalidConfig {
                reason: "self-hosted provider requires an endpoint URL".to_owned(),
            });
        }
        validate_url(&self.base_url, "base URL")?;
        if let Some(endpoint_url) = &self.endpoint_url {
            validate_url(endpoint_url, "endpoint URL")?;
        }
        if self.timeout.is_zero() {
            return Err(Error::InvalidConfig {
                reason: "timeout must be greater than zero".to_owned(),
            });
        }
        if self.retry.initial_backoff.is_zero() || self.retry.max_backoff.is_zero() {
            return Err(Error::InvalidConfig {
                reason: "retry backoffs must be greater than zero".to_owned(),
            });
        }
        if self.retry.max_retries > MAX_RETRIES {
            return Err(Error::InvalidConfig {
                reason: format!("max retries must not exceed {MAX_RETRIES}"),
            });
        }
        Ok(())
    }
}

fn validate_url(value: &str, label: &str) -> Result<()> {
    let url = reqwest::Url::parse(value).map_err(|_| Error::InvalidConfig {
        reason: format!("{label} must be an absolute HTTP URL"),
    })?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(Error::InvalidConfig {
            reason: format!("{label} must use HTTP or HTTPS"),
        });
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(Error::InvalidConfig {
            reason: format!("{label} must not contain credentials"),
        });
    }
    if url.query().is_some() || url.fragment().is_some() {
        return Err(Error::InvalidConfig {
            reason: format!("{label} must not contain a query or fragment"),
        });
    }
    if url.scheme() == "http"
        && !url
            .host_str()
            .and_then(|host| {
                host.trim_matches(['[', ']'])
                    .parse::<std::net::IpAddr>()
                    .ok()
            })
            .is_some_and(|address| address.is_loopback())
    {
        return Err(Error::InvalidConfig {
            reason: format!("HTTP {label} must use a literal loopback address"),
        });
    }
    Ok(())
}

pub(super) enum Failure {
    Terminal(Error),
    Retryable {
        error: Error,
        retry_after: Option<Duration>,
    },
}

fn classify_transport(source: reqwest::Error) -> Failure {
    if source.is_timeout() {
        Failure::Retryable {
            error: Error::Timeout,
            retry_after: None,
        }
    } else if source.is_connect() || source.is_body() {
        Failure::Retryable {
            error: Error::Transport { source },
            retry_after: None,
        }
    } else {
        Failure::Terminal(Error::Transport { source })
    }
}

fn classify_status(status: StatusCode, retry_after: Option<Duration>) -> Failure {
    match status {
        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
            Failure::Terminal(Error::Authentication)
        }
        StatusCode::UNPROCESSABLE_ENTITY | StatusCode::BAD_REQUEST => {
            Failure::Terminal(Error::Unprocessable)
        }
        StatusCode::REQUEST_TIMEOUT => Failure::Retryable {
            error: Error::Timeout,
            retry_after,
        },
        StatusCode::TOO_MANY_REQUESTS => Failure::Retryable {
            error: Error::RateLimited,
            retry_after,
        },
        status if status.as_u16() == 529 => Failure::Retryable {
            error: Error::Overloaded,
            retry_after,
        },
        status if status.is_server_error() => Failure::Retryable {
            error: Error::HttpStatus {
                status: status.as_u16(),
            },
            retry_after,
        },
        status => Failure::Terminal(Error::HttpStatus {
            status: status.as_u16(),
        }),
    }
}

fn parse_retry_after(value: Option<&reqwest::header::HeaderValue>) -> Option<Duration> {
    parse_retry_after_at(value, std::time::SystemTime::now())
}

fn parse_retry_after_at(
    value: Option<&reqwest::header::HeaderValue>,
    now: std::time::SystemTime,
) -> Option<Duration> {
    let value = value?.to_str().ok()?;
    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }
    httpdate::parse_http_date(value)
        .ok()?
        .duration_since(now)
        .ok()
}
