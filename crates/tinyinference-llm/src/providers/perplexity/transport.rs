use super::PerplexityConfig;
use crate::{Error, Result, model::ProviderError};
use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use serde_json::Value;
use std::{fmt, io::Write, sync::Arc, time::Duration};
use tokio::time::Instant;

#[derive(Clone, Copy)]
pub(super) enum Operation {
    Create,
    Read,
    Cancel,
}

#[async_trait]
pub(super) trait HttpTransport: Send + Sync {
    async fn send(&self, request: reqwest::Request) -> Result<reqwest::Response>;
}

struct ReqwestTransport {
    client: reqwest::Client,
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn send(&self, request: reqwest::Request) -> Result<reqwest::Response> {
        self.client.execute(request).await.map_err(|_| {
            failure(
                "transport_interrupted",
                "provider completion may be unknown",
                None,
            )
        })
    }
}

pub(super) struct Transport {
    pub config: PerplexityConfig,
    pub sender: Arc<dyn HttpTransport>,
    client: reqwest::Client,
    authorization: HeaderValue,
    secrets: Vec<String>,
}

impl fmt::Debug for Transport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PerplexityTransport")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl Transport {
    pub(super) fn new(key: String, mut config: PerplexityConfig) -> Result<Self> {
        config.selection.validate()?;
        if key.trim().is_empty() {
            return Err(Error::Validation(
                "Perplexity API key must not be blank".into(),
            ));
        }
        if [config.timeout, config.idle_timeout, config.connect_timeout]
            .iter()
            .any(|duration| duration.is_zero() || Instant::now().checked_add(*duration).is_none())
            || config.max_body_bytes == 0
            || config.max_event_bytes == 0
            || config.max_file_bytes == 0
            || config.max_retries > tinyinference_core::MAX_RETRIES
            || Instant::now().checked_add(config.timeout).is_none()
        {
            return Err(Error::Validation(
                "invalid Perplexity resource limits".into(),
            ));
        }
        let url = reqwest::Url::parse(config.base_url.trim())
            .map_err(|_| Error::Validation("invalid Perplexity endpoint".into()))?;
        let loopback = url.host_str().is_some_and(|host| {
            host == "localhost"
                || host
                    .trim_matches(['[', ']'])
                    .parse::<std::net::IpAddr>()
                    .is_ok_and(|ip| ip.is_loopback())
        });
        if url.host_str().is_none()
            || !(url.scheme() == "https" || (url.scheme() == "http" && loopback))
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            return Err(Error::Validation(
                "Perplexity requires HTTPS or loopback HTTP without userinfo, query, or fragment"
                    .into(),
            ));
        }
        config.base_url = url.as_str().trim_end_matches('/').into();
        let mut authorization = HeaderValue::from_str(&format!("Bearer {key}"))
            .map_err(|_| Error::Validation("invalid Perplexity authorization header".into()))?;
        authorization.set_sensitive(true);
        let client = reqwest::Client::builder()
            .connect_timeout(config.connect_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(|_| {
                failure(
                    "client_setup",
                    "could not construct Perplexity HTTP client",
                    None,
                )
            })?;
        let mut secrets = vec![key];
        for credential in config.remote_credentials.values() {
            if let Some(token) = &credential.authorization
                && !token.is_empty()
            {
                secrets.push(token.clone());
            }
            secrets.extend(
                credential
                    .headers
                    .values()
                    .filter(|v| !v.is_empty())
                    .cloned(),
            );
        }
        secrets.sort_by_key(|s| std::cmp::Reverse(s.len()));
        secrets.dedup();
        Ok(Self {
            config,
            authorization,
            sender: Arc::new(ReqwestTransport {
                client: client.clone(),
            }),
            client,
            secrets,
        })
    }

    pub(super) fn deadline(&self, override_ms: Option<u64>) -> Result<Instant> {
        let duration = override_ms
            .map(Duration::from_millis)
            .unwrap_or(self.config.timeout);
        if duration.is_zero() {
            return Err(Error::Validation("timeout must be positive".into()));
        }
        Instant::now()
            .checked_add(duration)
            .ok_or_else(|| Error::Validation("timeout is too large".into()))
    }

    pub(super) fn request(
        &self,
        method: reqwest::Method,
        path: &str,
        body: Option<&Value>,
    ) -> Result<reqwest::Request> {
        let mut builder = self
            .client
            .request(method, format!("{}{}", self.config.base_url, path))
            .header(AUTHORIZATION, self.authorization.clone());
        if let Some(body) = body {
            let mut writer = BoundedWriter {
                bytes: Vec::new(),
                limit: self.config.max_body_bytes,
            };
            serde_json::to_writer(&mut writer, body).map_err(|_| {
                Error::Validation("Perplexity request exceeds serialized size limit".into())
            })?;
            builder = builder
                .header(CONTENT_TYPE, "application/json")
                .body(writer.bytes);
        }
        builder
            .build()
            .map_err(|_| Error::Validation("invalid Perplexity request".into()))
    }

    pub(super) fn check_input_size(&self, request: &crate::model::ModelRequest) -> Result<()> {
        let mut counter = ByteCounter {
            count: 0,
            limit: self.config.max_body_bytes,
        };
        serde_json::to_writer(&mut counter, request)
            .map_err(|_| Error::Validation("Perplexity input exceeds byte limit".into()))
    }

    pub(super) async fn send(
        &self,
        request: reqwest::Request,
        operation: Operation,
        deadline: Instant,
    ) -> Result<reqwest::Response> {
        crate::network_guard::ensure_network_models_allowed()?;
        let work = async {
            let mut retries = 0;
            loop {
                if Instant::now() >= deadline {
                    return Err(timeout());
                }
                let outgoing = request
                    .try_clone()
                    .ok_or_else(|| Error::Validation("request cannot be replayed".into()))?;
                let response = self.sender.send(outgoing).await?;
                if response.status().is_success() {
                    return Ok(response);
                }
                let status = response.status().as_u16();
                let retry_after = response
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_owned);
                let may_retry = match operation {
                    Operation::Create => status == 429,
                    Operation::Read => matches!(status, 429 | 500 | 502 | 503 | 504),
                    Operation::Cancel => false,
                };
                if may_retry && retries < self.config.max_retries {
                    drop(response);
                    let delay =
                        tinyinference_core::backoff_ms_for_attempt(retries, retry_after.as_deref());
                    retries += 1;
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                    continue;
                }
                let body = read_limited(response, self.config.max_body_bytes.min(64 * 1024))
                    .await
                    .unwrap_or_default();
                let raw = serde_json::from_slice::<Value>(&body)
                    .ok()
                    .map(|v| self.scrub_value(v));
                let detail = raw
                    .as_ref()
                    .and_then(|v| v.get("error"))
                    .unwrap_or(&Value::Null);
                let message = detail
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("Perplexity HTTP request failed");
                return Err(Error::Provider(Box::new(ProviderError {
                    provider: "perplexity".into(),
                    status: Some(status),
                    code: detail
                        .get("code")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    message: self.scrub_text(message),
                    retryable: may_retry,
                    retry_after_ms: tinyinference_core::parse_retry_after_ms(
                        retry_after.as_deref(),
                    ),
                    raw,
                    ..Default::default()
                })));
            }
        };
        tokio::time::timeout_at(deadline, work)
            .await
            .map_err(|_| timeout())?
    }

    pub(super) async fn json(
        &self,
        response: reqwest::Response,
        deadline: Instant,
    ) -> Result<super::response::DecodedJson> {
        let bytes =
            tokio::time::timeout_at(deadline, read_limited(response, self.config.max_body_bytes))
                .await
                .map_err(|_| timeout())??;
        let mut decoded = super::response::decode(&bytes)?;
        decoded.value = self.scrub_value(decoded.value);
        if let Some(observer) = &self.config.on_response {
            observer(&decoded.value);
        }
        Ok(decoded)
    }

    pub(super) fn scrub_text(&self, text: &str) -> String {
        let mut text = text.to_owned();
        for secret in &self.secrets {
            text = text.replace(secret, "[REDACTED]");
        }
        tinyinference_core::sanitize::sanitize_api_error(&text)
    }

    pub(super) fn scrub_value(&self, mut value: Value) -> Value {
        fn walk(value: &mut Value, secrets: &[String]) {
            match value {
                Value::String(text) => {
                    for secret in secrets {
                        *text = text.replace(secret, "[REDACTED]");
                    }
                }
                Value::Array(items) => {
                    for item in items {
                        walk(item, secrets);
                    }
                }
                Value::Object(object) => {
                    for value in object.values_mut() {
                        walk(value, secrets);
                    }
                }
                _ => {}
            }
        }
        walk(&mut value, &self.secrets);
        value
    }
}

struct BoundedWriter {
    bytes: Vec<u8>,
    limit: usize,
}

struct ByteCounter {
    count: usize,
    limit: usize,
}
impl Write for ByteCounter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.count) {
            return Err(std::io::Error::other("input too large"));
        }
        self.count += bytes.len();
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
impl Write for BoundedWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
            return Err(std::io::Error::other("request too large"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(super) async fn read_limited(mut response: reqwest::Response, limit: usize) -> Result<Vec<u8>> {
    if response
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(failure(
            "response_too_large",
            "response exceeds byte limit",
            None,
        ));
    }
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await.map_err(|_| {
        failure(
            "transport_interrupted",
            "response body was interrupted",
            None,
        )
    })? {
        if chunk.len() > limit.saturating_sub(bytes.len()) {
            return Err(failure(
                "response_too_large",
                "response exceeds byte limit",
                None,
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

pub(super) fn timeout() -> Error {
    failure(
        "timeout",
        "Perplexity deadline expired; submitted work may still run",
        None,
    )
}
pub(super) fn failure(code: &str, message: &str, status: Option<u16>) -> Error {
    Error::Provider(Box::new(ProviderError {
        provider: "perplexity".into(),
        code: Some(code.into()),
        message: message.into(),
        status,
        ..Default::default()
    }))
}
