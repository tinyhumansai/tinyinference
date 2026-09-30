//! [`Http`]: the transport every probe and catalog read goes through.

use std::fmt::{self, Debug};
use std::time::Duration;

use async_trait::async_trait;

use crate::endpoint::redact_endpoint;
use crate::error::{HubError, PolicyViolation, TransportCondition, classify_transport};
use crate::policy::EndpointPolicy;
use crate::secret::LogOnly;

/// An HTTP method. The hub only ever reads and pings.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// `GET`.
    Get,
    /// `POST` (the one-token completion ping).
    Post,
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Get => "GET",
            Self::Post => "POST",
        })
    }
}

/// A request the hub asks its transport to make.
///
/// `Debug` never prints a header value or the body: the headers carry the
/// credential.
#[non_exhaustive]
#[derive(Clone)]
pub struct HubRequest {
    /// The method.
    pub method: Method,
    /// The URL. The transport applies the endpoint policy to it.
    pub url: String,
    /// Header `(name, value)` pairs, credential included.
    pub headers: Vec<(String, String)>,
    /// The body, for a `POST`.
    pub body: Option<Vec<u8>>,
    /// Total time allowed, including connect, TLS, redirects and the body.
    pub timeout: Duration,
    /// Bytes of the response body to keep; the rest is discarded and the
    /// response is marked truncated.
    pub body_cap: usize,
    /// Whether the request carries a credential. Drives the cleartext rule
    /// and the refusal to follow a redirect to another origin.
    pub credentialed: bool,
}

impl HubRequest {
    fn base(method: Method, url: impl Into<String>) -> Self {
        let policy = EndpointPolicy::hosted();
        Self {
            method,
            url: url.into(),
            headers: Vec::new(),
            body: None,
            timeout: policy.timeout,
            body_cap: policy.fail_body_cap,
            credentialed: false,
        }
    }

    /// A `GET` with the hosted policy's default timeout and failure-body cap.
    pub fn get(url: impl Into<String>) -> Self {
        Self::base(Method::Get, url)
    }

    /// A `POST` with a JSON body and a `content-type` header.
    pub fn post_json(url: impl Into<String>, body: &serde_json::Value) -> Self {
        let mut request = Self::base(Method::Post, url);
        request.body = Some(body.to_string().into_bytes());
        request
            .headers
            .push(("content-type".to_string(), "application/json".to_string()));
        request
    }

    /// Adds a header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), value.into()));
        self
    }

    /// Marks the request as carrying a credential.
    #[must_use]
    pub fn with_credentialed(mut self, credentialed: bool) -> Self {
        self.credentialed = credentialed;
        self
    }

    /// Sets the total time allowed.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets how many response bytes are kept.
    #[must_use]
    pub fn with_body_cap(mut self, cap: usize) -> Self {
        self.body_cap = cap;
        self
    }

    /// The first header called `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

impl Debug for HubRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.headers.iter().map(|(n, _)| n.as_str()).collect();
        f.debug_struct("HubRequest")
            .field("method", &self.method)
            .field("url", &redact_endpoint(&self.url))
            .field("header_names", &names)
            .field("body_len", &self.body.as_ref().map(Vec::len))
            .field("timeout", &self.timeout)
            .field("body_cap", &self.body_cap)
            .field("credentialed", &self.credentialed)
            .finish()
    }
}

/// What the transport got back, after any redirects.
#[non_exhaustive]
#[derive(Clone, PartialEq, Eq)]
pub struct HubResponse {
    /// The final status.
    pub status: u16,
    /// Response headers.
    pub headers: Vec<(String, String)>,
    /// The body, cut at the request's `body_cap`.
    pub body: Vec<u8>,
    /// The body ran past the cap and the rest was discarded.
    pub truncated: bool,
    /// The URL that answered (differs from the request's after a redirect).
    pub url: String,
}

impl HubResponse {
    /// A response with no headers, for a transport implementation or a test.
    pub fn new(status: u16, body: impl Into<Vec<u8>>, url: impl Into<String>) -> Self {
        Self {
            status,
            headers: Vec::new(),
            body: body.into(),
            truncated: false,
            url: url.into(),
        }
    }

    /// Whether the status is 2xx.
    pub fn is_success(&self) -> bool {
        (200..300).contains(&self.status)
    }

    /// The first header called `name` (case-insensitive).
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The headers as borrowed pairs, the shape [`classify`](crate::classify)
    /// takes.
    pub fn header_pairs(&self) -> Vec<(&str, &str)> {
        self.headers
            .iter()
            .map(|(n, v)| (n.as_str(), v.as_str()))
            .collect()
    }

    /// The body as text, invalid UTF-8 replaced.
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
}

impl Debug for HubResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HubResponse")
            .field("status", &self.status)
            .field("url", &redact_endpoint(&self.url))
            .field("body_len", &self.body.len())
            .field("truncated", &self.truncated)
            .finish()
    }
}

/// A request that produced no response.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HttpError {
    /// The request ran past its timeout.
    #[error("the request timed out")]
    Timeout,
    /// DNS failed, the connection was refused, or the handshake never
    /// completed: nothing usable is at that address.
    #[error("the connection failed")]
    ConnectFailed,
    /// The endpoint policy refused the URL, a redirect hop, or a resolved
    /// address.
    #[error("refused by the endpoint policy: {0}")]
    Policy(PolicyViolation),
    /// Anything else. The transport's text is log-only.
    #[error("the request failed")]
    Failed(LogOnly<String>),
}

impl HttpError {
    /// The hub error this becomes: a policy refusal is
    /// [`HubError::Policy`]; a transport failure is classified from its
    /// condition, never from its text.
    pub fn into_hub(self) -> HubError {
        match self {
            Self::Policy(violation) => HubError::Policy(violation),
            Self::Timeout => HubError::Provider(classify_transport(
                TransportCondition::Timeout,
                "the request timed out",
            )),
            Self::ConnectFailed => HubError::Provider(classify_transport(
                TransportCondition::ConnectFailed,
                "the connection failed",
            )),
            Self::Failed(detail) => HubError::Provider(classify_transport(
                TransportCondition::Other,
                detail.expose(),
            )),
        }
    }
}

/// The transport.
///
/// **An implementation MUST:**
///
/// 1. apply `policy` to the request URL and to every redirect hop (at most
///    `policy.max_redirects`), refusing with [`HttpError::Policy`];
/// 2. resolve DNS once, check every resolved address with
///    [`check_address`](crate::policy::check_address), and connect to that
///    checked address (anti DNS-rebinding);
/// 3. never forward a credential header to another origin (see
///    [`follow_redirects`](super::follow_redirects), which does 1 and 3 for
///    a transport that only implements a single hop);
/// 4. honour `timeout`, and keep at most `body_cap` response bytes, setting
///    [`HubResponse::truncated`].
#[async_trait]
pub trait Http: Send + Sync + Debug {
    /// Makes the request.
    ///
    /// # Errors
    ///
    /// An [`HttpError`] when no response was obtained.
    async fn send(
        &self,
        request: HubRequest,
        policy: &EndpointPolicy,
    ) -> Result<HubResponse, HttpError>;
}
