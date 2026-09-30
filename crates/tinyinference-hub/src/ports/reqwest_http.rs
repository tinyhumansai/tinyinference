//! [`ReqwestHttp`]: the reference [`Http`] over `reqwest` (feature
//! `http-reqwest`).
//!
//! It honours the whole [`Http`] contract: the endpoint policy is applied to
//! the first URL and to every redirect hop (through [`follow_redirects`]), DNS
//! is resolved **once per hop**, every resolved address is checked with
//! [`check_address`], and the connection is pinned to exactly the addresses
//! that were checked (`resolve_to_addrs`), so a name that answers public at
//! check time and private at connect time cannot get through (DNS rebinding).
//! Automatic redirects are off; credential headers never cross origins.
//!
//! **What is not covered by the crate's tests:** the two lines that touch the
//! network, [`SystemResolver::resolve`] (a blocking `getaddrinfo`) and
//! [`ReqwestExecutor::execute`] (the socket). Everything around them (the
//! address checks, the pin, the request and response conversion, the body cap,
//! the error mapping, the redirect chain) is exercised through the two seams
//! ([`Resolver`], [`Executor`]) with no socket. Deployments verify the network
//! path in their own integration environment.

use std::fmt::{self, Debug};
use std::net::{IpAddr, SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use url::{Host, Url};

use crate::error::{PolicyViolation, scrub_log_text};
use crate::policy::{EndpointPolicy, HeaderPolicy, check_address};
use crate::secret::LogOnly;

use super::clock::{Clock, SystemClock};
use super::http::{Http, HttpError, HubRequest, HubResponse, Method};
use super::redirect::follow_redirects;

/// The addresses a hop is pinned to: every one was checked against the policy.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pin {
    /// The name to pin, when the URL had one (an IP literal needs no pin).
    pub domain: Option<String>,
    /// The addresses the connection may go to.
    pub addrs: Vec<SocketAddr>,
}

/// Turns a name into addresses. The seam that lets DNS rebinding be tested.
#[async_trait]
pub trait Resolver: Send + Sync + Debug {
    /// Resolves `host` (a name, never an IP literal).
    ///
    /// # Errors
    ///
    /// [`HttpError::ConnectFailed`] when the name does not resolve.
    async fn resolve(
        &self,
        host: &str,
        port: u16,
        timeout: Duration,
    ) -> Result<Vec<IpAddr>, HttpError>;
}

/// Sends one prepared request over a connection pinned to `pin`.
#[async_trait]
pub trait Executor: Send + Sync + Debug {
    /// Executes `request` with no redirects, connecting only to `pin`.
    ///
    /// # Errors
    ///
    /// [`HttpError::Timeout`], [`HttpError::ConnectFailed`] or
    /// [`HttpError::Failed`].
    async fn execute(
        &self,
        pin: &Pin,
        request: reqwest::Request,
        timeout: Duration,
    ) -> Result<reqwest::Response, HttpError>;
}

/// The system resolver (`getaddrinfo` on a blocking thread).
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemResolver;

#[async_trait]
impl Resolver for SystemResolver {
    async fn resolve(
        &self,
        host: &str,
        port: u16,
        timeout: Duration,
    ) -> Result<Vec<IpAddr>, HttpError> {
        let name = host.to_string();
        let lookup = tokio::task::spawn_blocking(move || (name, port).to_socket_addrs());
        match tokio::time::timeout(timeout, lookup).await {
            Ok(Ok(Ok(found))) => Ok(found.map(|addr| addr.ip()).collect()),
            Err(_) => Err(HttpError::Timeout),
            _ => Err(HttpError::ConnectFailed),
        }
    }
}

/// The real executor: a `reqwest` client built for the pin, used once.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReqwestExecutor;

#[async_trait]
impl Executor for ReqwestExecutor {
    async fn execute(
        &self,
        pin: &Pin,
        request: reqwest::Request,
        timeout: Duration,
    ) -> Result<reqwest::Response, HttpError> {
        let client = build_client(pin, timeout)?;
        client.execute(request).await.map_err(map_reqwest_error)
    }
}

/// Builds a client that follows no redirects and resolves the pinned name to
/// exactly the checked addresses.
///
/// # Errors
///
/// [`HttpError::Failed`] when the client cannot be built.
pub fn build_client(pin: &Pin, timeout: Duration) -> Result<reqwest::Client, HttpError> {
    let mut builder = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout)
        .no_proxy();
    if let Some(domain) = &pin.domain {
        builder = builder.resolve_to_addrs(domain, &pin.addrs);
    }
    builder
        .build()
        .map_err(|error| HttpError::Failed(LogOnly::new(scrub_log_text(&error.to_string()))))
}

fn map_reqwest_error(error: reqwest::Error) -> HttpError {
    map_error_parts(error.is_timeout(), error.is_connect(), &error.to_string())
}

/// The error mapping, on plain facts so it can be tested without a network.
pub fn map_error_parts(is_timeout: bool, is_connect: bool, text: &str) -> HttpError {
    if is_timeout {
        HttpError::Timeout
    } else if is_connect {
        HttpError::ConnectFailed
    } else {
        HttpError::Failed(LogOnly::new(scrub_log_text(text)))
    }
}

/// Checks every resolved address against the policy and returns the socket
/// addresses to pin. One refused address refuses the hop: a name that answers
/// with a mix must not be connected to on the public half.
///
/// # Errors
///
/// [`HttpError::Policy`] for a refused address, [`HttpError::ConnectFailed`]
/// for an empty answer.
pub fn pin_addresses(
    ips: &[IpAddr],
    port: u16,
    policy: &EndpointPolicy,
) -> Result<Vec<SocketAddr>, HttpError> {
    if ips.is_empty() {
        return Err(HttpError::ConnectFailed);
    }
    for ip in ips {
        check_address(*ip, policy)
            .map_err(|refusal| HttpError::Policy(PolicyViolation::from(refusal)))?;
    }
    Ok(ips.iter().map(|ip| SocketAddr::new(*ip, port)).collect())
}

/// The reference [`Http`] over `reqwest`. Cheap to clone and share.
#[derive(Clone)]
pub struct ReqwestHttp {
    resolver: Arc<dyn Resolver>,
    executor: Arc<dyn Executor>,
    clock: Arc<dyn Clock>,
    headers: HeaderPolicy,
}

impl ReqwestHttp {
    /// The reference transport over the system resolver and a real `reqwest`
    /// client per hop.
    pub fn new() -> Self {
        Self::with_parts(Arc::new(SystemResolver), Arc::new(ReqwestExecutor))
    }

    /// A transport over custom seams. Used by the tests; also how a host with
    /// its own resolver (a DoH client) or executor keeps the rest.
    pub fn with_parts(resolver: Arc<dyn Resolver>, executor: Arc<dyn Executor>) -> Self {
        Self {
            resolver,
            executor,
            clock: Arc::new(SystemClock),
            headers: HeaderPolicy::builtin(),
        }
    }

    /// Measures the redirect chain's total timeout on `clock`.
    #[must_use]
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    async fn hop(
        &self,
        request: HubRequest,
        policy: &EndpointPolicy,
    ) -> Result<HubResponse, HttpError> {
        let url = Url::parse(&request.url).map_err(|_| {
            HttpError::Failed(LogOnly::new("the URL could not be parsed".to_string()))
        })?;
        let port = url.port_or_known_default().unwrap_or(443);
        // The request's timeout is what is left of the whole chain's; resolving
        // the name spends part of it, and the connection gets only the rest.
        let started = self.clock.now();
        let (domain, ips) = match url.host() {
            Some(Host::Domain(name)) => (
                Some(name.to_string()),
                self.resolver.resolve(name, port, request.timeout).await?,
            ),
            Some(Host::Ipv4(ip)) => (None, vec![IpAddr::V4(ip)]),
            Some(Host::Ipv6(ip)) => (None, vec![IpAddr::V6(ip)]),
            None => return Err(HttpError::ConnectFailed),
        };
        let pin = Pin {
            domain,
            addrs: pin_addresses(&ips, port, policy)?,
        };
        let prepared = build_request(&request, url)?;
        let left = request
            .timeout
            .checked_sub(self.clock.now().saturating_duration_since(started))
            .filter(|left| !left.is_zero())
            .ok_or(HttpError::Timeout)?;
        let response = self.executor.execute(&pin, prepared, left).await?;
        read_response(response, request.body_cap, &request.url).await
    }
}

impl Default for ReqwestHttp {
    fn default() -> Self {
        Self::new()
    }
}

impl Debug for ReqwestHttp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqwestHttp")
            .field("resolver", &self.resolver)
            .field("executor", &self.executor)
            .finish_non_exhaustive()
    }
}

/// Converts a hub request into a `reqwest` one.
///
/// # Errors
///
/// [`HttpError::Failed`] for a header name or value `reqwest` refuses.
pub fn build_request(request: &HubRequest, url: Url) -> Result<reqwest::Request, HttpError> {
    let method = match request.method {
        Method::Get => reqwest::Method::GET,
        Method::Post => reqwest::Method::POST,
    };
    let mut prepared = reqwest::Request::new(method, url);
    for (name, value) in &request.headers {
        let name = reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
            HttpError::Failed(LogOnly::new("a header name is not valid".to_string()))
        })?;
        let value = reqwest::header::HeaderValue::from_str(value).map_err(|_| {
            HttpError::Failed(LogOnly::new("a header value is not valid".to_string()))
        })?;
        prepared.headers_mut().append(name, value);
    }
    if let Some(body) = &request.body {
        *prepared.body_mut() = Some(reqwest::Body::from(body.clone()));
    }
    Ok(prepared)
}

/// Reads a response, keeping at most `cap` bytes of the body and saying so when
/// there was more.
///
/// # Errors
///
/// [`HttpError::Failed`] when the body stream breaks, [`HttpError::Timeout`]
/// when the client's timeout fires while reading.
pub async fn read_response(
    mut response: reqwest::Response,
    cap: usize,
    url: &str,
) -> Result<HubResponse, HttpError> {
    let status = response.status().as_u16();
    let headers: Vec<(String, String)> = response
        .headers()
        .iter()
        .filter_map(|(name, value)| {
            Some((name.as_str().to_string(), value.to_str().ok()?.to_string()))
        })
        .collect();
    let mut body: Vec<u8> = Vec::new();
    let mut truncated = false;
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| map_error_parts(error.is_timeout(), false, &error.to_string()))?
    {
        let room = cap.saturating_sub(body.len());
        if chunk.len() > room {
            body.extend_from_slice(&chunk[..room]);
            truncated = true;
            break;
        }
        body.extend_from_slice(&chunk);
    }
    let mut out = HubResponse::new(status, body, url);
    out.headers = headers;
    out.truncated = truncated;
    Ok(out)
}

#[async_trait]
impl Http for ReqwestHttp {
    async fn send(
        &self,
        request: HubRequest,
        policy: &EndpointPolicy,
    ) -> Result<HubResponse, HttpError> {
        follow_redirects(request, policy, &self.headers, &*self.clock, |hop| {
            self.hop(hop, policy)
        })
        .await
    }
}

#[cfg(test)]
#[path = "reqwest_http_test.rs"]
mod tests;
