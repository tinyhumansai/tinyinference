//! The scripting vocabulary of [`ScriptedHttp`](super::ScriptedHttp).

use std::net::IpAddr;
use std::time::Duration;

use crate::ports::{HubRequest, Method};

/// One scripted answer.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Scripted {
    /// An HTTP response.
    Status {
        /// The status.
        code: u16,
        /// Response headers.
        headers: Vec<(String, String)>,
        /// The body.
        body: Vec<u8>,
    },
    /// A redirect to `location`.
    Redirect {
        /// `301`, `302`, `303`, `307` or `308`.
        code: u16,
        /// The `Location` header value.
        location: String,
    },
    /// Takes `Duration` of fake time before the wrapped answer arrives. Past the
    /// request's timeout it becomes [`Scripted::Timeout`].
    Latency(Duration, Box<Scripted>),
    /// The request times out.
    Timeout,
    /// The connection is refused.
    ConnectRefused,
    /// DNS does not resolve.
    DnsFail,
    /// The host resolves to these addresses; each is checked against the
    /// policy before the wrapped answer is used (anti DNS-rebinding).
    ResolvesTo(Vec<IpAddr>, Box<Scripted>),
    /// A body of this many bytes of `x`, so a cap can be exercised.
    Oversize {
        /// The size.
        bytes: usize,
    },
    /// A 200 with arbitrary (usually malformed) bytes.
    Malformed(Vec<u8>),
    /// A 200 whose body arrives in chunks with `gap` between them, so a slow
    /// stream can run past a timeout.
    SlowStream {
        /// The chunks, concatenated into the body.
        chunks: Vec<Vec<u8>>,
        /// The pause before each chunk.
        gap: Duration,
    },
}

impl Scripted {
    /// A status with a JSON body and a JSON content type.
    pub fn json(code: u16, body: &serde_json::Value) -> Self {
        Self::Status {
            code,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
            body: body.to_string().into_bytes(),
        }
    }

    /// A status with a text body and no headers.
    pub fn text(code: u16, body: impl Into<String>) -> Self {
        Self::Status {
            code,
            headers: Vec::new(),
            body: body.into().into_bytes(),
        }
    }

    /// A status with headers and a text body.
    pub fn with_headers(code: u16, headers: &[(&str, &str)], body: impl Into<String>) -> Self {
        Self::Status {
            code,
            headers: headers
                .iter()
                .map(|(n, v)| ((*n).to_string(), (*v).to_string()))
                .collect(),
            body: body.into().into_bytes(),
        }
    }

    /// A redirect.
    pub fn redirect(code: u16, location: impl Into<String>) -> Self {
        Self::Redirect {
            code,
            location: location.into(),
        }
    }

    /// This answer after `by` of fake time.
    #[must_use]
    pub fn after(self, by: Duration) -> Self {
        Self::Latency(by, Box::new(self))
    }

    /// This answer once the host resolves to `ips`.
    #[must_use]
    pub fn resolving_to(self, ips: Vec<IpAddr>) -> Self {
        Self::ResolvesTo(ips, Box::new(self))
    }
}

/// How many answers a rule has and what happens when they run out.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Script {
    /// The same answer every time.
    Always(Scripted),
    /// These answers in order; a request after the last one is an unscripted
    /// request and panics, so a test cannot pass by accident.
    Sequence(Vec<Scripted>),
}

impl From<Scripted> for Script {
    fn from(scripted: Scripted) -> Self {
        Self::Always(scripted)
    }
}

/// How a rule names the requests it answers.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    method: Option<Method>,
    url: String,
    exact: bool,
    headers: Vec<(String, Option<String>)>,
    absent: Vec<String>,
}

impl Match {
    fn new(method: Option<Method>, url: impl Into<String>, exact: bool) -> Self {
        Self {
            method,
            url: url.into(),
            exact,
            headers: Vec::new(),
            absent: Vec::new(),
        }
    }

    /// A `GET` to exactly this URL (query string included).
    pub fn get(url: impl Into<String>) -> Self {
        Self::new(Some(Method::Get), url, true)
    }

    /// A `POST` to exactly this URL.
    pub fn post(url: impl Into<String>) -> Self {
        Self::new(Some(Method::Post), url, true)
    }

    /// Any method to a URL starting with this prefix.
    pub fn prefix(url: impl Into<String>) -> Self {
        Self::new(None, url, false)
    }

    /// Only requests carrying this header with exactly this value.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.push((name.into(), Some(value.into())));
        self
    }

    /// Only requests carrying this header, whatever its value.
    #[must_use]
    pub fn with_header_present(mut self, name: impl Into<String>) -> Self {
        self.headers.push((name.into(), None));
        self
    }

    /// Only requests **not** carrying this header.
    #[must_use]
    pub fn without_header(mut self, name: impl Into<String>) -> Self {
        self.absent.push(name.into());
        self
    }

    pub(super) fn matches(&self, request: &HubRequest) -> bool {
        if self.method.is_some_and(|m| m != request.method) {
            return false;
        }
        let url_ok = if self.exact {
            request.url == self.url
        } else {
            request.url.starts_with(&self.url)
        };
        url_ok
            && self
                .headers
                .iter()
                .all(|(name, value)| match (request.header(name), value) {
                    (Some(actual), Some(expected)) => actual == expected,
                    (Some(_), None) => true,
                    (None, _) => false,
                })
            && self
                .absent
                .iter()
                .all(|name| request.header(name).is_none())
    }
}
