//! [`ScriptedHttp`]: an [`Http`] implementation that answers from a script.

use std::fmt;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;

use crate::endpoint::redact_endpoint;
use crate::error::{PolicyViolation, scrub_log_text};
use crate::policy::{EndpointPolicy, HeaderPolicy, check_address};
use crate::ports::{Clock, Http, HttpError, HubRequest, HubResponse, Method, follow_redirects};
use crate::secret::{REDACTED, Secret, is_credential_name};

use super::clock::FakeClock;
use super::script::{Match, Script, Scripted};

/// A request as the hub sent it, with every credential replaced.
#[non_exhaustive]
#[derive(Clone)]
pub struct RecordedRequest {
    /// The method.
    pub method: Method,
    /// The URL, with credential-named query values masked.
    pub url: String,
    /// Header pairs, credential values replaced by `<redacted>`.
    pub headers: Vec<(String, String)>,
    /// The body, scrubbed of URLs' credentials and echoed auth values.
    pub body: Option<String>,
    /// Whether the request was marked as carrying a credential.
    pub credentialed: bool,
    /// The fake wall clock when it was sent.
    pub at_wall_ms: u64,
    raw_headers: Vec<(String, String)>,
}

impl RecordedRequest {
    /// The first header called `name`, credential values redacted.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Whether any header of this request carried `secret`.
    pub fn carried(&self, secret: &Secret) -> bool {
        !secret.is_empty()
            && self
                .raw_headers
                .iter()
                .any(|(_, value)| value.contains(secret.expose()))
    }
}

impl fmt::Debug for RecordedRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordedRequest")
            .field("method", &self.method)
            .field("url", &self.url)
            .field("headers", &self.headers)
            .field("credentialed", &self.credentialed)
            .finish()
    }
}

struct Rule {
    id: usize,
    matcher: Match,
    script: Script,
    cursor: usize,
    hits: usize,
}

impl Rule {
    fn available(&self) -> bool {
        match &self.script {
            Script::Always(_) => true,
            Script::Sequence(items) => self.cursor < items.len(),
        }
    }

    fn take(&mut self) -> Scripted {
        self.hits += 1;
        match &self.script {
            Script::Always(scripted) => scripted.clone(),
            Script::Sequence(items) => {
                let scripted = items[self.cursor].clone();
                self.cursor += 1;
                scripted
            }
        }
    }
}

#[derive(Default)]
struct State {
    rules: Vec<Rule>,
    next_id: usize,
    log: Vec<RecordedRequest>,
    refused: Vec<(String, PolicyViolation)>,
}

/// An [`Http`] that answers from rules instead of the network.
///
/// It applies the endpoint policy exactly as the port's contract requires: the
/// first URL, every redirect hop, every address a [`Scripted::ResolvesTo`]
/// names. A request no rule matches **panics**: an unscripted request is a
/// test bug, and a silent default would let a test pass by accident.
///
/// Newest rule wins, so a test can flip a provider's behaviour by adding a
/// rule for the same URL.
pub struct ScriptedHttp {
    clock: FakeClock,
    headers: HeaderPolicy,
    state: Mutex<State>,
}

impl ScriptedHttp {
    /// A transport with no rules, sharing `clock` (scripted latency advances it).
    pub fn new(clock: FakeClock) -> Self {
        Self {
            clock,
            headers: HeaderPolicy::builtin(),
            state: Mutex::new(State::default()),
        }
    }

    fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds a rule and returns its id (for [`ScriptedHttp::hits`]). It takes
    /// priority over every earlier rule.
    pub fn route(&self, matcher: Match, script: impl Into<Script>) -> usize {
        let mut state = self.state();
        let id = state.next_id;
        state.next_id += 1;
        state.rules.insert(
            0,
            Rule {
                id,
                matcher,
                script: script.into(),
                cursor: 0,
                hits: 0,
            },
        );
        id
    }

    /// Answers a `GET` to `url` with a 200 and this JSON body, every time.
    pub fn ok_json(&self, url: impl Into<String>, body: &serde_json::Value) -> usize {
        self.route(Match::get(url), Scripted::json(200, body))
    }

    /// Removes every rule (the request log is kept).
    pub fn clear_rules(&self) {
        self.state().rules.clear();
    }

    /// How many requests rule `id` has answered.
    pub fn hits(&self, id: usize) -> usize {
        self.state()
            .rules
            .iter()
            .find(|rule| rule.id == id)
            .map_or(0, |rule| rule.hits)
    }

    /// Every request sent, oldest first, one per hop.
    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.state().log.clone()
    }

    /// The requests sent from index `start` on (a cheap way to read only what
    /// is new since a previous look).
    pub fn requests_from(&self, start: usize) -> Vec<RecordedRequest> {
        self.state().log.iter().skip(start).cloned().collect()
    }

    /// How many requests the transport refused.
    pub fn refused_count(&self) -> usize {
        self.state().refused.len()
    }

    /// How many requests were sent, one per hop.
    pub fn request_count(&self) -> usize {
        self.state().log.len()
    }

    /// The requests the policy refused before sending, as
    /// `(redacted url, violation)`.
    pub fn refused(&self) -> Vec<(String, PolicyViolation)> {
        self.state().refused.clone()
    }

    fn record(&self, request: &HubRequest, at_wall_ms: u64) {
        let redacted: Vec<(String, String)> = request
            .headers
            .iter()
            .map(|(name, value)| {
                let secret = self.headers.is_credential_header(name) || is_credential_name(name);
                (
                    name.clone(),
                    if secret {
                        REDACTED.to_string()
                    } else {
                        value.clone()
                    },
                )
            })
            .collect();
        self.state().log.push(RecordedRequest {
            method: request.method,
            url: redact_endpoint(&request.url),
            headers: redacted,
            body: request
                .body
                .as_ref()
                .map(|b| scrub_log_text(&String::from_utf8_lossy(b))),
            credentialed: request.credentialed,
            at_wall_ms,
            raw_headers: request.headers.clone(),
        });
    }

    fn one_hop(
        &self,
        request: HubRequest,
        policy: &EndpointPolicy,
    ) -> Result<HubResponse, HttpError> {
        let scripted = {
            let mut state = self.state();
            // The first rule that matches answers, exhausted or not: a rule
            // that has run dry must not quietly hand the request to an older,
            // broader rule, or a driver making one request too many would pass.
            let position = state
                .rules
                .iter()
                .position(|rule| rule.matcher.matches(&request));
            let url = redact_endpoint(&request.url);
            let method = request.method;
            match position {
                Some(index) if state.rules[index].available() => state.rules[index].take(),
                Some(_) => {
                    drop(state);
                    panic!("ScriptedHttp: the script for {method} {url} is exhausted");
                }
                None => {
                    drop(state);
                    panic!("ScriptedHttp: no rule answers {method} {url}");
                }
            }
        };
        let at_wall_ms = self.clock.wall_ms();
        let result = self.answer(scripted, &request, policy, request.timeout);
        match &result {
            // Refused before connecting (a resolved address the policy forbids):
            // a real transport sends nothing, so nothing is logged as sent, and
            // the refusal names the URL of *this* hop.
            Err(HttpError::Policy(violation)) => {
                self.state()
                    .refused
                    .push((redact_endpoint(&request.url), violation.clone()));
            }
            _ => self.record(&request, at_wall_ms),
        }
        result
    }

    fn answer(
        &self,
        scripted: Scripted,
        request: &HubRequest,
        policy: &EndpointPolicy,
        remaining: Duration,
    ) -> Result<HubResponse, HttpError> {
        let cap = request.body_cap;
        let respond = |code: u16, headers: Vec<(String, String)>, mut body: Vec<u8>| {
            let truncated = body.len() > cap;
            body.truncate(cap);
            let mut response = HubResponse::new(code, body, request.url.clone());
            response.headers = headers;
            response.truncated = truncated;
            Ok(response)
        };
        match scripted {
            Scripted::Status {
                code,
                headers,
                body,
            } => respond(code, headers, body),
            Scripted::Redirect { code, location } => {
                respond(code, vec![("location".to_string(), location)], Vec::new())
            }
            Scripted::Latency(by, inner) => {
                if by >= remaining {
                    self.clock.advance(remaining);
                    return Err(HttpError::Timeout);
                }
                self.clock.advance(by);
                self.answer(*inner, request, policy, remaining - by)
            }
            Scripted::Timeout => {
                self.clock.advance(remaining);
                Err(HttpError::Timeout)
            }
            Scripted::ConnectRefused | Scripted::DnsFail => Err(HttpError::ConnectFailed),
            Scripted::ResolvesTo(ips, inner) => {
                if ips.is_empty() {
                    return Err(HttpError::ConnectFailed);
                }
                for ip in ips {
                    if let Err(refusal) = check_address(ip, policy) {
                        return Err(HttpError::Policy(PolicyViolation::from(refusal)));
                    }
                }
                self.answer(*inner, request, policy, remaining)
            }
            Scripted::Oversize { bytes } => {
                let mut response =
                    HubResponse::new(200, vec![b'x'; bytes.min(cap)], request.url.clone());
                response.truncated = bytes > cap;
                Ok(response)
            }
            Scripted::Malformed(bytes) => respond(200, Vec::new(), bytes),
            Scripted::SlowStream { chunks, gap } => {
                let total = gap.saturating_mul(u32::try_from(chunks.len()).unwrap_or(u32::MAX));
                if total >= remaining {
                    self.clock.advance(remaining);
                    return Err(HttpError::Timeout);
                }
                self.clock.advance(total);
                respond(200, Vec::new(), chunks.concat())
            }
        }
    }
}

impl fmt::Debug for ScriptedHttp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = self.state();
        f.debug_struct("ScriptedHttp")
            .field("rules", &state.rules.len())
            .field("requests", &state.log.len())
            .finish()
    }
}

#[async_trait]
impl Http for ScriptedHttp {
    async fn send(
        &self,
        request: HubRequest,
        policy: &EndpointPolicy,
    ) -> Result<HubResponse, HttpError> {
        let url = request.url.clone();
        let refused_before = self.state().refused.len();
        let result = follow_redirects(request, policy, &self.headers, &self.clock, |hop| {
            let result = self.one_hop(hop, policy);
            async move { result }
        })
        .await;
        // A refusal by the redirect loop itself (the first URL, a hop's target)
        // is kept too, so a test can assert "this was refused" without
        // inspecting the error. A resolved-address refusal was already kept by
        // `one_hop`, under the URL of the hop that hit it.
        if let Err(HttpError::Policy(violation)) = &result
            && self.state().refused.len() == refused_before
        {
            self.state()
                .refused
                .push((redact_endpoint(&url), violation.clone()));
        }
        result
    }
}
