//! Client configuration and measured result types.

use std::{fmt, sync::Arc, time::Duration};

use crate::{Error, EvaluationResponse};

const DEFAULT_SYSTEM_ONE_PATH: &str = "v1/systemone";
const TINYHUMANS_SYSTEM_ONE_PATH: &str = "agent-integrations/openrouter/systemone";
const OPENJEV_BASE_URL: &str = "https://api.openjev.sh";
const SURPLUS_BASE_URL: &str = "https://api.surplusintelligence.ai";
const SURPLUS_DECISIONS_PATH: &str = "v1/decisions";

/// Hosted or self-hosted System One API provider.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum Provider {
    /// `TypeSafe`'s first-party System One API.
    #[default]
    TypeSafe,
    /// `OpenRouter`'s compatible System One API.
    OpenRouter,
    /// OpenJEV's public System One API.
    OpenJev,
    /// An operator-declared, Jev-compatible System One endpoint, such as a
    /// self-hosted open decision model.
    ///
    /// It has no conventional route, so an exact endpoint URL is required. The
    /// response must echo the requested model id, as with `TypeSafe`. The API
    /// key may be empty for an unauthenticated local server, in which case no
    /// `Authorization` header is sent.
    SelfHosted,
    /// Surplus Intelligence's decisions API, which serves `TypeSafe`'s Jev at
    /// `/v1/decisions`.
    ///
    /// Responses name the resolved release in the `typesafe/` namespace, as
    /// with `OpenRouter`. Surplus rejects Box questions, so the client
    /// emulates them (see [`BoxSupport`]).
    Surplus,
}

impl Provider {
    /// Whether this provider answers [`Question::Box`](crate::Question::Box)
    /// natively. A provider that does not gets Box questions emulated with
    /// Noul questions over image crops unless the caller overrides it with
    /// [`ClientConfig::with_box_support`].
    #[must_use]
    pub fn answers_box(self) -> bool {
        !matches!(self, Self::Surplus)
    }
}

/// How the client answers [`Question::Box`](crate::Question::Box).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BoxSupport {
    /// Send Box questions to the provider as they are.
    Native,
    /// Replace each Box question with Noul questions about crops of its image
    /// and build the Box answer from theirs. The caller still sends Box
    /// questions and receives [`Answer::Box`](crate::Answer::Box) answers.
    Emulated(BoxEmulation),
}

impl BoxSupport {
    /// The default for `provider`: [`Self::Native`] when it
    /// [answers Box](Provider::answers_box), else emulation with
    /// [`BoxEmulation::default`].
    #[must_use]
    pub fn for_provider(provider: Provider) -> Self {
        if provider.answers_box() {
            Self::Native
        } else {
            Self::Emulated(BoxEmulation::default())
        }
    }
}

/// Settings for emulating Box questions with Noul questions over image crops.
///
/// Each Box question's image is cut into a `grid` by `grid` set of cells, and
/// every cell is asked about as its own image. With `refine`, the box found is
/// cut into the same grid again in a second, sequential round of requests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoxEmulation {
    /// Cells per side, from 2 to 8; each round asks `grid * grid` Noul
    /// questions per Box question.
    pub grid: u32,
    /// Whether to run the second, refining round.
    pub refine: bool,
}

impl Default for BoxEmulation {
    /// A 4 by 4 grid with refinement: up to 32 Noul questions per Box
    /// question, and a box as fine as one sixteenth of a cell per side.
    fn default() -> Self {
        Self {
            grid: 4,
            refine: true,
        }
    }
}

/// Async Jev-compatible System One client.
#[derive(Clone)]
pub struct Client {
    pub(super) config: ClientConfig,
    pub(super) http: reqwest::Client,
    pub(super) transport: Arc<dyn super::HttpTransport>,
}

impl fmt::Debug for Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Client")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// HTTP client configuration.
#[derive(Clone)]
pub struct ClientConfig {
    pub(super) api_key: ApiKey,
    /// API origin or path prefix without the System One endpoint path.
    pub base_url: String,
    /// Provider-specific System One endpoint path.
    pub(super) system_one_path: &'static str,
    /// Exact evaluation endpoint, when the provider is exposed elsewhere.
    pub(super) endpoint_url: Option<String>,
    /// Provider-specific response validation behavior.
    pub provider: Provider,
    /// Total timeout for one HTTP attempt.
    pub timeout: Duration,
    /// Transient failure retry policy.
    pub retry: RetryPolicy,
    /// Sanitized product attribution for the exact `TinyHumans` proxy endpoint.
    pub(super) sdk_name: Option<String>,
    /// Caller override of the provider's Box support.
    pub(super) box_support: Option<BoxSupport>,
}

impl ClientConfig {
    /// Create production configuration for an API key.
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: ApiKey(api_key.into()),
            base_url: "https://api.typesafe.ai".to_owned(),
            system_one_path: DEFAULT_SYSTEM_ONE_PATH,
            endpoint_url: None,
            provider: Provider::TypeSafe,
            timeout: Duration::from_secs(30),
            retry: RetryPolicy::default(),
            sdk_name: None,
            box_support: None,
        }
    }

    /// Create configuration for `OpenRouter`'s System One API.
    #[must_use]
    pub fn openrouter(api_key: impl Into<String>) -> Self {
        Self {
            api_key: ApiKey(api_key.into()),
            base_url: "https://openrouter.ai/api".to_owned(),
            system_one_path: DEFAULT_SYSTEM_ONE_PATH,
            endpoint_url: None,
            provider: Provider::OpenRouter,
            timeout: Duration::from_secs(30),
            retry: RetryPolicy::default(),
            sdk_name: None,
            box_support: None,
        }
    }

    /// Create configuration for OpenJEV's System One API.
    #[must_use]
    pub fn openjev(api_key: impl Into<String>) -> Self {
        Self {
            api_key: ApiKey(api_key.into()),
            base_url: OPENJEV_BASE_URL.to_owned(),
            system_one_path: DEFAULT_SYSTEM_ONE_PATH,
            endpoint_url: None,
            provider: Provider::OpenJev,
            timeout: Duration::from_secs(30),
            retry: RetryPolicy::default(),
            sdk_name: None,
            box_support: None,
        }
    }

    /// Create configuration for Surplus Intelligence's decisions API at
    /// `https://api.surplusintelligence.ai/v1/decisions`.
    ///
    /// Surplus serves Jev under release ids such as `jev-1.14`; set
    /// [`EvaluationRequest::model`](crate::EvaluationRequest::model)
    /// accordingly. Box questions are emulated because Surplus rejects them.
    #[must_use]
    pub fn surplus(api_key: impl Into<String>) -> Self {
        Self {
            api_key: ApiKey(api_key.into()),
            base_url: SURPLUS_BASE_URL.to_owned(),
            system_one_path: SURPLUS_DECISIONS_PATH,
            endpoint_url: None,
            provider: Provider::Surplus,
            timeout: Duration::from_secs(30),
            retry: RetryPolicy::default(),
            sdk_name: None,
            box_support: None,
        }
    }

    /// Create configuration for a self-hosted, Jev-compatible System One
    /// endpoint.
    ///
    /// `endpoint_url` is the exact evaluation URL; no provider path is
    /// appended. It must be HTTPS, or plain HTTP on a literal loopback
    /// address. `api_key` is sent as a bearer token only to that endpoint and
    /// may be empty when the server needs no authentication. Self-hosted
    /// models have no default model id, so set
    /// [`EvaluationRequest::model`](crate::EvaluationRequest::model) to the
    /// model the server answers as.
    #[must_use]
    pub fn self_hosted(endpoint_url: impl Into<String>, api_key: impl Into<String>) -> Self {
        let endpoint_url = endpoint_url.into();
        Self {
            api_key: ApiKey(api_key.into()),
            base_url: endpoint_url.clone(),
            system_one_path: DEFAULT_SYSTEM_ONE_PATH,
            endpoint_url: Some(endpoint_url),
            provider: Provider::SelfHosted,
            timeout: Duration::from_secs(30),
            retry: RetryPolicy::default(),
            sdk_name: None,
            box_support: None,
        }
    }

    /// Create configuration for the `TinyHumans` `OpenRouter` System One proxy.
    ///
    /// The proxy accepts a `TinyHumans` API key and forwards typed Jev requests
    /// to `OpenRouter` while applying the caller's `TinyHumans` account limits.
    #[must_use]
    pub fn tinyhumans_openrouter(api_key: impl Into<String>) -> Self {
        Self {
            api_key: ApiKey(api_key.into()),
            base_url: "https://api.tinyhumans.ai".to_owned(),
            system_one_path: TINYHUMANS_SYSTEM_ONE_PATH,
            endpoint_url: None,
            provider: Provider::OpenRouter,
            timeout: Duration::from_secs(30),
            retry: RetryPolicy::default(),
            sdk_name: None,
            box_support: None,
        }
    }

    /// Replace the API key without exposing it through a public field.
    #[must_use]
    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = ApiKey(api_key.into());
        self
    }

    /// Use an exact evaluation endpoint instead of the provider's conventional
    /// System One path.
    #[must_use]
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Attribute `TinyHumans` proxy requests to a product. Invalid characters
    /// are dropped, the value is lowercased and capped at 64 ASCII bytes.
    /// Other endpoints never receive this header.
    #[must_use]
    pub fn with_sdk_name(mut self, raw: &str) -> Self {
        let name = raw
            .trim()
            .chars()
            .filter(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '.' | '_' | '-')
            })
            .take(64)
            .map(|character| character.to_ascii_lowercase())
            .collect::<String>();
        self.sdk_name = (!name.is_empty()).then_some(name);
        self
    }

    /// Answer Box questions as `support` says instead of as the provider's
    /// default ([`BoxSupport::for_provider`]).
    #[must_use]
    pub fn with_box_support(mut self, support: BoxSupport) -> Self {
        self.box_support = Some(support);
        self
    }

    /// How this configuration answers Box questions: the override from
    /// [`Self::with_box_support`], else the provider's default.
    #[must_use]
    pub fn box_support(&self) -> BoxSupport {
        self.box_support
            .unwrap_or_else(|| BoxSupport::for_provider(self.provider))
    }

    /// Returns the exact endpoint override, when configured.
    #[must_use]
    pub fn endpoint_url(&self) -> Option<&str> {
        self.endpoint_url.as_deref()
    }
}

impl fmt::Debug for ClientConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClientConfig")
            .field("api_key", &"[REDACTED]")
            .field("base_url", &redacted_url_origin(&self.base_url))
            .field("system_one_path", &self.system_one_path)
            .field(
                "endpoint_url",
                &self.endpoint_url.as_deref().map(redacted_url_origin),
            )
            .field("provider", &self.provider)
            .field("timeout", &self.timeout)
            .field("retry", &self.retry)
            .field("sdk_name", &self.sdk_name)
            .field("box_support", &self.box_support())
            .finish()
    }
}

fn redacted_url_origin(value: &str) -> String {
    reqwest::Url::parse(value)
        .map(|url| format!("{}/[REDACTED]", url.origin().ascii_serialization()))
        .unwrap_or_else(|_| "[REDACTED]".to_owned())
}

#[derive(Clone)]
pub(super) struct ApiKey(String);

impl ApiKey {
    pub(super) fn expose(&self) -> &str {
        &self.0
    }
}

/// Retry limits for transient failures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RetryPolicy {
    /// Additional attempts after the initial request.
    pub max_retries: u32,
    /// Delay used after the first retryable failure.
    pub initial_backoff: Duration,
    /// Maximum delay, including provider-requested delays.
    pub max_backoff: Duration,
}

impl RetryPolicy {
    pub(super) fn delay(self, attempts: u32) -> Duration {
        let exponent = attempts.saturating_sub(1).min(31);
        self.initial_backoff
            .saturating_mul(1_u32.checked_shl(exponent).unwrap_or(u32::MAX))
            .min(self.max_backoff)
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(2),
        }
    }
}

/// A validated response with transport measurements.
#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationResult {
    /// Typed, validated provider response.
    pub response: EvaluationResponse,
    /// Provider request id, when returned as a header.
    pub request_id: Option<String>,
    /// HTTP attempts including the successful or terminal attempt.
    pub attempts: u32,
    /// End-to-end elapsed time including retry delays.
    pub latency: Duration,
}

/// A failed evaluation with the attempts and elapsed time it spent.
#[derive(Debug, thiserror::Error)]
#[error("{error}")]
pub struct EvaluationFailure {
    /// Classified terminal failure.
    #[source]
    pub error: Box<Error>,
    /// HTTP attempts made before failure; zero for local request validation.
    pub attempts: u32,
    /// End-to-end elapsed time including retry delays.
    pub latency: Duration,
}
