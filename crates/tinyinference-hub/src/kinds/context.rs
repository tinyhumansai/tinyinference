//! What a driver is handed to do its work: the ports, and the provider being
//! talked to.

use std::fmt;

use crate::catalogue::ANTHROPIC_VERSION;
use crate::descriptor::ProviderDescriptor;
use crate::error::{HubError, ProviderFailure, TransportCondition, classify_transport};
use crate::ids::{KindId, ModelId, Slug};
use crate::policy::{EndpointPolicy, HeaderPolicy};
use crate::ports::{Clock, Http, HubRequest, HubResponse};
use crate::secret::Secret;
use crate::taxonomy::{AuthStyle, ProviderGroup};

use super::KindDriver;

/// The ports and policies one operation runs under. Cheap to build; borrows
/// everything.
#[non_exhaustive]
#[derive(Clone, Copy)]
pub struct DriverContext<'a> {
    /// The transport.
    pub http: &'a dyn Http,
    /// The endpoint policy every request is held to.
    pub policy: &'a EndpointPolicy,
    /// Time.
    pub clock: &'a dyn Clock,
    /// Which headers may go where.
    pub headers: &'a HeaderPolicy,
    /// The product-identity header (`name`, `value`), sent only to first-party
    /// hosts (guard G26). `None` means the host sends none.
    pub product: Option<&'a (String, String)>,
}

impl<'a> DriverContext<'a> {
    /// A context with no product header.
    pub fn new(
        http: &'a dyn Http,
        policy: &'a EndpointPolicy,
        clock: &'a dyn Clock,
        headers: &'a HeaderPolicy,
    ) -> Self {
        Self {
            http,
            policy,
            clock,
            headers,
            product: None,
        }
    }

    /// This context with a product-identity header.
    #[must_use]
    pub fn with_product(mut self, product: &'a (String, String)) -> Self {
        self.product = Some(product);
        self
    }
}

impl fmt::Debug for DriverContext<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DriverContext")
            .field("policy", self.policy)
            .field("product_header", &self.product.map(|(name, _)| name))
            .finish()
    }
}

/// The provider an operation is about: everything a driver needs and nothing it
/// should not have (no record, no store).
#[non_exhaustive]
#[derive(Clone, Copy)]
pub struct Target<'a> {
    /// The routing key, named in `SignedOut` and in logs.
    pub slug: &'a Slug,
    /// The catalogue kind.
    pub kind: &'a KindId,
    /// The group.
    pub group: ProviderGroup,
    /// The endpoint, normalised.
    pub base_url: &'a str,
    /// How the credential is presented.
    pub auth: &'a AuthStyle,
    /// The credential, if one resolved.
    pub credential: Option<&'a Secret>,
    /// The model to use for a completion ping.
    pub model: Option<&'a ModelId>,
}

impl fmt::Debug for Target<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Target")
            .field("slug", self.slug)
            .field("kind", self.kind)
            .field("base_url", &crate::endpoint::redact_endpoint(self.base_url))
            .field("credential", &self.credential)
            .finish_non_exhaustive()
    }
}

impl<'a> Target<'a> {
    /// A target with no credential and no model.
    pub fn new(
        slug: &'a Slug,
        kind: &'a KindId,
        group: ProviderGroup,
        base_url: &'a str,
        auth: &'a AuthStyle,
    ) -> Self {
        Self {
            slug,
            kind,
            group,
            base_url,
            auth,
            credential: None,
            model: None,
        }
    }

    /// This target presenting `credential`.
    #[must_use]
    pub fn with_credential(mut self, credential: &'a Secret) -> Self {
        self.credential = Some(credential);
        self
    }

    /// This target with a model for a completion ping.
    #[must_use]
    pub fn with_model(mut self, model: &'a ModelId) -> Self {
        self.model = Some(model);
        self
    }
}

impl Target<'_> {
    /// The endpoint without a trailing slash, a query or a fragment.
    pub fn base(&self) -> &str {
        let base = self.without_fragment();
        let end = base.find('?').unwrap_or(base.len());
        base[..end].trim_end_matches('/')
    }

    /// The endpoint without a fragment: the part a request can carry.
    fn without_fragment(&self) -> &str {
        let base = self.base_url.trim();
        &base[..base.find('#').unwrap_or(base.len())]
    }

    /// The endpoint's own query string (`api-version=preview` on an Azure
    /// resource), without the `?`. A `?` inside a fragment is not a query.
    fn base_query(&self) -> Option<&str> {
        let (_, query) = self.without_fragment().split_once('?')?;
        Some(query).filter(|query| !query.is_empty())
    }

    /// `path` (with an optional `?query`) appended to the endpoint, keeping the
    /// endpoint's own query: `https://x/v1?api-version=preview` plus
    /// `/models?limit=5` is `https://x/v1/models?limit=5&api-version=preview`.
    /// String concatenation would put the path after the query and produce a
    /// URL the server cannot route.
    pub fn join(&self, path: &str) -> String {
        let (path, path_query) = match path.split_once('?') {
            Some((path, query)) => (path, Some(query)),
            None => (path, None),
        };
        let mut url = format!("{}{path}", self.base());
        let query: Vec<&str> = path_query
            .into_iter()
            .chain(self.base_query())
            .filter(|part| !part.is_empty())
            .collect();
        if !query.is_empty() {
            url.push('?');
            url.push_str(&query.join("&"));
        }
        url
    }

    /// The credential, trimmed, when it is non-empty.
    pub fn key(&self) -> Option<&str> {
        self.credential
            .map(|secret| secret.expose().trim())
            .filter(|key| !key.is_empty())
    }
}

/// Adds the header(s) `auth` uses to present `key`, returning whether a
/// credential was attached.
///
/// A missing or blank key attaches nothing (sending an empty header is worse
/// than sending none), and [`AuthStyle::None`] attaches nothing even when a key
/// is supplied: a keyless local runtime may be *given* a key, but Ollama in
/// particular answers spurious `401`s to one.
pub(crate) fn apply_auth(
    headers: &mut Vec<(String, String)>,
    auth: &AuthStyle,
    key: Option<&str>,
) -> bool {
    let Some(key) = key.map(str::trim).filter(|k| !k.is_empty()) else {
        return false;
    };
    match auth {
        AuthStyle::None => return false,
        AuthStyle::Bearer | AuthStyle::SessionJwt => {
            headers.push(("authorization".to_string(), format!("Bearer {key}")));
        }
        AuthStyle::XApiKey => headers.push(("x-api-key".to_string(), key.to_string())),
        AuthStyle::Anthropic => {
            headers.push(("x-api-key".to_string(), key.to_string()));
            headers.push((
                "anthropic-version".to_string(),
                ANTHROPIC_VERSION.to_string(),
            ));
        }
        AuthStyle::Custom(name) => {
            let name = name.trim();
            if name.is_empty() {
                return false;
            }
            headers.push((name.to_ascii_lowercase(), key.to_string()));
        }
    }
    true
}

impl DriverContext<'_> {
    /// A request to `url` for `target`: credential in the style the provider
    /// expects, the descriptor's extra headers, and the product header only if
    /// `url` is first-party. `credentialed` is set from what was attached, so
    /// the transport applies the cleartext and cross-origin rules.
    pub(crate) fn request(
        &self,
        descriptor: &ProviderDescriptor,
        target: &Target<'_>,
        mut request: HubRequest,
    ) -> HubRequest {
        request.credentialed = apply_auth(&mut request.headers, target.auth, target.key());
        for (name, value) in descriptor.extra_headers {
            request
                .headers
                .push(((*name).to_string(), (*value).to_string()));
        }
        if let Some((name, value)) = self.product {
            let mut product = vec![(name.clone(), value.clone())];
            self.headers
                .strip_product_header_unless_first_party(&mut product, &request.url);
            request.headers.extend(product);
        }
        // The policy is the authority on how long. (How much is the caller's to
        // say: a listing asks for the catalog cap, a ping for the answer cap.)
        request.timeout = self.policy.timeout;
        request
    }

    /// A `GET` for one request of a model-list read that began at `started`:
    /// the catalog size cap, and a timeout clamped to what the list deadline has
    /// left, so a chain of fallbacks (scoped then public, native then
    /// compatible) is bounded as a whole, not per request.
    ///
    /// # Errors
    ///
    /// A `timeout` failure once the deadline has passed.
    pub(crate) fn list_request(
        &self,
        descriptor: &ProviderDescriptor,
        target: &Target<'_>,
        url: String,
        started: std::time::Instant,
    ) -> Result<HubRequest, HubError> {
        let left = self.time_left(started)?;
        let mut request = self.request(
            descriptor,
            target,
            HubRequest::get(url).with_body_cap(self.policy.catalog_cap),
        );
        request.timeout = request.timeout.min(left);
        Ok(request)
    }

    /// How much of the list deadline is left for a read that started at
    /// `started`.
    ///
    /// # Errors
    ///
    /// A `timeout` failure once the deadline has passed: a paged read that keeps
    /// finding one more page must not hold its cache lock (and everyone queued
    /// on it) for pages times the per-request timeout.
    pub(crate) fn time_left(
        &self,
        started: std::time::Instant,
    ) -> Result<std::time::Duration, HubError> {
        let spent = self.clock.now().saturating_duration_since(started);
        self.policy
            .list_deadline
            .checked_sub(spent)
            .filter(|left| !left.is_zero())
            .ok_or_else(|| {
                HubError::Provider(classify_transport(
                    TransportCondition::Timeout,
                    "the model list took longer than the list deadline",
                ))
            })
    }

    /// Sends `request`, turning a transport failure into a hub error and a
    /// non-2xx answer into a classified [`ProviderFailure`].
    ///
    /// A failure body is cut to the policy's failure cap before it is
    /// classified (an error message is never legitimately large), and a
    /// redirect the transport did not follow is an endpoint failure: the
    /// endpoint did not serve where it said it would.
    pub(crate) async fn call<D: KindDriver + ?Sized>(
        &self,
        driver: &D,
        request: HubRequest,
    ) -> Result<HubResponse, HubError> {
        self.call_with(
            &|status, headers, body| driver.classify(status, headers, body),
            request,
        )
        .await
    }

    /// [`DriverContext::call`] with the classifier passed as a function, so the
    /// body is compiled once rather than once per driver type.
    pub(crate) async fn call_with(
        &self,
        classify: &Classifier<'_>,
        request: HubRequest,
    ) -> Result<HubResponse, HubError> {
        let response = self
            .http
            .send(request, self.policy)
            .await
            .map_err(crate::ports::HttpError::into_hub)?;
        if response.is_success() {
            return Ok(response);
        }
        if (300..400).contains(&response.status) {
            return Err(HubError::Provider(
                classify_transport(
                    TransportCondition::RedirectRefused,
                    "a redirect was not followed",
                )
                .with_status(response.status),
            ));
        }
        let cap = self.policy.fail_body_cap.min(response.body.len());
        let body = String::from_utf8_lossy(&response.body[..cap]).into_owned();
        let failure: ProviderFailure = classify(response.status, &response.header_pairs(), &body);
        Err(HubError::Provider(failure.with_truncated(
            response.truncated || cap < response.body.len(),
        )))
    }
}

/// A failure classifier: `(status, headers, body)` to a [`ProviderFailure`].
pub(crate) type Classifier<'a> =
    dyn Fn(u16, &[(&str, &str)], &str) -> ProviderFailure + Send + Sync + 'a;
