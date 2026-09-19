//! Reusable, provider-neutral [`ChatModel`] decorators and observers.

use std::sync::Arc;

use async_trait::async_trait;

use super::{
    ChatModel, ModelCallCorrelation, ModelProfile, ModelRequest, ModelResponse, ModelStream,
    ModelStreamItem, ResolvedModelRoute,
};
use crate::Result;
use crate::usage::Usage;

/// Request defaults applied only when the caller leaves a field unset.
///
/// This deliberately contains only model-boundary knobs. Provider credentials,
/// routing policy, and pricing remain host responsibilities.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelRequestDefaults {
    /// Default provider model or host route identifier.
    pub model: Option<String>,
    /// Default sampling temperature.
    pub temperature: Option<f64>,
}

impl ModelRequestDefaults {
    /// Applies defaults without replacing caller-supplied values.
    #[must_use]
    pub fn apply(&self, mut request: ModelRequest) -> ModelRequest {
        if request.model.is_none() {
            request.model.clone_from(&self.model);
        }
        if request.temperature.is_none() {
            request.temperature = self.temperature;
        }
        request
    }
}

/// Decorates a model with request defaults and an optional profile override.
pub struct ProfileOverrideModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    profile: ModelProfile,
    defaults: ModelRequestDefaults,
}

impl<State: Send + Sync> std::fmt::Debug for ProfileOverrideModel<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProfileOverrideModel")
            .field("profile", &self.profile)
            .field("defaults", &self.defaults)
            .finish_non_exhaustive()
    }
}

impl<State: Send + Sync> ProfileOverrideModel<State> {
    /// Wraps `inner`, returning `profile` from [`ChatModel::profile`].
    #[must_use]
    pub fn new(inner: Arc<dyn ChatModel<State>>, profile: ModelProfile) -> Self {
        Self {
            inner,
            profile,
            defaults: ModelRequestDefaults::default(),
        }
    }

    /// Adds defaults applied only where a request has no explicit value.
    #[must_use]
    pub fn with_defaults(mut self, defaults: ModelRequestDefaults) -> Self {
        self.defaults = defaults;
        self
    }

    /// Sets a default request model/route identifier.
    #[must_use]
    pub fn with_request_model(mut self, model: impl Into<String>) -> Self {
        self.defaults.model = Some(model.into());
        self
    }

    /// Sets a default sampling temperature.
    #[must_use]
    pub fn with_request_temperature(mut self, temperature: f64) -> Self {
        self.defaults.temperature = Some(temperature);
        self
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for ProfileOverrideModel<State> {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }

    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse> {
        self.inner.invoke(state, self.defaults.apply(request)).await
    }

    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        self.inner.stream(state, self.defaults.apply(request)).await
    }
}

/// Decorates a model with a hard maximum-output-token cap.
pub struct MaxTokensModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    max_tokens: u32,
}

impl<State: Send + Sync> std::fmt::Debug for MaxTokensModel<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MaxTokensModel")
            .field("max_tokens", &self.max_tokens)
            .finish_non_exhaustive()
    }
}

impl<State: Send + Sync> MaxTokensModel<State> {
    /// Wraps `inner` with a maximum output-token cap.
    #[must_use]
    pub fn new(inner: Arc<dyn ChatModel<State>>, max_tokens: u32) -> Self {
        Self { inner, max_tokens }
    }

    fn cap(&self, mut request: ModelRequest) -> ModelRequest {
        request.max_tokens = Some(
            request
                .max_tokens
                .map_or(self.max_tokens, |current| current.min(self.max_tokens)),
        );
        request
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for MaxTokensModel<State> {
    fn profile(&self) -> Option<&ModelProfile> {
        self.inner.profile()
    }

    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }

    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse> {
        self.inner.invoke(state, self.cap(request)).await
    }

    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        self.inner.stream(state, self.cap(request)).await
    }
}

/// Decorates a model by recording its concrete provider/model/route identity.
///
/// The decorator only stamps facts supplied by its host. It never chooses a
/// route, performs a fallback, or accesses host credentials.
pub struct RouteRecordingModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    route: ResolvedModelRoute,
}

impl<State: Send + Sync> std::fmt::Debug for RouteRecordingModel<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RouteRecordingModel")
            .field("route", &self.route)
            .finish_non_exhaustive()
    }
}

impl<State: Send + Sync> RouteRecordingModel<State> {
    /// Wraps `inner` and stamps every success response and stream with `route`.
    #[must_use]
    pub fn new(inner: Arc<dyn ChatModel<State>>, route: ResolvedModelRoute) -> Self {
        Self { inner, route }
    }

    fn stamp_response(
        &self,
        mut response: ModelResponse,
        correlation: Option<ModelCallCorrelation>,
    ) -> ModelResponse {
        if response.correlation.is_none() {
            response.correlation = correlation;
        }
        response.resolved_route = Some(self.route.clone());
        response
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for RouteRecordingModel<State> {
    fn profile(&self) -> Option<&ModelProfile> {
        self.inner.profile()
    }

    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }

    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse> {
        let correlation = request.correlation.clone();
        let response = self.inner.invoke(state, request).await?;
        Ok(self.stamp_response(response, correlation))
    }

    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        let correlation = request.correlation.clone();
        let stream = self
            .inner
            .stream(state, request)
            .await?
            .with_resolved_route(self.route.clone());
        let stream = if let Some(correlation) = correlation.clone() {
            stream.with_correlation(correlation)
        } else {
            stream
        };
        let route = self.route.clone();
        Ok(stream.map_items(move |item| match item {
            ModelStreamItem::Completed(mut response) => {
                if response.correlation.is_none() {
                    response.correlation = correlation.clone();
                }
                response.resolved_route = Some(route.clone());
                ModelStreamItem::Completed(response)
            }
            other => other,
        }))
    }
}

/// The terminal observation for one logical model call.
#[derive(Clone, Debug, PartialEq)]
pub enum ModelCallObservation {
    /// A non-cached successful call that did not change route.
    Succeeded {
        /// Stable run/model-call correlation, when the host supplied one.
        correlation: Option<ModelCallCorrelation>,
        /// Concrete route, when a route decorator or provider supplied one.
        route: Option<ResolvedModelRoute>,
        /// Provider-reported usage, when available.
        usage: Option<Usage>,
    },
    /// A successful response served from a cache.
    CacheHit {
        /// Stable run/model-call correlation, when the host supplied one.
        correlation: Option<ModelCallCorrelation>,
        /// Concrete route, when available.
        route: Option<ResolvedModelRoute>,
        /// Provider-reported usage, when available.
        usage: Option<Usage>,
    },
    /// A success where the resolved route differs from the requested route.
    Fallback {
        /// Stable run/model-call correlation, when the host supplied one.
        correlation: Option<ModelCallCorrelation>,
        /// Route requested by the caller.
        requested_route: String,
        /// Concrete route that handled the call.
        resolved_route: ResolvedModelRoute,
        /// Provider-reported usage, when available.
        usage: Option<Usage>,
    },
    /// A model invocation or terminal stream failure.
    Failed {
        /// Stable run/model-call correlation, when the host supplied one.
        correlation: Option<ModelCallCorrelation>,
        /// Credential-safe normalized failure text.
        message: String,
    },
}

/// Receives exactly one terminal observation for each completed model call.
pub trait ModelObserver: Send + Sync {
    /// Handles a terminal observation. Implementations must not panic.
    fn observe(&self, observation: ModelCallObservation);
}

/// Decorates a model to emit one generic terminal observation per call.
pub struct ObservingModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    observer: Arc<dyn ModelObserver>,
}

/// The small request projection needed after a model call completes.
#[derive(Clone, Debug, Default)]
struct ObservedRequest {
    correlation: Option<ModelCallCorrelation>,
    requested_route: Option<String>,
}

impl ObservedRequest {
    fn from_request(request: &ModelRequest) -> Self {
        Self {
            correlation: request.correlation.clone(),
            requested_route: request.requested_route.clone(),
        }
    }
}

impl<State: Send + Sync> std::fmt::Debug for ObservingModel<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ObservingModel")
            .finish_non_exhaustive()
    }
}

impl<State: Send + Sync> ObservingModel<State> {
    /// Wraps `inner` and sends its terminal outcomes to `observer`.
    #[must_use]
    pub fn new(inner: Arc<dyn ChatModel<State>>, observer: Arc<dyn ModelObserver>) -> Self {
        Self { inner, observer }
    }
}

fn observation_for_response(
    request: &ObservedRequest,
    response: &ModelResponse,
) -> ModelCallObservation {
    let correlation = response
        .correlation
        .clone()
        .or_else(|| request.correlation.clone());
    let route = response.resolved_route.clone();
    if response.served_from_cache {
        return ModelCallObservation::CacheHit {
            correlation,
            route,
            usage: response.usage,
        };
    }
    if let (Some(requested_route), Some(resolved_route)) =
        (request.requested_route.clone(), route.clone())
        && requested_route != resolved_route.route
    {
        return ModelCallObservation::Fallback {
            correlation,
            requested_route,
            resolved_route,
            usage: response.usage,
        };
    }
    ModelCallObservation::Succeeded {
        correlation,
        route,
        usage: response.usage,
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for ObservingModel<State> {
    fn profile(&self) -> Option<&ModelProfile> {
        self.inner.profile()
    }

    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }

    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse> {
        let observed_request = ObservedRequest::from_request(&request);
        match self.inner.invoke(state, request).await {
            Ok(response) => {
                self.observer
                    .observe(observation_for_response(&observed_request, &response));
                Ok(response)
            }
            Err(error) => {
                self.observer.observe(ModelCallObservation::Failed {
                    correlation: observed_request.correlation,
                    message: error.to_string(),
                });
                Err(error)
            }
        }
    }

    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        let observed_request = ObservedRequest::from_request(&request);
        let stream = match self.inner.stream(state, request).await {
            Ok(stream) => stream,
            Err(error) => {
                self.observer.observe(ModelCallObservation::Failed {
                    correlation: observed_request.correlation,
                    message: error.to_string(),
                });
                return Err(error);
            }
        };
        let observer = self.observer.clone();
        let stream_metadata = stream.metadata().clone();
        let mut observed = false;
        Ok(stream.map_items(move |item| {
            if observed {
                return item;
            }
            match &item {
                ModelStreamItem::Completed(response) => {
                    observed = true;
                    let mut response = response.clone();
                    if response.correlation.is_none() {
                        response.correlation = stream_metadata.correlation.clone();
                    }
                    if response.resolved_route.is_none() {
                        response.resolved_route = stream_metadata.resolved_route.clone();
                    }
                    observer.observe(observation_for_response(&observed_request, &response));
                }
                ModelStreamItem::Failed(message) => {
                    observed = true;
                    observer.observe(ModelCallObservation::Failed {
                        correlation: stream_metadata
                            .correlation
                            .clone()
                            .or_else(|| observed_request.correlation.clone()),
                        message: message.clone(),
                    });
                }
                ModelStreamItem::ProviderFailed(error) => {
                    observed = true;
                    observer.observe(ModelCallObservation::Failed {
                        correlation: stream_metadata
                            .correlation
                            .clone()
                            .or_else(|| observed_request.correlation.clone()),
                        message: error.to_string(),
                    });
                }
                _ => {}
            }
            item
        }))
    }
}
