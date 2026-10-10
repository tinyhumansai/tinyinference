//! Ordered model failover before an attempt produces any content.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;

use super::{
    ChatModel, DeferredHandle, DeferredStatus, InputModality, InputSource, ModelProfile,
    ModelRequest, ModelResponse, ModelStream, ModelStreamItem,
};
use crate::{Error, Result};

/// Tries models in order, retaining the first successful attempt.
///
/// Requests are cloned without rewriting options, correlation or routing. Only
/// model/provider failures trigger another attempt; validation, unsupported input
/// and serialization errors are returned immediately. Streaming withholds opening
/// and usage events until content or a terminal success arrives. A failed opening
/// attempt is discarded, but once any content or tool boundary is exposed this
/// decorator never switches models. Dropping the returned stream drops the selected
/// provider stream, preserving its cancellation guard.
///
/// Identical profiles are retained. Heterogeneous profiles advertise only common
/// modalities and token limits, with optional capabilities disabled: one provider's
/// schema transforms, tool-id restrictions or prompt dialect must not be applied
/// to another. Transport input support is the intersection of every adapter.
pub struct FallbackModel<State: Send + Sync> {
    models: Vec<Arc<dyn ChatModel<State>>>,
    profile: ModelProfile,
}

const DEFERRED_ROUTE_KEY: &str = "_tinyinference_fallback_route";

#[derive(serde::Serialize, serde::Deserialize)]
struct DeferredRoute {
    version: u8,
    adapter: usize,
    identity: serde_json::Value,
    original: DeferredHandle,
}

fn adapter_identity<State: Send + Sync>(model: &dyn ChatModel<State>) -> serde_json::Value {
    serde_json::json!({ "profile": model.profile(), "cache_identity": model.cache_identity() })
}

fn route_deferred(item: &mut ModelStreamItem, adapter: usize, identity: &serde_json::Value) {
    if let ModelStreamItem::Deferred(handle) = item {
        let route = DeferredRoute {
            version: 1,
            adapter,
            identity: identity.clone(),
            original: handle.clone(),
        };
        // Retain the original handle, including a provider's use of this metadata
        // key. Nested fallback chains can therefore unwrap one decorator at a time.
        handle.metadata.insert(
            DEFERRED_ROUTE_KEY.into(),
            serde_json::to_value(route).expect("deferred route is JSON"),
        );
    }
}

impl<State: Send + Sync> std::fmt::Debug for FallbackModel<State> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("FallbackModel")
            .field("attempts", &self.models.len())
            .field("profile", &self.profile)
            .finish_non_exhaustive()
    }
}

impl<State: Send + Sync> FallbackModel<State> {
    /// Creates a chain with a required primary model and ordered alternatives.
    #[must_use]
    pub fn new(
        primary: Arc<dyn ChatModel<State>>,
        fallbacks: Vec<Arc<dyn ChatModel<State>>>,
    ) -> Self {
        let mut models = Vec::with_capacity(fallbacks.len() + 1);
        models.push(primary);
        models.extend(fallbacks);
        let profiles: Vec<_> = models
            .iter()
            .map(|model| model.profile().cloned().unwrap_or_default())
            .collect();
        let mut profile = profiles[0].clone();
        if profiles.iter().any(|other| other != &profile) {
            profile = ModelProfile::default();
            for other in &profiles {
                profile.modalities.text_in &= other.modalities.text_in;
                profile.modalities.text_out &= other.modalities.text_out;
            }
            // Start optional modalities at true, then intersect. Unknown profiles
            // default these to false, so uncertainty never becomes a guarantee.
            profile.modalities.image_in = profiles.iter().all(|p| p.modalities.image_in);
            profile.modalities.image_out = profiles.iter().all(|p| p.modalities.image_out);
            profile.modalities.audio_in = profiles.iter().all(|p| p.modalities.audio_in);
            profile.modalities.audio_out = profiles.iter().all(|p| p.modalities.audio_out);
            profile.modalities.video_in = profiles.iter().all(|p| p.modalities.video_in);
            profile.modalities.video_out = profiles.iter().all(|p| p.modalities.video_out);
            profile.modalities.document_in = profiles.iter().all(|p| p.modalities.document_in);
            profile.max_input_tokens = profiles
                .iter()
                .map(|p| p.max_input_tokens)
                .collect::<Option<Vec<_>>>()
                .and_then(|limits| limits.into_iter().min());
            profile.max_output_tokens = profiles
                .iter()
                .map(|p| p.max_output_tokens)
                .collect::<Option<Vec<_>>>()
                .and_then(|limits| limits.into_iter().min());
            // Hoisting is a restriction, not an optional capability.
            profile.hoists_system_messages = profiles.iter().any(|p| p.hoists_system_messages);
        }
        Self { models, profile }
    }
}

fn can_fallback(error: &Error) -> bool {
    match error {
        Error::Model(_) => true,
        Error::Provider(error) => {
            // retryable governs retrying this adapter/account. Authentication and
            // quota failures can still be served by an independent fallback.
            !matches!(error.status, Some(400 | 413 | 415 | 422))
                && !matches!(
                    error.code.as_deref(),
                    Some(
                        "invalid_request_error"
                            | "invalid_argument"
                            | "unsupported_operation"
                            | "context_length_exceeded"
                            | "context_window_exceeded"
                    )
                )
        }
        _ => false,
    }
}

fn visible(item: &ModelStreamItem) -> bool {
    match item {
        ModelStreamItem::MessageDelta(delta) => {
            !delta.text.is_empty() || !delta.reasoning.is_empty() || delta.tool_call.is_some()
        }
        ModelStreamItem::ToolCallDelta(_)
        | ModelStreamItem::BlockStart { .. }
        | ModelStreamItem::BlockDelta { .. }
        | ModelStreamItem::BlockEnd { .. } => true,
        _ => false,
    }
}

fn terminal(item: &ModelStreamItem) -> bool {
    matches!(
        item,
        ModelStreamItem::Completed(_)
            | ModelStreamItem::Failed(_)
            | ModelStreamItem::ProviderFailed(_)
            | ModelStreamItem::Deferred(_)
    )
}

/// Keep the original stream inside the wrapper, including its abort guard.
fn selected_stream(
    mut prefix: Vec<ModelStreamItem>,
    stream: ModelStream,
    identity: serde_json::Value,
    adapter: usize,
) -> ModelStream {
    for item in &mut prefix {
        route_deferred(item, adapter, &identity);
    }
    let metadata = stream.metadata().clone();
    let ended = prefix.last().is_some_and(terminal);
    let tail = futures::stream::unfold((stream, ended), move |(mut stream, ended)| {
        let identity = identity.clone();
        async move {
            if ended {
                return None;
            }
            match stream.next().await {
                Some(mut item) => {
                    route_deferred(&mut item, adapter, &identity);
                    let ended = terminal(&item);
                    Some((item, (stream, ended)))
                }
                None => Some((
                    ModelStreamItem::Failed("provider stream ended without a terminal item".into()),
                    (stream, true),
                )),
            }
        }
    });
    ModelStream::new(Box::pin(futures::stream::iter(prefix).chain(tail))).with_metadata(metadata)
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for FallbackModel<State> {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    fn supports_input(
        &self,
        modality: InputModality,
        mime_type: &str,
        source: InputSource,
    ) -> bool {
        self.models
            .iter()
            .all(|model| model.supports_input(modality, mime_type, source))
    }

    /// Returns the selected response unchanged, or the last eligible failure.
    async fn invoke(&self, state: &State, request: ModelRequest) -> Result<ModelResponse> {
        let mut last = Error::Model("model chain is empty".into());
        for model in &self.models {
            match model.invoke(state, request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) if can_fallback(&error) => last = error,
                Err(error) => return Err(error),
            }
        }
        Err(last)
    }

    /// Selects an attempt before exposing content, preserving stream metadata.
    ///
    /// # Errors
    /// Returns the final opening failure when no attempt produces content or a
    /// terminal success. After content is exposed, failures remain stream items.
    async fn stream(&self, state: &State, request: ModelRequest) -> Result<ModelStream> {
        let mut last = Error::Model("model chain is empty".into());
        for (adapter, model) in self.models.iter().enumerate() {
            let mut stream = match model.stream(state, request.clone()).await {
                Ok(stream) => stream,
                Err(error) if can_fallback(&error) => {
                    last = error;
                    continue;
                }
                Err(error) => return Err(error),
            };
            let mut prefix = Vec::new();
            loop {
                let item = match stream.next().await {
                    Some(item) => item,
                    None => {
                        last = Error::Model("provider stream ended without a terminal item".into());
                        break;
                    }
                };
                let failure = match item {
                    ModelStreamItem::Failed(message) => Some(Error::Model(message)),
                    ModelStreamItem::ProviderFailed(error) => {
                        Some(Error::Provider(Box::new(error)))
                    }
                    item => {
                        let selected = visible(&item) || terminal(&item);
                        prefix.push(item);
                        if selected {
                            return Ok(selected_stream(
                                prefix,
                                stream,
                                adapter_identity(model.as_ref()),
                                adapter,
                            ));
                        }
                        None
                    }
                };
                if let Some(error) = failure {
                    if !can_fallback(&error) {
                        return Err(error);
                    }
                    last = error;
                    break;
                }
            }
        }
        Err(last)
    }

    /// Polls the adapter that accepted this handle, without failing over the job.
    ///
    /// Routing is serialized in handle metadata, preserving every original
    /// provider field for delegation. Persisted handles can be polled after
    /// reconstruction with the same ordered adapters. Index and advertised
    /// identity must match; adapters without identity require the host to restore
    /// the same configuration. Unknown handles are rejected.
    async fn fetch_deferred(&self, handle: &DeferredHandle) -> Result<DeferredStatus> {
        let encoded = handle.metadata.get(DEFERRED_ROUTE_KEY).ok_or_else(|| {
            Error::Unsupported("deferred handle has no fallback routing metadata".into())
        })?;
        let route: DeferredRoute = serde_json::from_value(encoded.clone())?;
        let model = self.models.get(route.adapter).ok_or_else(|| {
            Error::Validation("deferred adapter is absent from this fallback chain".into())
        })?;
        if route.version != 1 || route.identity != adapter_identity(model.as_ref()) {
            return Err(Error::Validation(
                "deferred adapter identity or route version does not match".into(),
            ));
        }
        model.fetch_deferred(&route.original).await
    }
}
