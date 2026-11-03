//! Host supplied provider construction overrides.
//!
//! The host keeps provider registration and scope. This module accepts three
//! independent lookup ports so a host can preserve live, family-specific
//! lookup behavior (for example a scoped slot with a process-wide fallback).
//! Host configuration, credentials, approval state, and HTTP policy travel in
//! the caller's context and are never loaded or interpreted here.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;
use tinyinference_embeddings::{EmbeddingModel, Result as EmbeddingResult};

use crate::Result;
use crate::model::ChatModel;
use crate::model::discover::DiscoveryRequest;

/// A boxed future whose result is safe to move between executor threads.
pub type BoxFuture<T> = Pin<Box<dyn Future<Output = T> + Send + 'static>>;

/// Inputs a host chat factory uses to build a provider model.
///
/// Host-specific settings belong in `context`; the other fields describe the
/// resolved model construction and remain independent of a model turn's state
/// type (`ChatModel<()>`).
pub struct ChatModelRequest<'a, Ctx> {
    /// Workload role for which the model is being constructed.
    pub role: &'a str,
    /// Concrete model requested by the host, when one was selected.
    pub model_override: Option<&'a str>,
    /// Fallback model selected by the host.
    pub default_model: Option<&'a str>,
    /// Whether native tool calling is enabled for this model.
    pub native_tool_calling: bool,
    /// Conversation served by this model, when construction is turn-specific.
    pub thread_id: Option<&'a str>,
    /// Borrowed host context such as configuration, egress, and approvals.
    pub context: &'a Ctx,
}

impl<Ctx> std::fmt::Debug for ChatModelRequest<'_, Ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatModelRequest")
            .field("role", &self.role)
            .field("model_override", &self.model_override)
            .field("default_model", &self.default_model)
            .field("native_tool_calling", &self.native_tool_calling)
            .field("thread_id", &self.thread_id)
            .field("context", &"<host context>")
            .finish()
    }
}

/// Builds a chat model and reports the model id it resolved to.
pub type ChatModelFactory<Ctx> = Arc<
    dyn for<'a> Fn(&ChatModelRequest<'a, Ctx>) -> Result<(Arc<dyn ChatModel<()>>, String)>
        + Send
        + Sync,
>;

/// Inputs to an optional provider readiness check.
pub struct ReadinessRequest<Ctx> {
    /// Workload role to probe.
    pub role: String,
    /// Concrete model id resolved by the host.
    pub model: String,
    /// Owned host context needed by the probe.
    pub context: Ctx,
}

impl<Ctx> std::fmt::Debug for ReadinessRequest<Ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ReadinessRequest")
            .field("role", &self.role)
            .field("model", &self.model)
            .field("context", &"<host context>")
            .finish()
    }
}

/// Probes whether a provider can serve a role and model now.
pub type ReadinessProbe<Ctx> =
    Arc<dyn Fn(ReadinessRequest<Ctx>) -> BoxFuture<std::result::Result<(), String>> + Send + Sync>;

/// A host supplied chat provider factory and optional readiness probe.
pub struct ChatModelEntry<Ctx> {
    /// Builds the chat model.
    pub build: ChatModelFactory<Ctx>,
    /// Optionally checks whether the provider can serve a resolved model.
    pub probe: Option<ReadinessProbe<Ctx>>,
    /// Provider-owned model vocabulary, defaults, and capabilities.
    pub models: Option<ProviderModelMetadata<Ctx>>,
}

impl<Ctx> std::fmt::Debug for ChatModelEntry<Ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ChatModelEntry")
            .field("has_readiness_probe", &self.probe.is_some())
            .field("has_model_metadata", &self.models.is_some())
            .finish_non_exhaustive()
    }
}

impl<Ctx> Clone for ChatModelEntry<Ctx> {
    fn clone(&self) -> Self {
        Self {
            build: Arc::clone(&self.build),
            probe: self.probe.as_ref().map(Arc::clone),
            models: self.models.clone(),
        }
    }
}

/// Callback that selects a workload role for a provider model alias.
pub type ModelRoleLookup = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;
/// Callback that resolves a workload role using a host-selected context.
pub type ContextModelResolver<Ctx> =
    Arc<dyn Fn(&str, &Ctx, Option<&str>) -> Option<String> + Send + Sync>;
/// Callback that reports a provider model's static context window.
pub type ModelContextWindowLookup = Arc<dyn Fn(&str) -> Option<u64> + Send + Sync>;

/// Provider-owned model behavior consulted by a consuming host.
///
/// The host owns registration and decides whether a lookup is process-wide,
/// task-local, or tied to an explicit context. This value carries only the
/// selected provider's model metadata and callbacks; it does not store or
/// select providers itself.
pub struct ProviderModelMetadata<Ctx> {
    /// The provider's default model, absent when it has no default.
    pub default_model: Arc<dyn Fn() -> Option<String> + Send + Sync>,
    /// Resolves a provider-specific model alias to its workload role.
    pub role_for_model: ModelRoleLookup,
    /// Whether a model is a provider-specific alias rather than a concrete id.
    pub is_alias: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    /// Whether a model is a concrete id suitable for pass-through.
    pub is_raw_model: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    /// Whether a model accepts image input according to provider metadata.
    pub supports_vision: Arc<dyn Fn(&str) -> bool + Send + Sync>,
    /// Resolves a workload role and optional pin using the host's selected context.
    pub resolve_model: ContextModelResolver<Ctx>,
    /// The model's static context window, if the provider knows one.
    pub context_window: ModelContextWindowLookup,
    /// Provider default pricing as (model id, input USD/1M, output USD/1M).
    pub default_pricing: Arc<dyn Fn() -> Option<(String, f64, f64)> + Send + Sync>,
    /// Applies a provider-specific request hint without exposing its wire key.
    pub without_reasoning:
        Arc<dyn Fn(crate::model::ModelRequest) -> crate::model::ModelRequest + Send + Sync>,
}

impl<Ctx> std::fmt::Debug for ProviderModelMetadata<Ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderModelMetadata")
            .finish_non_exhaustive()
    }
}

impl<Ctx> Clone for ProviderModelMetadata<Ctx> {
    fn clone(&self) -> Self {
        Self {
            default_model: Arc::clone(&self.default_model),
            role_for_model: Arc::clone(&self.role_for_model),
            is_alias: Arc::clone(&self.is_alias),
            is_raw_model: Arc::clone(&self.is_raw_model),
            supports_vision: Arc::clone(&self.supports_vision),
            resolve_model: Arc::clone(&self.resolve_model),
            context_window: Arc::clone(&self.context_window),
            default_pricing: Arc::clone(&self.default_pricing),
            without_reasoning: Arc::clone(&self.without_reasoning),
        }
    }
}

/// Inputs a host embedding factory uses to construct an embedding model.
pub struct EmbeddingRequest<Ctx> {
    /// Embedding model id selected by the host.
    pub model: String,
    /// Vector dimensionality selected by the host.
    pub dimensions: usize,
    /// Owned host context such as configuration and credential scope.
    pub context: Ctx,
}

impl<Ctx> std::fmt::Debug for EmbeddingRequest<Ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("EmbeddingRequest")
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .field("context", &"<host context>")
            .finish()
    }
}

/// A provider embedding factory's async result.
pub type EmbeddingFactoryFuture = BoxFuture<EmbeddingResult<Box<dyn EmbeddingModel>>>;

/// Builds a provider embedding model.
pub type EmbeddingFactory<Ctx> =
    Arc<dyn Fn(EmbeddingRequest<Ctx>) -> EmbeddingFactoryFuture + Send + Sync>;

/// Owned host context passed to a catalog listing implementation.
pub struct CatalogRequest<Ctx> {
    /// Host-resolved catalog context, including any key and configured client.
    pub context: Ctx,
}

impl<Ctx> std::fmt::Debug for CatalogRequest<Ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CatalogRequest")
            .field("context", &"<host context>")
            .finish()
    }
}

/// Lists models using a host configured catalog implementation.
pub type CatalogListing<Ctx> =
    Arc<dyn Fn(CatalogRequest<Ctx>) -> BoxFuture<std::result::Result<Value, String>> + Send + Sync>;

/// Async result of a provider limit-discovery request.
pub type CatalogLimitsFuture = BoxFuture<Option<DiscoveryRequest>>;

/// Input to an async provider catalog limits resolver.
#[derive(Debug)]
pub struct CatalogLimitsRequest<Ctx> {
    /// Model whose supported limits are being discovered.
    pub model: String,
    /// Owned host context such as configuration and credential sources.
    pub context: Ctx,
}

/// Builds the provider limit discovery request for a model, if supported.
pub type CatalogLimits<Ctx> =
    Arc<dyn Fn(CatalogLimitsRequest<Ctx>) -> CatalogLimitsFuture + Send + Sync>;

/// A provider catalog listing and optional model-limit resolver.
pub struct CatalogEntry<Ctx> {
    /// Lists the provider's models as a neutral JSON payload.
    pub list: CatalogListing<Ctx>,
    /// Resolves provider-specific discovery inputs for one model.
    pub limits: Option<CatalogLimits<Ctx>>,
}

impl<Ctx> std::fmt::Debug for CatalogEntry<Ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CatalogEntry")
            .field("has_limits_resolver", &self.limits.is_some())
            .finish_non_exhaustive()
    }
}

impl<Ctx> Clone for CatalogEntry<Ctx> {
    fn clone(&self) -> Self {
        Self {
            list: Arc::clone(&self.list),
            limits: self.limits.as_ref().map(Arc::clone),
        }
    }
}

type ChatLookup<Ctx> = Arc<dyn Fn(&str) -> Option<ChatModelEntry<Ctx>> + Send + Sync>;
type EmbeddingLookup<Ctx> = Arc<dyn Fn(&str) -> Option<EmbeddingFactory<Ctx>> + Send + Sync>;
type CatalogLookup<Ctx> = Arc<dyn Fn(&str) -> Option<CatalogEntry<Ctx>> + Send + Sync>;

/// Independent, host-owned lookups for provider-specific construction.
///
/// Each closure is consulted live and independently. The host may back these
/// lookups with scoped registrations and process-wide fallbacks without
/// snapshotting them into TinyInference-owned storage.
pub struct ProviderOverrides<Ctx> {
    chat: ChatLookup<Ctx>,
    embeddings: EmbeddingLookup<Ctx>,
    catalogs: CatalogLookup<Ctx>,
}

impl<Ctx> std::fmt::Debug for ProviderOverrides<Ctx> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProviderOverrides")
            .finish_non_exhaustive()
    }
}

impl<Ctx> Clone for ProviderOverrides<Ctx> {
    fn clone(&self) -> Self {
        Self {
            chat: Arc::clone(&self.chat),
            embeddings: Arc::clone(&self.embeddings),
            catalogs: Arc::clone(&self.catalogs),
        }
    }
}

impl<Ctx> ProviderOverrides<Ctx> {
    /// Creates the three independent provider lookup ports.
    pub fn new<C, E, L>(chat: C, embeddings: E, catalogs: L) -> Self
    where
        C: Fn(&str) -> Option<ChatModelEntry<Ctx>> + Send + Sync + 'static,
        E: Fn(&str) -> Option<EmbeddingFactory<Ctx>> + Send + Sync + 'static,
        L: Fn(&str) -> Option<CatalogEntry<Ctx>> + Send + Sync + 'static,
    {
        Self {
            chat: Arc::new(chat),
            embeddings: Arc::new(embeddings),
            catalogs: Arc::new(catalogs),
        }
    }

    /// Returns the live chat entry for `provider_id`, if the host has one.
    pub fn chat_entry(&self, provider_id: &str) -> Option<ChatModelEntry<Ctx>> {
        (self.chat)(provider_id)
    }

    /// Returns the live embedding factory for `provider_id`, if the host has one.
    pub fn embedding_factory(&self, provider_id: &str) -> Option<EmbeddingFactory<Ctx>> {
        (self.embeddings)(provider_id)
    }

    /// Returns the live catalog entry for `provider_id`, if the host has one.
    pub fn catalog_entry(&self, provider_id: &str) -> Option<CatalogEntry<Ctx>> {
        (self.catalogs)(provider_id)
    }

    /// Builds an override when registered, otherwise delegates to `builtin`.
    ///
    /// # Errors
    /// Returns the selected factory's construction error.
    pub fn build_chat<B>(
        &self,
        provider_id: &str,
        request: &ChatModelRequest<'_, Ctx>,
        builtin: B,
    ) -> Result<(Arc<dyn ChatModel<()>>, String)>
    where
        B: FnOnce(&ChatModelRequest<'_, Ctx>) -> Result<(Arc<dyn ChatModel<()>>, String)>,
    {
        match self.chat_entry(provider_id) {
            Some(entry) => (entry.build)(request),
            None => builtin(request),
        }
    }

    /// Builds an override when registered, otherwise delegates to `builtin`.
    ///
    /// # Errors
    /// Returns the selected factory's construction error.
    pub async fn build_embedding<B>(
        &self,
        provider_id: &str,
        request: EmbeddingRequest<Ctx>,
        builtin: B,
    ) -> EmbeddingResult<Box<dyn EmbeddingModel>>
    where
        B: FnOnce(EmbeddingRequest<Ctx>) -> EmbeddingFactoryFuture,
    {
        match self.embedding_factory(provider_id) {
            Some(factory) => factory(request).await,
            None => builtin(request).await,
        }
    }

    /// Lists models through the registered provider, or uses the built-in path.
    ///
    /// # Errors
    /// Returns the selected catalog implementation's error.
    pub async fn list_catalog<B>(
        &self,
        provider_id: &str,
        request: CatalogRequest<Ctx>,
        builtin: B,
    ) -> std::result::Result<Value, String>
    where
        B: FnOnce(CatalogRequest<Ctx>) -> BoxFuture<std::result::Result<Value, String>>,
    {
        match self.catalog_entry(provider_id) {
            Some(entry) => (entry.list)(request).await,
            None => builtin(request).await,
        }
    }
}
