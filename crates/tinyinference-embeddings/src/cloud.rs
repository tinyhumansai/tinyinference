//! Bearer-authenticated OpenAI-compatible cloud embedding model.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;

use super::{EmbeddingModel, EmbeddingUsage, OpenAiEmbeddingModel};
use crate::{Error, Result};

/// Default model id for the host-authenticated cloud endpoint.
pub const DEFAULT_CLOUD_MODEL: &str = "embedding-v1";
/// Default vector dimensionality for the cloud model.
pub const DEFAULT_CLOUD_DIMENSIONS: usize = 1024;

/// Resolves the current bearer token for each request.
pub type BearerResolver =
    Arc<dyn Fn() -> Pin<Box<dyn Future<Output = Result<String>> + Send>> + Send + Sync>;

/// Host policy hook invoked immediately before an embedding batch may leave
/// the process. This keeps privacy and audit policy in the host without
/// requiring a host-owned embedding implementation.
pub type EmbeddingEgressGuard = Arc<dyn Fn(&str, usize) -> Result<()> + Send + Sync>;

/// Cloud model whose credential lifecycle remains owned by the host.
pub struct CloudEmbeddingModel {
    client: reqwest::Client,
    base_url: String,
    model: String,
    dimensions: usize,
    bearer: BearerResolver,
    egress_guard: Option<EmbeddingEgressGuard>,
}

impl CloudEmbeddingModel {
    /// Creates a cloud embedding adapter using a host-owned bearer resolver.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        dimensions: usize,
        bearer: BearerResolver,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.into().trim().trim_end_matches('/').to_owned(),
            model: model.into(),
            dimensions,
            bearer,
            egress_guard: None,
        }
    }

    /// Installs a host-owned egress policy hook.
    #[must_use]
    pub fn with_egress_guard(mut self, guard: EmbeddingEgressGuard) -> Self {
        self.egress_guard = Some(guard);
        self
    }
}

impl std::fmt::Debug for CloudEmbeddingModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CloudEmbeddingModel")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("dimensions", &self.dimensions)
            .finish_non_exhaustive()
    }
}

impl CloudEmbeddingModel {
    /// Validates `texts`, resolves the bearer, and builds the request model.
    ///
    /// `None` for an empty batch — the one case that answers without a
    /// credential, so an empty call must not fail on a missing session.
    ///
    /// Shared by both embed entry points so the validation order stays one
    /// thing: refuse blank input before the bearer is resolved, and refuse a
    /// blank bearer before the texts leave the process.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when an input is empty or whitespace, when the
    /// bearer resolver fails, or when it resolves to a blank token.
    async fn delegate(&self, texts: &[String]) -> Result<Option<OpenAiEmbeddingModel>> {
        if texts.is_empty() {
            return Ok(None);
        }
        if let Some(index) = texts.iter().position(|text| text.trim().is_empty()) {
            return Err(Error::Validation(format!(
                "cloud embed: refusing empty/whitespace input at index {index} of {} (model={})",
                texts.len(),
                self.model
            )));
        }
        if let Some(guard) = &self.egress_guard {
            guard(&self.model, texts.len())?;
        }
        let bearer = (self.bearer)().await?;
        if bearer.trim().is_empty() {
            return Err(Error::Validation(
                "No backend session for cloud embeddings".into(),
            ));
        }
        Ok(Some(
            OpenAiEmbeddingModel::new(bearer)
                .with_client(self.client.clone())
                .with_base_url(&self.base_url)
                .with_model(&self.model)
                .with_dimensions(self.dimensions)
                .with_send_dimensions(false)
                .with_required_api_key(true),
        ))
    }
}

#[async_trait]
impl EmbeddingModel for CloudEmbeddingModel {
    fn name(&self) -> &str {
        "cloud"
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let Some(delegate) = self.delegate(texts).await? else {
            return Ok(Vec::new());
        };
        delegate.embed(texts).await
    }

    /// Forwarded, not defaulted: the endpoint is OpenAI-compatible, so the
    /// delegate already reads the usage this host is billed on.
    async fn embed_with_usage(
        &self,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, Option<EmbeddingUsage>)> {
        let Some(delegate) = self.delegate(texts).await? else {
            return Ok((Vec::new(), None));
        };
        delegate.embed_with_usage(texts).await
    }
}

#[cfg(test)]
#[path = "cloud_tests.rs"]
mod tests;
