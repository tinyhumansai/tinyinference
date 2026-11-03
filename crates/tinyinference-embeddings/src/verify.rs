//! Setup-time endpoint verification and host-neutral connection tests.

use std::time::Duration;

use crate::{
    CredentialResolver, EmbeddingFactorySettings, EmbeddingModel, ManagedModelFactory,
    create_configured_embedding_model, format_embedding_signature,
    probe::{
        EmbedProbe, classify_embed_probe, final_probe_dims, probe_custom_embeddings,
        validate_custom_embeddings_endpoint,
    },
    served_models::{ModelNotServed, check_requested_model_served, fetch_served_model_ids},
};

/// Serializable rejection from custom endpoint verification.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(untagged)]
pub enum VerifyRejection {
    /// Probe rejection with stable remediation fields.
    Probe(crate::probe::EmbeddingProbeRejection),
    /// Positive evidence the selected model is not served.
    ModelNotServed(Box<ModelNotServed>),
}

impl VerifyRejection {
    /// Stable error code used by the host's settings response.
    pub fn error(&self) -> &str {
        match self {
            Self::Probe(value) => value.error,
            Self::ModelNotServed(value) => value.error,
        }
    }
    /// User-facing remediation text.
    pub fn message(&self) -> &str {
        match self {
            Self::Probe(value) => value.message,
            Self::ModelNotServed(value) => &value.message,
        }
    }
}

/// Dimensions and signature proven by a successful endpoint probe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifiedDims {
    /// Actual vector width returned by the endpoint.
    pub dimensions: usize,
    /// Signature bytes for persisted provider `custom:<endpoint>`.
    pub signature: String,
}

impl VerifiedDims {
    /// Whether a prior persisted signature belongs to a different embedding space.
    #[must_use]
    pub fn signature_changed(&self, previous_signature: Option<&str>) -> bool {
        previous_signature.is_some_and(|previous| previous != self.signature)
    }
}

/// Verify a custom OpenAI-compatible endpoint and return its actual vector width.
///
/// # Errors
/// Returns a structured rejection when probing or positive model guidance fails.
pub async fn verify_custom_endpoint(
    endpoint: &str,
    key: &str,
    model: &str,
    dims: usize,
) -> Result<VerifiedDims, VerifyRejection> {
    verify_custom_endpoint_with_probe(
        endpoint,
        key,
        model,
        dims,
        Duration::from_secs(10),
        probe_custom_embeddings(endpoint, key, model, dims),
    )
    .await
}

async fn verify_custom_endpoint_with_probe<F>(
    endpoint: &str,
    key: &str,
    model: &str,
    dims: usize,
    probe_timeout: Duration,
    probe_future: F,
) -> Result<VerifiedDims, VerifyRejection>
where
    F: std::future::Future<Output = crate::Result<Vec<Vec<f32>>>>,
{
    // Fail closed before best-effort guidance: /models must never receive a bearer
    // for an endpoint rejected by the probe transport's URL policy.
    validate_custom_embeddings_endpoint(endpoint).map_err(|error| {
        classify_embed_probe(EmbedProbe::Failed(error.to_string()))
            .map(VerifyRejection::Probe)
            .unwrap_or_else(|| {
                VerifyRejection::Probe(crate::probe::EmbeddingProbeRejection {
                    error: "EMBEDDINGS_VERIFICATION_FAILED",
                    message: "The custom embeddings endpoint could not be verified.",
                    summary: "invalid custom embeddings endpoint — not saved",
                    detail: Some(error.to_string()),
                })
            })
    })?;
    let probe = tokio::time::timeout(probe_timeout, probe_future).await;
    let (vectors, outcome) = match probe {
        Ok(Ok(vectors)) => (Some(vectors.clone()), EmbedProbe::Returned(vectors)),
        Ok(Err(error)) => (None, EmbedProbe::Failed(error.to_string())),
        Err(_) => (None, EmbedProbe::TimedOut),
    };
    if let Some(rejection) = classify_embed_probe(outcome) {
        return Err(rejection_with_model_guidance(endpoint, key, model, rejection).await);
    }
    let vectors = vectors.expect("a passing probe returned vectors");
    let actual = final_probe_dims(model, dims, vectors.first().map_or(0, Vec::len));
    let signature = format_embedding_signature(&format!("custom:{endpoint}"), model, actual);
    let verified = VerifiedDims {
        dimensions: actual,
        signature,
    };
    Ok(verified)
}

async fn rejection_with_model_guidance(
    endpoint: &str,
    key: &str,
    model: &str,
    rejection: crate::probe::EmbeddingProbeRejection,
) -> VerifyRejection {
    if let Ok(Ok(served)) = tokio::time::timeout(
        Duration::from_secs(5),
        fetch_served_model_ids(endpoint, key),
    )
    .await
        && let Some(rejection) = check_requested_model_served(model, &served)
    {
        return VerifyRejection::ModelNotServed(Box::new(rejection));
    }
    VerifyRejection::Probe(rejection)
}

/// Result of a provider connection test.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectionTestResult {
    /// Actual width returned by the provider.
    pub actual_dims: usize,
}

/// Identifies whether a provider connection test failed while constructing the
/// selected model or while invoking that model.
#[derive(Debug)]
pub enum ConnectionTestError {
    /// The selected provider or its configuration could not produce a model.
    Build(crate::Error),
    /// The model was constructed, but the embedding request failed.
    Embed(crate::Error),
}

impl std::fmt::Display for ConnectionTestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Build(error) | Self::Embed(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ConnectionTestError {}

/// Build the selected provider using host-supplied credential/model factories and test it.
/// A custom endpoint uses the dedicated portable setup probe.
pub async fn test_embedding_connection(
    settings: &EmbeddingFactorySettings,
    credentials: &CredentialResolver,
    managed: &ManagedModelFactory,
    custom_endpoint: Option<&str>,
) -> crate::Result<ConnectionTestResult> {
    test_embedding_connection_detailed(settings, credentials, managed, custom_endpoint)
        .await
        .map_err(|error| match error {
            ConnectionTestError::Build(error) | ConnectionTestError::Embed(error) => error,
        })
}

/// Test a provider connection while preserving model construction versus
/// embedding invocation failures for hosts with distinct error contracts.
pub async fn test_embedding_connection_detailed(
    settings: &EmbeddingFactorySettings,
    credentials: &CredentialResolver,
    managed: &ManagedModelFactory,
    custom_endpoint: Option<&str>,
) -> std::result::Result<ConnectionTestResult, ConnectionTestError> {
    let stored_endpoint = settings.provider.strip_prefix("custom:");
    if let Some(endpoint) = custom_endpoint.or(stored_endpoint) {
        let key = credentials("custom");
        let vectors = probe_custom_embeddings(endpoint, &key, &settings.model, settings.dimensions)
            .await
            .map_err(ConnectionTestError::Embed)?;
        return Ok(ConnectionTestResult {
            actual_dims: vectors.first().map_or(0, Vec::len),
        });
    }
    let model = create_configured_embedding_model(settings, credentials, managed)
        .await
        .map_err(ConnectionTestError::Build)?;
    let vectors = model
        .embed(&["connection test".to_owned()])
        .await
        .map_err(ConnectionTestError::Embed)?;
    let actual_dims = vectors.first().map_or(0, Vec::len);
    Ok(ConnectionTestResult { actual_dims })
}

/// Resolve persisted provider selection using host-resolved state.
#[must_use]
pub fn effective_embedder_slug(
    opt_out: bool,
    local_model: Option<&str>,
    picker: Option<&str>,
    session_exists: bool,
) -> String {
    if opt_out {
        return "none".into();
    }
    if local_model
        .map(str::trim)
        .is_some_and(|value| !value.is_empty())
    {
        return "ollama".into();
    }
    let provider = picker.map(str::trim).unwrap_or_default();
    if provider == "openai" || provider == "custom" || provider.starts_with("custom:") {
        return "custom".into();
    }
    if session_exists {
        "cloud".into()
    } else {
        "unconfigured".into()
    }
}

/// A model placeholder that consistently reports a host-provided unavailable error.
#[derive(Debug)]
pub struct UnavailableEmbeddingModel {
    model: String,
    dimensions: usize,
    message: String,
}

impl UnavailableEmbeddingModel {
    /// Create a placeholder preserving the host's error text and configured identity.
    pub fn new(model: impl Into<String>, dimensions: usize, message: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            dimensions,
            message: message.into(),
        }
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for UnavailableEmbeddingModel {
    fn name(&self) -> &str {
        "cloud"
    }
    fn model_id(&self) -> &str {
        &self.model
    }
    fn dimensions(&self) -> usize {
        self.dimensions
    }
    fn signature(&self) -> String {
        format!("unavailable:{}:{}", self.model, self.dimensions)
    }
    async fn embed(&self, _texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        Err(crate::Error::Validation(self.message.clone()))
    }
    async fn embed_with_usage(
        &self,
        _texts: &[String],
    ) -> crate::Result<(Vec<Vec<f32>>, Option<crate::EmbeddingUsage>)> {
        Err(crate::Error::Validation(self.message.clone()))
    }
}

#[cfg(test)]
#[path = "verify_tests.rs"]
mod tests;
