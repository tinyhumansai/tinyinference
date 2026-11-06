//! Construction policy for OCI Generative AI chat models.

use std::sync::Arc;

use super::OciNativeModel;
use super::auth::{OciAuth, ResolvedAuth};
use super::signer::OciRequestSigner;
use crate::model::ChatModel;
use crate::providers::openai::{AuthStyle, OpenAiModel};
use crate::{Error, Result};

/// Which OpenAI-compatible surface to call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OciOpenAiBase {
    /// `https://inference.generativeai.<region>.oci.oraclecloud.com/openai/v1`
    #[default]
    OpenAi,
    /// `.../20231130/actions/v1`
    Actions,
}

/// Native chat API format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OciApiFormat {
    /// `apiFormat: GENERIC` (Meta, xAI, Google, OpenAI-family models, ...).
    #[default]
    Generic,
    /// `apiFormat: COHERE` (Cohere command models).
    Cohere,
}

/// Which wire protocol to speak.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OciTransport {
    /// Reuse the OpenAI Chat Completions adapter against an OCI base path.
    OpenAiCompat {
        /// Which OpenAI-compatible base path to use.
        base: OciOpenAiBase,
    },
    /// `POST /20231130/actions/chat`.
    Native {
        /// GENERIC or COHERE request/response shape.
        api_format: OciApiFormat,
    },
}

/// How the model is addressed.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum OciServing {
    /// On-demand serving by model id ([`OciConfig::model`]).
    #[default]
    OnDemand,
    /// A dedicated AI cluster endpoint.
    Dedicated {
        /// The endpoint OCID.
        endpoint_id: String,
    },
}

/// Fully resolved configuration for an OCI Generative AI chat model.
#[derive(Debug, Clone)]
pub struct OciConfig {
    /// OCI region identifier (for example `us-chicago-1`). Ignored when
    /// [`Self::endpoint`] is set; falls back to the credential's region.
    pub region: Option<String>,
    /// Full service endpoint override (`https://...`, no path).
    pub endpoint: Option<String>,
    /// How requests authenticate.
    pub auth: OciAuth,
    /// Wire protocol.
    pub transport: OciTransport,
    /// Compartment OCID. Required for the native transport and for IAM-signed
    /// calls on either OpenAI-compatible base, where it is sent as
    /// `opc-compartment-id`.
    pub compartment_id: Option<String>,
    /// Generative AI project OCID, sent as `OpenAI-Project` on `/openai/v1`.
    pub project_id: Option<String>,
    /// Model addressing.
    pub serving: OciServing,
    /// Default model id.
    pub model: String,
    /// Static headers attached to every request.
    pub extra_headers: Vec<(String, String)>,
    /// Sampling temperature that replaces the request's own, like
    /// [`OpenAiConfig::temperature_override`](crate::providers::openai::OpenAiConfig::temperature_override).
    /// OpenAI reasoning models still receive none.
    pub temperature_override: Option<f64>,
}

impl OciConfig {
    /// Starts a configuration with the OpenAI-compatible `/openai/v1`
    /// transport and on-demand serving.
    pub fn new(auth: OciAuth, model: impl Into<String>) -> Self {
        Self {
            region: None,
            endpoint: None,
            auth,
            transport: OciTransport::OpenAiCompat {
                base: OciOpenAiBase::OpenAi,
            },
            compartment_id: None,
            project_id: None,
            serving: OciServing::OnDemand,
            model: model.into(),
            extra_headers: Vec::new(),
            temperature_override: None,
        }
    }

    /// Resolves the service endpoint (no trailing slash, no path).
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when neither an endpoint nor a region is known,
    /// or the endpoint is not `https://`.
    pub fn service_endpoint(&self) -> Result<String> {
        if let Some(endpoint) = self.endpoint.as_deref().filter(|e| !e.trim().is_empty()) {
            let endpoint = endpoint.trim().trim_end_matches('/');
            if !endpoint.starts_with("https://") && !is_loopback_http(endpoint) {
                return Err(Error::Validation(
                    "OCI endpoint must start with https:// (plain http is allowed only for loopback hosts)".into(),
                ));
            }
            return Ok(endpoint.to_string());
        }
        let region = self
            .region
            .clone()
            .or_else(|| self.auth.implied_region())
            .filter(|region| !region.trim().is_empty())
            .ok_or_else(|| Error::Validation("OCI region or endpoint is required".into()))?;
        let region = region.trim();
        if !region
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            return Err(Error::Validation(format!("invalid OCI region `{region}`")));
        }
        Ok(format!(
            "https://inference.generativeai.{region}.oci.oraclecloud.com"
        ))
    }
}

fn is_loopback_http(endpoint: &str) -> bool {
    reqwest::Url::parse(endpoint).is_ok_and(|url| {
        url.scheme() == "http"
            && matches!(url.host_str(), Some("localhost" | "127.0.0.1" | "[::1]"))
    })
}

/// Builds an OCI Generative AI chat model from fully resolved configuration.
///
/// # Errors
///
/// [`Error::Validation`] for missing region/compartment or unusable keys,
/// [`Error::Unsupported`] for auth flows that are not implemented.
pub fn build_oci_model(config: OciConfig) -> Result<Arc<dyn ChatModel<()>>> {
    let endpoint = config.service_endpoint()?;
    let resolved = config.auth.resolve()?;
    tracing::debug!(
        target: "tinyinference::oci",
        transport = ?config.transport,
        serving = ?config.serving,
        "building oci model"
    );
    match config.transport {
        OciTransport::OpenAiCompat { base } => Ok(Arc::new(openai_compat_model(
            &config, &endpoint, base, resolved,
        ))),
        OciTransport::Native { api_format } => {
            let compartment_id = config
                .compartment_id
                .clone()
                .filter(|c| !c.trim().is_empty())
                .ok_or_else(|| {
                    Error::Validation("OCI native chat requires a compartment_id".into())
                })?;
            Ok(Arc::new(
                OciNativeModel::new(
                    endpoint,
                    resolved,
                    compartment_id,
                    config.serving,
                    api_format,
                    config.model,
                    config.extra_headers,
                )
                .with_temperature_override(config.temperature_override),
            ))
        }
    }
}

fn openai_compat_model(
    config: &OciConfig,
    endpoint: &str,
    base: OciOpenAiBase,
    resolved: ResolvedAuth,
) -> OpenAiModel {
    let path = match base {
        OciOpenAiBase::OpenAi => "/openai/v1",
        OciOpenAiBase::Actions => "/20231130/actions/v1",
    };
    let model_id = match &config.serving {
        OciServing::OnDemand => config.model.clone(),
        OciServing::Dedicated { endpoint_id } => endpoint_id.clone(),
    };
    let base_url = format!("{endpoint}{path}");
    let mut model = match resolved {
        ResolvedAuth::Bearer(key) => OpenAiModel::new(key),
        ResolvedAuth::Signed(source) => OpenAiModel::new("")
            .with_auth_style(AuthStyle::None)
            .with_request_signer(Arc::new(OciRequestSigner::new(source))),
    }
    .with_provider("oci")
    .with_base_url(base_url)
    .with_model(model_id.clone());
    // OpenAI reasoning models reject a non-default temperature on OCI, the
    // same as on the native API (see `request.rs`).
    if crate::providers::openai::is_reasoning_model(&config.model) {
        model = model.with_temperature_unsupported_models([model_id]);
    }
    if config.temperature_override.is_some() {
        model = model.with_temperature_override(config.temperature_override);
    }
    if base == OciOpenAiBase::OpenAi
        && let Some(project) = config.project_id.as_deref().filter(|p| !p.is_empty())
    {
        model = model.with_header("OpenAI-Project", project);
    }
    // IAM-signed calls name the billed compartment on both bases (`/openai/v1`
    // answers "Compartment ID must be provided" without it); a bearer key
    // already carries its own, and the header is harmless alongside it.
    if let Some(compartment) = config.compartment_id.as_deref().filter(|c| !c.is_empty()) {
        model = model.with_header("opc-compartment-id", compartment);
    }
    for (name, value) in &config.extra_headers {
        model = model.with_header(name.clone(), value.clone());
    }
    model
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
