//! Oracle Cloud Infrastructure (OCI) Generative AI chat provider.
//!
//! Two transports share one configuration ([`OciConfig`]):
//!
//! * [`OciTransport::OpenAiCompat`] reuses the OpenAI Chat Completions
//!   adapter against `/openai/v1` or `/20231130/actions/v1`, adding OCI auth
//!   (a bearer API key, or an HTTP Signature applied through
//!   [`RequestSigner`]).
//! * [`OciTransport::Native`] speaks `POST /20231130/actions/chat` with the
//!   GENERIC or COHERE `chatRequest` shapes, including SSE streaming.
//!
//! # Authentication
//!
//! | [`OciAuth`] | Status |
//! |---|---|
//! | `ApiKey` | Implemented (`Authorization: Bearer`) |
//! | `UserPrincipal`, `SessionToken`, `SessionTokenFile` | Implemented (HTTP Signature) |
//! | `ResourcePrincipal` (v2.2, env) | Implemented |
//! | `TokenSource` | Implemented (host-supplied) |
//! | `InstancePrincipal`, `OkeWorkloadIdentity` | Implemented behind the `oci-principals` cargo feature (OpenSSL); otherwise [`Error::Unsupported`] |

mod auth;
mod config;
mod config_file;
mod principals;
mod request;
mod response;
mod signer;
mod stream;

#[cfg(test)]
#[path = "fixtures.rs"]
mod fixtures;

pub use auth::{OciAuth, OciSessionMaterial, OciTokenSource};
pub use config::{
    OciApiFormat, OciConfig, OciOpenAiBase, OciServing, OciTransport, build_oci_model,
};
pub use config_file::{OciProfile, default_config_path, load_profile, parse_profile};
#[cfg(feature = "oci-principals")]
pub use principals::{
    InstancePrincipalSettings, InstancePrincipalSource, OkeWorkloadIdentitySettings,
    OkeWorkloadIdentitySource,
};

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::model::{
    ChatModel, Modalities, ModelProfile, ModelRequest, ModelResponse, ModelStream, ProviderError,
};
use crate::providers::RequestSigner;
use crate::{Error, Result};

use auth::ResolvedAuth;
use signer::OciRequestSigner;

pub(super) const PROVIDER: &str = "oci";
/// [`crate::message::MessageOrigin::api`] stamped on native responses.
pub(super) const API_NATIVE: &str = "actions-chat";
const CONNECT_TIMEOUT_SECS: u64 = 30;

enum NativeAuth {
    Bearer(String),
    Signed(OciRequestSigner),
}

/// A chat model backed by OCI Generative AI's native chat API.
///
/// Build one through [`build_oci_model`].
pub struct OciNativeModel {
    client: reqwest::Client,
    url: String,
    auth: NativeAuth,
    compartment_id: String,
    serving: OciServing,
    api_format: OciApiFormat,
    model: String,
    profile: ModelProfile,
    extra_headers: Vec<(String, String)>,
    temperature_override: Option<f64>,
}

impl std::fmt::Debug for OciNativeModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OciNativeModel")
            .field("url", &self.url)
            .field("serving", &self.serving)
            .field("api_format", &self.api_format)
            .field("model", &self.model)
            .field("auth", &"[redacted]")
            .finish()
    }
}

impl OciNativeModel {
    pub(super) fn new(
        endpoint: String,
        auth: ResolvedAuth,
        compartment_id: String,
        serving: OciServing,
        api_format: OciApiFormat,
        model: String,
        extra_headers: Vec<(String, String)>,
    ) -> Self {
        let auth = match auth {
            ResolvedAuth::Bearer(key) => NativeAuth::Bearer(key),
            ResolvedAuth::Signed(source) => NativeAuth::Signed(OciRequestSigner::new(source)),
        };
        let profile = ModelProfile {
            provider: Some(PROVIDER.to_string()),
            model: Some(model.clone()),
            modalities: Modalities {
                text_in: true,
                text_out: true,
                image_in: api_format == OciApiFormat::Generic,
                ..Modalities::default()
            },
            tool_calling: true,
            parallel_tool_calls: api_format == OciApiFormat::Generic,
            streaming: true,
            streaming_tool_chunks: true,
            ..ModelProfile::default()
        };
        Self {
            client: reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("OCI default HTTP client configuration is valid"),
            url: format!("{endpoint}/20231130/actions/chat"),
            auth,
            compartment_id,
            serving,
            api_format,
            model,
            profile,
            extra_headers,
            temperature_override: None,
        }
    }

    /// Replaces each request's temperature with `temperature` when set.
    pub(super) fn with_temperature_override(mut self, temperature: Option<f64>) -> Self {
        self.temperature_override = temperature;
        self
    }

    fn request_model<'a>(&'a self, request: &'a ModelRequest) -> &'a str {
        request.model.as_deref().unwrap_or(&self.model)
    }

    fn provider_error(
        &self,
        request: &ModelRequest,
        message: String,
        status: Option<u16>,
        code: Option<String>,
        raw: Option<Value>,
        retry_after: Option<&str>,
    ) -> ProviderError {
        let retryable =
            crate::failure::classify_provider_failure(status, code.as_deref(), &message)
                .is_retryable();
        ProviderError {
            provider: PROVIDER.to_string(),
            model: Some(self.request_model(request).to_string()),
            status,
            code,
            message,
            retryable,
            retry_after_ms: tinyinference_core::parse_retry_after_ms(retry_after),
            raw,
            partial_message: None,
            stop_reason: None,
        }
    }

    /// Builds the fully authenticated (bearer or signed) HTTP request.
    fn prepare(&self, request: &ModelRequest, streaming: bool) -> Result<reqwest::Request> {
        let overridden;
        let request = match self.temperature_override {
            Some(temperature) => {
                overridden = ModelRequest {
                    temperature: Some(temperature),
                    ..request.clone()
                };
                &overridden
            }
            None => request,
        };
        let body = request::chat_details(
            request,
            &self.model,
            &self.serving,
            &self.compartment_id,
            self.api_format,
            streaming,
        );
        // Serialize first: the signature covers the exact bytes sent.
        let bytes = serde_json::to_vec(&body)?;
        let mut builder = self
            .client
            .post(&self.url)
            .header(reqwest::header::CONTENT_TYPE, "application/json")
            .header(
                reqwest::header::ACCEPT,
                if streaming {
                    "text/event-stream"
                } else {
                    "application/json"
                },
            );
        for (name, value) in &self.extra_headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        if let NativeAuth::Bearer(key) = &self.auth {
            let mut value = reqwest::header::HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|_| Error::Validation("OCI API key is not a valid header value".into()))?;
            value.set_sensitive(true);
            builder = builder.header(reqwest::header::AUTHORIZATION, value);
        }
        builder = builder.body(bytes);
        if let (false, Some(timeout_ms)) = (streaming, request.timeout_ms) {
            builder = builder.timeout(Duration::from_millis(timeout_ms));
        }
        let mut built = builder.build().map_err(|error| {
            Error::Validation(format!("OCI request could not be built: {error}"))
        })?;
        if let NativeAuth::Signed(signer) = &self.auth {
            signer.sign(&mut built)?;
        }
        Ok(built)
    }

    async fn post(&self, request: &ModelRequest, streaming: bool) -> Result<reqwest::Response> {
        crate::network_guard::ensure_network_models_allowed()?;
        let built = self.prepare(request, streaming)?;
        tracing::debug!(
            target: "tinyinference::oci",
            streaming,
            model = self.request_model(request),
            "oci native chat request"
        );
        let response = self.client.execute(built).await.map_err(|error| {
            Error::Provider(Box::new(self.provider_error(
                request,
                format!("oci request failed: {error}"),
                None,
                None,
                None,
                None,
            )))
        })?;
        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }
        let retry_after = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let text = response.text().await.unwrap_or_default();
        Err(Error::Provider(Box::new(self.parse_error_body(
            request,
            status.as_u16(),
            &text,
            retry_after.as_deref(),
        ))))
    }

    /// OCI errors are `{"code":"...","message":"..."}`.
    fn parse_error_body(
        &self,
        request: &ModelRequest,
        status: u16,
        text: &str,
        retry_after: Option<&str>,
    ) -> ProviderError {
        let raw = serde_json::from_str::<Value>(text).ok();
        let message = raw
            .as_ref()
            .and_then(|value| value["message"].as_str())
            .filter(|message| !message.trim().is_empty())
            .unwrap_or(text)
            .to_string();
        let code = raw
            .as_ref()
            .and_then(|value| value["code"].as_str())
            .map(str::to_string);
        self.provider_error(
            request,
            format!("oci returned HTTP {status}: {message}"),
            Some(status),
            code,
            raw,
            retry_after,
        )
    }

    fn origin_for(&self, request: &ModelRequest) -> crate::message::MessageOrigin {
        crate::message::MessageOrigin {
            provider: PROVIDER.to_string(),
            api: API_NATIVE.to_string(),
            model: self.request_model(request).to_string(),
        }
    }
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for OciNativeModel {
    fn profile(&self) -> Option<&ModelProfile> {
        Some(&self.profile)
    }

    fn cache_identity(&self) -> Option<String> {
        Some(format!("oci:{}:{}", self.url, self.model))
    }

    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        let response = self.post(&request, false).await?;
        let body: Value = response
            .json()
            .await
            .map_err(|error| Error::Model(format!("oci response was not JSON: {error}")))?;
        let origin = self.origin_for(&request);
        response::parse_response(body).map(|mut response| {
            response.message.origin = Some(origin);
            response.inherit_correlation(request.correlation)
        })
    }

    async fn stream(&self, _state: &State, request: ModelRequest) -> Result<ModelStream> {
        let response = self.post(&request, true).await?;
        let model = self.request_model(&request).to_string();
        let stream = stream::into_model_stream(response, model);
        Ok(match request.correlation {
            Some(correlation) => stream.with_correlation(correlation),
            None => stream,
        })
    }
}
