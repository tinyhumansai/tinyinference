//! OKE workload identity: a Kubernetes service-account token exchanged at the
//! cluster's proxymux for a session token (oci-go-sdk
//! `federation_client_oke_workload_identity.go`, `resource_principal_key_provider.go`).

use std::path::PathBuf;
use std::time::Duration;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use serde_json::{Value, json};

use super::keys::generate_session_keypair;
use super::token::{TokenCache, block_on, now_secs};
use crate::providers::oci::auth::{OciSessionMaterial, OciTokenSource};
use crate::{Error, Result};

const SA_TOKEN_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/token";
const SA_CA_PATH: &str = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt";
const CA_PATH_ENV: &str = "OCI_KUBERNETES_SERVICE_ACCOUNT_CERT_PATH";
const SERVICE_HOST_ENV: &str = "KUBERNETES_SERVICE_HOST";
const REGION_ENV: &str = "OCI_RESOURCE_PRINCIPAL_REGION";
const PROXYMUX_PORT: &str = "12250";
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Locations for the workload-identity exchange; all injectable.
#[derive(Debug, Clone)]
pub struct OkeWorkloadIdentitySettings {
    /// Full proxymux URL, normally
    /// `https://$KUBERNETES_SERVICE_HOST:12250/resourcePrincipalSessionTokens`.
    pub proxymux_url: String,
    /// Service-account token file (re-read on every renewal; it rotates).
    pub service_account_token_path: PathBuf,
    /// CA bundle trusted for the proxymux (the only roots used, as in the SDK).
    pub ca_cert_path: PathBuf,
}

impl OkeWorkloadIdentitySettings {
    /// Settings from the pod environment (`KUBERNETES_SERVICE_HOST`,
    /// `OCI_KUBERNETES_SERVICE_ACCOUNT_CERT_PATH`) and standard paths.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] when `KUBERNETES_SERVICE_HOST` is unset.
    pub fn from_env() -> Result<Self> {
        let host = std::env::var(SERVICE_HOST_ENV)
            .ok()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| Error::Validation(format!("{SERVICE_HOST_ENV} is not set")))?;
        let ca = std::env::var(CA_PATH_ENV)
            .ok()
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| SA_CA_PATH.to_string());
        Ok(Self {
            proxymux_url: format!("https://{host}:{PROXYMUX_PORT}/resourcePrincipalSessionTokens"),
            service_account_token_path: PathBuf::from(SA_TOKEN_PATH),
            ca_cert_path: PathBuf::from(ca),
        })
    }
}

/// An [`OciTokenSource`] for OKE workload identity. Caches the session token
/// and renews it at half its lifetime (never later than five minutes before
/// expiry); safe to share across threads.
#[derive(Debug)]
pub struct OkeWorkloadIdentitySource {
    settings: OkeWorkloadIdentitySettings,
    cache: TokenCache,
}

impl OkeWorkloadIdentitySource {
    /// Creates a source; nothing is fetched until the first request.
    pub fn new(settings: OkeWorkloadIdentitySettings) -> Self {
        Self {
            settings,
            cache: TokenCache::default(),
        }
    }
}

impl OciTokenSource for OkeWorkloadIdentitySource {
    fn fetch(&self) -> Result<OciSessionMaterial> {
        self.cache
            .get_or_renew(now_secs(), true, || block_on(renew(&self.settings)))
    }
}

/// Region from `OCI_RESOURCE_PRINCIPAL_REGION` (required by the SDK flow).
pub(super) fn region_from_env() -> Option<String> {
    std::env::var(REGION_ENV)
        .ok()
        .filter(|v| !v.trim().is_empty())
}

/// Request body: `{"podKey": "<session public key PEM>"}` (unsanitized PEM).
pub(super) fn proxymux_body(public_pem: &str) -> Value {
    json!({ "podKey": public_pem })
}

/// Decodes the proxymux reply: a JSON string holding base64 of
/// `{"token": "<3-char prefix><jwt>"}`; the first three characters of the
/// token are dropped (the SDK does `token[3:]`).
pub(super) fn parse_proxymux_response(body: &str) -> Result<String> {
    let invalid = |what: &str| Error::Model(format!("invalid workload identity response: {what}"));
    let inner: String = serde_json::from_str(body.trim())
        .or_else(|_| {
            // The SDK strips the first and last byte instead of JSON-decoding.
            let trimmed = body.trim();
            trimmed
                .get(1..trimmed.len().saturating_sub(1))
                .map(str::to_string)
                .ok_or(())
        })
        .map_err(|_| invalid("body"))?;
    let decoded = B64
        .decode(inner.as_bytes())
        .map_err(|_| invalid("base64"))?;
    let value: Value = serde_json::from_slice(&decoded).map_err(|_| invalid("json"))?;
    let token = value
        .get("token")
        .or_else(|| value.get("Token"))
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("missing token"))?;
    if token.is_empty() {
        return Err(invalid("empty token"));
    }
    token
        .get(3..)
        .filter(|rest| !rest.is_empty())
        .map(str::to_string)
        .ok_or_else(|| invalid("token too short"))
}

fn http_client(ca_path: &std::path::Path) -> Result<reqwest::Client> {
    let mut builder = reqwest::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none());
    let pem = std::fs::read(ca_path).map_err(|error| {
        Error::Validation(format!(
            "could not read Kubernetes CA {}: {}",
            ca_path.display(),
            error.kind()
        ))
    })?;
    let certs = reqwest::Certificate::from_pem_bundle(&pem)
        .map_err(|_| Error::Validation("Kubernetes CA bundle is not valid PEM".into()))?;
    builder = builder.tls_built_in_root_certs(false);
    for cert in certs {
        builder = builder.add_root_certificate(cert);
    }
    builder
        .build()
        .map_err(|error| Error::Model(format!("could not build HTTP client: {error}")))
}

async fn renew(settings: &OkeWorkloadIdentitySettings) -> Result<OciSessionMaterial> {
    let client = http_client(&settings.ca_cert_path)?;
    let sa_token = std::fs::read_to_string(&settings.service_account_token_path)
        .map_err(|error| {
            Error::Validation(format!(
                "could not read service account token {}: {}",
                settings.service_account_token_path.display(),
                error.kind()
            ))
        })?
        .trim()
        .to_string();
    let session = generate_session_keypair()?;
    let mut authorization =
        reqwest::header::HeaderValue::from_str(&format!("Bearer {sa_token}"))
            .map_err(|_| Error::Validation("service account token is not a valid header".into()))?;
    authorization.set_sensitive(true);
    tracing::debug!(target: "tinyinference::oci", "oke workload identity exchange");
    let response = client
        .post(&settings.proxymux_url)
        .header(reqwest::header::AUTHORIZATION, authorization)
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(serde_json::to_vec(&proxymux_body(&session.public_pem))?)
        .send()
        .await
        .map_err(|error| Error::Model(format!("workload identity request failed: {error}")))?;
    let status = response.status();
    let text = response
        .text()
        .await
        .map_err(|error| Error::Model(format!("workload identity response unreadable: {error}")))?;
    if status == reqwest::StatusCode::FORBIDDEN {
        return Err(Error::Model(
            "workload identity request was forbidden; ensure the cluster type is enhanced".into(),
        ));
    }
    if !status.is_success() {
        return Err(Error::Model(format!(
            "workload identity exchange failed with HTTP {status}"
        )));
    }
    Ok(OciSessionMaterial {
        token: parse_proxymux_response(&text)?,
        private_key_pem: session.private_pem,
    })
}

#[cfg(test)]
#[path = "oke_tests.rs"]
mod tests;
