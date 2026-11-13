//! Instance principal: IMDS certificates exchanged at the X.509 federation
//! endpoint for a session token (oci-go-sdk `instance_principal_key_provider.go`,
//! `federation_client.go`, `certificate_retriever.go`, `utils.go`).

use std::time::Duration;

use serde_json::{Value, json};

use super::keys::{generate_session_keypair, parse_leaf_cert, sanitize_pem};
use super::token::{TokenCache, block_on, now_secs};
use crate::providers::oci::auth::{OciSessionMaterial, OciTokenSource};
use crate::providers::oci::signer::{SigningKey, sign_request_with};
use crate::{Error, Result};

const DEFAULT_METADATA_BASE_URL: &str = "http://169.254.169.254/opc/v2";
const METADATA_BASE_URL_ENV: &str = "OCI_METADATA_BASE_URL";
const REGION_PATH: &str = "/instance/region";
const LEAF_CERT_PATH: &str = "/identity/cert.pem";
const LEAF_KEY_PATH: &str = "/identity/key.pem";
const INTERMEDIATE_PATH: &str = "/identity/intermediate.pem";
const HTTP_TIMEOUT: Duration = Duration::from_secs(30);

/// Endpoints for the instance-principal exchange; all injectable for tests
/// and non-standard realms.
#[derive(Debug, Clone)]
pub struct InstancePrincipalSettings {
    /// IMDS v2 base URL (default `http://169.254.169.254/opc/v2`, or
    /// `OCI_METADATA_BASE_URL`).
    pub metadata_base_url: String,
    /// Federation service origin, e.g. `https://auth.us-phoenix-1.oraclecloud.com`.
    /// Derived from the region (commercial realm) when `None`.
    pub federation_endpoint: Option<String>,
    /// Region identifier; read from IMDS when `None`.
    pub region: Option<String>,
}

impl InstancePrincipalSettings {
    /// Defaults, honouring `OCI_METADATA_BASE_URL`.
    pub fn from_env() -> Self {
        let base = std::env::var(METADATA_BASE_URL_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| DEFAULT_METADATA_BASE_URL.to_string());
        Self {
            metadata_base_url: base,
            federation_endpoint: None,
            region: None,
        }
    }
}

/// An [`OciTokenSource`] for instance principals. Caches the session token
/// and renews it five minutes before expiry; safe to share across threads.
#[derive(Debug)]
pub struct InstancePrincipalSource {
    settings: InstancePrincipalSettings,
    cache: TokenCache,
}

impl InstancePrincipalSource {
    /// Creates a source; nothing is fetched until the first request.
    pub fn new(settings: InstancePrincipalSettings) -> Self {
        Self {
            settings,
            cache: TokenCache::default(),
        }
    }
}

impl OciTokenSource for InstancePrincipalSource {
    fn fetch(&self) -> Result<OciSessionMaterial> {
        self.cache.get_or_renew(now_secs(), false, || {
            block_on(renew(&self.settings, now_secs()))
        })
    }
}

/// Maps IMDS short region codes (`phx`) to full identifiers. Values that
/// already look like full identifiers pass through; this table is partial.
pub(super) fn normalize_region(raw: &str) -> String {
    let raw = raw.trim().to_ascii_lowercase();
    let full = match raw.as_str() {
        "phx" => "us-phoenix-1",
        "iad" => "us-ashburn-1",
        "ord" => "us-chicago-1",
        "fra" => "eu-frankfurt-1",
        "lhr" => "uk-london-1",
        "ams" => "eu-amsterdam-1",
        "zrh" => "eu-zurich-1",
        "yyz" => "ca-toronto-1",
        "yul" => "ca-montreal-1",
        "nrt" => "ap-tokyo-1",
        "kix" => "ap-osaka-1",
        "icn" => "ap-seoul-1",
        "bom" => "ap-mumbai-1",
        "syd" => "ap-sydney-1",
        "sin" => "ap-singapore-1",
        "gru" => "sa-saopaulo-1",
        "dxb" => "me-dubai-1",
        "jed" => "me-jeddah-1",
        other => other,
    };
    full.to_string()
}

/// Federation endpoint for a region (commercial realm `oraclecloud.com`).
pub(super) fn federation_origin(settings: &InstancePrincipalSettings, region: &str) -> String {
    match &settings.federation_endpoint {
        Some(endpoint) => endpoint.trim_end_matches('/').to_string(),
        None => format!("https://auth.{region}.oraclecloud.com"),
    }
}

fn client() -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(HTTP_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| Error::Model(format!("could not build HTTP client: {error}")))
}

/// IMDS GET with the required `Authorization: Bearer Oracle` header.
async fn imds_get(client: &reqwest::Client, url: &str) -> Result<String> {
    let response = client
        .get(url)
        .header(reqwest::header::AUTHORIZATION, "Bearer Oracle")
        .send()
        .await
        .map_err(|error| Error::Model(format!("IMDS request to {url} failed: {error}")))?;
    let status = response.status();
    if !status.is_success() {
        return Err(Error::Model(format!("IMDS {url} returned HTTP {status}")));
    }
    response
        .text()
        .await
        .map_err(|error| Error::Model(format!("IMDS response from {url} unreadable: {error}")))
}

/// Reads the region from IMDS (blocking).
pub(super) fn fetch_region(settings: &InstancePrincipalSettings) -> Result<String> {
    block_on(async {
        let url = format!(
            "{}{REGION_PATH}",
            settings.metadata_base_url.trim_end_matches('/')
        );
        Ok(normalize_region(&imds_get(&client()?, &url).await?))
    })
}

/// The JSON body of the X.509 federation request (`X509FederationDetails`).
pub(super) fn federation_body(
    leaf_pem: &str,
    intermediate_pems: &[String],
    public_pem: &str,
) -> Value {
    json!({
        "certificate": sanitize_pem(leaf_pem),
        "publicKey": sanitize_pem(public_pem),
        "intermediateCertificates": intermediate_pems.iter().map(|p| sanitize_pem(p)).collect::<Vec<_>>(),
        "fingerprintAlgorithm": "SHA256",
    })
}

/// Builds the federation request signed with the leaf key. The keyId is
/// `<tenancy>/fed-x509-sha256/<sha256 fingerprint>` and `host` is not signed.
pub(super) fn signed_federation_request(
    client: &reqwest::Client,
    origin: &str,
    body: &Value,
    tenancy: &str,
    fingerprint: &str,
    leaf_key: &SigningKey,
    date: &str,
) -> Result<reqwest::Request> {
    let mut request = client
        .post(format!("{origin}/v1/x509"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(serde_json::to_vec(body)?)
        .build()
        .map_err(|error| Error::Model(format!("federation request build failed: {error}")))?;
    let key_id = format!("{tenancy}/fed-x509-sha256/{fingerprint}");
    sign_request_with(&mut request, &key_id, leaf_key, date, false)?;
    Ok(request)
}

/// Extracts `token` from the federation response body.
pub(super) fn parse_federation_response(body: &str) -> Result<String> {
    let value: Value = serde_json::from_str(body)
        .map_err(|_| Error::Model("federation response was not JSON".into()))?;
    value["token"]
        .as_str()
        .filter(|token| !token.is_empty())
        .map(str::to_string)
        .ok_or_else(|| Error::Model("federation response carried no token".into()))
}

async fn renew(settings: &InstancePrincipalSettings, now: i64) -> Result<OciSessionMaterial> {
    let client = client()?;
    let base = settings.metadata_base_url.trim_end_matches('/');
    let leaf_pem = imds_get(&client, &format!("{base}{LEAF_CERT_PATH}")).await?;
    let leaf_key_pem = imds_get(&client, &format!("{base}{LEAF_KEY_PATH}")).await?;
    let intermediate = imds_get(&client, &format!("{base}{INTERMEDIATE_PATH}")).await?;
    let region = match &settings.region {
        Some(region) => region.clone(),
        None => normalize_region(&imds_get(&client, &format!("{base}{REGION_PATH}")).await?),
    };
    let leaf = parse_leaf_cert(&leaf_pem)?;
    let leaf_key = SigningKey::from_pem(&leaf_key_pem)?;
    let session = generate_session_keypair()?;
    let body = federation_body(&leaf_pem, &[intermediate], &session.public_pem);
    let origin = federation_origin(settings, &region);
    let date = httpdate::fmt_http_date(
        std::time::UNIX_EPOCH + Duration::from_secs(u64::try_from(now).unwrap_or(0)),
    );
    let mut last_error = Error::Model("federation exchange not attempted".into());
    for attempt in 0..3u32 {
        let request = signed_federation_request(
            &client,
            &origin,
            &body,
            &leaf.tenancy,
            &leaf.fingerprint,
            &leaf_key,
            &date,
        )?;
        tracing::debug!(target: "tinyinference::oci", attempt, "instance principal federation request");
        match client.execute(request).await {
            Ok(response) if response.status().is_success() => {
                let text = response.text().await.map_err(|error| {
                    Error::Model(format!("federation response unreadable: {error}"))
                })?;
                return Ok(OciSessionMaterial {
                    token: parse_federation_response(&text)?,
                    private_key_pem: session.private_pem,
                });
            }
            Ok(response) if response.status().is_client_error() => {
                // The SDK does not retry 4xx from the auth service.
                return Err(Error::Model(format!(
                    "federation endpoint rejected the request with HTTP {}",
                    response.status()
                )));
            }
            Ok(response) => {
                last_error = Error::Model(format!(
                    "federation endpoint returned HTTP {}",
                    response.status()
                ));
            }
            Err(error) => {
                last_error = Error::Model(format!("federation request failed: {error}"));
            }
        }
        tokio::time::sleep(Duration::from_millis(1000 << attempt)).await;
    }
    Err(last_error)
}

#[cfg(test)]
#[path = "instance_tests.rs"]
mod tests;
