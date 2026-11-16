//! OCI HTTP Signature v1 (`Authorization: Signature version="1",...`).
//!
//! The signature is RSA-SHA256 (PKCS#1 v1.5) over a signing string made of
//! `name: value` lines joined by `\n`. GET/DELETE/HEAD sign
//! `(request-target) date host`; requests with a body (POST/PUT/PATCH) also
//! sign `x-content-sha256 content-type content-length`, where
//! `x-content-sha256` is the base64 SHA-256 of the body (required even when
//! the body is empty). The `keyId` is `<tenancy>/<user>/<fingerprint>` for API
//! keys and `ST$<token>` for session / principal tokens.

use std::sync::Arc;
use std::time::SystemTime;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use reqwest::header::{CONTENT_LENGTH, CONTENT_TYPE, HeaderName, HeaderValue};
use ring::rand::SystemRandom;
use ring::signature::{RSA_PKCS1_SHA256, RsaKeyPair};
use sha2::{Digest, Sha256};

use super::auth::KeySource;
use crate::providers::RequestSigner;
use crate::{Error, Result};

/// An RSA private key able to sign OCI requests.
pub(crate) struct SigningKey {
    pair: RsaKeyPair,
}

impl std::fmt::Debug for SigningKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("SigningKey([redacted])")
    }
}

impl SigningKey {
    /// Parses an unencrypted PKCS#1 (`RSA PRIVATE KEY`) or PKCS#8
    /// (`PRIVATE KEY`) PEM document.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for encrypted, malformed, or unsupported keys.
    pub(crate) fn from_pem(pem: &str) -> Result<Self> {
        let (label, der) = decode_pem(pem)?;
        let pair = match label.as_str() {
            "PRIVATE KEY" => RsaKeyPair::from_pkcs8(&der),
            "RSA PRIVATE KEY" => RsaKeyPair::from_der(&der),
            other => {
                return Err(Error::Validation(format!(
                    "unsupported OCI private key PEM type `{other}` (expected PRIVATE KEY or RSA PRIVATE KEY)"
                )));
            }
        }
        .map_err(|_| {
            Error::Validation(
                "OCI private key is not a valid RSA key (2048 to 8192 bits required)".to_string(),
            )
        })?;
        Ok(Self { pair })
    }

    /// Signs `message` and returns the base64 signature.
    pub(crate) fn sign_b64(&self, message: &[u8]) -> Result<String> {
        let mut signature = vec![0u8; self.pair.public().modulus_len()];
        self.pair
            .sign(
                &RSA_PKCS1_SHA256,
                &SystemRandom::new(),
                message,
                &mut signature,
            )
            .map_err(|_| Error::Model("OCI request signing failed".to_string()))?;
        Ok(B64.encode(signature))
    }

    /// DER `RSAPublicKey` for the key (used by tests to verify signatures).
    #[cfg(test)]
    pub(crate) fn public_der(&self) -> Vec<u8> {
        use ring::signature::KeyPair;
        self.pair.public_key().as_ref().to_vec()
    }
}

/// Decodes the first PEM block, rejecting encrypted keys with a clear error.
fn decode_pem(pem: &str) -> Result<(String, Vec<u8>)> {
    let mut label: Option<String> = None;
    let mut body = String::new();
    for line in pem.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix("-----BEGIN ") {
            let name = rest.trim_end_matches('-').trim().to_string();
            if name == "ENCRYPTED PRIVATE KEY" {
                return Err(encrypted_key_error());
            }
            label = Some(name);
        } else if line.starts_with("-----END ") {
            break;
        } else if label.is_some() && !line.is_empty() {
            if line.starts_with("Proc-Type:") && line.contains("ENCRYPTED") {
                return Err(encrypted_key_error());
            }
            if line.contains(':') {
                // Other RFC 1421 header (e.g. DEK-Info); not key material.
                continue;
            }
            body.push_str(line);
        }
    }
    let label =
        label.ok_or_else(|| Error::Validation("OCI private key is not PEM encoded".to_string()))?;
    let der = B64
        .decode(body.as_bytes())
        .map_err(|_| Error::Validation("OCI private key PEM body is not valid base64".into()))?;
    Ok((label, der))
}

fn encrypted_key_error() -> Error {
    Error::Validation(
        "encrypted OCI private keys are not supported; decrypt the key (openssl rsa -in key.pem -out key_plain.pem) or use an unencrypted key".to_string(),
    )
}

/// The headers and signing string for one request, before signing.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SigningMaterial {
    /// Space-separated `headers="..."` list.
    pub(crate) header_names: String,
    /// The exact string that is signed.
    pub(crate) signing_string: String,
}

/// Builds the signing string. `body` is `Some((x_content_sha256, content_type,
/// content_length))` for body-bearing methods.
pub(crate) fn signing_material(
    method: &str,
    target: &str,
    date: &str,
    host: Option<&str>,
    body: Option<(&str, &str, usize)>,
) -> SigningMaterial {
    let mut names = vec!["(request-target)", "date"];
    let mut lines = vec![
        format!(
            "(request-target): {} {}",
            method.to_ascii_lowercase(),
            target
        ),
        format!("date: {date}"),
    ];
    if let Some(host) = host {
        names.push("host");
        lines.push(format!("host: {host}"));
    }
    if let Some((sha, content_type, length)) = body {
        names.extend(["x-content-sha256", "content-type", "content-length"]);
        lines.push(format!("x-content-sha256: {sha}"));
        lines.push(format!("content-type: {content_type}"));
        lines.push(format!("content-length: {length}"));
    }
    SigningMaterial {
        header_names: names.join(" "),
        signing_string: lines.join("\n"),
    }
}

/// Formats the `Authorization` header value.
pub(crate) fn authorization_header(key_id: &str, header_names: &str, signature: &str) -> String {
    format!(
        "Signature version=\"1\",keyId=\"{key_id}\",algorithm=\"rsa-sha256\",headers=\"{header_names}\",signature=\"{signature}\""
    )
}

fn has_body(method: &reqwest::Method) -> bool {
    matches!(
        *method,
        reqwest::Method::POST | reqwest::Method::PUT | reqwest::Method::PATCH
    )
}

/// Signs `request` in place with an explicit `date` (RFC 7231 GMT), signing
/// the `host` header as OCI services expect.
pub(crate) fn sign_request_at(
    request: &mut reqwest::Request,
    key_id: &str,
    key: &SigningKey,
    date: &str,
) -> Result<()> {
    sign_request_with(request, key_id, key, date, true)
}

/// Like [`sign_request_at`], with control over whether `host` is signed. The
/// X.509 federation endpoint signs `date (request-target)` plus the body
/// headers and no `host` (oci-go-sdk `federation_client.go`: `genericHeaders`).
pub(crate) fn sign_request_with(
    request: &mut reqwest::Request,
    key_id: &str,
    key: &SigningKey,
    date: &str,
    include_host: bool,
) -> Result<()> {
    let url = request.url().clone();
    let host = match (url.host_str(), url.port()) {
        (Some(host), Some(port)) => format!("{host}:{port}"),
        (Some(host), None) => host.to_string(),
        (None, _) => return Err(Error::Validation("OCI request URL has no host".into())),
    };
    let target = match url.query() {
        Some(query) => format!("{}?{query}", url.path()),
        None => url.path().to_string(),
    };
    let method = request.method().clone();
    let body_parts = if has_body(&method) {
        let bytes: &[u8] = match request.body() {
            None => &[],
            Some(body) => body.as_bytes().ok_or_else(|| {
                Error::Validation("OCI signing requires a buffered request body".into())
            })?,
        };
        let sha = B64.encode(Sha256::digest(bytes));
        let content_type = request
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("application/json")
            .to_string();
        Some((sha, content_type, bytes.len()))
    } else {
        None
    };
    let material = signing_material(
        method.as_str(),
        &target,
        date,
        include_host.then_some(host.as_str()),
        body_parts
            .as_ref()
            .map(|(sha, ct, len)| (sha.as_str(), ct.as_str(), *len)),
    );
    let signature = key.sign_b64(material.signing_string.as_bytes())?;
    let authorization = authorization_header(key_id, &material.header_names, &signature);

    let headers = request.headers_mut();
    let set = |headers: &mut reqwest::header::HeaderMap, name: HeaderName, value: &str| {
        let value = HeaderValue::from_str(value)
            .map_err(|_| Error::Validation("OCI signature header was not valid".into()))?;
        headers.insert(name, value);
        Ok::<(), Error>(())
    };
    set(headers, HeaderName::from_static("date"), date)?;
    if let Some((sha, content_type, length)) = &body_parts {
        set(headers, HeaderName::from_static("x-content-sha256"), sha)?;
        set(headers, CONTENT_TYPE, content_type)?;
        set(headers, CONTENT_LENGTH, &length.to_string())?;
    }
    let mut value = HeaderValue::from_str(&authorization)
        .map_err(|_| Error::Validation("OCI authorization header was not valid".into()))?;
    value.set_sensitive(true);
    headers.insert(reqwest::header::AUTHORIZATION, value);
    tracing::debug!(
        target: "tinyinference::oci",
        method = method.as_str(),
        headers = material.header_names.as_str(),
        "oci request signed"
    );
    Ok(())
}

/// A [`RequestSigner`] that signs with OCI credentials resolved per request.
#[derive(Debug)]
pub(crate) struct OciRequestSigner {
    source: Arc<KeySource>,
}

impl OciRequestSigner {
    pub(crate) fn new(source: KeySource) -> Self {
        Self {
            source: Arc::new(source),
        }
    }
}

impl RequestSigner for OciRequestSigner {
    fn sign(&self, request: &mut reqwest::Request) -> Result<()> {
        let (key_id, key) = self.source.current()?;
        let date = httpdate::fmt_http_date(SystemTime::now());
        sign_request_at(request, &key_id, &key, &date)
    }
}

#[cfg(test)]
#[path = "signer_tests.rs"]
mod tests;
