//! OpenSSL-backed helpers: RSA session keypairs and leaf-certificate parsing.

use openssl::hash::{MessageDigest, hash};
use openssl::pkey::PKey;
use openssl::rsa::Rsa;
use openssl::x509::X509;

use crate::{Error, Result};

/// A fresh RSA-2048 session keypair.
pub(super) struct SessionKeypair {
    /// PKCS#1 `RSA PRIVATE KEY` PEM (consumed by the signer).
    pub(super) private_pem: String,
    /// SPKI `PUBLIC KEY` PEM (what the exchange endpoints bind the token to).
    pub(super) public_pem: String,
}

pub(super) fn generate_session_keypair() -> Result<SessionKeypair> {
    let fail = |what: &str| Error::Model(format!("session key generation failed ({what})"));
    let rsa = Rsa::generate(2048).map_err(|_| fail("generate"))?;
    let private_pem = String::from_utf8(rsa.private_key_to_pem().map_err(|_| fail("private pem"))?)
        .map_err(|_| fail("private utf8"))?;
    let public = PKey::from_rsa(rsa).map_err(|_| fail("pkey"))?;
    let public_pem = String::from_utf8(public.public_key_to_pem().map_err(|_| fail("public pem"))?)
        .map_err(|_| fail("public utf8"))?;
    Ok(SessionKeypair {
        private_pem,
        public_pem,
    })
}

/// Facts extracted from the instance leaf certificate.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct LeafCert {
    /// Tenancy OCID from the `opc-tenant:` subject attribute.
    pub(super) tenancy: String,
    /// Lowercase colon-separated SHA-256 of the DER certificate.
    pub(super) fingerprint: String,
}

pub(super) fn parse_leaf_cert(pem: &str) -> Result<LeafCert> {
    let cert = X509::from_pem(pem.as_bytes())
        .map_err(|_| Error::Model("instance leaf certificate is not valid PEM X.509".into()))?;
    let tenancy = cert
        .subject_name()
        .entries()
        .filter_map(|entry| entry.data().to_string().ok())
        .find_map(|value| value.strip_prefix("opc-tenant:").map(str::to_string))
        .ok_or_else(|| {
            Error::Model("instance leaf certificate has no opc-tenant subject attribute".into())
        })?;
    let der = cert
        .to_der()
        .map_err(|_| Error::Model("could not encode leaf certificate".into()))?;
    let digest = hash(MessageDigest::sha256(), &der)
        .map_err(|_| Error::Model("could not hash leaf certificate".into()))?;
    let fingerprint = digest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(":");
    Ok(LeafCert {
        tenancy,
        fingerprint,
    })
}

/// Strips PEM armor and newlines, as the federation request expects
/// (`sanitizeCertificateString` in the SDK).
pub(super) fn sanitize_pem(pem: &str) -> String {
    pem.replace("-----BEGIN CERTIFICATE-----", "")
        .replace("-----END CERTIFICATE-----", "")
        .replace("-----BEGIN PUBLIC KEY-----", "")
        .replace("-----END PUBLIC KEY-----", "")
        .replace('\n', "")
}

#[cfg(test)]
#[path = "keys_tests.rs"]
mod tests;
