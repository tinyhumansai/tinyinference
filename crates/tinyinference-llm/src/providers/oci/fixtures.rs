//! Test key material, generated with OpenSSL once per test process. No key
//! or certificate is committed to the repository.

use std::sync::LazyLock;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use openssl::hash::MessageDigest;
use openssl::pkey::{PKey, Private};
use openssl::rsa::Rsa;
use openssl::sign::Signer;

/// Tenancy OCID embedded in the test certificate.
#[cfg(feature = "oci-principals")]
pub(super) const TEST_TENANCY: &str = "ocid1.tenancy.oc1..testtenant";

struct Material {
    key: PKey<Private>,
    pkcs8_pem: String,
    pkcs1_pem: String,
    #[cfg(feature = "oci-principals")]
    leaf_cert_pem: String,
    #[cfg(feature = "oci-principals")]
    leaf_fingerprint: String,
}

static MATERIAL: LazyLock<Material> = LazyLock::new(|| {
    let rsa = Rsa::generate(2048).expect("generate test key");
    let pkcs1_pem = String::from_utf8(rsa.private_key_to_pem().unwrap()).unwrap();
    let key = PKey::from_rsa(rsa).unwrap();
    let pkcs8_pem = String::from_utf8(key.private_key_to_pem_pkcs8().unwrap()).unwrap();
    #[cfg(feature = "oci-principals")]
    let (leaf_cert_pem, leaf_fingerprint) = leaf_cert(&key);
    Material {
        key,
        pkcs8_pem,
        pkcs1_pem,
        #[cfg(feature = "oci-principals")]
        leaf_cert_pem,
        #[cfg(feature = "oci-principals")]
        leaf_fingerprint,
    }
});

/// Self-signed leaf certificate for the test key, with subject attributes
/// `opc-tenant:<TEST_TENANCY>` and `opc-identity:...`, plus the lowercase
/// colon-separated SHA-256 fingerprint of its DER form.
#[cfg(feature = "oci-principals")]
fn leaf_cert(key: &PKey<Private>) -> (String, String) {
    use openssl::asn1::Asn1Time;
    use openssl::x509::{X509, X509NameBuilder};

    let mut name = X509NameBuilder::new().unwrap();
    name.append_entry_by_text("O", "Oracle").unwrap();
    name.append_entry_by_text("OU", &format!("opc-tenant:{TEST_TENANCY}"))
        .unwrap();
    name.append_entry_by_text("OU", "opc-identity:ocid1.instance.oc1..testinst")
        .unwrap();
    name.append_entry_by_text("CN", "test-leaf").unwrap();
    let name = name.build();

    let mut cert = X509::builder().unwrap();
    cert.set_version(2).unwrap();
    cert.set_subject_name(&name).unwrap();
    cert.set_issuer_name(&name).unwrap();
    cert.set_pubkey(key).unwrap();
    cert.set_not_before(&Asn1Time::days_from_now(0).unwrap())
        .unwrap();
    cert.set_not_after(&Asn1Time::days_from_now(1).unwrap())
        .unwrap();
    cert.sign(key, MessageDigest::sha256()).unwrap();
    let cert = cert.build();

    let fingerprint = cert
        .digest(MessageDigest::sha256())
        .unwrap()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join(":");
    (
        String::from_utf8(cert.to_pem().unwrap()).unwrap(),
        fingerprint,
    )
}

/// PKCS#8 (`PRIVATE KEY`) PEM of the test key.
pub(super) fn test_pkcs8_pem() -> &'static str {
    &MATERIAL.pkcs8_pem
}

/// PKCS#1 (`RSA PRIVATE KEY`) PEM of the same key.
pub(super) fn test_pkcs1_pem() -> &'static str {
    &MATERIAL.pkcs1_pem
}

/// Base64 RSA-SHA256 (PKCS#1 v1.5) signature of `message` under the test key,
/// computed by OpenSSL, independently of the signer under test.
pub(super) fn openssl_signature_b64(message: &[u8]) -> String {
    let mut signer = Signer::new(MessageDigest::sha256(), &MATERIAL.key).unwrap();
    signer.update(message).unwrap();
    B64.encode(signer.sign_to_vec().unwrap())
}

#[cfg(feature = "oci-principals")]
pub(super) fn test_leaf_cert_pem() -> &'static str {
    &MATERIAL.leaf_cert_pem
}

#[cfg(feature = "oci-principals")]
pub(super) fn test_leaf_fingerprint() -> &'static str {
    &MATERIAL.leaf_fingerprint
}

/// A PEM document with `label` armor around `body`. Tests build armor this
/// way so the source never contains a literal private-key header.
pub(super) fn pem(label: &str, body: &str) -> String {
    let dashes = "-".repeat(5);
    format!("{dashes}BEGIN {label}{dashes}\n{body}\n{dashes}END {label}{dashes}\n")
}
