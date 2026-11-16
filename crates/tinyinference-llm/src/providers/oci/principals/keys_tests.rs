use super::*;
use crate::providers::oci::fixtures::{
    TEST_TENANCY, pem, test_leaf_cert_pem, test_leaf_fingerprint,
};
use crate::providers::oci::signer::SigningKey;

#[test]
fn session_keypair_is_usable_by_the_signer() {
    let pair = generate_session_keypair().unwrap();
    assert!(pair.public_pem.starts_with("-----BEGIN PUBLIC KEY-----"));
    let header = pem("RSA PRIVATE KEY", "");
    assert!(pair.private_pem.starts_with(header.lines().next().unwrap()));
    let key = SigningKey::from_pem(&pair.private_pem).unwrap();
    assert!(!key.sign_b64(b"x").unwrap().is_empty());
}

#[test]
fn leaf_cert_yields_tenancy_and_sha256_fingerprint() {
    let leaf = parse_leaf_cert(test_leaf_cert_pem()).unwrap();
    assert_eq!(leaf.tenancy, TEST_TENANCY);
    assert_eq!(leaf.fingerprint, test_leaf_fingerprint());
}

#[test]
fn bad_certificates_are_rejected() {
    assert!(parse_leaf_cert("nope").is_err());
}

#[test]
fn sanitize_strips_armor_and_newlines() {
    assert_eq!(
        sanitize_pem("-----BEGIN CERTIFICATE-----\nAAA\nBBB\n-----END CERTIFICATE-----\n"),
        "AAABBB"
    );
    assert_eq!(
        sanitize_pem("-----BEGIN PUBLIC KEY-----\nCC\n-----END PUBLIC KEY-----"),
        "CC"
    );
}
