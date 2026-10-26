use super::*;
use crate::providers::oci::fixtures::{openssl_signature_b64, pem, test_pkcs1_pem, test_pkcs8_pem};
use ring::signature::{RSA_PKCS1_2048_8192_SHA256, UnparsedPublicKey};

const DATE: &str = "Thu, 05 Jan 2014 21:31:40 GMT";

fn verify(key: &SigningKey, message: &str, signature_b64: &str) -> bool {
    let signature = B64.decode(signature_b64).unwrap();
    UnparsedPublicKey::new(&RSA_PKCS1_2048_8192_SHA256, key.public_der())
        .verify(message.as_bytes(), &signature)
        .is_ok()
}

#[test]
fn pkcs8_and_pkcs1_keys_produce_the_openssl_signature() {
    for pem in [test_pkcs8_pem(), test_pkcs1_pem()] {
        let key = SigningKey::from_pem(pem).expect("key parses");
        assert_eq!(
            key.sign_b64(b"line1\nline2").unwrap(),
            openssl_signature_b64(b"line1\nline2")
        );
    }
}

#[test]
fn encrypted_keys_are_rejected_with_a_clear_error() {
    let pkcs8 = pem("ENCRYPTED PRIVATE KEY", "AAAA");
    let pkcs1 = pem(
        "RSA PRIVATE KEY",
        "Proc-Type: 4,ENCRYPTED\nDEK-Info: AES-128-CBC,00\n\nAAAA",
    );
    for pem in [pkcs8, pkcs1] {
        let error = SigningKey::from_pem(&pem).unwrap_err().to_string();
        assert!(error.contains("encrypted"), "{error}");
    }
}

#[test]
fn malformed_pem_is_a_validation_error() {
    assert!(SigningKey::from_pem("not pem").is_err());
    assert!(SigningKey::from_pem(&pem("PRIVATE KEY", "!!!")).is_err());
    assert!(
        SigningKey::from_pem("-----BEGIN PUBLIC KEY-----\nAAAA\n-----END PUBLIC KEY-----").is_err()
    );
}

#[test]
fn get_signs_request_target_date_and_host() {
    let material = signing_material(
        "GET",
        "/20231130/models?x=1",
        DATE,
        Some("h.example.com"),
        None,
    );
    assert_eq!(material.header_names, "(request-target) date host");
    assert_eq!(
        material.signing_string,
        "(request-target): get /20231130/models?x=1\ndate: Thu, 05 Jan 2014 21:31:40 GMT\nhost: h.example.com"
    );
}

#[test]
fn post_additionally_signs_body_headers() {
    let material = signing_material(
        "POST",
        "/20231130/actions/chat",
        DATE,
        Some("h.example.com"),
        Some(("SHA=", "application/json", 12)),
    );
    assert_eq!(
        material.header_names,
        "(request-target) date host x-content-sha256 content-type content-length"
    );
    assert_eq!(
        material.signing_string,
        "(request-target): post /20231130/actions/chat\n\
         date: Thu, 05 Jan 2014 21:31:40 GMT\n\
         host: h.example.com\n\
         x-content-sha256: SHA=\n\
         content-type: application/json\n\
         content-length: 12"
    );
}

#[test]
fn authorization_header_has_the_documented_shape() {
    assert_eq!(
        authorization_header("t/u/fp", "(request-target) date host", "SIG"),
        "Signature version=\"1\",keyId=\"t/u/fp\",algorithm=\"rsa-sha256\",headers=\"(request-target) date host\",signature=\"SIG\""
    );
}

fn header<'a>(request: &'a reqwest::Request, name: &str) -> &'a str {
    request.headers().get(name).unwrap().to_str().unwrap()
}

fn signature_of(authorization: &str) -> &str {
    authorization
        .split("signature=\"")
        .nth(1)
        .unwrap()
        .trim_end_matches('"')
}

#[test]
fn post_request_is_signed_with_a_verifiable_signature() {
    let key = SigningKey::from_pem(test_pkcs8_pem()).unwrap();
    let body = br#"{"hello":"world"}"#.to_vec();
    let mut request = reqwest::Client::new()
        .post(
            "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/20231130/actions/chat",
        )
        .body(body.clone())
        .build()
        .unwrap();
    sign_request_at(&mut request, "ST$tok", &key, DATE).unwrap();

    let sha = B64.encode(Sha256::digest(&body));
    assert_eq!(header(&request, "x-content-sha256"), sha);
    assert_eq!(header(&request, "content-length"), body.len().to_string());
    assert_eq!(header(&request, "content-type"), "application/json");
    assert_eq!(header(&request, "date"), DATE);

    let authorization = header(&request, "authorization");
    assert!(
        authorization
            .starts_with("Signature version=\"1\",keyId=\"ST$tok\",algorithm=\"rsa-sha256\"")
    );
    let expected = format!(
        "(request-target): post /20231130/actions/chat\ndate: {DATE}\nhost: inference.generativeai.us-chicago-1.oci.oraclecloud.com\nx-content-sha256: {sha}\ncontent-type: application/json\ncontent-length: {}",
        body.len()
    );
    assert!(verify(&key, &expected, signature_of(authorization)));
}

#[test]
fn empty_post_body_still_carries_the_content_digest() {
    let key = SigningKey::from_pem(test_pkcs1_pem()).unwrap();
    let mut request = reqwest::Client::new()
        .post("https://example.com/x")
        .build()
        .unwrap();
    sign_request_at(&mut request, "k", &key, DATE).unwrap();
    assert_eq!(
        header(&request, "x-content-sha256"),
        B64.encode(Sha256::digest(b""))
    );
    assert_eq!(header(&request, "content-length"), "0");
}

#[test]
fn get_request_signs_only_the_basic_headers_and_keeps_ports() {
    let key = SigningKey::from_pem(test_pkcs8_pem()).unwrap();
    let mut request = reqwest::Client::new()
        .get("https://example.com:8443/a/b?q=1")
        .build()
        .unwrap();
    sign_request_at(&mut request, "k", &key, DATE).unwrap();
    assert!(request.headers().get("x-content-sha256").is_none());
    let authorization = header(&request, "authorization");
    assert!(authorization.contains("headers=\"(request-target) date host\""));
    let expected = format!("(request-target): get /a/b?q=1\ndate: {DATE}\nhost: example.com:8443");
    assert!(verify(&key, &expected, signature_of(authorization)));
}

#[test]
fn debug_output_never_contains_key_material() {
    let key = SigningKey::from_pem(test_pkcs8_pem()).unwrap();
    assert_eq!(format!("{key:?}"), "SigningKey([redacted])");
}

#[test]
fn federation_style_signing_omits_host() {
    let key = SigningKey::from_pem(test_pkcs8_pem()).unwrap();
    let mut request = reqwest::Client::new()
        .post("https://auth.us-phoenix-1.oraclecloud.com/v1/x509")
        .body("{}")
        .build()
        .unwrap();
    sign_request_with(&mut request, "t/fed-x509-sha256/fp", &key, DATE, false).unwrap();
    let authorization = header(&request, "authorization");
    assert!(authorization.contains(
        "headers=\"(request-target) date x-content-sha256 content-type content-length\""
    ));
}
