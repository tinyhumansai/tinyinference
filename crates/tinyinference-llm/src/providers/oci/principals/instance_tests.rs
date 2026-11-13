use super::*;
use crate::providers::oci::fixtures::{
    TEST_TENANCY, test_leaf_cert_pem, test_leaf_fingerprint, test_pkcs8_pem,
};
use crate::providers::oci::principals::mock_http::{self, jwt};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;

const DATE: &str = "Thu, 05 Jan 2014 21:31:40 GMT";

#[test]
fn region_codes_are_normalized() {
    assert_eq!(normalize_region("phx\n"), "us-phoenix-1");
    assert_eq!(normalize_region("us-chicago-1"), "us-chicago-1");
}

#[test]
fn federation_origin_prefers_the_override() {
    let mut settings = InstancePrincipalSettings::from_env();
    assert_eq!(
        federation_origin(&settings, "us-phoenix-1"),
        "https://auth.us-phoenix-1.oraclecloud.com"
    );
    settings.federation_endpoint = Some("http://127.0.0.1:1/".into());
    assert_eq!(federation_origin(&settings, "x"), "http://127.0.0.1:1");
}

#[test]
fn federation_body_matches_the_sdk_shape() {
    let body = federation_body(
        "-----BEGIN CERTIFICATE-----\nAA\n-----END CERTIFICATE-----\n",
        &["-----BEGIN CERTIFICATE-----\nBB\n-----END CERTIFICATE-----\n".to_string()],
        "-----BEGIN PUBLIC KEY-----\nCC\n-----END PUBLIC KEY-----\n",
    );
    assert_eq!(
        body,
        serde_json::json!({
            "certificate": "AA",
            "publicKey": "CC",
            "intermediateCertificates": ["BB"],
            "fingerprintAlgorithm": "SHA256",
        })
    );
}

#[test]
fn federation_request_uses_fed_x509_sha256_key_id_and_no_host() {
    let key = SigningKey::from_pem(test_pkcs8_pem()).unwrap();
    let request = signed_federation_request(
        &reqwest::Client::new(),
        "https://auth.us-phoenix-1.oraclecloud.com",
        &serde_json::json!({"a": 1}),
        TEST_TENANCY,
        test_leaf_fingerprint(),
        &key,
        DATE,
    )
    .unwrap();
    assert_eq!(request.url().path(), "/v1/x509");
    let authorization = request.headers()["authorization"].to_str().unwrap();
    assert!(authorization.contains(&format!(
        "keyId=\"{TEST_TENANCY}/fed-x509-sha256/{}\"",
        test_leaf_fingerprint()
    )));
    assert!(authorization.contains(
        "headers=\"(request-target) date x-content-sha256 content-type content-length\""
    ));
}

#[test]
fn federation_response_requires_a_token() {
    assert_eq!(
        parse_federation_response(r#"{"token":"abc"}"#).unwrap(),
        "abc"
    );
    assert!(parse_federation_response(r#"{"token":""}"#).is_err());
    assert!(parse_federation_response("nope").is_err());
}

#[test]
fn full_exchange_against_a_local_mock_and_caching() {
    let token = jwt(now_secs(), now_secs() + 3600);
    let reply = serde_json::json!({ "token": token }).to_string();
    let (url, log) = mock_http::start(move |req| match req.path.as_str() {
        "/opc/v2/identity/cert.pem" => (200, test_leaf_cert_pem().to_string()),
        "/opc/v2/identity/key.pem" => (200, test_pkcs8_pem().to_string()),
        "/opc/v2/identity/intermediate.pem" => (200, test_leaf_cert_pem().to_string()),
        "/opc/v2/instance/region" => (200, "phx".to_string()),
        "/v1/x509" => (200, reply.clone()),
        _ => (404, String::new()),
    });
    let source = InstancePrincipalSource::new(InstancePrincipalSettings {
        metadata_base_url: format!("{url}/opc/v2"),
        federation_endpoint: Some(url.clone()),
        region: None,
    });
    let first = source.fetch().unwrap();
    assert_eq!(first.token, token);
    assert!(first.private_key_pem.contains("RSA PRIVATE KEY"));
    let second = source.fetch().unwrap();
    assert_eq!(second.private_key_pem, first.private_key_pem);

    let requests = log.lock().unwrap().clone();
    let federation: Vec<_> = requests.iter().filter(|r| r.path == "/v1/x509").collect();
    assert_eq!(federation.len(), 1, "second fetch must hit the cache");
    for imds in requests.iter().filter(|r| r.path.starts_with("/opc/")) {
        assert_eq!(imds.header("authorization"), Some("Bearer Oracle"));
    }
    let body: serde_json::Value = serde_json::from_str(&federation[0].body).unwrap();
    assert_eq!(body["fingerprintAlgorithm"], "SHA256");
    assert!(body["publicKey"].as_str().unwrap().len() > 100);
    assert!(
        federation[0]
            .header("authorization")
            .unwrap()
            .contains("/fed-x509-sha256/")
    );
    assert!(B64.decode(body["certificate"].as_str().unwrap()).is_ok());
}

#[test]
fn client_errors_from_the_federation_endpoint_are_not_retried() {
    let (url, log) = mock_http::start(|req| match req.path.as_str() {
        "/opc/v2/identity/cert.pem" | "/opc/v2/identity/intermediate.pem" => {
            (200, test_leaf_cert_pem().to_string())
        }
        "/opc/v2/identity/key.pem" => (200, test_pkcs8_pem().to_string()),
        "/v1/x509" => (401, "{}".into()),
        _ => (404, String::new()),
    });
    let source = InstancePrincipalSource::new(InstancePrincipalSettings {
        metadata_base_url: format!("{url}/opc/v2"),
        federation_endpoint: Some(url),
        region: Some("us-phoenix-1".into()),
    });
    let error = source.fetch().unwrap_err().to_string();
    assert!(error.contains("401"), "{error}");
    let calls = log
        .lock()
        .unwrap()
        .iter()
        .filter(|r| r.path == "/v1/x509")
        .count();
    assert_eq!(calls, 1);
}

#[test]
fn imds_failure_is_reported() {
    let (url, _log) = mock_http::start(|_| (500, "boom".into()));
    let source = InstancePrincipalSource::new(InstancePrincipalSettings {
        metadata_base_url: format!("{url}/opc/v2"),
        federation_endpoint: None,
        region: Some("us-phoenix-1".into()),
    });
    assert!(source.fetch().unwrap_err().to_string().contains("500"));
}
