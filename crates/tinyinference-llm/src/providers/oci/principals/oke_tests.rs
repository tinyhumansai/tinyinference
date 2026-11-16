use super::*;
use crate::providers::oci::fixtures::test_leaf_cert_pem;
use crate::providers::oci::principals::mock_http::{self, jwt};

fn wrap(token: &str) -> String {
    let inner = B64.encode(json!({ "token": token }).to_string());
    format!("\"{inner}\"")
}

#[test]
fn body_carries_the_pem_public_key_unsanitized() {
    let pem = "-----BEGIN PUBLIC KEY-----\nAA\n-----END PUBLIC KEY-----\n";
    assert_eq!(proxymux_body(pem), json!({ "podKey": pem }));
}

#[test]
fn response_is_unwrapped_decoded_and_prefix_stripped() {
    assert_eq!(parse_proxymux_response(&wrap("ST$a.b.c")).unwrap(), "a.b.c");
}

#[test]
fn legacy_capitalised_token_field_is_accepted() {
    let inner = B64.encode(r#"{"Token":"ST$x.y.z"}"#);
    assert_eq!(
        parse_proxymux_response(&format!("\"{inner}\"")).unwrap(),
        "x.y.z"
    );
}

#[test]
fn bad_responses_are_errors() {
    let empty = B64.encode(r#"{"token":""}"#);
    let short = B64.encode(r#"{"token":"ST"}"#);
    for body in [
        "".to_string(),
        "\"!!!\"".to_string(),
        format!("\"{}\"", B64.encode("not json")),
        format!("\"{empty}\""),
        format!("\"{short}\""),
        format!("\"{}\"", B64.encode("{}")),
    ] {
        assert!(parse_proxymux_response(&body).is_err(), "{body}");
    }
}

fn settings(dir: &std::path::Path, url: &str) -> OkeWorkloadIdentitySettings {
    std::fs::write(dir.join("token"), "sa-token-value\n").unwrap();
    std::fs::write(dir.join("ca.crt"), test_leaf_cert_pem()).unwrap();
    OkeWorkloadIdentitySettings {
        proxymux_url: format!("{url}/resourcePrincipalSessionTokens"),
        service_account_token_path: dir.join("token"),
        ca_cert_path: dir.join("ca.crt"),
    }
}

#[test]
fn full_exchange_against_a_local_mock_and_caching() {
    let dir = tempfile::tempdir().unwrap();
    let now = now_secs();
    let jwt = jwt(now, now + 7200);
    let reply = wrap(&format!("ST${jwt}"));
    let (url, log) = mock_http::start(move |_| (200, reply.clone()));
    let source = OkeWorkloadIdentitySource::new(settings(dir.path(), &url));

    let first = source.fetch().unwrap();
    assert_eq!(first.token, jwt);
    assert!(first.private_key_pem.contains("RSA PRIVATE KEY"));
    source.fetch().unwrap();

    let requests = log.lock().unwrap().clone();
    assert_eq!(requests.len(), 1, "second fetch must hit the cache");
    let request = &requests[0];
    assert_eq!(request.method, "POST");
    assert_eq!(request.path, "/resourcePrincipalSessionTokens");
    assert_eq!(
        request.header("authorization"),
        Some("Bearer sa-token-value")
    );
    assert_eq!(request.header("content-type"), Some("application/json"));
    let body: Value = serde_json::from_str(&request.body).unwrap();
    assert!(
        body["podKey"]
            .as_str()
            .unwrap()
            .starts_with("-----BEGIN PUBLIC KEY-----")
    );
}

#[test]
fn tokens_past_half_life_are_renewed() {
    let dir = tempfile::tempdir().unwrap();
    let now = now_secs();
    // Issued long ago and expiring soon enough that the soft expiry passed.
    let jwt = jwt(now - 3000, now + 1200);
    let reply = wrap(&format!("ST${jwt}"));
    let (url, log) = mock_http::start(move |_| (200, reply.clone()));
    let source = OkeWorkloadIdentitySource::new(settings(dir.path(), &url));
    source.fetch().unwrap();
    source.fetch().unwrap();
    assert_eq!(log.lock().unwrap().len(), 2);
}

#[test]
fn forbidden_and_missing_files_have_clear_errors() {
    let dir = tempfile::tempdir().unwrap();
    let (url, _log) = mock_http::start(|_| (403, "no".into()));
    let source = OkeWorkloadIdentitySource::new(settings(dir.path(), &url));
    assert!(source.fetch().unwrap_err().to_string().contains("enhanced"));

    let missing = OkeWorkloadIdentitySource::new(OkeWorkloadIdentitySettings {
        proxymux_url: "http://127.0.0.1:1/x".into(),
        service_account_token_path: dir.path().join("none"),
        ca_cert_path: dir.path().join("ca.crt"),
    });
    assert!(
        missing
            .fetch()
            .unwrap_err()
            .to_string()
            .contains("service account token")
    );
}
