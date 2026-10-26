use super::*;
use crate::message::Message;
use crate::model::ModelRequest;
use crate::providers::oci::fixtures::test_pkcs8_pem;

fn api_key() -> OciAuth {
    OciAuth::ApiKey {
        bearer: "sk-test".into(),
    }
}

fn signed() -> OciAuth {
    OciAuth::SessionToken {
        token: "tok".into(),
        private_key_pem: test_pkcs8_pem().into(),
    }
}

fn config(auth: OciAuth) -> OciConfig {
    let mut config = OciConfig::new(auth, "openai.gpt-4o");
    config.region = Some("us-chicago-1".into());
    config
}

#[test]
fn endpoint_derives_from_region_or_override() {
    assert_eq!(
        config(api_key()).service_endpoint().unwrap(),
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com"
    );
    let mut overridden = config(api_key());
    overridden.endpoint = Some("https://proxy.example/".into());
    assert_eq!(
        overridden.service_endpoint().unwrap(),
        "https://proxy.example"
    );
    overridden.endpoint = Some("http://example.com".into());
    assert!(overridden.service_endpoint().is_err());
    overridden.endpoint = Some("http://127.0.0.1:9999".into());
    assert!(overridden.service_endpoint().is_ok());
    let mut missing = config(api_key());
    missing.region = None;
    assert!(missing.service_endpoint().is_err());
    missing.region = Some("bad/region".into());
    assert!(missing.service_endpoint().is_err());
}

#[test]
fn openai_base_uses_bearer_and_project_header() {
    let mut cfg = config(api_key());
    cfg.project_id = Some("ocid1.generativeaiproject.p".into());
    cfg.compartment_id = Some("ocid1.compartment.c".into());
    let endpoint = cfg.service_endpoint().unwrap();
    let ResolvedAuth::Bearer(_) = cfg.auth.resolve().unwrap() else {
        panic!("bearer expected");
    };
    let model = openai_compat_model(
        &cfg,
        &endpoint,
        OciOpenAiBase::OpenAi,
        cfg.auth.resolve().unwrap(),
    );
    assert_eq!(
        model.base_url(),
        "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com/openai/v1"
    );
    assert_eq!(model.provider(), "oci");
    assert_eq!(model.model(), "openai.gpt-4o");
    assert_eq!(*model.auth_style(), AuthStyle::Bearer);
    assert!(!model.has_request_signer());
    assert_eq!(
        model.extra_headers(),
        [
            (
                "OpenAI-Project".to_string(),
                "ocid1.generativeaiproject.p".to_string()
            ),
            (
                "opc-compartment-id".to_string(),
                "ocid1.compartment.c".to_string()
            )
        ]
    );
}

fn compat_body(model_id: &str) -> serde_json::Value {
    let mut cfg = config(api_key());
    cfg.model = model_id.into();
    let endpoint = cfg.service_endpoint().unwrap();
    let model = openai_compat_model(
        &cfg,
        &endpoint,
        OciOpenAiBase::OpenAi,
        cfg.auth.resolve().unwrap(),
    );
    let mut request = ModelRequest::new(vec![Message::user("hi")]).with_max_tokens(64);
    request.temperature = Some(0.7);
    serde_json::to_value(model.translate_request(&request).unwrap()).unwrap()
}

#[test]
fn compat_path_drops_temperature_and_max_tokens_for_openai_reasoning_models() {
    let body = compat_body("openai.gpt-5.5");
    assert!(body.get("temperature").is_none(), "{body}");
    assert!(body.get("max_tokens").is_none(), "{body}");
    assert_eq!(body["max_completion_tokens"], 64);

    let body = compat_body("openai.gpt-oss-120b");
    assert_eq!(body["temperature"], 0.7);
    assert_eq!(body["max_tokens"], 64);
    assert!(body.get("max_completion_tokens").is_none(), "{body}");
}

#[test]
fn compat_path_applies_the_temperature_override_except_on_reasoning_models() {
    for (model_id, expected) in [("openai.gpt-oss-120b", Some(0.2)), ("openai.gpt-5.5", None)] {
        let mut cfg = config(api_key());
        cfg.model = model_id.into();
        cfg.temperature_override = Some(0.2);
        let endpoint = cfg.service_endpoint().unwrap();
        let model = openai_compat_model(
            &cfg,
            &endpoint,
            OciOpenAiBase::OpenAi,
            cfg.auth.resolve().unwrap(),
        );
        let mut request = ModelRequest::new(vec![Message::user("hi")]);
        request.temperature = Some(0.7);
        let body = serde_json::to_value(model.translate_request(&request).unwrap()).unwrap();
        assert_eq!(
            body.get("temperature").and_then(|t| t.as_f64()),
            expected,
            "{body}"
        );
    }
}

#[test]
fn native_path_applies_the_temperature_override() {
    let body = |model: &OciNativeModel| {
        let mut request = ModelRequest::new(vec![Message::user("hi")]);
        request.temperature = Some(0.7);
        let request = model.prepare(&request, false).unwrap();
        serde_json::from_slice::<serde_json::Value>(request.body().unwrap().as_bytes().unwrap())
            .unwrap()
    };
    let plain = native(api_key(), OciApiFormat::Generic);
    assert_eq!(body(&plain)["chatRequest"]["temperature"], 0.7);
    let overridden = native(api_key(), OciApiFormat::Generic).with_temperature_override(Some(0.2));
    assert_eq!(body(&overridden)["chatRequest"]["temperature"], 0.2);
}

#[test]
fn actions_base_with_signature_auth_has_no_header_credential() {
    let mut cfg = config(signed());
    cfg.compartment_id = Some("ocid1.compartment.c".into());
    cfg.project_id = Some("ignored-on-actions".into());
    let endpoint = cfg.service_endpoint().unwrap();
    let model = openai_compat_model(
        &cfg,
        &endpoint,
        OciOpenAiBase::Actions,
        cfg.auth.resolve().unwrap(),
    );
    assert!(model.base_url().ends_with("/20231130/actions/v1"));
    assert_eq!(*model.auth_style(), AuthStyle::None);
    assert!(model.has_request_signer());
    assert_eq!(
        model.extra_headers(),
        [(
            "opc-compartment-id".to_string(),
            "ocid1.compartment.c".to_string()
        )]
    );
}

#[test]
fn dedicated_serving_addresses_the_endpoint_on_the_compat_path() {
    let mut cfg = config(api_key());
    cfg.serving = OciServing::Dedicated {
        endpoint_id: "ocid1.generativeaiendpoint.e".into(),
    };
    let endpoint = cfg.service_endpoint().unwrap();
    let model = openai_compat_model(
        &cfg,
        &endpoint,
        OciOpenAiBase::OpenAi,
        cfg.auth.resolve().unwrap(),
    );
    assert_eq!(model.model(), "ocid1.generativeaiendpoint.e");
}

fn native(auth: OciAuth, format: OciApiFormat) -> OciNativeModel {
    let mut cfg = config(auth);
    cfg.endpoint = Some("http://127.0.0.1:1".into());
    cfg.compartment_id = Some("ocid1.compartment.c".into());
    cfg.transport = OciTransport::Native { api_format: format };
    let resolved = cfg.auth.resolve().unwrap();
    OciNativeModel::new(
        cfg.service_endpoint().unwrap(),
        resolved,
        "ocid1.compartment.c".into(),
        OciServing::OnDemand,
        format,
        cfg.model,
        vec![],
    )
}

#[test]
fn native_bearer_request_carries_the_key_and_no_signature() {
    let model = native(api_key(), OciApiFormat::Generic);
    let request = model
        .prepare(&ModelRequest::new(vec![Message::user("hi")]), false)
        .unwrap();
    assert_eq!(request.url().path(), "/20231130/actions/chat");
    assert_eq!(request.headers()["authorization"], "Bearer sk-test");
    assert!(request.headers().get("x-content-sha256").is_none());
    assert_eq!(request.headers()["accept"], "application/json");
}

#[test]
fn native_signed_request_uses_signature_auth_and_digest() {
    let model = native(signed(), OciApiFormat::Cohere);
    let request = model
        .prepare(&ModelRequest::new(vec![Message::user("hi")]), true)
        .unwrap();
    let authorization = request.headers()["authorization"].to_str().unwrap();
    assert!(authorization.starts_with("Signature version=\"1\",keyId=\"ST$tok\""));
    assert!(authorization.contains("x-content-sha256 content-type content-length"));
    assert!(request.headers().get("x-content-sha256").is_some());
    assert_eq!(request.headers()["accept"], "text/event-stream");
    let body: serde_json::Value =
        serde_json::from_slice(request.body().unwrap().as_bytes().unwrap()).unwrap();
    assert_eq!(body["chatRequest"]["apiFormat"], "COHERE");
}

#[test]
fn native_requires_a_compartment_and_unsupported_auth_fails_to_build() {
    let mut cfg = config(api_key());
    cfg.transport = OciTransport::Native {
        api_format: OciApiFormat::Generic,
    };
    assert!(build_oci_model(cfg.clone()).is_err());
    cfg.compartment_id = Some("ocid1.compartment.c".into());
    assert!(build_oci_model(cfg).is_ok());
    #[cfg(not(feature = "oci-principals"))]
    {
        let unsupported = config(OciAuth::InstancePrincipal);
        assert!(matches!(
            build_oci_model(unsupported),
            Err(Error::Unsupported(_))
        ));
    }
}

#[test]
fn native_debug_does_not_leak_the_key() {
    let model = native(api_key(), OciApiFormat::Generic);
    assert!(!format!("{model:?}").contains("sk-test"));
}
