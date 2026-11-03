use super::*;

async fn respond_to_requests(
    listener: tokio::net::TcpListener,
    responses: Vec<(&'static str, u16, &'static str)>,
) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    for (expected_request, status, body) in responses {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut chunk = [0; 1024];
        let header_end = loop {
            let read = stream.read(&mut chunk).await.unwrap();
            assert_ne!(read, 0, "client closed before sending request headers");
            request.extend_from_slice(&chunk[..read]);
            if let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                break header_end + 4;
            }
        };
        let headers = String::from_utf8_lossy(&request[..header_end]);
        assert!(
            headers.starts_with(expected_request),
            "unexpected request: {headers}"
        );
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().unwrap())
            })
            .unwrap_or(0);
        while request.len() - header_end < content_length {
            let read = stream.read(&mut chunk).await.unwrap();
            assert_ne!(read, 0, "client closed before sending the request body");
            request.extend_from_slice(&chunk[..read]);
        }
        let reason = if status == 200 { "OK" } else { "Unavailable" };
        let response = format!(
            "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
    }
}

#[test]
fn effective_slug_preserves_precedence_and_trims_values() {
    assert_eq!(
        effective_embedder_slug(true, Some(" local "), Some("openai"), true),
        "none"
    );
    assert_eq!(
        effective_embedder_slug(false, Some(" local "), Some("openai"), true),
        "ollama"
    );
    assert_eq!(
        effective_embedder_slug(false, None, Some(" openai "), true),
        "custom"
    );
    assert_eq!(
        effective_embedder_slug(false, None, Some("custom:https://x"), true),
        "custom"
    );
    assert_eq!(effective_embedder_slug(false, None, None, true), "cloud");
    assert_eq!(
        effective_embedder_slug(false, None, None, false),
        "unconfigured"
    );
}

#[test]
fn rejection_is_serializable_with_stable_fields() {
    let rejection = VerifyRejection::Probe(crate::probe::EmbeddingProbeRejection {
        error: "EMBEDDINGS_VERIFICATION_FAILED",
        message: "retry",
        summary: "failed",
        detail: None,
    });
    let json = serde_json::to_value(rejection).unwrap();
    assert_eq!(json["error"], "EMBEDDINGS_VERIFICATION_FAILED");
    assert_eq!(json["message"], "retry");
}

#[tokio::test]
async fn invalid_userinfo_endpoint_never_reaches_models_guidance() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let endpoint = format!("http://secret@{address}");
    let result = verify_custom_endpoint(&endpoint, "sensitive-token", "model", 3).await;
    assert!(matches!(result, Err(VerifyRejection::Probe(_))));
    assert!(
        matches!(listener.accept(), Err(error) if error.kind() == std::io::ErrorKind::WouldBlock)
    );
}

#[tokio::test]
async fn unavailable_model_preserves_identity_message_and_both_errors() {
    let model =
        UnavailableEmbeddingModel::new("embedding-v1", 1024, "BACKEND_UNAVAILABLE: offline");
    assert_eq!(model.name(), "cloud");
    assert_eq!(model.signature(), "unavailable:embedding-v1:1024");
    assert_eq!(model.model_id(), "embedding-v1");
    assert_eq!(model.dimensions(), 1024);
    let error = model.embed(&["text".into()]).await.unwrap_err().to_string();
    assert_eq!(error, "validation error: BACKEND_UNAVAILABLE: offline");
    let error = model
        .embed_with_usage(&["text".into()])
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(error, "validation error: BACKEND_UNAVAILABLE: offline");
}

#[tokio::test]
async fn connection_test_uses_host_managed_factory_and_returns_actual_width() {
    let settings = EmbeddingFactorySettings {
        provider: "cloud".into(),
        model: "configured-model".into(),
        dimensions: 4,
        ollama_base_url: "http://localhost:11434".into(),
    };
    let credentials: CredentialResolver = std::sync::Arc::new(|_| String::new());
    let managed: ManagedModelFactory = std::sync::Arc::new(|_, dims| {
        Box::pin(async move {
            Ok(Box::new(crate::MockEmbeddingModel::new(dims)) as Box<dyn crate::EmbeddingModel>)
        })
    });
    let result = test_embedding_connection(&settings, &credentials, &managed, None)
        .await
        .unwrap();
    assert_eq!(result.actual_dims, 4);
}

#[tokio::test]
async fn detailed_connection_test_identifies_model_build_failures() {
    let settings = EmbeddingFactorySettings {
        provider: "not-a-provider".into(),
        model: "model".into(),
        dimensions: 4,
        ollama_base_url: "http://localhost:11434".into(),
    };
    let credentials: CredentialResolver = std::sync::Arc::new(|_| String::new());
    let managed: ManagedModelFactory =
        std::sync::Arc::new(|_, _| Box::pin(async { unreachable!() }));

    assert!(matches!(
        test_embedding_connection_detailed(&settings, &credentials, &managed, None).await,
        Err(ConnectionTestError::Build(crate::Error::Validation(_)))
    ));
}

#[tokio::test]
async fn detailed_connection_test_identifies_embedding_failures() {
    struct FailingModel;

    #[async_trait::async_trait]
    impl crate::EmbeddingModel for FailingModel {
        fn name(&self) -> &str {
            "managed"
        }

        fn model_id(&self) -> &str {
            "model"
        }

        fn dimensions(&self) -> usize {
            4
        }

        async fn embed(&self, _texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
            Err(crate::Error::Embedding("offline".into()))
        }
    }

    let settings = EmbeddingFactorySettings {
        provider: "managed".into(),
        model: "model".into(),
        dimensions: 4,
        ollama_base_url: "http://localhost:11434".into(),
    };
    let credentials: CredentialResolver = std::sync::Arc::new(|_| String::new());
    let managed: ManagedModelFactory = std::sync::Arc::new(|_, _| {
        Box::pin(async { Ok(Box::new(FailingModel) as Box<dyn crate::EmbeddingModel>) })
    });

    assert!(matches!(
        test_embedding_connection_detailed(&settings, &credentials, &managed, None).await,
        Err(ConnectionTestError::Embed(crate::Error::Embedding(message))) if message == "offline"
    ));
}

#[tokio::test]
async fn connection_test_preserves_noop_provider_success() {
    let settings = EmbeddingFactorySettings {
        provider: "none".into(),
        model: "none".into(),
        dimensions: 0,
        ollama_base_url: String::new(),
    };
    let credentials: CredentialResolver = std::sync::Arc::new(|_| String::new());
    let managed: ManagedModelFactory =
        std::sync::Arc::new(|_, _| Box::pin(async { unreachable!() }));
    let result = test_embedding_connection(&settings, &credentials, &managed, None)
        .await
        .unwrap();
    assert_eq!(result.actual_dims, 0);
}

#[tokio::test]
async fn public_custom_verification_uses_native_width_and_skips_models_on_success() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let mut request = Vec::new();
        let mut chunk = [0; 1024];
        loop {
            let read = stream.read(&mut chunk).await.unwrap();
            request.extend_from_slice(&chunk[..read]);
            if request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                break;
            }
        }
        assert!(String::from_utf8_lossy(&request).starts_with("POST /v1/embeddings "));
        let body = r#"{"data":[{"embedding":[0.1,0.2]}]}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        stream.write_all(response.as_bytes()).await.unwrap();
        tokio::select! {
            biased;
            unexpected = listener.accept() => { drop(unexpected); true }
            _ = done_rx => false,
        }
    });
    let verified = verify_custom_endpoint(&endpoint, "test-key", "nomic-embed-text", 99)
        .await
        .unwrap();
    done_tx.send(()).unwrap();
    assert!(
        !server.await.unwrap(),
        "successful verification must not request /models"
    );
    assert_eq!(verified.dimensions, 2);
    assert_eq!(
        verified.signature,
        format!("provider=custom:{endpoint};model=nomic-embed-text;dims=2")
    );
    assert!(!verified.signature_changed(Some(&verified.signature)));
    assert!(verified.signature_changed(Some("previous-signature")));
}

#[tokio::test]
async fn public_custom_verification_prefers_positive_model_guidance_after_rejection() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let server = tokio::spawn(respond_to_requests(
        listener,
        vec![
            ("POST /v1/embeddings ", 400, r#"{"error":"failed"}"#),
            (
                "GET /v1/models ",
                200,
                r#"{"data":[{"id":"served-model"}]}"#,
            ),
        ],
    ));
    let result = verify_custom_endpoint(&endpoint, "test-key", "missing-model", 3).await;
    server.await.unwrap();
    assert!(
        matches!(result, Err(VerifyRejection::ModelNotServed(value)) if value.available_models == ["served-model"])
    );
}

#[tokio::test]
async fn public_custom_verification_keeps_generic_rejection_when_models_fail() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let server = tokio::spawn(respond_to_requests(
        listener,
        vec![
            ("POST /v1/embeddings ", 400, r#"{"error":"failed"}"#),
            ("GET /v1/models ", 503, "{}"),
        ],
    ));
    let result = verify_custom_endpoint(&endpoint, "test-key", "missing-model", 3).await;
    server.await.unwrap();
    assert!(
        matches!(result, Err(VerifyRejection::Probe(value)) if value.error == "EMBEDDINGS_VERIFICATION_FAILED")
    );
}

#[tokio::test]
async fn public_custom_verification_keeps_generic_rejection_for_empty_model_list() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let server = tokio::spawn(respond_to_requests(
        listener,
        vec![
            ("POST /v1/embeddings ", 400, r#"{"error":"failed"}"#),
            ("GET /v1/models ", 200, r#"{"data":[]}"#),
        ],
    ));
    let result = verify_custom_endpoint(&endpoint, "test-key", "missing-model", 3).await;
    server.await.unwrap();
    assert!(
        matches!(result, Err(VerifyRejection::Probe(value)) if value.error == "EMBEDDINGS_VERIFICATION_FAILED")
    );
}

#[tokio::test]
async fn custom_connection_infers_stored_endpoint_and_resolves_custom_credential_slug() {
    use std::sync::{Arc, Mutex};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let server = tokio::spawn(respond_to_requests(
        listener,
        vec![(
            "POST /v1/embeddings ",
            200,
            r#"{"data":[{"embedding":[0.1,0.2]}]}"#,
        )],
    ));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let capture = seen.clone();
    let settings = EmbeddingFactorySettings {
        provider: format!("custom:{endpoint}"),
        model: "configured-model".into(),
        dimensions: 4,
        ollama_base_url: String::new(),
    };
    let credentials: CredentialResolver = Arc::new(move |slug| {
        capture.lock().unwrap().push(slug.to_owned());
        "key".into()
    });
    let managed: ManagedModelFactory = Arc::new(|_, _| Box::pin(async { unreachable!() }));
    let result = test_embedding_connection(&settings, &credentials, &managed, None)
        .await
        .unwrap();
    server.await.unwrap();
    assert_eq!(result.actual_dims, 2);
    assert_eq!(*seen.lock().unwrap(), ["custom"]);
}

#[tokio::test]
async fn timed_out_probe_still_uses_positive_models_guidance() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(respond_to_requests(
        listener,
        vec![("GET /v1/models ", 200, r#"{"data":[{"id":"other-model"}]}"#)],
    ));
    let result = verify_custom_endpoint_with_probe(
        &format!("http://{address}/v1"),
        "key",
        "requested-model",
        3,
        Duration::ZERO,
        std::future::pending(),
    )
    .await;
    server.await.unwrap();
    assert!(
        matches!(result, Err(VerifyRejection::ModelNotServed(value)) if value.requested_model == "requested-model")
    );
}

#[test]
fn signature_comparison_requires_and_uses_explicit_old_state() {
    let result = VerifiedDims {
        dimensions: 7,
        signature: "provider=custom:https://host/v1;model=m;dims=7".into(),
    };
    assert!(!result.signature_changed(None));
    assert!(!result.signature_changed(Some(&result.signature)));
    assert!(result.signature_changed(Some("old")));
}
