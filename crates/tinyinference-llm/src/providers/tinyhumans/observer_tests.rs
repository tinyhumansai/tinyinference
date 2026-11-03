use super::*;
use crate::message::Message;
use crate::model::ModelRequest;
use std::sync::atomic::AtomicUsize;

#[tokio::test]
async fn observer_reports_unary_provider_failure_once_and_preserves_success() {
    let fixture = Fixture::default();
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        content_type: "application/json",
        body: json!({ "error": { "message": "server marker" } }).to_string(),
    };
    let base = serve(fixture.clone()).await;
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_from_callback = Arc::clone(&observed);
    let model = fixture_model(&base).with_error_observer(move |error, point| {
        let fields = match error {
            Error::Provider(provider) => (provider.status, provider.message.clone()),
            other => (None, other.to_string()),
        };
        observed_from_callback.lock().unwrap().push((point, fields));
    });

    let error = match model
        .invoke(&(), ModelRequest::new(vec![Message::user("fail")]))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("fixture is a provider failure"),
    };
    match error {
        Error::Provider(provider) => {
            assert_eq!(provider.status, Some(500));
            assert_eq!(provider.message, "server marker");
        }
        other => panic!("expected provider error, got {other}"),
    }
    assert_eq!(
        *observed.lock().unwrap(),
        vec![(
            TinyHumansFailurePoint::Invoke,
            (Some(500), "server marker".to_owned()),
        )]
    );

    *fixture.response.lock().unwrap() = FixtureResponse::default();
    model
        .invoke(&(), ModelRequest::new(vec![Message::user("succeed")]))
        .await
        .expect("fixture success");
    assert_eq!(observed.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn observer_reports_stream_open_failure_without_changing_auth_callbacks() {
    let fixture = Fixture::default();
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::FORBIDDEN,
        content_type: "application/json",
        body: json!({ "error": { "message": "forbidden marker" } }).to_string(),
    };
    let base = serve(fixture).await;
    let observations = Arc::new(AtomicUsize::new(0));
    let observer_count = Arc::clone(&observations);
    let legacy = Arc::new(AtomicUsize::new(0));
    let legacy_count = Arc::clone(&legacy);
    let managed = Arc::new(AtomicUsize::new(0));
    let managed_count = Arc::clone(&managed);
    let model = fixture_model(&base)
        .with_error_observer(move |error, point| {
            assert_eq!(point, TinyHumansFailurePoint::StreamOpen);
            assert!(matches!(error, Error::Provider(provider) if provider.status == Some(403)));
            observer_count.fetch_add(1, Ordering::SeqCst);
        })
        .with_auth_failure_handler(move |_, operation| {
            assert_eq!(operation, TinyHumansOperation::Stream);
            legacy_count.fetch_add(1, Ordering::SeqCst);
        })
        .with_managed_auth_failure_handler(move |_, failure| {
            assert_eq!(failure.operation(), TinyHumansOperation::Stream);
            assert_eq!(failure.kind(), TinyHumansAuthFailureKind::Forbidden);
            managed_count.fetch_add(1, Ordering::SeqCst);
        });

    let error = match model
        .stream(&(), ModelRequest::new(vec![Message::user("stream")]))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("fixture rejects stream opening"),
    };
    assert!(matches!(error, Error::Provider(provider) if provider.status == Some(403)));
    assert_eq!(observations.load(Ordering::SeqCst), 1);
    assert_eq!(legacy.load(Ordering::SeqCst), 1);
    assert_eq!(managed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn observer_reports_in_band_failure_once_and_keeps_the_stream_item() {
    let fixture = Fixture::default();
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::OK,
        content_type: "text/event-stream",
        body: "data: {\"error\":{\"message\":\"stream marker\"}}\n\n".into(),
    };
    let base = serve(fixture.clone()).await;
    let observed = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed_from_callback = Arc::clone(&observed);
    let legacy = Arc::new(AtomicUsize::new(0));
    let legacy_count = Arc::clone(&legacy);
    let managed = Arc::new(AtomicUsize::new(0));
    let managed_count = Arc::clone(&managed);
    let model = fixture_model(&base)
        .with_error_observer(move |error, point| {
            let fields = match error {
                Error::Provider(provider) => (provider.status, provider.message.clone()),
                _ => panic!("expected normalized provider error"),
            };
            observed_from_callback.lock().unwrap().push((point, fields));
        })
        .with_auth_failure_handler(move |_, _| {
            legacy_count.fetch_add(1, Ordering::SeqCst);
        })
        .with_managed_auth_failure_handler(move |error, failure| {
            assert_eq!(error.message, "Unauthorized in-band marker");
            assert_eq!(failure.operation(), TinyHumansOperation::Stream);
            assert_eq!(failure.kind(), TinyHumansAuthFailureKind::Other);
            managed_count.fetch_add(1, Ordering::SeqCst);
        });

    let items = model
        .stream(&(), ModelRequest::new(vec![Message::user("stream")]))
        .await
        .expect("HTTP stream opens")
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(items.last(), Some(ModelStreamItem::ProviderFailed(error)) if error.message == "stream marker")
    );
    assert_eq!(
        *observed.lock().unwrap(),
        vec![(
            TinyHumansFailurePoint::StreamItem,
            (None, "stream marker".to_owned()),
        )]
    );
    assert_eq!(
        legacy.load(Ordering::SeqCst),
        0,
        "non-auth failure is not demoted"
    );

    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::OK,
        content_type: "text/event-stream",
        body: "data: {\"error\":{\"message\":\"Unauthorized in-band marker\",\"code\":\"unauthorized\"}}\n\n".into(),
    };
    let items = model
        .stream(&(), ModelRequest::new(vec![Message::user("auth stream")]))
        .await
        .expect("auth stream opens")
        .collect::<Vec<_>>()
        .await;
    assert!(
        matches!(items.last(), Some(ModelStreamItem::ProviderFailed(error)) if error.message == "Unauthorized in-band marker")
    );
    assert_eq!(
        *observed.lock().unwrap(),
        vec![
            (
                TinyHumansFailurePoint::StreamItem,
                (None, "stream marker".to_owned()),
            ),
            (
                TinyHumansFailurePoint::StreamItem,
                (None, "Unauthorized in-band marker".to_owned()),
            ),
        ]
    );
    assert_eq!(legacy.load(Ordering::SeqCst), 1);
    assert_eq!(managed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn observer_classifies_readiness_and_non_provider_failures() {
    let fixture = Fixture::default();
    *fixture.response.lock().unwrap() = FixtureResponse {
        status: StatusCode::INTERNAL_SERVER_ERROR,
        content_type: "application/json",
        body: json!({ "error": { "message": "readiness marker" } }).to_string(),
    };
    let base = serve(fixture).await;
    let readiness = Arc::new(std::sync::Mutex::new(Vec::new()));
    let readiness_seen = Arc::clone(&readiness);
    fixture_model(&base)
        .with_error_observer(move |error, point| {
            readiness_seen
                .lock()
                .unwrap()
                .push((point, error.to_string()));
        })
        .probe_readiness()
        .await
        .expect("readiness fails open for a provider error");
    assert_eq!(readiness.lock().unwrap().len(), 1);
    assert_eq!(
        readiness.lock().unwrap()[0].0,
        TinyHumansFailurePoint::Readiness
    );
    assert!(readiness.lock().unwrap()[0].1.contains("readiness marker"));

    let resolver_calls = Arc::new(AtomicUsize::new(0));
    let calls = Arc::clone(&resolver_calls);
    let errors = Arc::new(std::sync::Mutex::new(Vec::new()));
    let errors_seen = Arc::clone(&errors);
    let model = TinyHumansModel::new("http://127.0.0.1:9", "model", "default-model", move || {
        let calls = Arc::clone(&calls);
        Box::pin(async move {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(Error::Model("resolver marker".into()))
        })
    })
    .unwrap()
    .with_error_observer(move |error, point| {
        errors_seen.lock().unwrap().push((point, error.to_string()));
    });
    let error = match model
        .invoke(&(), ModelRequest::new(vec![Message::user("invoke")]))
        .await
    {
        Err(error) => error,
        Ok(_) => panic!("resolver failure"),
    };
    assert!(matches!(error, Error::Model(_)));
    assert_eq!(resolver_calls.load(Ordering::SeqCst), 1);
    assert_eq!(errors.lock().unwrap().len(), 1);
    assert_eq!(errors.lock().unwrap()[0].0, TinyHumansFailurePoint::Invoke);
    assert!(errors.lock().unwrap()[0].1.contains("resolver marker"));
}
