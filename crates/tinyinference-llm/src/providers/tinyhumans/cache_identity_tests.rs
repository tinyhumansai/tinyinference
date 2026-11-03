use super::*;
use crate::model::ChatModel;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn cache_namespace_is_trimmed_and_blank_input_keeps_the_current_value() {
    let base = "https://api.example.test";
    let default = fixture_model(base);
    assert_eq!(
        default.cache_identity().as_deref(),
        Some("tinyhumans:https://api.example.test/openai/v1:default-model")
    );
    assert_eq!(
        default
            .with_cache_namespace("  ")
            .cache_identity()
            .as_deref(),
        Some("tinyhumans:https://api.example.test/openai/v1:default-model")
    );

    let custom = fixture_model(base).with_cache_namespace("  openhuman  ");
    assert_eq!(
        custom.cache_identity().as_deref(),
        Some("openhuman:https://api.example.test/openai/v1:default-model")
    );
    assert_eq!(
        custom
            .with_cache_namespace("\t ")
            .cache_identity()
            .as_deref(),
        Some("openhuman:https://api.example.test/openai/v1:default-model")
    );
}

#[test]
fn cache_identity_keeps_endpoint_and_default_model_dimensions() {
    let first = TinyHumansModel::new("https://one.example.test", "", "model-a", || {
        Box::pin(async { Ok("credential-a".into()) })
    })
    .unwrap()
    .with_cache_namespace("openhuman");
    let other_endpoint = TinyHumansModel::new("https://two.example.test", "", "model-a", || {
        Box::pin(async { Ok("credential-b".into()) })
    })
    .unwrap()
    .with_cache_namespace("openhuman");
    let other_model = TinyHumansModel::new("https://one.example.test", "", "model-b", || {
        Box::pin(async { Ok("credential-c".into()) })
    })
    .unwrap()
    .with_cache_namespace("openhuman");

    assert_eq!(
        first.cache_identity().as_deref(),
        Some("openhuman:https://one.example.test/openai/v1:model-a")
    );
    assert_ne!(first.cache_identity(), other_endpoint.cache_identity());
    assert_ne!(first.cache_identity(), other_model.cache_identity());
}

#[test]
fn reading_cache_identity_never_resolves_or_includes_credentials() {
    let reads = Arc::new(AtomicUsize::new(0));
    let resolver_reads = Arc::clone(&reads);
    let model = TinyHumansModel::new("https://api.example.test", "", "default-model", move || {
        let resolver_reads = Arc::clone(&resolver_reads);
        Box::pin(async move {
            resolver_reads.fetch_add(1, Ordering::SeqCst);
            Ok("must-not-appear-in-cache-identity".into())
        })
    })
    .unwrap()
    .with_cache_namespace("openhuman");

    let identity = model.cache_identity().expect("stable managed identity");
    assert_eq!(
        identity,
        "openhuman:https://api.example.test/openai/v1:default-model"
    );
    assert!(!identity.contains("must-not-appear-in-cache-identity"));
    assert_eq!(reads.load(Ordering::SeqCst), 0);
}
