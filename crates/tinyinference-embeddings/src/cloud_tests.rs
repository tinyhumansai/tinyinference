use super::*;

fn missing_bearer() -> BearerResolver {
    Arc::new(|| {
        Box::pin(async {
            Err(Error::Validation(
                "No backend session for cloud embeddings".into(),
            ))
        })
    })
}

#[test]
fn identity_matches_host_contract() {
    let model = CloudEmbeddingModel::new(
        "https://api.example/openai/v1/",
        DEFAULT_CLOUD_MODEL,
        DEFAULT_CLOUD_DIMENSIONS,
        missing_bearer(),
    );
    assert_eq!(model.name(), "cloud");
    assert_eq!(
        model.signature(),
        "provider=cloud;model=embedding-v1;dims=1024"
    );
}

#[tokio::test]
async fn validation_precedes_bearer_resolution() {
    let model = CloudEmbeddingModel::new(
        "https://api.example/openai/v1",
        DEFAULT_CLOUD_MODEL,
        DEFAULT_CLOUD_DIMENSIONS,
        missing_bearer(),
    );
    assert!(model.embed(&[]).await.unwrap().is_empty());
    let error = model.embed(&[" ".into()]).await.unwrap_err();
    assert!(error.to_string().contains("empty/whitespace"));
}

#[tokio::test]
async fn empty_and_blank_batches_skip_egress_guard_and_bearer_for_both_entrypoints() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let guard_calls = Arc::new(AtomicUsize::new(0));
    let bearer_calls = Arc::new(AtomicUsize::new(0));
    let guard_count = guard_calls.clone();
    let bearer_count = bearer_calls.clone();
    let model = CloudEmbeddingModel::new(
        "https://api.example",
        "embedding-v1",
        8,
        Arc::new(move || {
            bearer_count.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok("token".into()) })
        }),
    )
    .with_egress_guard(Arc::new(move |_, _| {
        guard_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }));
    assert!(model.embed(&[]).await.unwrap().is_empty());
    assert!(model.embed_with_usage(&[]).await.unwrap().0.is_empty());
    assert!(model.embed(&["  ".into()]).await.is_err());
    assert!(model.embed_with_usage(&["\t".into()]).await.is_err());
    assert_eq!(guard_calls.load(Ordering::SeqCst), 0);
    assert_eq!(bearer_calls.load(Ordering::SeqCst), 0);
}
