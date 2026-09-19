//! Unit tests for the embeddings + retrieval module.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde_json::json;

use super::*;

struct ShortEmbeddingModel;

struct WrongDimensionsEmbeddingModel;

struct DiscoveringEmbeddingModel {
    dimensions: AtomicUsize,
}

#[async_trait::async_trait]
impl EmbeddingModel for DiscoveringEmbeddingModel {
    fn name(&self) -> &str {
        "discovering"
    }

    fn model_id(&self) -> &str {
        "discovering"
    }

    async fn embed(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        self.dimensions.store(2, Ordering::Release);
        Ok(vec![vec![1.0, 0.0]; texts.len()])
    }

    fn dimensions(&self) -> usize {
        self.dimensions.load(Ordering::Acquire)
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for ShortEmbeddingModel {
    fn name(&self) -> &str {
        "short"
    }

    fn model_id(&self) -> &str {
        "short"
    }

    async fn embed(&self, _texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![1.0, 0.0]])
    }

    fn dimensions(&self) -> usize {
        2
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for WrongDimensionsEmbeddingModel {
    fn name(&self) -> &str {
        "wrong-dimensions"
    }

    fn model_id(&self) -> &str {
        "wrong-dimensions"
    }

    async fn embed(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![1.0; 3]; texts.len()])
    }

    fn dimensions(&self) -> usize {
        2
    }
}

struct UsageEmbeddingModel;

struct FailingCancelledEmbeddingModel {
    cancellation: EmbeddingCancellation,
}

struct SlowEmbeddingModel {
    started: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    dropped: Arc<AtomicBool>,
}

struct InFlightEmbedding(Arc<AtomicBool>);

impl Drop for InFlightEmbedding {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for SlowEmbeddingModel {
    fn name(&self) -> &str {
        "slow"
    }

    fn model_id(&self) -> &str {
        "slow"
    }

    async fn embed(&self, _texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        let _in_flight = InFlightEmbedding(self.dropped.clone());
        self.started
            .lock()
            .unwrap()
            .take()
            .expect("one slow embedding invocation")
            .send(())
            .expect("test waits for the invocation to start");
        std::future::pending::<crate::Result<Vec<Vec<f32>>>>().await
    }

    fn dimensions(&self) -> usize {
        2
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for UsageEmbeddingModel {
    fn name(&self) -> &str {
        "usage"
    }

    fn model_id(&self) -> &str {
        "usage"
    }

    async fn embed(&self, texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        Ok(vec![vec![1.0, 0.0]; texts.len()])
    }

    async fn embed_with_usage(
        &self,
        texts: &[String],
    ) -> crate::Result<(Vec<Vec<f32>>, Option<EmbeddingUsage>)> {
        Ok((
            vec![vec![1.0, 0.0]; texts.len()],
            Some(EmbeddingUsage::new(17)),
        ))
    }

    fn dimensions(&self) -> usize {
        2
    }
}

#[async_trait::async_trait]
impl EmbeddingModel for FailingCancelledEmbeddingModel {
    fn name(&self) -> &str {
        "failing-cancelled"
    }

    fn model_id(&self) -> &str {
        "failing-cancelled"
    }

    async fn embed(&self, _texts: &[String]) -> crate::Result<Vec<Vec<f32>>> {
        self.cancellation.cancel();
        Err(crate::Error::Embedding(
            "provider failed after cancellation".to_string(),
        ))
    }

    fn dimensions(&self) -> usize {
        2
    }
}

#[tokio::test]
async fn embedding_request_validates_order_dimensions_usage_and_cancellation() {
    let model = UsageEmbeddingModel;
    let response = model
        .embed_request(EmbeddingRequest::new(vec![
            "first".to_string(),
            "second".to_string(),
        ]))
        .await
        .unwrap();
    assert_eq!(response.vectors, vec![vec![1.0, 0.0], vec![1.0, 0.0]]);
    assert_eq!(response.dimensions, 2);
    assert_eq!(response.usage, Some(EmbeddingUsage::new(17)));

    let cancellation = EmbeddingCancellation::new();
    cancellation.cancel();
    let error = model
        .embed_request(
            EmbeddingRequest::new(vec!["never dispatched".to_string()])
                .with_cancellation(cancellation),
        )
        .await
        .unwrap_err();
    assert!(matches!(error, crate::Error::Cancelled));
}

#[tokio::test]
async fn embedding_request_rejects_wrong_batch_count_and_dimensions() {
    let inputs = vec!["a".to_string(), "b".to_string()];
    let count_error = ShortEmbeddingModel
        .embed_request(EmbeddingRequest::new(inputs.clone()))
        .await
        .unwrap_err();
    assert!(matches!(count_error, crate::Error::Validation(_)));

    let dimension_error = WrongDimensionsEmbeddingModel
        .embed_request(EmbeddingRequest::new(inputs))
        .await
        .unwrap_err();
    assert!(matches!(dimension_error, crate::Error::Validation(_)));
}

#[tokio::test]
async fn embedding_request_cancellation_drops_an_in_flight_provider_operation() {
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let dropped = Arc::new(AtomicBool::new(false));
    let model = Arc::new(SlowEmbeddingModel {
        started: Mutex::new(Some(started_tx)),
        dropped: dropped.clone(),
    });
    let cancellation = EmbeddingCancellation::new();
    let request =
        EmbeddingRequest::new(vec!["slow".to_string()]).with_cancellation(cancellation.clone());
    let task = tokio::spawn({
        let model = model.clone();
        async move { model.embed_request(request).await }
    });

    started_rx.await.expect("provider operation starts");
    cancellation.cancel();
    let error = task.await.unwrap().unwrap_err();
    assert!(matches!(error, crate::Error::Cancelled));
    assert!(dropped.load(Ordering::Acquire));
}

#[tokio::test]
async fn embedding_request_prefers_cancellation_when_provider_fails_after_cancelling() {
    let cancellation = EmbeddingCancellation::new();
    let model = FailingCancelledEmbeddingModel {
        cancellation: cancellation.clone(),
    };

    let error = model
        .embed_request(
            EmbeddingRequest::new(vec!["input".to_string()]).with_cancellation(cancellation),
        )
        .await
        .unwrap_err();

    assert!(matches!(error, crate::Error::Cancelled));
}

#[test]
fn cosine_similarity_identical_is_one() {
    assert_eq!(cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]), 1.0);
    assert_eq!(cosine_similarity(&[3.0, 4.0], &[6.0, 8.0]), 1.0);
}

#[test]
fn cosine_similarity_orthogonal_is_zero() {
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]), 0.0);
}

#[test]
fn cosine_similarity_opposite_is_negative_one() {
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]), -1.0);
}

#[test]
fn cosine_similarity_known_value() {
    // 45 degrees between (1,0) and (1,1) -> cos = 1/sqrt(2).
    let s = cosine_similarity(&[1.0, 0.0], &[1.0, 1.0]);
    assert!((s - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
}

#[test]
fn cosine_similarity_degenerate_inputs_return_zero() {
    assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[1.0], &[1.0, 0.0]), 0.0);
    assert_eq!(cosine_similarity(&[], &[]), 0.0);
}

#[tokio::test]
async fn mock_model_is_deterministic_and_correct_shape() {
    let model = MockEmbeddingModel::new(24);
    assert_eq!(model.dimensions(), 24);
    let a = model.embed(&["hello world".to_string()]).await.unwrap();
    let b = model.embed(&["hello world".to_string()]).await.unwrap();
    assert_eq!(a, b, "identical text must yield identical vectors");
    assert_eq!(a[0].len(), 24);

    // A text embedded against itself has cosine similarity 1.0.
    assert!((cosine_similarity(&a[0], &b[0]) - 1.0).abs() < 1e-6);

    // Different texts produce different vectors.
    let c = model
        .embed(&["totally different".to_string()])
        .await
        .unwrap();
    assert_ne!(a[0], c[0]);
}

#[tokio::test]
async fn mock_model_batches_in_order() {
    let model = MockEmbeddingModel::new(8);
    let texts = vec!["one".to_string(), "two".to_string(), "three".to_string()];
    let vectors = model.embed(&texts).await.unwrap();
    assert_eq!(vectors.len(), 3);
    // Each batched vector matches the individually-embedded vector.
    for (text, v) in texts.iter().zip(vectors.iter()) {
        let single = model.embed(std::slice::from_ref(text)).await.unwrap();
        assert_eq!(&single[0], v);
    }
}

#[tokio::test]
async fn mock_model_empty_input_returns_empty() {
    let model = MockEmbeddingModel::new(8);
    let vectors = model.embed(&[]).await.unwrap();
    assert!(vectors.is_empty());
}

#[tokio::test]
async fn vector_store_ranks_by_cosine_similarity() {
    let store = InMemoryVectorStore::new();
    assert!(store.is_empty());
    store
        .add("a".into(), vec![1.0, 0.0], json!({"k": "a"}))
        .await
        .unwrap();
    store
        .add("b".into(), vec![0.0, 1.0], json!({"k": "b"}))
        .await
        .unwrap();
    store
        .add("c".into(), vec![0.9, 0.1], json!({"k": "c"}))
        .await
        .unwrap();
    assert_eq!(store.len(), 3);

    let hits = store.query(&[1.0, 0.0], 2).await.unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].id, "a");
    assert_eq!(hits[1].id, "c");
    assert!(hits[0].score >= hits[1].score);
    assert_eq!(hits[0].metadata, json!({"k": "a"}));
}

#[tokio::test]
async fn vector_store_top_k_zero_and_overflow() {
    let store = InMemoryVectorStore::new();
    store
        .add("a".into(), vec![1.0, 0.0], json!({}))
        .await
        .unwrap();
    assert!(store.query(&[1.0, 0.0], 0).await.unwrap().is_empty());
    // Requesting more than stored returns all entries.
    assert_eq!(store.query(&[1.0, 0.0], 10).await.unwrap().len(), 1);
}

#[tokio::test]
async fn vector_store_add_replaces_existing_id() {
    let store = InMemoryVectorStore::new();
    store
        .add("x".into(), vec![1.0, 0.0], json!({"v": 1}))
        .await
        .unwrap();
    store
        .add("x".into(), vec![0.0, 1.0], json!({"v": 2}))
        .await
        .unwrap();
    assert_eq!(store.len(), 1);
    let hits = store.query(&[0.0, 1.0], 1).await.unwrap();
    assert_eq!(hits[0].id, "x");
    assert_eq!(hits[0].metadata, json!({"v": 2}));
}

#[tokio::test]
async fn vector_store_rejects_mismatched_query_dimension() {
    let store = InMemoryVectorStore::new();
    store
        .add("a".into(), vec![1.0, 0.0], json!({}))
        .await
        .unwrap();

    let err = store.query(&[1.0, 0.0, 0.0], 1).await.unwrap_err();
    assert!(matches!(err, crate::Error::Validation(_)), "{err:?}");
    assert!(err.to_string().contains("dimensions"), "{err}");
}

#[tokio::test]
async fn vector_store_rejects_mismatched_or_empty_add() {
    let store = InMemoryVectorStore::new();
    // Zero-dimensional vectors are rejected outright.
    let err = store.add("z".into(), vec![], json!({})).await.unwrap_err();
    assert!(matches!(err, crate::Error::Validation(_)), "{err:?}");

    store
        .add("a".into(), vec![1.0, 0.0], json!({}))
        .await
        .unwrap();
    // The first vector fixes the store's dimensionality.
    let err = store
        .add("b".into(), vec![1.0, 0.0, 0.0], json!({}))
        .await
        .unwrap_err();
    assert!(matches!(err, crate::Error::Validation(_)), "{err:?}");
    assert_eq!(store.len(), 1, "rejected vectors must not be stored");
}

#[tokio::test]
async fn vector_store_empty_store_accepts_any_query_dimension() {
    let store = InMemoryVectorStore::new();
    // No stored dimensionality to compare against: any query returns no hits.
    assert!(store.query(&[1.0, 2.0, 3.0], 5).await.unwrap().is_empty());
    assert!(store.query(&[], 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn retriever_rejects_query_of_wrong_dimension() {
    // Index with a 8-dim model, then retrieve with a 4-dim model over the same
    // store: the mismatch must surface as a Validation error, not zero-score
    // arbitrary hits.
    let store = Arc::new(InMemoryVectorStore::new());
    let indexer = Retriever::new(Arc::new(MockEmbeddingModel::new(8)), store.clone());
    indexer
        .index(vec![("doc".into(), "some text".into(), json!({}))])
        .await
        .unwrap();

    let querier = Retriever::new(Arc::new(MockEmbeddingModel::new(4)), store);
    let err = querier.retrieve("some text", 1).await.unwrap_err();
    assert!(matches!(err, crate::Error::Validation(_)), "{err:?}");
}

#[tokio::test]
async fn retriever_index_and_retrieve_most_similar_first() {
    let retriever = Retriever::new(
        Arc::new(MockEmbeddingModel::new(64)),
        Arc::new(InMemoryVectorStore::new()),
    );
    retriever
        .index(vec![
            (
                "cats".into(),
                "cats are great pets".into(),
                json!({"topic": "animals"}),
            ),
            (
                "dogs".into(),
                "dogs are loyal companions".into(),
                json!({"topic": "animals"}),
            ),
            (
                "finance".into(),
                "the stock market crashed today".into(),
                json!({"topic": "finance"}),
            ),
        ])
        .await
        .unwrap();

    // Querying with the exact text of an indexed doc ranks it first (cosine 1.0).
    let hits = retriever.retrieve("cats are great pets", 3).await.unwrap();
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0].id, "cats");
    assert!((hits[0].score - 1.0).abs() < 1e-6);
    assert_eq!(hits[0].metadata, json!({"topic": "animals"}));
}

#[tokio::test]
async fn retriever_discovers_dimensions_during_first_index() {
    let model = Arc::new(DiscoveringEmbeddingModel {
        dimensions: AtomicUsize::new(0),
    });
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = Retriever::new(model.clone(), store.clone());

    retriever
        .index(vec![("doc".into(), "text".into(), json!({}))])
        .await
        .unwrap();

    assert_eq!(model.dimensions(), 2);
    assert_eq!(store.len(), 1);
    assert_eq!(retriever.retrieve("text", 1).await.unwrap().len(), 1);
}

#[tokio::test]
async fn retriever_empty_index_is_noop() {
    let retriever = Retriever::new(
        Arc::new(MockEmbeddingModel::new(8)),
        Arc::new(InMemoryVectorStore::new()),
    );
    retriever.index(vec![]).await.unwrap();
    assert!(retriever.retrieve("anything", 5).await.unwrap().is_empty());
}

#[tokio::test]
async fn retriever_rejects_short_batches_before_writing() {
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = Retriever::new(Arc::new(ShortEmbeddingModel), store.clone());
    let error = retriever
        .index(vec![
            ("one".into(), "one".into(), json!({})),
            ("two".into(), "two".into(), json!({})),
        ])
        .await
        .unwrap_err();
    assert!(error.to_string().contains("1 vectors for 2 documents"));
    assert!(
        store.is_empty(),
        "partial batches must not mutate the store"
    );
}

#[tokio::test]
async fn noop_retriever_disables_semantic_index_and_query() {
    let store = Arc::new(InMemoryVectorStore::new());
    let retriever = Retriever::new(Arc::new(NoopEmbeddingModel), store.clone());
    retriever
        .index(vec![("one".into(), "one".into(), json!({}))])
        .await
        .unwrap();
    assert!(store.is_empty());
    assert!(retriever.retrieve("one", 1).await.unwrap().is_empty());
}

#[test]
fn embedding_debug_output_redacts_credentials() {
    for output in [
        format!("{:?}", OpenAiEmbeddingModel::new("openai-secret")),
        format!("{:?}", CohereEmbeddingModel::new("cohere-secret")),
        format!("{:?}", VoyageEmbeddingModel::new("voyage-secret")),
    ] {
        assert!(output.contains("[REDACTED]"), "{output}");
        assert!(!output.contains("secret"), "{output}");
    }
}

#[test]
fn openai_embeddings_follow_response_indices() {
    let value = json!({
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]}
        ]
    });
    let vectors = super::openai::parse_vectors(&value, 2, 2).unwrap();
    assert_eq!(vectors, vec![vec![1.0, 0.0], vec![0.0, 1.0]]);

    let duplicate = json!({
        "data": [
            {"index": 0, "embedding": [1.0, 0.0]},
            {"index": 0, "embedding": [0.0, 1.0]}
        ]
    });
    assert!(super::openai::parse_vectors(&duplicate, 2, 2).is_err());
}

#[tokio::test]
async fn retriever_accessors_expose_collaborators() {
    let retriever = Retriever::new(
        Arc::new(MockEmbeddingModel::new(8)),
        Arc::new(InMemoryVectorStore::new()),
    );
    assert_eq!(retriever.model().dimensions(), 8);
    let _ = retriever.store();
}
#[test]
fn embedding_identity_signature_is_stable() {
    let model = MockEmbeddingModel::new(8);
    assert_eq!(model.name(), "mock");
    assert_eq!(model.model_id(), "deterministic-hash");
    assert_eq!(
        model.signature(),
        "provider=mock;model=deterministic-hash;dims=8"
    );
    assert_eq!(
        format_embedding_signature("openai", "text-embedding-3-small", 1536),
        "provider=openai;model=text-embedding-3-small;dims=1536"
    );
}

#[test]
fn known_ollama_widths_are_resolved_without_guessing_unknown_models() {
    assert_eq!(known_ollama_embedding_dimensions("bge-m3"), Some(1024));
    assert_eq!(
        known_ollama_embedding_dimensions("all-minilm:latest"),
        Some(384)
    );
    assert_eq!(
        known_ollama_embedding_dimensions("nomic-embed-text"),
        Some(768)
    );
    assert_eq!(
        known_ollama_embedding_dimensions("user-managed-model"),
        None
    );
}

#[test]
fn mmr_preserves_negative_similarity_as_diversity() {
    let anchor = [1.0, 0.0];
    let orthogonal = [0.0, 1.0];
    let anti_correlated = [-1.0, 0.0];
    let candidates = [
        MmrCandidate {
            index: 0,
            embedding: &anchor,
            relevance: 1.0,
        },
        MmrCandidate {
            index: 1,
            embedding: &orthogonal,
            relevance: 0.5,
        },
        MmrCandidate {
            index: 2,
            embedding: &anti_correlated,
            relevance: 0.5,
        },
    ];
    let picked = mmr_select(&candidates, 2, 0.5);
    assert_eq!(
        picked.iter().map(|item| item.index).collect::<Vec<_>>(),
        [0, 2]
    );
}

#[test]
fn incremental_mean_replaces_an_incompatible_centroid() {
    assert_eq!(incremental_mean_embedding(&[], &[1.0, 2.0], 0), [1.0, 2.0]);
    assert_eq!(
        incremental_mean_embedding(&[1.0], &[1.0, 2.0], 3),
        [1.0, 2.0]
    );
    assert_eq!(
        incremental_mean_embedding(&[0.0, 0.0], &[1.0, 1.0], 1),
        [0.5, 0.5]
    );
}

#[test]
fn embedding_token_estimate_rounds_up_across_the_batch() {
    assert_eq!(estimate_embedding_input_tokens(&[]), 0);
    assert_eq!(estimate_embedding_input_tokens(&["hello".into()]), 2);
    assert_eq!(
        estimate_embedding_input_tokens(&["abc".into(), "defgh".into()]),
        2
    );
}

#[tokio::test]
async fn voyage_and_noop_identity_match_host_contract() {
    let voyage = VoyageEmbeddingModel::new("test-key");
    assert_eq!(voyage.name(), "voyage");
    assert_eq!(voyage.model_id(), VOYAGE_DEFAULT_MODEL);
    assert_eq!(
        voyage.signature(),
        "provider=voyage;model=voyage-3-large;dims=1024"
    );

    let noop = NoopEmbeddingModel;
    assert_eq!(noop.signature(), "provider=none;model=none;dims=0");
    assert_eq!(
        noop.embed(&["first".into(), "second".into()])
            .await
            .unwrap(),
        vec![Vec::<f32>::new(), Vec::<f32>::new()]
    );
}

#[test]
fn gemini_openai_compatible_model_id_is_not_rewritten() {
    let model = OpenAiEmbeddingModel::new("test-key")
        .with_base_url("https://generativelanguage.googleapis.com/v1beta/openai")
        .with_model("gemini-embedding-001");
    assert_eq!(model.model(), "gemini-embedding-001");
}

#[test]
fn openai_zero_dimensions_accepts_provider_vector_length() {
    let value = json!({
        "data": [{"index": 0, "embedding": [1.0, 0.0, 0.5]}]
    });
    let vectors = super::openai::parse_vectors(&value, 1, 0).unwrap();
    assert_eq!(vectors, vec![vec![1.0, 0.0, 0.5]]);
}

#[test]
fn openai_dimension_discovery_updates_the_trait_contract() {
    let model = OpenAiEmbeddingModel::new("test-key").with_dimensions(0);
    model
        .adopt_discovered_dimensions(&[vec![1.0, 0.0, 0.5]])
        .unwrap();
    assert_eq!(model.dimensions(), 3);
    assert!(
        model
            .adopt_discovered_dimensions(&[vec![1.0, 0.0]])
            .is_err()
    );
}

// ── EmbeddingUsage ────────────────────────────────────────────────────────────

#[test]
fn embedding_usage_serializes_as_stable_token_metadata() {
    let usage = EmbeddingUsage::new(128);
    let encoded = serde_json::to_value(usage).expect("embedding usage serializes");
    assert_eq!(encoded, json!({ "input_tokens": 128 }));
    assert_eq!(
        serde_json::from_value::<EmbeddingUsage>(encoded).unwrap(),
        usage
    );
}

#[test]
fn openai_usage_reads_prompt_tokens() {
    let value = json!({"data": [], "usage": {"prompt_tokens": 128, "total_tokens": 128}});
    assert_eq!(
        super::openai::parse_usage(&value),
        Some(EmbeddingUsage::new(128))
    );
}

#[test]
fn voyage_usage_falls_back_to_total_tokens() {
    // Voyage answers on the OpenAI response shape but reports only a total.
    let value = json!({"data": [], "usage": {"total_tokens": 4096}});
    assert_eq!(
        super::openai::parse_usage(&value),
        Some(EmbeddingUsage::new(4096))
    );
}

#[test]
fn usage_prefers_prompt_tokens_over_total() {
    let value = json!({"data": [], "usage": {"prompt_tokens": 90, "total_tokens": 100}});
    assert_eq!(
        super::openai::parse_usage(&value),
        Some(EmbeddingUsage::new(90))
    );
}

#[test]
fn absent_or_unusable_usage_reports_none() {
    // No usage object at all — every provider that omits the field.
    assert!(super::openai::parse_usage(&json!({"data": []})).is_none());
    // Present but empty.
    assert!(super::openai::parse_usage(&json!({"usage": {}})).is_none());
    // Non-numeric, rather than a silent zero.
    assert!(super::openai::parse_usage(&json!({"usage": {"prompt_tokens": "128"}})).is_none());
    // Zero is indistinguishable from unpopulated, so it is not a measurement.
    assert!(super::openai::parse_usage(&json!({"usage": {"prompt_tokens": 0}})).is_none());
}

#[tokio::test]
async fn default_embed_with_usage_reports_no_usage() {
    // A model that does not override the method still answers, and says
    // nothing about cost rather than claiming zero.
    let (vectors, usage) = ShortEmbeddingModel
        .embed_with_usage(&["hello".to_owned()])
        .await
        .unwrap();
    assert_eq!(vectors, vec![vec![1.0, 0.0]]);
    assert!(usage.is_none());
}

#[tokio::test]
async fn empty_batch_reports_no_usage() {
    let model = OpenAiEmbeddingModel::new("key").with_base_url("http://127.0.0.1:1");
    let (vectors, usage) = model.embed_with_usage(&[]).await.unwrap();
    assert!(vectors.is_empty());
    assert!(usage.is_none());
}
