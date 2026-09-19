//! Type definitions for the harness embeddings + retrieval module.
//!
//! These are the building blocks of retrieval-augmented context — the
//! [`EmbeddingModel`] / [`VectorStore`] / [`Retriever`] triad that lets a
//! recursive agent fetch external knowledge on demand rather than carrying it
//! all in-context.
//!
//! All public types declared here are re-exported through [`super`] so callers
//! import them from `tinyinference_embeddings` directly.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::watch;

use crate::Result;

/// Cooperative cancellation for an embedding request.
///
/// The signal is cloneable so a host can retain one handle while an embedding
/// request owns another. The common request adapter races every provider future
/// against this signal and drops an active transport on cancellation.
#[derive(Clone, Debug)]
pub struct EmbeddingCancellation {
    cancelled: Arc<AtomicBool>,
    signal: watch::Sender<bool>,
}

impl Default for EmbeddingCancellation {
    fn default() -> Self {
        let (signal, _) = watch::channel(false);
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            signal,
        }
    }
}

impl EmbeddingCancellation {
    /// Creates a non-cancelled request signal.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Requests cooperative cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.signal.send_replace(true);
    }

    /// Returns whether cancellation has been requested.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    async fn cancelled(&self) {
        if self.is_cancelled() {
            return;
        }
        let mut receiver = self.signal.subscribe();
        while !*receiver.borrow_and_update() {
            if receiver.changed().await.is_err() {
                return;
            }
        }
    }
}

/// Provider-neutral batched embedding request.
#[derive(Clone, Debug)]
pub struct EmbeddingRequest {
    /// Texts to embed in the exact order the caller expects results.
    pub inputs: Vec<String>,
    /// Cooperative cancellation shared with the caller.
    pub cancellation: EmbeddingCancellation,
}

impl EmbeddingRequest {
    /// Creates a request with a fresh cancellation signal.
    #[must_use]
    pub fn new(inputs: Vec<String>) -> Self {
        Self {
            inputs,
            cancellation: EmbeddingCancellation::new(),
        }
    }

    /// Replaces the request cancellation signal.
    #[must_use]
    pub fn with_cancellation(mut self, cancellation: EmbeddingCancellation) -> Self {
        self.cancellation = cancellation;
        self
    }
}

/// Validated embedding result for one batch.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbeddingResponse {
    /// One vector for each request input, in the same order.
    pub vectors: Vec<Vec<f32>>,
    /// Dimension shared by every returned vector.
    pub dimensions: usize,
    /// Provider-reported batch usage, when available.
    pub usage: Option<EmbeddingUsage>,
}

// ── EmbeddingModel ────────────────────────────────────────────────────────────

/// Provider-neutral embedding model.
///
/// An embedding model turns text into dense [`f32`] vectors that downstream
/// vector stores and retrievers can compare with a distance metric (this module
/// uses cosine similarity). Implementations must be `Send + Sync` so they can be
/// shared across async task boundaries behind an [`Arc`].
///
/// The harness keeps embedding generation separate from the chat model
/// abstraction: chat models produce messages, embedding models produce vectors.
///
/// # Contract
/// - [`embed`](EmbeddingModel::embed) returns exactly one vector per input
///   text, in the same order as the inputs.
/// - Every returned vector has length [`dimensions`](EmbeddingModel::dimensions).
/// - Embedding the same text twice should produce the same vector for
///   deterministic implementations such as [`MockEmbeddingModel`].
#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    /// Whether semantic indexing and retrieval are enabled for this model.
    ///
    /// A zero value from [`Self::dimensions`] may mean dimensions have not yet
    /// been discovered, so callers must use this explicit capability instead
    /// of treating zero dimensions as disabled.
    fn is_enabled(&self) -> bool {
        true
    }

    /// Stable provider identifier, such as `"openai"` or `"ollama"`.
    fn name(&self) -> &str;

    /// Stable model identifier within the provider.
    fn model_id(&self) -> &str;

    /// Embeds a batch of texts, returning one vector per input in input order.
    ///
    /// Returning an empty `Vec` for empty input is valid.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embeds a batch and also reports what the provider said the batch cost.
    ///
    /// Same vectors, same contract, same single request as [`Self::embed`] —
    /// the only difference is that the provider's own token accounting
    /// survives the call instead of being dropped with the rest of the
    /// response. A caller metering embedding spend needs the provider's number
    /// rather than a local estimate, and this is the only place it exists.
    ///
    /// `None` means the provider reported nothing, which is not the same as
    /// zero: it is the honest answer for a local model, and for any endpoint
    /// that omits the field. A caller must not substitute an estimate for it.
    ///
    /// The default implementation delegates to [`Self::embed`] and reports
    /// `None`, so an implementation that cannot report usage needs no change.
    ///
    /// # Errors
    ///
    /// Whatever [`Self::embed`] would return for the same batch. Usage is
    /// reported alongside a success and never turns one into a failure: a
    /// response whose vectors parsed but whose usage did not is a successful
    /// embed with no usage.
    async fn embed_with_usage(
        &self,
        texts: &[String],
    ) -> Result<(Vec<Vec<f32>>, Option<EmbeddingUsage>)> {
        Ok((self.embed(texts).await?, None))
    }

    /// Embeds a validated batch with stable ordering, dimensionality, usage,
    /// and cooperative cancellation.
    ///
    /// The default races every provider operation against cancellation at the
    /// common model boundary. Cancelling drops the in-flight provider future;
    /// this interrupts Reqwest transports and any provider future that releases
    /// its resources on drop, without requiring each adapter to duplicate the
    /// cancellation mechanism.
    ///
    /// # Errors
    /// Returns [`crate::Error::Cancelled`] when cancellation is requested,
    /// propagates provider failures, and returns
    /// [`crate::Error::Validation`] when a provider violates the one-vector-per-
    /// input or common-dimension contract.
    async fn embed_request(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        if request.cancellation.is_cancelled() {
            return Err(crate::Error::Cancelled);
        }
        let cancellation = request.cancellation.clone();
        let result = tokio::select! {
            result = self.embed_with_usage(&request.inputs) => result,
            () = cancellation.cancelled() => return Err(crate::Error::Cancelled),
        };
        if request.cancellation.is_cancelled() {
            return Err(crate::Error::Cancelled);
        }
        let (vectors, usage) = result?;
        if vectors.len() != request.inputs.len() {
            return Err(crate::Error::Validation(format!(
                "embedding batch returned {} vectors for {} inputs",
                vectors.len(),
                request.inputs.len()
            )));
        }
        let dimensions = self.dimensions();
        let response_dimensions = vectors.first().map_or(dimensions, Vec::len);
        if dimensions > 0 && response_dimensions != dimensions {
            return Err(crate::Error::Validation(format!(
                "embedding batch returned {response_dimensions} dimensions; model advertises {dimensions}"
            )));
        }
        if vectors
            .iter()
            .any(|vector| vector.len() != response_dimensions)
        {
            return Err(crate::Error::Validation(
                "embedding batch returned inconsistent vector dimensions".to_string(),
            ));
        }
        Ok(EmbeddingResponse {
            vectors,
            dimensions: response_dimensions,
            usage,
        })
    }

    /// Embeds a retrieval query. Asymmetric providers can override this;
    /// symmetric models reuse [`Self::embed`].
    async fn embed_query(&self, query: &str) -> Result<Vec<f32>> {
        let mut vectors = self.embed(&[query.to_owned()]).await?;
        Ok(vectors.pop().unwrap_or_default())
    }

    /// Returns the fixed dimensionality of every vector this model produces.
    fn dimensions(&self) -> usize;

    /// Stable embedding-space identity used to partition persisted vectors.
    fn signature(&self) -> String {
        format_embedding_signature(self.name(), self.model_id(), self.dimensions())
    }
}

/// Format the canonical embedding-space signature.
///
/// This must remain byte-identical to OpenHuman's persisted signature contract.
pub fn format_embedding_signature(name: &str, model_id: &str, dimensions: usize) -> String {
    format!("provider={name};model={model_id};dims={dimensions}")
}

// ── EmbeddingUsage ────────────────────────────────────────────────────────────

/// What a provider reported an embedding request cost, in tokens.
///
/// Normalized across providers that each name it differently — OpenAI's
/// `usage.prompt_tokens`, Voyage's `usage.total_tokens`, Cohere's
/// `meta.billed_units.input_tokens` — so a caller meters spend without
/// matching on a provider.
///
/// Only ever constructed from a provider's own numbers. There is no estimating
/// constructor on purpose: a value of this type is a measurement, and a caller
/// that receives one can price it without qualifying the figure.
///
/// Embeddings generate no completion, so there is no output-token counterpart.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct EmbeddingUsage {
    /// Input tokens the provider billed for the batch.
    pub input_tokens: u64,
}

impl EmbeddingUsage {
    /// A usage report of `input_tokens`.
    #[must_use]
    pub fn new(input_tokens: u64) -> Self {
        Self { input_tokens }
    }
}

// ── MockEmbeddingModel ────────────────────────────────────────────────────────

/// Deterministic, offline embedding model for tests and examples.
///
/// `MockEmbeddingModel` hashes the input text to derive a stable vector without
/// any network access. Identical text always maps to an identical vector (so
/// the cosine similarity of a text with itself is exactly `1.0`), while
/// different texts map to different vectors. This makes retrieval behaviour
/// testable offline: querying with the exact text of an indexed document ranks
/// that document first.
///
/// The vectors are **not** semantically meaningful — this model exists purely
/// for deterministic shape/retrieval tests, mirroring LangChain's
/// `DeterministicFakeEmbedding`.
///
/// # Example
/// ```
/// use tinyinference_embeddings::{EmbeddingModel, MockEmbeddingModel};
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let model = MockEmbeddingModel::new(16);
/// let vectors = model.embed(&["hello".to_string()]).await.unwrap();
/// assert_eq!(vectors.len(), 1);
/// assert_eq!(vectors[0].len(), 16);
/// # });
/// ```
#[derive(Clone, Copy, Debug)]
pub struct MockEmbeddingModel {
    /// Fixed dimensionality of every produced vector.
    pub(crate) dimensions: usize,
}

impl MockEmbeddingModel {
    /// Creates a deterministic mock embedding model producing vectors of length
    /// `dimensions`.
    ///
    /// # Panics
    /// Panics if `dimensions` is `0`, since a zero-length vector cannot be
    /// compared with cosine similarity.
    pub fn new(dimensions: usize) -> Self {
        assert!(dimensions > 0, "embedding dimensions must be non-zero");
        Self { dimensions }
    }

    /// Computes the deterministic vector for a single `text`.
    ///
    /// Each component is derived by hashing `(text, component_index)` and
    /// mapping the result into the half-open range `[-1.0, 1.0)`. The mapping is
    /// pure, so repeated calls with the same `text` always return the same
    /// vector.
    pub(crate) fn embed_one(&self, text: &str) -> Vec<f32> {
        (0..self.dimensions)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                text.hash(&mut hasher);
                i.hash(&mut hasher);
                let raw = hasher.finish();
                // Map into [0, 1) then shift/scale into [-1, 1).
                let frac = (raw % 1_000_000) as f32 / 1_000_000.0;
                frac * 2.0 - 1.0
            })
            .collect()
    }
}

#[async_trait]
impl EmbeddingModel for MockEmbeddingModel {
    fn name(&self) -> &str {
        "mock"
    }

    fn model_id(&self) -> &str {
        "deterministic-hash"
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_one(t)).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// ── ScoredDoc ─────────────────────────────────────────────────────────────────

/// A document returned from a vector-store or retriever query, with its
/// relevance score.
///
/// `score` is a cosine similarity in `[-1.0, 1.0]` where **higher is more
/// similar**. Results from this module's stores are returned in descending
/// score order (most similar first).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoredDoc {
    /// Caller-supplied identifier of the matched document.
    pub id: String,
    /// Cosine similarity to the query vector; higher is more similar.
    pub score: f32,
    /// Arbitrary metadata associated with the document at index time.
    pub metadata: Value,
}

// ── VectorStore ───────────────────────────────────────────────────────────────

/// A store of dense vectors that supports nearest-neighbour search.
///
/// Implementations associate an `id` and arbitrary `metadata` with each vector,
/// and answer top-`k` similarity queries. This module's [`InMemoryVectorStore`]
/// ranks results by cosine similarity.
///
/// Implementations must be `Send + Sync` so they can be shared behind an
/// [`Arc`].
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Adds (or overwrites, when `id` already exists) a vector with associated
    /// `metadata`.
    ///
    /// Implementations should reject vectors whose dimensionality does not
    /// match the vectors already stored (and zero-length vectors) with
    /// [`Error::Validation`](crate::Error::Validation),
    /// so a store never mixes incomparable vectors.
    async fn add(&self, id: String, vector: Vec<f32>, metadata: Value) -> Result<()>;

    /// Returns up to `top_k` documents most similar to `vector`, sorted by
    /// descending similarity score.
    ///
    /// Returns fewer than `top_k` documents when the store holds fewer entries,
    /// and an empty `Vec` when `top_k` is `0` or the store is empty (an empty
    /// store has no dimensionality to validate against, so any query vector is
    /// answered with no hits). Implementations should reject a query vector
    /// whose dimensionality does not match the stored vectors with
    /// [`Error::Validation`](crate::Error::Validation)
    /// instead of returning meaningless scores.
    async fn query(&self, vector: &[f32], top_k: usize) -> Result<Vec<ScoredDoc>>;
}

// ── InMemoryVectorStore ───────────────────────────────────────────────────────

/// A single stored vector together with its id and metadata.
#[derive(Clone, Debug)]
pub(crate) struct VectorEntry {
    /// Caller-supplied document id.
    pub(crate) id: String,
    /// The stored dense vector.
    pub(crate) vector: Vec<f32>,
    /// Arbitrary metadata attached at insert time.
    pub(crate) metadata: Value,
}

/// Shared interior state of an [`InMemoryVectorStore`]: the entries in
/// insertion order plus an id → index map for O(1) upsert-by-id.
///
/// Entries are never removed, only appended or replaced in place, so the
/// indices in `index` stay valid for the lifetime of the store.
#[derive(Debug, Default)]
pub(crate) struct VectorStoreInner {
    /// All stored entries, in insertion order.
    pub(crate) entries: Vec<VectorEntry>,
    /// Maps each entry id to its position in `entries`.
    pub(crate) index: HashMap<String, usize>,
}

/// Thread-safe in-process [`VectorStore`] backed by a plain [`Vec`].
///
/// Search is a linear scan computing cosine similarity against every stored
/// vector, which is appropriate for tests, examples, and small corpora. The
/// store is cheaply clonable through the inner [`Arc`]; clones share the same
/// underlying data.
///
/// Adding a vector whose `id` already exists **replaces** the previous entry
/// (an O(1) id-indexed upsert), so re-indexing a document updates it in place.
///
/// The store's dimensionality is fixed by the first vector added: later `add`s
/// and non-empty-store `query`s with a different vector length are rejected
/// with [`Error::Validation`](crate::Error::Validation),
/// as are zero-length vectors, so every stored comparison is meaningful.
///
/// # Example
/// ```
/// use tinyinference_embeddings::{InMemoryVectorStore, VectorStore};
/// use serde_json::json;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let store = InMemoryVectorStore::new();
/// store.add("a".into(), vec![1.0, 0.0], json!({})).await.unwrap();
/// store.add("b".into(), vec![0.0, 1.0], json!({})).await.unwrap();
/// let hits = store.query(&[1.0, 0.0], 1).await.unwrap();
/// assert_eq!(hits[0].id, "a");
/// # });
/// ```
#[derive(Clone, Debug, Default)]
pub struct InMemoryVectorStore {
    /// Entries plus their id index, protected by a standard mutex.
    pub(crate) inner: Arc<Mutex<VectorStoreInner>>,
}

// ── Retriever ─────────────────────────────────────────────────────────────────

/// Query-to-document component tying an [`EmbeddingModel`] to a [`VectorStore`].
///
/// A `Retriever` embeds documents at index time and embeds queries at retrieval
/// time using the **same** embedding model, then delegates nearest-neighbour
/// search to the vector store. Both collaborators are held behind [`Arc`] so a
/// retriever is cheap to clone and share.
///
/// # Example
/// ```
/// use std::sync::Arc;
/// use tinyinference_embeddings::{InMemoryVectorStore, MockEmbeddingModel, Retriever};
/// use serde_json::json;
///
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let retriever = Retriever::new(
///     Arc::new(MockEmbeddingModel::new(32)),
///     Arc::new(InMemoryVectorStore::new()),
/// );
/// retriever
///     .index(vec![
///         ("d1".into(), "cats are great".into(), json!({})),
///         ("d2".into(), "the stock market crashed".into(), json!({})),
///     ])
///     .await
///     .unwrap();
/// let hits = retriever.retrieve("cats are great", 1).await.unwrap();
/// assert_eq!(hits[0].id, "d1");
/// # });
/// ```
#[derive(Clone)]
pub struct Retriever {
    /// Embedding model used for both documents and queries.
    pub(crate) model: Arc<dyn EmbeddingModel>,
    /// Backing vector store searched at retrieval time.
    pub(crate) store: Arc<dyn VectorStore>,
}

impl std::fmt::Debug for Retriever {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Retriever")
            .field("provider", &self.model.name())
            .field("model", &self.model.model_id())
            .field("dimensions", &self.model.dimensions())
            .finish_non_exhaustive()
    }
}
