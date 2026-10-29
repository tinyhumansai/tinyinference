# TinyInference

TinyInference is the provider-facing Rust layer shared by TinyHumans AI agent
runtimes. It owns model and embedding API concerns without owning an agent loop,
graph runtime, middleware system, capability registry, or workspace policy.

The workspace provides:

- provider-neutral messages, tool-call shapes, requests, responses, usage, and
  capability profiles;
- a `ChatModel<State>` abstraction with real asynchronous streaming;
- OpenAI Chat Completions, OpenAI Responses, and OpenAI-compatible provider
  adapters for Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together,
  and Mistral;
- native Perplexity Agent API model/preset selection, cited and hosted-tool
  output, incremental streaming, signed function continuation, background
  recovery, and generated-file retrieval;
- OpenAI, Cohere, Ollama, Voyage, cloud, no-op, and deterministic mock
  embeddings;
- standalone document reranking through Voyage, with original-position results,
  cancellation, bounded retries, and provider-reported usage;
- request caching, stream accumulation, normalized provider failures,
  provider-neutral retry classification and `Retry-After` parsing;
- conservative context-window and vision-capability hints for raw model ids
  when a provider cannot supply an authoritative model profile;
- normalized provider model-catalog parsing and local runtime model, vision,
  embedding, speech model, and voice resolution;
- embedding-provider catalogs, and endpoint-only local inference against a
  runtime the user installs and runs (Ollama, LM Studio, MLX, OMLX, or any
  OpenAI-compatible server): reachability probes, installed-model discovery,
  and inference. TinyInference never downloads models or voices, installs a
  runtime, or starts/stops one;
- reusable provider OAuth/PKCE, OpenAI Codex authentication, credential-file
  parsing, and deterministic provider-error classification;
- OpenAI-compatible hosted transcription, Piper synthesis, local-LLM
  transcript cleanup, and bounded PCM streaming helpers.
- `tinyinference-decisions`, typed Jev/System One and Levanto Sage APIs;
  see [`docs/tinyinference-decisions.md`](docs/tinyinference-decisions.md).

## Use

```rust
use tinyinference_llm::message::Message;
use tinyinference_llm::model::{ChatModel, ModelRequest};
use tinyinference_llm::providers::MockModel;

tokio::runtime::Runtime::new().unwrap().block_on(async {
let model = MockModel::echo();
let response = model
    .invoke(&(), ModelRequest::new(vec![Message::user("hello")]))
    .await
    .unwrap();
assert_eq!(response.text(), "hello");
});
```

TinyAgents vendors this repository at `vendor/tinyinference` and re-exports the
public modules through its historical `tinyagents::harness::*` paths. New code
can depend on `tinyinference-llm` for language models,
`tinyinference-embeddings` for vector generation, retrieval, and reranking,
`tinyinference-local` for endpoint-only local inference,
`tinyinference-providers` for provider authentication and routing primitives,
`tinyinference-voice` for speech inference and streaming-audio mechanics,
`tinyinference-image` for image generation and the shared media-reference
standards and OpenRouter media transport,
`tinyinference-video` for asynchronous video generation (submit, poll,
download, resume), `tinyinference-decisions` for typed Jev and Sage decisions,
and `tinyinference-core` only for shared infrastructure.

### Perplexity Agent API

Use one Perplexity key with an explicit model or server-managed preset:

```rust,no_run
use tinyinference_llm::{ChatModel, Message, ModelRequest, PerplexityModel, PerplexitySelection};

# async fn example(key: String) -> tinyinference_llm::Result<()> {
let model = PerplexityModel::new(key, PerplexitySelection::preset("low"))?;
let response = model.invoke(&(), ModelRequest::new(vec![
    Message::user("Explain the latest battery research with sources."),
])).await?;
println!("{}", response.text());
// response.output contains typed citations, hosted results and file notices.
# Ok(())
# }
```

`PerplexitySelection::Model("provider/model".into())` selects a specific model;
`Models` configures a provider-managed fallback list. Presets retain server
defaults unless you explicitly override them. Tool settings merge with preset
tools on the provider. Explicit Anthropic models require an output-token cap.

`ModelResponse.output` preserves ordered rich output; `message` is the familiar
text/reasoning/local-function projection. Hosted MCP, search and sandbox activity
is never dispatched as a local function call. Stream-only progress survives in
`execution.progress`; incomplete and cancelled states remain explicit.

Use `submit_background`, `retrieve_response`, `resume_background`, and
`cancel_background` for long jobs. List and explicitly download generated files
by response/file ID; filenames are metadata, not local paths. A disconnect never
silently starts a replacement run. Remote MCP credentials are bound to their
configured HTTPS server and kept out of serializable model requests.

See [Perplexity contracts and migration](docs/migrations/perplexity-agent.md)
for configuration, replay, storage semantics, limits, and public Rust type changes.

### Voyage reranking

`tinyinference-embeddings` exports `Reranker`, `RerankRequest`,
`VoyageReranker`, and `VoyageRerankConfig`. Supply your Voyage API key explicitly:

```rust,no_run
use tinyinference_embeddings::{Reranker, RerankRequest, VoyageReranker};

# async fn example(api_key: String) -> tinyinference_embeddings::Result<()> {
let model = VoyageReranker::new(api_key)?;
let documents = vec!["Reset your password in Settings.".into(), "Shipping takes two days.".into()];
let mut request = RerankRequest::new("How do I reset my password?", documents);
request.top_k = Some(1);
let response = model.rerank(request).await?;
for result in response.results {
    println!("original position: {}, score: {}", result.index, result.relevance_score);
}
# Ok(())
# }
```

Use candidate texts from any search system. Our `Retriever` does not preserve
source text: resolve its returned IDs against your own document storage first.
Missing text should fail before the request rather than silently removing a
candidate. Returned indices address the exact submitted list. Keep reranking
scores separate from `ScoredDoc.score`, which remains vector similarity.
The `rerank` module's compiled rustdoc shows this mapping and caller-owned fallback.

Defaults are `rerank-3`, a 30-second total deadline, up to three retries of
429/500/502/503/504 responses, 8 MiB serialized requests, and 2 MiB responses.
`VoyageRerankConfig` makes these configurable. Requests accept up to 1,000
documents; zero results or empty candidates require no provider call.
Blank queries/documents are rejected for nonempty requests. Provider token
limits still apply. Text shortening is disabled unless `request.truncate` is
explicitly enabled; large batches are never silently split or dropped.

Clone `request.cancellation` to cancel while waiting for pacing, a response,
or a retry. The deadline covers all attempts. Transport interruptions are not
replayed; the provider may still finish and bill an interrupted request.
Explicit status retries can repeat billable work; set `max_retries` to zero to
disable them. Missing usage is unknown, and reported usage covers only the
successful response. Decide in your app whether provider failure preserves
original search order; propagate cancellation rather than treating it as fallback.

Errors use `Error::Rerank(RerankError)` for provider, timeout, and response
failures; `Validation` and `Cancelled` retain their existing roles. Downstream
exhaustive matches on the crate's `Error` must add the new `Rerank` arm.

### Levanto Sage

Sage uses typed decisions. A missing verdict means Sage is unsure and should
be handled as a valid answer. Create an API key in Levanto and pass it from
your application's secret storage; the client never reads ambient credentials.

```rust,no_run
use tinyinference_decisions::sage::{
    DecisionQuestion, DecisionRequest, DecisionResponse, SageClient, YesNoAnswer,
};

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let key = std::env::var("SAGE_API_KEY")?;
let sage = SageClient::new(key)?;
let request = DecisionRequest::new(
    "Marketing copy promises guaranteed returns.",
    DecisionQuestion::YesNo {
        id: "needs_review".into(),
        instructions: "Does this copy need compliance review?".into(),
    },
);
if let DecisionResponse::YesNo { result, .. } = sage.decide(&request).await? {
    match result.answer {
        Some(YesNoAnswer::Yes) => println!("review"),
        Some(YesNoAnswer::No) => println!("send"),
        None => println!("escalate"),
    }
}
# Ok(())
# }
```

## Layout

```text
Cargo.toml
crates/tinyinference-core/
└── src/
    ├── retry_after.rs shared Retry-After parsing and bounded backoff
    └── sanitize.rs    credential-safe diagnostic formatting
crates/tinyinference-llm/
└── src/
    ├── cache/       request fingerprints and response-cache contracts
    ├── catalog/     provider model-catalog types and response parsing
    ├── message/    provider-neutral message and content blocks
    ├── model/      ChatModel, request/response, profiles, and streaming
    ├── providers/  mock and OpenAI-compatible transports
    ├── error.rs    crate-wide Error and Result
    ├── failure.rs  provider-failure classification and retry hints
    ├── tool.rs     model-visible tool schemas and call/delta shapes
    └── usage/      normalized token accounting
crates/tinyinference-embeddings/
└── src/            embedding clients, rerank/, vector store, and retriever
crates/tinyinference-local/
└── src/            local endpoint probing, model selection, and inference
crates/tinyinference-providers/
└── src/            OAuth/PKCE flows and provider error classification
crates/tinyinference-voice/
└── src/            hosted STT, Piper TTS, cleanup, and PCM streaming helpers
crates/tinyinference-image/
└── src/            ImageGenerator, media references and output-shape
                    normalization, OpenRouter media transport, capabilities
crates/tinyinference-video/
└── src/            VideoGenerator, submit/poll/download job loop, resume by id
crates/tinyinference-decisions/
└── src/            typed Jev and Sage decisions and HTTP clients
```

### Media generation

`tinyinference-image` and `tinyinference-video` speak OpenRouter's media wire
format (`POST /images`, `POST /videos`, `GET /videos/{id}`,
`GET /videos/{id}/content`). The same generators run against OpenRouter
directly (`MediaAuth::ApiKey`) or against a host backend that proxies those
routes verbatim (`MediaAuth::Bearer` with `MediaTransport::with_base_url`).
A generator returns delivered media or an error — never an empty success —
and every error after a billed submit names the job and says not to resubmit.
Live smoke tests: `cargo run -p tinyinference-image --example
live_openrouter_image` and `cargo run -p tinyinference-video --example
live_openrouter_video` (skip without `OPENROUTER_API_KEY`).

## Development

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets --all-features
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Tests are deterministic and offline. Provider integration tests operate on
wire payloads and synthetic byte streams; constructing a hosted provider does
not make a network call.

## 0.3 migration

Version 0.3 intentionally makes the model-boundary contract explicit.
`ModelStream` is now a struct, so construct custom streams with
`ModelStream::new(Box::pin(stream))` rather than returning a boxed stream
directly. Set a host route through `ModelRequest::with_requested_route`; do not
put it in `model`, which identifies a provider model or alias. Consult
[`docs/migrations/0.3.md`](docs/migrations/0.3.md) for the complete migration
map.

## License

GPL-3.0-only. See [LICENSE](LICENSE).
