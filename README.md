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
- OpenAI, Cohere, Ollama, Voyage, cloud, no-op, and deterministic mock
  embeddings;
- request caching, stream accumulation, normalized provider failures,
  provider-neutral retry classification and `Retry-After` parsing;
- conservative context-window and vision-capability hints for raw model ids
  when a provider cannot supply an authoritative model profile;
- normalized provider model-catalog parsing and local runtime model, vision,
  embedding, speech model, voice, and quantization resolution;
- embedding-provider catalogs, local model-tier presets, Ollama installation,
  Piper binary/voice installation, local runtime lifecycle and inference;
- reusable provider OAuth/PKCE, OpenAI Codex authentication, credential-file
  parsing, and deterministic provider-error classification;
- OpenAI-compatible hosted transcription, Piper synthesis, local-LLM
  transcript cleanup, and bounded PCM streaming helpers.

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
`tinyinference-embeddings` for vector generation and retrieval,
`tinyinference-local` for local runtimes and installers,
`tinyinference-providers` for provider authentication and routing primitives,
`tinyinference-voice` for speech inference and streaming-audio mechanics,
`tinyinference-image` for image generation and the shared media-reference
standards and OpenRouter media transport,
`tinyinference-video` for asynchronous video generation (submit, poll,
download, resume), and `tinyinference-core` only for shared infrastructure.

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
└── src/            embedding clients, vector store, and retriever
crates/tinyinference-local/
└── src/            device profiling, local runtimes, model selection, and installers
crates/tinyinference-providers/
└── src/            OAuth/PKCE flows and provider error classification
crates/tinyinference-voice/
└── src/            hosted STT, Piper TTS, cleanup, and PCM streaming helpers
crates/tinyinference-image/
└── src/            ImageGenerator, media references and output-shape
                    normalization, OpenRouter media transport, capabilities
crates/tinyinference-video/
└── src/            VideoGenerator, submit/poll/download job loop, resume by id
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
