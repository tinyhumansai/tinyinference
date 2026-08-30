# TinyInference

TinyInference is the provider-facing Rust layer shared by TinyHumans AI agent
runtimes. It owns model and embedding API concerns without owning an agent loop,
graph runtime, middleware system, capability registry, or workspace policy.

The crate provides:

- provider-neutral messages, tool-call shapes, requests, responses, usage, and
  capability profiles;
- a `ChatModel<State>` abstraction with real asynchronous streaming;
- OpenAI Chat Completions, OpenAI Responses, and OpenAI-compatible provider
  adapters for Anthropic, Ollama, DeepSeek, Groq, xAI, OpenRouter, Together,
  and Mistral;
- OpenAI, Cohere, Ollama, Voyage, cloud, no-op, and deterministic mock
  embeddings;
- request caching, stream accumulation, normalized provider failures, and
  provider-neutral retry classification.

## Use

```rust
use tinyinference::message::Message;
use tinyinference::model::{ChatModel, ModelRequest};
use tinyinference::providers::MockModel;

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
that only needs inference can depend on TinyInference directly.

## Layout

```text
Cargo.toml
crates/tinyinference/
└── src/
    ├── cache/       request fingerprints and response-cache contracts
    ├── embeddings/ embedding clients, vector store, and retriever
    ├── message/    provider-neutral message and content blocks
    ├── model/      ChatModel, request/response, profiles, and streaming
    ├── providers/  mock and OpenAI-compatible transports
    ├── error.rs    crate-wide Error and Result
    ├── failure.rs  normalized provider-failure classification
    ├── tool.rs     model-visible tool schemas and call/delta shapes
    └── usage/      normalized token accounting
```

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

## License

GPL-3.0-only. See [LICENSE](LICENSE).
