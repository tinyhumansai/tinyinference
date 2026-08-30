# Repository Guidelines

## Scope

TinyInference is a Rust 2024 library workspace. It owns provider-neutral model,
message, tool-call, usage, streaming, provider transport, cache, and embedding
APIs. Agent loops, middleware, graphs, registries, orchestration, and workspace
policy belong in consuming runtimes such as TinyAgents.

## Structure

The public crate is `crates/tinyinference`. Keep feature areas in module
directories with `mod.rs`, `types.rs`, and `test.rs` where the area is large.
Centralize deliberate exports in `src/lib.rs`. Keep provider wire types private
unless callers must construct them.

## Workflow

Make new implementation changes on a feature branch. Prefer direct execution
for clear tasks. Preserve unrelated work and do not rewrite or squash existing
commits.

Run from the repository root:

```sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --all-targets --all-features
cargo test --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Do not redirect Cargo output to a temporary target directory. Use the normal
workspace target configuration.

## Code and API

Use Rust 2024 idioms and standard rustfmt. Public fallible APIs return the
crate-wide `Result<T>`. Add a typed `Error` variant when callers need to
distinguish a failure. Never expose provider-specific JSON above the provider
adapter when a normalized type can represent it.

`ChatModel` and `EmbeddingModel` implementations must be `Send + Sync`. Stream
adapters emit terminal `Completed` or `ProviderFailed` items and must preserve
tool-call ids, incremental arguments, reasoning, finish reasons, and usage.
Never log or expose API keys; custom `Debug` implementations must redact them.

Every public item needs rustdoc. Document `# Errors` and `# Panics` where
applicable. Keep Markdown files at 500 lines or fewer.

## Tests

Keep unit tests beside their module in dedicated `test.rs` files or an existing
module-local test submodule. Tests must not depend on network access, wall-clock
timing, ambient credentials, or mutable process environment. Use synthetic HTTP
payloads and byte streams for provider behavior. Add tests for serialization,
stream reconstruction, malformed tool calls, error classification, and vector
dimension contracts whenever those surfaces change.

## Commits and Pull Requests

Keep commits small and focused. Pull requests should summarize API/behavior
changes and list exact verification commands. Open ready-for-review PRs against
`tinyhumansai/tinyinference`; use drafts only for genuinely incomplete work.
