# Changelog

## Unreleased

- OpenAI-compatible provider: a Claude model id (`claude-...`, or one with an
  `anthropic` path segment) now reports `hoists_system_messages`, as the native
  Anthropic adapter does. A relay serving Claude over Chat Completions joins
  every `role: "system"` message into Anthropic's one top-level `system`, so a
  note a harness added mid-turn as a system message rewrote the cached prefix
  twice: when it appeared and when it went. Callers that read the flag now put
  such notes in the conversation tail, and the request builder drops system
  messages that render no text. Seen on `anthropic/claude-haiku-4.5` through
  OpenRouter: two budget notices in an 80-call turn wrote about 276k prompt
  tokens to the cache again.
- OpenAI-compatible provider: a streamed response now keeps the vendor fields
  of `usage` (OpenRouter's `cost`, `cost_details`, `is_byok`) on `raw`, under
  `usage`, where the non-streaming response already has them in its wire
  body. A caller that reads a reported charge from `/usage/cost` saw it only
  when it did not stream. The token counts the normalized usage reads stay off
  `raw`, so a plain OpenAI stream still has `raw: None`.
- Depend on `tinystoragedrivers-core` / `-secrets` v0.7.0 (was v0.5.0), so a host that vendors the v0.7.0 storage crates holds one `SecretStore` and one `BlobStore` trait instead of two (tinyhumansai/openhuman#7641).

## 0.4.1

### Breaking changes

- Storage goes through tinystoragedrivers ports: `OAuthFlow::new` takes an
  `Arc<dyn SecretStore>`, `start`/`clear_pending` are async, `pending_filename`
  is now `pending_secret`; `GeneratedMedia::persist` writes to a `BlobStore`
  and returns `BlobMeta` (new `Error::Storage`).

### Fixed

- OpenAI-compatible provider: an Anthropic-family request that asks for
  prompt-cache breakpoints now also marks its newest text-bearing message. In a
  tool loop the last user message stops moving, so the assistant tool calls and
  tool results behind it were re-sent uncached on every call. Measured on
  `anthropic/claude-haiku-5.5` through OpenRouter over an 8-round tool loop:
  31% of prompt tokens read from cache before, 82% after. Other models keep the
  system and last-user breakpoints (Gemini honours only the last breakpoint).

## 0.3.1

### Breaking changes

- Removed the stale copy of the harness response cache from
  `tinyinference_llm::cache`: `cache_key`, the `ResponseCache` trait,
  `InMemoryResponseCache`, `PromptCacheLayout` and `CacheLayoutEvent`. Nothing
  used them; the maintained versions (scoped keys, SQLite store, singleflight,
  prompt-layout guard) live in `tinyagents-harness`'s `cache` module.
  `CachePolicy` stays at `tinyinference_llm::cache::CachePolicy`, unchanged.
  `canonical_value` is now public there. The crate no longer depends on `sha2`.

### Added

- OpenAI-compatible provider: `OpenAiConfig` accepts a host-configured
  `reqwest::Client`, used for every transport path, so a custom provider can
  trust private CA certificates (openhuman#6905).
- Model discovery reads an Ollama server's context window from the native
  `POST /api/show` when its `/v1/models` listing carries none
  (`LimitSource::NativeApi`, `DiscoveryRequest::with_ollama_native`,
  `ModelListingFetcher::post_json`). The smaller of `num_ctx` and the
  architecture `context_length` is used (openhuman#7099).
- `tinyinference-image`: the `ImageGenerator` trait, `OpenRouterImageGenerator`
  (`POST /images`), `MockImageGenerator`, media-reference standards
  (URL, `data:` URL, bytes, local path → OpenRouter content parts), aspect-ratio,
  resolution and size normalization, per-model capability pre-flight checks, and
  a billing-aware OpenRouter media transport usable directly or through a
  proxying backend.
- `tinyinference-video`: the `VideoGenerator` trait, `OpenRouterVideoGenerator`
  (`POST /videos`, `GET /videos/{id}`, `GET /videos/{id}/content`),
  `wait_for_job` (resume by job id), and `MockVideoGenerator`. A `completed`
  job with no outputs keeps polling instead of failing.
- OpenAI-compatible chat: a `ReasoningConfig::budget_tokens` sent to an
  OpenRouter endpoint is now emitted as `reasoning: {"max_tokens": N}` (and
  `reasoning_effort` is omitted, since OpenRouter takes one or the other).
  Other OpenAI-compatible endpoints still drop the budget, and an explicit
  `reasoning` provider option still wins.
- OpenAI-compatible chat: adaptive parameter omission. A 400 whose message
  names an optional field the request sent (`temperature`, `top_p`, `seed`,
  `max_tokens`, `max_completion_tokens`, `reasoning_effort`) drops that field,
  retries once, and remembers the omission for that endpoint and model. The
  evidence rule and the process-wide store are public as
  `providers::omission::{parameter_blamed_by, remember_omit, is_omitted}`.
- `providers::BearerSource` and `OpenAiModel::with_bearer_source`: a credential
  read per request (honouring `AuthStyle`, invalidated on 401) so a rotating
  token no longer forces rebuilding the model.
- Anthropic: a request carrying both a reasoning `effort` and `budget_tokens`
  now keeps adaptive thinking with that effort, byte-identical to the
  effort-only request. Fixed-budget thinking is used only when no effort is
  set. (Previously the budget won.)

### Fixed

- `scrub_credentials` no longer redacts ordinary source code. Keys must be whole
  identifiers ending in a sensitive word, key/operator/value never span a
  newline, `==` comparisons are ignored, and the value must look like a secret:
  known provider prefixes (`sk-`, `ghp_`, `github_pat_`, `xox?-`, `AKIA`, JWT
  `eyJ`) always redact, while placeholders (`"<eos>"`), whitespace, signed or
  fractional numbers, digit-less identifiers (`None`, `self.vocab`) and values
  followed by `(`, `[` or member access pass through.

## 0.3.0

### Breaking changes

- `ModelStream` is a metadata-owning stream struct. This source-breaking change
  means custom `ChatModel` implementations must replace direct
  `Ok(Box::pin(stream))` returns with `Ok(ModelStream::new(Box::pin(stream)))`.
- `ModelRequest` now distinguishes `model` from `requested_route`. Hosts must
  use `with_requested_route` when fallback observability needs a route name.
- `ModelResponse` includes `correlation` and `resolved_route`; custom response
  literals must initialize both fields.

### Added

- Typed model-call correlation, route metadata, fixed-point charged usage, and
  context-window usage fields.
- Abort-on-drop streams, generic model decorators and terminal observers.
- Cancellable, validated embedding requests and responses.

See [`docs/migrations/0.3.md`](docs/migrations/0.3.md) for migration details.
