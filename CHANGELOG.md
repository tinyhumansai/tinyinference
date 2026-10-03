# Changelog

## Unreleased

### Added

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
