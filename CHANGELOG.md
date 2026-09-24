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
