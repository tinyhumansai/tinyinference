# Changelog

## Unreleased

### Added

- `tinyinference-hub`: a new leaf crate that gives every host one provider hub.
  A `Hub` facade over four ports (credential store, compare-and-swap config
  store, `Http`, `Clock`) offers connect (with rollback on a rejected key), add,
  edit, remove, enable, key set and clear, draft and stored probes at three
  depths, cached single-flight model lists, folded health fed by probes and real
  turns, defaults, per-agent pins, per-workload routes, turn resolution and a
  `ChatModel` that resolves its credential on every call. It also ships one
  taxonomy and catalogue (managed, 26 cloud, 5 local, 2 CLI kinds), a typed
  `HubError` with stable reason codes and a vendor-body classifier that
  separates spend caps from rate limits, redacting `Secret` and validated
  identifiers, the endpoint SSRF policy, an ordered credential chain,
  scope-partitioned model caches, pure readers for OpenCompany's and OpenHuman's
  stored shapes with loss reports, local and environment detection, CLI-login
  readiness (`cli`), OAuth types (`oauth`), a reference `ReqwestHttp`
  (`http-reqwest`), and a no-socket `testing` kit with a seeded scenario runner.
  Nothing else in the workspace depends on it and no existing public item
  changed.
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
