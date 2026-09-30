# tinyinference-hub

One provider taxonomy, catalogue, typed error taxonomy, and endpoint policy for
every TinyInference host (OpenCompany, OpenHuman, the TUI, CLIs, future
servers).

Two hosts used to implement provider management separately: two catalogues of
the same hosted vendors, two error classifiers, and safety invariants (SSRF
policy, credential redaction) that only one had. This crate is the shared
answer. It is a **leaf**: it depends on `tinyinference-llm` (plus small
utility crates), nothing depends on it, and no existing public item of any
other crate changed.

This first slice ships the foundations. Probing, model catalogs, health,
operations, route resolution and the `ChatModel` factory follow in later slices
(see "Roadmap").

## What is here

| Module | What it gives you |
|---|---|
| `error` | `HubError`, the stable `ReasonCode` wire vocabulary, `Retry`, and `classify(status, headers, body)`, which turns a vendor response into a `ProviderFailure`. A spend cap is `quota` and never retried; a rate limit is `rate_limited` and carries the provider's own delay. |
| `Secret`, `LogOnly` | Wrappers that redact themselves in `Debug` and `Display`. `Secret` has no `Serialize`; `LogOnly` holds raw upstream text that may echo request material. |
| `ids` | `Slug`, `ModelId`, `KindId`, `ScopeKey`, `AgentKey`, `WorkloadKey`, and the validators ported from OpenCompany (`slugify`, `check_provider_name`, `check_slug`, `check_model_id`). |
| `taxonomy` | Groups, transports, protocols, `AuthStyle`, catalog shapes, `TestDepth`, `CliKind`, and `LocalRuntime`, which reconciles the four local-runtime enums by conversion. |
| `catalogue`, `descriptor` | Every built-in kind as data: the managed kind, 26 cloud providers, 5 local runtimes, 2 CLI logins, with typed quirks. `ProviderRecord` is a configured instance and has no credential field. |
| `policy`, `endpoint` | `EndpointPolicy` presets (`hosted`, `desktop`, `local_only`), `check_endpoint`, `check_address`, redirect checks, `HeaderPolicy`, and endpoint credential refusal, redaction and scrubbing. |

## Guarantees

- **A credential is never printed or stored on a record.** `Secret` redacts in
  `Debug` and `Display` and does not implement `Serialize`; `ProviderRecord` has
  no credential field.
- **Raw upstream error text is log-only.** It never reaches `Display`, `Debug`,
  or `HubError::user_message`.
- **Only a rejected credential rolls back an add** (a local runtime also rolls
  back when it is unreachable). The classifier's auth branch is a positive list
  of phrases, so a body it does not recognise keeps the key.
- **SSRF policy on every URL and address.** Link-local and metadata addresses
  are refused under every policy; alternative IPv4 spellings, `localhost` by
  name, and IPv4 embedded in IPv6 (mapped, NAT64, compatible) get the answer for
  the address a client would actually connect to.

## Example

```rust
use tinyinference_hub::{ReasonCode, Retry, classify};

// An Anthropic spend cap arrives as a 429. It is a quota problem, not a
// cooldown: never retried.
let failure = classify(
    429,
    &[],
    r#"{"type":"error","error":{"type":"rate_limit_error","message":"You have reached your specified API usage limits."}}"#,
);
assert_eq!(failure.reason, ReasonCode::Quota);
assert_eq!(failure.retry, Retry::Never);
```

## Features

`default = []`.

| Feature | Effect |
|---|---|
| `local-bridge` | `From`/`TryFrom` between `LocalRuntime` and `tinyinference-local`'s `LocalProviderKind` and `LocalAiProvider`. |
| `cli`, `oauth`, `testing`, `http-reqwest` | Reserved for later slices; they add nothing yet. `oauth` will only ever define types: no OAuth flow is enabled. |

## Known limits of this slice

- `Http` is a port the host implements (a later slice). Turn traffic uses
  `tinyinference-llm`'s own transport, so the hub's per-hop redirect and
  IP-pinning guarantees will cover probes and catalogs, not chat turns.
- The managed kind's endpoint, catalog shape and query are supplied by the host,
  because OpenCompany and OpenHuman reach different backends.

## Roadmap

1. Foundations (this crate as it stands): errors, secrets, identifiers,
   taxonomy, catalogue, endpoint policy.
2. Engine: ports, credential chain, model catalog cache, probing, health, kind
   drivers.
3. Operations: config with compare-and-swap, the operation set, route
   resolution, import readers, the `ChatModel` factory, and the no-socket
   simulation harness.
