# Perplexity Agent API contracts and migration

The native provider lives in `tinyinference_llm::providers::perplexity` and is
also exported from the crate root. It uses `/v1/agent` by default; an explicit
`PerplexityConfig.responses_alias` enables `/v1/responses` for creation.
Lifecycle and file operations use the canonical Agent routes.

## Choosing models and presets

Construct `PerplexityModel::new(key, selection)` or use `with_config` for explicit
limits, tools and callbacks. Credentials are supplied by the application; the
adapter does not read environment variables. `build_perplexity_model` returns
an `Arc<dyn ChatModel<()>>` when the host needs the existing trait-object interface.

Selections are `Model`, `Models` (one to five fallback IDs), and `Preset` with
optional model/fallback override. Preset names are not a frozen whitelist.
Current documented names are fast, low, medium, high and xhigh. Preset contents
can change server-side. Explicit Anthropic selections require a positive
`ModelRequest.max_tokens` or configured output cap.

Use `PerplexityOptions::apply_to` for typed per-call options. Those override
configured options; `ModelRequest.model` overrides a single model while retaining
a preset. A model override alongside a fallback list is rejected as ambiguous.
Unset instructions/tools/steps remain absent so server defaults survive.
System messages replace preset instructions; hosted-tool overrides merge per tool.
Use an explicit tool-choice override when replacing a preset's auto/required policy.

Hosted configuration includes web/image/finance/people search, URL fetching,
sandbox, remote MCP and existing connectors. MCP credentials are configured
separately by server label and exact server URL; they do not enter serialized
`ModelRequest.provider_options`. Remote calls execute on Perplexity's side.
The API does not enforce an MCP `require_approval` option; this adapter rejects it.
Do not use an empty remote allowlist to mean no tools: the provider treats it as all tools.

Unsupported settings fail explicitly: profile/skill administration or request
configuration, seed/stop sequences, unsupported reasoning settings, local media
paths, audio/video input, and unresolved required-capability guarantees. Standard
text, image, and URL/base64 document inputs use existing message/media types.
The caller resolves local files. Provider-specific model/media limits still apply.

Payload callbacks may adjust validated step/output budgets and sampling values.
Use typed options for other fields; callbacks cannot alter input, tools, credentials,
or execution mode behind the validated request. Response callbacks see sanitized data.

## Rich results and streams

`ModelResponse.output` is authoritative for providers with rich output. Each item
retains its index, optional item ID/status, and a `ModelOutputKind`. Known kinds
cover messages with annotations, reasoning, custom functions, search sources,
images, fetched pages, financial content, sandbox results, remote tool discovery
and execution, namespaces, and generated-file notices.

`message` remains the compatibility projection: visible text, supplied reasoning,
and only functions for the host to execute. Hosted tools never become local
`ToolCall` values. Text markers and source IDs are preserved without rewriting.
Citation offsets carry `provider_characters`; they are not safe Rust byte indices.
Unknown items/content/annotations use the explicit provider-extension boundary.
Malformed known items return an error rather than an empty or opaque success.

`ModelResponse.execution` records response ID, actual model, status, reconnect
cursor, reported costs/tool counts and stream-only `ModelProgress`. Stream items
include `OutputEvent` alongside existing text/tool/usage deltas. Collecting a stream
retains progress and unknown events; authoritative snapshots replace incremental
state without adding the same text, function arguments or usage twice.

`Completed` marks stream termination, not necessarily a fully completed task:
inspect execution status and finish reason for incomplete output. Failed/cancelled
foreground runs produce `ProviderFailed`. That error retains `partial_response`,
including received output and recovery identifiers. EOF without a terminal response
is a failure. Refusals and unfamiliar content are preserved as extensions.

Usage remains optional. Missing usage or charges are unknown, not zero. Provider
decimal costs retain their original number text inside the adapter; valid USD
totals are also projected to checked, nearest micro-dollars in
`Usage.charged_amount`. No local pricing calculation is performed. Tool counts and
cost breakdowns remain separate from token totals.

## Functions and conversation state

Declare application functions with the existing `ToolSchema`. A returned
`ToolCall` preserves `call_id` as `id` and keeps the distinct provider item ID,
original argument text and thought signature in optional `replay` metadata.
Malformed arguments remain explicitly invalid; the library never runs functions.

Create results with `ToolMessage::for_call(&call, content)`. For manual history,
include the preceding assistant call and matching result; native function items
are sent instead of text imitations. Duplicate/unmatched/conflicting calls or
results fail validation. Signed replay requires its original provider/model.
Pin the response's actual model when continuing a dynamically selected preset.

For custom function results, keep the original request's conversation and tools,
append `Message::Assistant(first.message)` followed by the result from `for_call`,
set `request.model` to the returned model, and leave `continuation_id` unset.
Retain the original call rather than reconstructing it: its argument bytes and
signature must survive. Change a forced tool choice back to `Auto` for another
turn, or `None` when only a final answer is wanted. Execute each function once;
resending its result does not require executing it again.

The complete [function replay example](../../crates/tinyinference-llm/examples/perplexity_function_replay.rs)
uses these existing types. With `PERPLEXITY_API_KEY` set, run:

```sh
cargo run -p tinyinference-llm --example perplexity_function_replay
```

For ordinary chat continuation, put the prior execution ID in
`ModelRequest.continuation_id` and send only the new user turn. The provider
requires a completed same-account response. Do not resend full assistant history
alongside the continuation ID. Unsupported signed reasoning/history blocks
are rejected rather than being silently stripped.

Live checks on 2026-10-04 isolated a broader source-response problem: referencing
a response ending in a custom function call returned HTTP 400 for both
`google/gemini-3.1-flash-lite` and `openai/gpt-6-luna`, even for a plain user
follow-up. Stored assistant-text responses continued successfully. Direct HTTP
probes, including completed background calls, reproduced this independently of
the adapter. Full call/result replay works; the internal server cause remains
unconfirmed.
The adapter preserves the error and never silently substitutes another request.

`store` is omitted by default. In Perplexity's documented behavior, false hides
retrieval but does **not** disable server persistence or continuation. Follow your
provider agreement for retention requirements.

## Background recovery and generated files

`submit_background` returns a serializable `DeferredHandle`. Persist it in the
calling application. Each retrieval is one explicit read; the library starts no
polling loop. `fetch_deferred` maps to pending, completed (including explicit
incomplete status), structured provider failure, or confirmed cancellation.

`resume_background(handle, cursor)` reconnects after the last sequence number.
It emits only new events and returns the full authoritative final response.
If the reconnect window has expired, inspect `retrieve_response` instead of
creating a replacement job. Handles are validated against the configured endpoint;
they cannot redirect credentials to a URL embedded in persisted metadata.

`cancel_background` returns `Cancelling` when acknowledged. Poll to observe
`Cancelled`; acknowledgement is not proof that execution has stopped. Dropping
a local stream stops local reads but does not silently cancel a durable remote job.
Cancellation is never automatically replayed.

The live cancellation check received `cancelling`, followed by `incomplete`.
Inspect the returned status; cancellation acknowledgement is not proof of a
terminal `cancelled` response.

Use `response_handle` for a foreground response's generated files. `list_response_files`
returns response-scoped file IDs, filenames, reported byte sizes and creation times.
`download_response_file` returns bounded bytes and media type. The library does not
write files, interpret filenames as paths, or fetch arbitrary output URLs. A failed
download does not regenerate the run. `share_file` notices preserve undocumented
metadata; the file-list endpoint provides authoritative descriptors.

## Limits and retries

Defaults are configurable: 600 seconds total, 30 seconds to connect, 60 seconds
of stream inactivity, 32 MiB for input/JSON/stream traffic, 4 MiB per SSE event,
and 64 MiB per downloaded file. `ModelRequest.timeout_ms` overrides the total
call budget. Limits are enforced during serialization and reads. Stream collection
is pull-driven; no detached producer continues after the caller drops the stream.

Creation retries only explicit HTTP 429, up to three retries within the original
deadline. Transport and 5xx creation failures can have unknown completion/billing
and are not replayed. Read operations can retry 429/500/502/503/504. Use zero retries
when the caller needs to manage all repetition. Recover acknowledged jobs by ID.

The adapter provides no automatic response-cache identity: presets evolve,
searches change, and remote tools may have side effects. The host owns caching
policy. Existing response serialization and cache storage preserve the rich fields.

## Rust compatibility

Existing serialized records remain readable through defaulted optional fields.
Rust struct literals and exhaustive matches require source updates:

- `ModelResponse`: add `output: Vec::new()` and `execution: None` for legacy providers.
- `ProviderError`: add `partial_response: None`, or use its existing default.
- `ToolCall`: add `replay: None`, or use `ToolCall::new`/`invalid`.
- `ToolMessage`: add `call_context: None`, or use existing constructors/`for_call`.
- `ModelStreamItem`: handle `OutputEvent` while retaining the existing delta channels.
- `DeferredStatus`: handle structured `ProviderFailed` and `Cancelled` states.
- `ProviderKind`: handle `Perplexity`; the OpenAI-compatible constructor explicitly
  rejects it so native requests cannot accidentally use Chat Completions.

Other providers keep their wire behavior. No workspace release version is changed
by this feature. Library fixtures verify protocol behavior; live provider behavior,
model access and billing require a separately authorized Perplexity account test.
