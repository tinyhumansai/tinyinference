# Shared provider-call budgets

`tinyinference_llm::model::budget` supplies `Budget`, `SpendLimits`,
`CallBudget` and `BudgetedModel`. Hosts own policy and select conservative
per-call input/output token and microdollar bounds for every permitted route.

Wrap concrete providers **below** retry, fallback and cache adapters. Every
invocation atomically reserves against its ledger and all ancestor ledgers.
`Budget::child` gives a turn a local ceiling while charging the shared run.
Completed spend and live reservations share one lock, so concurrent callers
cannot each reserve the same capacity. `Error::BudgetExceeded` carries the
refusing limits, requested amount, completed spend and reservations.

The wrapper caps output tokens and refuses pass-through output cap overrides.
It currently admits text requests only, using complete serialized request
bytes, including response schemas and provider prompt options as a conservative input bound; leave room for provider framing. An
incorrect price or token bound is not a provider-enforced billing cap. Actual
usage exceeding it is retained and stops further calls.

Success reconciles reported usage. Missing or negative charges retain the
reserved cost. Missing token usage retains the token reservation. Provider
errors, dropped streams and cancelled calls retain the complete reservation
because a lost response may have been billed. Cache replays charge nothing.
Streaming items and transport metadata pass through unchanged; a completed
stream reconciles exactly once. Terminal failures charge their reservation
when observed, even if the consumer retains the stream. The physical-attempt
scope remains active while lazy streams are polled.

OpenAI's shared transport tail admits internal request-shape retry attempts
under the invocation's physical-call scope. The initial attempt owns the outer
reservation; each extra attempt reserves again before HTTP and conservatively
retains that reservation. Provider adapters which introduce internal retries
must use this same admission hook before every extra physical send. Ordinary
harness retries enter the wrapper once per attempt.

Tests in `model/budget_tests.rs` cover concurrent admission, ancestor ceilings,
reconciliation, cancellation, streaming completion/drop, cache replay, unknown
charges, overflow, output cap overrides, oversized schemas/provider prompts
and provider-internal retries. The
OpenHuman facade additionally verifies the HTTP boundary with wiremock.
