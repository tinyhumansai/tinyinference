# tinyinference-hub

One provider hub for every TinyInference host (OpenCompany, OpenHuman, the TUI,
CLIs, future servers).

Two hosts used to implement provider management separately: two catalogues of
the same hosted vendors, two error classifiers, two managed paths, and safety
invariants (SSRF policy, tenant-scoped model cache, "only a rejected key rolls a
new key back") that only one had. This crate is the shared answer. It is a
**leaf**: it depends on `tinyinference-llm` (plus small utility crates), nothing
depends on it, and no existing public item of any other crate changed.

## Plug it in (four ports)

```rust
use tinyinference_hub::ports::memory::{MemoryConfig, MemoryCredentials};
use tinyinference_hub::ports::SystemClock;
use tinyinference_hub::{
    ConnectOptions, EndpointPolicy, Hub, ModelId, ProviderDraft, ScopeKey, Secret, TurnQuery,
};

let hub = Hub::builder()
    .credentials(MemoryCredentials::new()) // wrap your keychain or vault
    .config(MemoryConfig::new())           // a file, a database row
    .http(my_http)                         // ReqwestHttp (feature `http-reqwest`) or your own
    .clock(SystemClock)
    .policy(EndpointPolicy::desktop())     // `hosted()` in a multi-tenant server
    .build()?;

let me = ScopeKey::new("user:local");
let draft = ProviderDraft::new("openai")
    .with_key(Secret::new(std::env::var("OPENAI_API_KEY")?))
    .with_model(ModelId::parse("gpt-5")?);
hub.connect(&me, draft, ConnectOptions::default()).await?;   // checks the key, keeps the row
let turn = hub.resolve_for_turn(&me, &TurnQuery::new()).await?; // no credential inside
let model = hub.chat_model(&me, &turn).await?;               // Arc<dyn ChatModel<()>>
```

`examples/minimal_host.rs` is a runnable, network-free walk-through
(`cargo run -p tinyinference-hub --example minimal_host --features testing`); its
output is a golden file.

| Port | What it is |
|---|---|
| `CredentialStore` | where keys live. `get`/`set`/`delete`; an error means *unreadable*, never *no key* |
| `ConfigStore` | the persisted `HubConfig`, with compare-and-swap |
| `Http` | the transport for probes and catalogs; applies the endpoint policy on every redirect hop and pins the address it checked |
| `Clock` | the only source of time |

Optional: `HealthStore` and `EventSink` (in-memory or no-op by default),
`TokenSource` (a rotating managed token), `EnvSource`, `Detector`, `UsageQuery`
(references only your host knows about), `ProcessSpawner` (feature `cli`),
`ModelFactory` (how the model behind a turn is built), `ModelMetadataSource`.

## What the hub does

| Area | Operations |
|---|---|
| Providers | `connect` (add, check, roll back on a rejected key), `add`, `edit`, `remove`, `set_enabled`, `set_key`, `clear_key` |
| Checking | `probe_draft` (nothing stored), `test` at `KeyOnly`, `Catalog` or `Completion` depth, `list_models` (cached, single-flight, `refresh`), `retest_down` |
| State | `health`, `status` (managed first), `record_outcome` |
| Choosing | `set_default`, `clear_default`, `pin_agent`, `set_workload_route`, `resolve_for_turn` |
| Using | `chat_model` (resolves the credential on every call; feeds real turns back into health) |
| Discovering | `detect` (local runtimes by fingerprint, provider keys in the environment; never persisted), `local_status` |
| Feature-gated | `cli_readiness` (`cli`), `oauth_start`/`oauth_complete` (`oauth`, types only) |

Every change is *load, check the guards, save with the version loaded*, retried
on a lost compare-and-swap with the guards re-run. The state machine is written
out once, in the `hub` module docs.

## Guarantees

- **A credential is never printed or stored on a record.** `Secret` redacts in
  `Debug` and `Display` and has no `Serialize`; `ProviderRecord` has no
  credential field and refuses credential-shaped fields at any depth on load.
- **Raw upstream error text is log-only.** It never reaches `Display`, `Debug`
  or `HubError::user_message`.
- **Only a rejected credential rolls back an add** (a local runtime also when it
  is unreachable), never with `add_anyway`, and the previous key is restored.
- **SSRF policy on every URL and address.** Link-local and metadata addresses are
  refused under every policy; alternative IPv4 spellings, `localhost` by name and
  IPv4 embedded in IPv6 get the answer for the address a client would connect to;
  a credential never crosses an origin.
- **Tenants never see each other.** A model list cached for one scope is never
  served to another; a rejected key or a `403` is never remembered.
- **Signed out is a typed state**, not an empty list.
- **The default is never silently changed.** Removing a provider leaves the
  default, every pin and every route in place, where they fail closed.

## Migration readers

`import::oc` and `import::oh` are pure functions from the stored shapes of
OpenCompany and OpenHuman (plain structs that also `Deserialize` from the stored
JSON) to a `HubConfig` plus a `LossReport` naming every step that was dropped,
ambiguous, normalised, synthesised or fail-closed. `route::legacy_oc` and
`route::legacy_oh` read and write both string route grammars. Nothing is written
back to either host by the hub.

## Modules

| Module | What it gives you |
|---|---|
| `hub` | `Hub`, `HubBuilder`, `ManagedConfig`, `HubPolicy` and the value types the operations return |
| `error` | `HubError`, `ReasonCode`, `Retry`, `classify` |
| `ids`, `Secret`, `LogOnly` | validated identifiers and redacting wrappers |
| `taxonomy`, `catalogue`, `descriptor` | every built-in kind as data; `LocalRuntime` reconciles the four local-runtime enums |
| `policy`, `endpoint` | `EndpointPolicy` presets, redirect and address checks, endpoint redaction |
| `ports` | the traits, in-memory defaults, `follow_redirects`, `ReqwestHttp` (feature) |
| `credential` | the ordered `CredentialChain` and its sources |
| `catalog`, `probe`, `health`, `kinds` | parsers and cache, three-depth probes, folded health, kind drivers |
| `config`, `route`, `import` | the persisted document, structured routes, migration readers |
| `client`, `detect`, `cli`, `oauth` | the `ChatModel`, detection, CLI readiness, OAuth types |
| `testkit` (feature `testing`) | `FakeClock`, `ScriptedHttp`, `MemoryPorts`, `ScenarioRunner`, the contract suite |

## Features

`default = []`.

| Feature | Effect |
|---|---|
| `testing` | the no-socket simulation kit (also always available to this crate's own tests) |
| `cli` | the `ProcessSpawner` port and `Hub::cli_readiness`; a CLI login is a route target, never a record |
| `oauth` | OAuth **types only**; every flow answers `Unsupported`. Claude subscription OAuth is excluded permanently: Claude is a CLI login |
| `http-reqwest` | `ReqwestHttp`: no automatic redirects, DNS resolved once per hop, every address checked, connection pinned to the checked addresses, proxies ignored |
| `local-bridge` | `From`/`TryFrom` between `LocalRuntime` and `tinyinference-local`'s enums |

## Known limits

- Turn traffic uses `tinyinference-llm`'s own transport, so per-redirect policy
  and address pinning cover probes and catalogs, not chat turns. The resolved
  endpoint is re-checked against the policy at `resolve_for_turn`.
- `ReqwestHttp` builds a `reqwest` client per hop (no connection reuse) and
  ignores proxy environment variables, because a proxy would bypass the pinned
  address; a host behind a corporate proxy supplies its own `Http`.
- `ReqwestHttp`'s two network touch points (`SystemResolver::resolve` and
  `ReqwestExecutor::execute`) are not exercised by this crate's tests, which open
  no sockets; everything around them is.
- The managed provider's backend differs between OpenCompany and OpenHuman; the
  host names its endpoint and catalog shape.
- An ephemeral route parses but is not resolved by the hub.
- Every operation that touches a provider's key slot holds one lock per
  `(scope, slug)`, so within **one** `Hub` (and its clones) a key change, an
  origin move, an add and a removal of the same provider are ordered. Two hubs
  over one store are ordered only by the stores' own guarantees: there a
  `set_key` racing a `remove` in the instant between two stores can still leave
  one orphan key slot; the hub re-checks and cleans up, which narrows the window
  to two adjacent calls but cannot close it across two stores.
- An edit that moves the endpoint to another origin **with** a key deletes the
  old key, commits the record at the new origin *disabled*, writes the new key,
  then switches the record back on, so no credential the hub stores can meet an
  origin it was not entered for, and a credential a host source supplies is never
  offered to the new origin while the record is parked (a route to it fails
  closed; a `HubModel` kept from before the move refuses with `stale_route`; a
  probe or listing waits for the move). A failure rolls back in the order
  empty-slot, record back, old key back. If only the last step (switching it back
  on) fails the edit **succeeds with a warning** (`SavedWithWarning`; call
  `set_enabled`). If the record cannot be moved back it is left disabled at the
  new origin holding at most the key entered for it: the caller gets the reason
  the move failed, the stuck state is logged and announced (`ProviderEdited`,
  `KeyChanged`), and if it has no key one must be entered (`set_key`) before the
  provider is tested, listed or switched on.
- Operations that change two stores (a key and a record: `add`, `edit`, `remove`,
  `set_key`, `clear_key`, and above all an endpoint move) are **not
  cancellation-safe**. A future dropped between their awaits (a request timeout,
  a `select!`) leaves the state a failure would leave, minus the undo and the
  announcement. Run them to completion (spawn them; do not race them against a
  timer). After a failed endpoint move that could not be undone the provider is
  disabled at the new endpoint holding at most the new key: if it has none, enter
  a key (`set_key`) before switching it on.
- Failover, budgets, cooldowns and streaming health are a later crate; the hub
  exposes the signals (`record_outcome`, health, `Retry`).

## Testing

`cargo test -p tinyinference-hub --all-features` runs the unit, contract,
property, golden, compat and simulated-e2e suites; `PROPTEST_CASES=6000` widens
the property tests. `testkit::ScenarioRunner` plays seeded random sessions over
a full `Hub` and checks eleven invariants after every step (the eleventh: a
credential is only ever sent to the origin it was entered for, including while
an edit is parked mid-way by the `Hold` interleaving hook); a seed that ever
failed is kept in `tests/golden/sim_seeds.txt` and replayed forever. The long
soak (`tests/sim_soak.rs`: 20,000 seeds, each plain and with flaky stores) is `#[ignore]`d: run it with
`cargo test -p tinyinference-hub --all-features --release --test sim_soak -- --ignored`.
