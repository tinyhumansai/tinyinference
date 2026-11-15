# TinyInference Decisions

`tinyinference-decisions` provides typed Rust clients for TypeSafe AI's System One API and Jev model, and Levanto Sage. Jev requests supply shared state and independent Choice, Score, and Noul questions. Sage supports Yes/No, Choice, Scale, Sort, and Tags decisions, including images, reasoning, grounding, and batches.

The client keeps execution and policy outside the model. A Choice selects only from caller supplied values, a Score rates one described dimension, and a Noul reports the probability of a yes/no condition. Callers own confidence thresholds, escalation, state transitions, and side effects.

```rust,no_run
use std::collections::BTreeMap;
use serde_json::json;
use tinyinference_decisions::{Choice, Client, EvaluationRequest, Question};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let request = EvaluationRequest::jev(
    json!({"ticket": "I was charged twice"}),
    BTreeMap::from([("route".to_owned(), Question::Choice(Choice {
        instructions: json!("Which team should handle this ticket?"),
        criteria: BTreeMap::from([
            ("billing".to_owned(), None),
            ("technical".to_owned(), None),
        ]),
    }))]),
);
let result = Client::from_env()?.evaluate(&request).await?;
println!("{:?}", result.response.answers["route"]);
# Ok(())
# }
```

The API key is read from `TYPESAFE_API_KEY` or supplied through `ClientConfig`. Keys are redacted from `Debug` output and never included in errors. The client supports TypeSafe, OpenRouter, OpenJEV, and Surplus Intelligence endpoints, bounded retries, request and response validation, and explicit per-call failure metadata.

For OpenRouter, construct `ClientConfig::openrouter("<key>")`. Tiny Humans proxy users can use `ClientConfig::tinyhumans_openrouter("<key>")`. For OpenJEV, use `ClientConfig::openjev("<key>")` with `EvaluationRequest::openjev(...)`, which selects OpenJEV's `openjev` model id. For Surplus Intelligence, use `ClientConfig::surplus("<key>")` with an explicit release such as `EvaluationRequest::new("jev-1.14", ...)`; it posts to `https://api.surplusintelligence.ai/v1/decisions`, and the response may name the resolved release (`typesafe/jev-1.14-20261014`), as on OpenRouter. Custom endpoints can be configured with `.with_endpoint_url(...)`. For a self-hosted Jev-compatible decision model, use `ClientConfig::self_hosted("<endpoint>", "<key or empty>")` and set `EvaluationRequest::model` to the model the server answers as; the response must echo it.

The live example spends a real API call:

```sh
TYPESAFE_API_KEY='<key>' cargo run -p tinyinference-decisions --example basic
```

The crate is part of the TinyInference workspace and is licensed GPL-3.0-only.

## Images and Box questions

Jev 1.13 and later read images, and Jev 1.14 (`jev-preview`) adds the Box
question, which returns where one thing is in an image. Attach an image with
`EvaluationRequest::with_image(key, ImageAsset::png(&bytes))` and put `key`
somewhere in `state` (for example `{"page": key}`), so the model knows where it
belongs. Questions point at the state field in backticks.

```rust,no_run
use std::collections::BTreeMap;
use serde_json::json;
use tinyinference_decisions::{
    Answer, BoxQuestion, Client, EvaluationRequest, ImageAsset, Question,
};

# async fn run(png: Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
let request = EvaluationRequest::new(
    "jev-preview",
    json!({"page": "shot-1"}),
    BTreeMap::from([(
        "button".to_owned(),
        Question::Box(BoxQuestion { instructions: json!("the Verify button in `page`") }),
    )]),
)
.with_image("shot-1", ImageAsset::png(&png));
let result = Client::from_env()?.evaluate(&request).await?;
if let Some(Answer::Box(answer)) = result.response.answers.get("button") {
    let (x, y) = answer.edges.centre(); // fractions of width and height
    println!("{x} {y} at confidence {}", answer.confidence);
}
# Ok(())
# }
```

The client sends `TypeSafe-Danger-Zone-Images: true` on any request with an
image or a Box question, as the provider requires. `validate` checks the
documented limits before anything is sent: PNG or JPEG whose header matches the
declared format, 280 to 1568 pixels per side, 16 MiB per image and 32 MiB in
all, and every asset key referenced by the state or a question. Images are
neither resized nor rotated; that stays with the caller. A Box answer is always
one box, so ask a Noul alongside it when the thing may be absent.

## Emulated Box questions

Some providers serve Jev without the Box question: Surplus Intelligence
answers a request containing one with 400 (`questions.<id> has a type that is
not one of noul | choice | score`), while Noul questions about images work.
`Provider::answers_box()` reports this, and `ClientConfig::box_support()`
resolves it to `BoxSupport::Native` or `BoxSupport::Emulated(BoxEmulation)`.
`ClientConfig::with_box_support(...)` overrides the provider's default in
either direction. Callers do not change anything else: they send Box
questions and receive `Answer::Box` answers whose ids match their questions.

Emulation works inside `Client::evaluate`:

1. Each Box question's image is the asset whose key its instructions mention;
   else the asset held by a top-level state field it names in backticks
   (`` `page` `` with `{"page": "shot-1"}`); else the request's only image.
   Anything else is an `Error::InvalidRequest` naming the question, raised
   before any request is sent.
2. The image is decoded and cut into a `grid` by `grid` set of cells (4 by 4
   by default). Each cell is upscaled (bilinear) until its shorter side is
   280 pixels, encoded as PNG, and asked about as a Noul: does this tile show
   all or part of the original instructions?
3. Probes are packed into requests holding at most 32 MiB of images; a crop is
   at most 16 MiB. They share a request with the caller's other questions
   unless that request carries the image the crops were cut from, so a tile
   is never judged next to the whole picture. Sub-requests run one after
   another, each with the normal retry policy and response validation.
4. The box is the bounding rectangle of the cells with probability 0.5 or more
   that connect (diagonals included) to the best cell, so a second instance
   elsewhere does not stretch it. With no such cell it is the best cell
   alone. Confidence is the best cell's probability. Edges are fractions of
   the original image.
5. With `refine` (the default), the box is cut into the same grid again and
   asked about in one more round. The refined box replaces the first only when
   one of its cells reaches 0.5. Refinement is skipped when the box already
   covers the whole image or its cells would be under 32 pixels; a box more
   than 4 times longer than wide (or the reverse) is first grown about its
   centre so its cells can be upscaled within the 1568-pixel maximum.

Probe question ids and asset keys start with `__tinyinference_box`, which
emulating requests reserve. The result's `usage` sums every sub-request (a
field stays absent only when no sub-request reported it), `attempts` counts
every HTTP attempt, `latency` covers them all, and `request_id` is the first
one returned. A Box question costs up to `2 * grid * grid` Noul questions with
an image each, and up to three sequential HTTP requests in the usual case
(the caller's other questions with the original image, then each round). The
precision is one cell, or one cell of a cell after refinement; this is a
coarse locator, not a native detector.

## Local decision models

Use `ClientConfig::self_hosted` for a locally served OpenJEV, Decider, Surogate
decisions-v1, or another server implementing the Jev-compatible wire contract.
The URL is the complete POST endpoint: `/v1/systemone`, `/v1/decisions`, or a
server-specific path. The client adds no path and follows no redirects.
HTTPS is supported; plain HTTP requires a literal loopback address such as
`127.0.0.1` or `[::1]`. URLs cannot contain credentials, queries, or fragments.

```rust,no_run
use std::collections::BTreeMap;
use serde_json::json;
use tinyinference_decisions::{Client, ClientConfig, EvaluationRequest, Noul, Question};

# async fn run() -> tinyinference_decisions::Result<()> {
let client = Client::new(ClientConfig::self_hosted(
    "http://127.0.0.1:8000/v1/systemone",
    "", // No Authorization header for a server without authentication.
))?;
let request = EvaluationRequest::new(
    "openjev", // Use the exact model id reported by your server.
    json!({"ticket": "I was charged twice"}),
    BTreeMap::from([("duplicate_charge".into(), Question::Noul(Noul {
        instructions: json!("Was the customer charged twice?"),
        criteria: None,
    }))]),
);
let result = client.evaluate(&request).await?;
println!("{:?}", result.response.answers["duplicate_charge"]);
# Ok(())
# }
```

The server must return the requested model id, all requested question ids,
typed answers, and a `usage` object (which may be empty). Choice distributions,
Score legends and weighted scores, and Noul probabilities use the same
validation as hosted Jev. The client preserves server probabilities unchanged.
An OpenAI chat-completions endpoint or Ollama chat endpoint needs a decisions
adapter; a model name alone does not make it serve this protocol.

Run the three-question live smoke example against your own server:

```sh
cargo run -p tinyinference-decisions --example live_local -- \
  http://127.0.0.1:8000/v1/systemone openjev
```

It evaluates two synthetic tickets with Choice, Noul, and Score, printing
validated answers, usage, attempts, and latency. Set `LOCAL_DECISIONS_API_KEY`
when your server needs a bearer token. Supply the server's own model id in
place of `openjev`. Runtime installation and model downloads stay outside
the library.

## Levanto Sage

Sage is available through `tinyinference_decisions::sage::SageClient`. Its
`DecisionQuestion` and `DecisionResponse` types preserve each decision kind's
result, including `None` when Sage is unsure. `decide_batch` groups questions
by shared content, while `estimate_decision` and `estimate_batch` predict billed
input tokens without running the model. Image content, reasoning modes,
grounding, and choice latency modes follow Sage's native API.

The Sage client accepts an explicit API key. See the root README for a complete
example. Rust imports use `tinyinference_decisions` because Rust identifiers
cannot contain hyphens; the Cargo package is `tinyinference-decisions`.

An opt-in live example exercises readiness, model listing, token estimation,
reasoning, image input, and a two-question batch with synthetic content. It
uses decision allowance and reads the key from `SAGE_API_KEY`:

```sh
cargo run -p tinyinference-decisions --example live_sage
```
