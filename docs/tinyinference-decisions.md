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

The API key is read from `TYPESAFE_API_KEY` or supplied through `ClientConfig`. Keys are redacted from `Debug` output and never included in errors. The client supports TypeSafe, OpenRouter, and OpenJEV endpoints, bounded retries, request and response validation, and explicit per-call failure metadata.

For OpenRouter, construct `ClientConfig::openrouter("<key>")`. Tiny Humans proxy users can use `ClientConfig::tinyhumans_openrouter("<key>")`. For OpenJEV, use `ClientConfig::openjev("<key>")` with `EvaluationRequest::openjev(...)`, which selects OpenJEV's `openjev` model id. Custom endpoints can be configured with `.with_endpoint_url(...)`. For a self-hosted Jev-compatible decision model, use `ClientConfig::self_hosted("<endpoint>", "<key or empty>")` and set `EvaluationRequest::model` to the model the server answers as; the response must echo it.

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
