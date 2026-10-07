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
