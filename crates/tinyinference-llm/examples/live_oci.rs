//! Live smoke test: the OCI Generative AI provider against the real service.
//!
//! Network- and credential-gated. Runs a completion, a streamed completion,
//! and a streamed tool round trip for each case; exits cleanly (status 0,
//! "skipped") unless `OCI_LIVE_PROFILE` or `OCI_LIVE_API_KEY` is set, and
//! with status 1 when any check fails.
//!
//! ```bash
//! OCI_LIVE_PROFILE=MY_PROFILE OCI_LIVE_REGION=us-chicago-1 \
//! OCI_LIVE_COMPARTMENT=ocid1.compartment... \
//! OCI_LIVE_CASES='native_generic=openai.gpt-5.5,native_cohere=cohere.command-a-03-2025' \
//!   cargo run -p tinyinference-llm --example live_oci
//! ```
//!
//! | Variable               | Meaning                                                         |
//! | ---------------------- | --------------------------------------------------------------- |
//! | `OCI_LIVE_PROFILE`     | `~/.oci/config` profile (API key or session token)              |
//! | `OCI_LIVE_CONFIG`      | config file path, default `~/.oci/config`                       |
//! | `OCI_LIVE_API_KEY`     | GenAI API key; used instead of the profile when set             |
//! | `OCI_LIVE_REGION`      | region, default the profile's region                            |
//! | `OCI_LIVE_COMPARTMENT` | compartment OCID, default the profile's tenancy                 |
//! | `OCI_LIVE_PROJECT`     | GenAI project OCID, required by the `openai` (`/openai/v1`) base |
//! | `OCI_LIVE_CASES`       | comma list of `<transport>=<model>`                             |
//!
//! Transports: `native_generic`, `native_cohere`, `actions` (`/20231130/actions/v1`),
//! `openai` (`/openai/v1`).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde_json::json;
use tinyinference_llm::model::collect_model_stream;
use tinyinference_llm::providers::oci::{
    OciApiFormat, OciAuth, OciConfig, OciOpenAiBase, OciTransport, build_oci_model, load_profile,
};
use tinyinference_llm::{ChatModel, Message, ModelRequest, ToolSchema};

const DEFAULT_CASES: &str = "native_generic=meta.llama-3.3-70b-instruct,\
native_cohere=cohere.command-a-03-2025";

struct Env {
    auth: OciAuth,
    region: Option<String>,
    compartment: Option<String>,
    project: Option<String>,
    cases: Vec<(String, String)>,
}

fn env() -> Option<Env> {
    let var = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
    let (auth, profile_region, tenancy) = if let Some(bearer) = var("OCI_LIVE_API_KEY") {
        (OciAuth::ApiKey { bearer }, None, None)
    } else {
        let profile = var("OCI_LIVE_PROFILE")?;
        let path = var("OCI_LIVE_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var("HOME").expect("HOME")).join(".oci/config")
            });
        let loaded = load_profile(&path, &profile).expect("load OCI profile");
        (loaded.auth, loaded.region, loaded.tenancy)
    };
    let cases = var("OCI_LIVE_CASES")
        .unwrap_or_else(|| DEFAULT_CASES.to_string())
        .split(',')
        .filter_map(|pair| {
            let (transport, model) = pair.trim().split_once('=')?;
            Some((transport.trim().to_string(), model.trim().to_string()))
        })
        .collect();
    Some(Env {
        auth,
        region: var("OCI_LIVE_REGION").or(profile_region),
        compartment: var("OCI_LIVE_COMPARTMENT").or(tenancy),
        project: var("OCI_LIVE_PROJECT"),
        cases,
    })
}

fn transport(name: &str) -> OciTransport {
    match name {
        "native_generic" => OciTransport::Native {
            api_format: OciApiFormat::Generic,
        },
        "native_cohere" => OciTransport::Native {
            api_format: OciApiFormat::Cohere,
        },
        "actions" => OciTransport::OpenAiCompat {
            base: OciOpenAiBase::Actions,
        },
        "openai" => OciTransport::OpenAiCompat {
            base: OciOpenAiBase::OpenAi,
        },
        other => panic!("unknown OCI_LIVE_CASES transport `{other}`"),
    }
}

fn model_for(env: &Env, transport_name: &str, model: &str) -> Arc<dyn ChatModel<()>> {
    let mut config = OciConfig::new(env.auth.clone(), model);
    config.region = env.region.clone();
    config.compartment_id = env.compartment.clone();
    config.project_id = env.project.clone();
    config.transport = transport(transport_name);
    build_oci_model(config).expect("build OCI model")
}

fn weather_tool() -> ToolSchema {
    ToolSchema::new(
        "get_weather",
        "Get the current weather for a city.",
        json!({
            "type": "object",
            "properties": { "city": { "type": "string", "description": "City name" } },
            "required": ["city"]
        }),
    )
}

/// Sends a temperature the way a host default does, so the provider's
/// per-model sampling rules are exercised against the live service.
fn request(messages: Vec<Message>) -> ModelRequest {
    ModelRequest {
        messages,
        max_tokens: Some(512),
        temperature: Some(0.7),
        ..ModelRequest::default()
    }
}

/// One plain completion: non-empty text and reported usage.
async fn check_invoke(model: &Arc<dyn ChatModel<()>>) -> Result<String, String> {
    let response = model
        .invoke(
            &(),
            request(vec![Message::user("Reply with exactly the word: pong")]),
        )
        .await
        .map_err(|e| e.to_string())?;
    let text = Message::Assistant(response.message.clone()).text();
    if !text.to_lowercase().contains("pong") {
        return Err(format!("unexpected text {text:?}"));
    }
    let usage = response.usage.ok_or("no usage reported")?;
    Ok(format!("text={text:?} usage={usage:?}"))
}

/// Streaming: the accumulated stream yields text and a finish reason.
async fn check_stream(model: &Arc<dyn ChatModel<()>>) -> Result<String, String> {
    let stream = model
        .stream(
            &(),
            request(vec![Message::user("Count from 1 to 5, comma separated.")]),
        )
        .await
        .map_err(|e| e.to_string())?;
    let response = collect_model_stream(stream)
        .await
        .map_err(|e| e.to_string())?;
    let text = Message::Assistant(response.message.clone()).text();
    if !text.contains('3') {
        return Err(format!("stream text missing content: {text:?}"));
    }
    Ok(format!(
        "text={text:?} finish={:?} usage={}",
        response.finish_reason,
        response.usage.is_some()
    ))
}

/// Tool round trip, streamed: the model calls the tool, we answer, it uses it.
async fn check_tools(model: &Arc<dyn ChatModel<()>>) -> Result<String, String> {
    let ask = Message::user("What is the weather in Paris? Use the get_weather tool.");
    let mut first = request(vec![ask.clone()]);
    first.tools = vec![weather_tool()];
    let stream = model.stream(&(), first).await.map_err(|e| e.to_string())?;
    let response = collect_model_stream(stream)
        .await
        .map_err(|e| e.to_string())?;
    let call = response
        .message
        .tool_calls
        .first()
        .cloned()
        .ok_or("model made no tool call")?;
    if call.name != "get_weather" || call.invalid.is_some() {
        return Err(format!("bad tool call {call:?}"));
    }
    let city = call.arguments["city"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    if !city.to_lowercase().contains("paris") {
        return Err(format!("tool arguments missing city: {}", call.arguments));
    }

    let mut second = request(vec![
        ask,
        Message::Assistant(response.message.clone()),
        Message::tool(call.id.clone(), r#"{"temperature_c": 17, "sky": "cloudy"}"#),
    ]);
    second.tools = vec![weather_tool()];
    let answer = model.invoke(&(), second).await.map_err(|e| e.to_string())?;
    let text = Message::Assistant(answer.message).text();
    if !text.contains("17") {
        return Err(format!("final answer ignored the tool result: {text:?}"));
    }
    Ok(format!("args={} answer={text:?}", call.arguments))
}

#[tokio::main]
async fn main() {
    let Some(env) = env() else {
        eprintln!("[oci_live] skipped: set OCI_LIVE_PROFILE or OCI_LIVE_API_KEY");
        return;
    };
    tinyinference_llm::allow_network_models();
    let mut failures = Vec::new();
    for (transport_name, model_id) in &env.cases {
        let model = model_for(&env, transport_name, model_id);
        for (check, outcome) in [
            ("invoke", timed(check_invoke(&model)).await),
            ("stream", timed(check_stream(&model)).await),
            ("tools", timed(check_tools(&model)).await),
        ] {
            let (result, ms) = outcome;
            match result {
                Ok(detail) => {
                    println!("[oci_live] PASS {transport_name} {model_id} {check} {ms}ms {detail}")
                }
                Err(error) => {
                    println!("[oci_live] FAIL {transport_name} {model_id} {check} {ms}ms {error}");
                    failures.push(format!("{transport_name} {model_id} {check}: {error}"));
                }
            }
        }
    }
    if !failures.is_empty() {
        eprintln!("OCI live failures:\n{}", failures.join("\n"));
        std::process::exit(1);
    }
    println!("[oci_live] all checks passed");
}

async fn timed<F: std::future::Future<Output = Result<String, String>>>(
    future: F,
) -> (Result<String, String>, u128) {
    let start = Instant::now();
    let result = future.await;
    (result, start.elapsed().as_millis())
}
