//! Local runtime detection: fingerprint a port, report only what answers as
//! itself.

use std::time::Duration;

use crate::catalogue;
use crate::config::ProviderDraft;
use crate::policy::EndpointPolicy;
use crate::ports::{DetectOptions, Http, HubRequest};
use crate::taxonomy::LocalRuntime;

/// How to recognise one runtime.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    /// The runtime.
    pub runtime: LocalRuntime,
    /// Its usual port.
    pub port: u16,
    /// A path only this runtime answers.
    pub path: &'static str,
    /// The JSON member (or, for a list, the array member) that proves it.
    pub proof: Proof,
}

/// What in the answer proves the runtime.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proof {
    /// A top-level string member with this name (`version`).
    String(&'static str),
    /// Any of these top-level members exists.
    Any(&'static [&'static str]),
    /// A top-level array member with this name (LM Studio's `data`).
    Array(&'static str),
}

/// The fingerprints, in the port order detection reports them.
pub fn fingerprints() -> &'static [Fingerprint] {
    &[
        Fingerprint {
            runtime: LocalRuntime::Ollama,
            port: 11434,
            path: "/api/version",
            proof: Proof::String("version"),
        },
        Fingerprint {
            runtime: LocalRuntime::LmStudio,
            port: 1234,
            path: "/api/v0/models",
            proof: Proof::Array("data"),
        },
        Fingerprint {
            runtime: LocalRuntime::Vllm,
            port: 8000,
            path: "/version",
            proof: Proof::String("version"),
        },
        Fingerprint {
            runtime: LocalRuntime::LlamaCpp,
            port: 8080,
            path: "/props",
            proof: Proof::Any(&["default_generation_settings", "total_slots", "build_info"]),
        },
    ]
}

fn proves(proof: Proof, body: &[u8]) -> Option<Option<String>> {
    let value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let object = value.as_object()?;
    match proof {
        Proof::String(name) => object
            .get(name)
            .and_then(|v| v.as_str())
            .map(|v| Some(v.to_string())),
        Proof::Any(names) => names
            .iter()
            .any(|n| object.contains_key(*n))
            .then_some(None),
        Proof::Array(name) => object.get(name).filter(|v| v.is_array()).map(|_| None),
    }
}

const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const PROBE_BODY_CAP: usize = 64 * 1024;

/// Asks one fingerprint's port whether its runtime answers. Returns the version
/// when the answer carried one.
pub(crate) async fn fingerprint_at(
    http: &dyn Http,
    policy: &EndpointPolicy,
    root: &str,
    print: &Fingerprint,
) -> Option<Option<String>> {
    let request = HubRequest::get(format!("{}{}", root.trim_end_matches('/'), print.path))
        .with_timeout(PROBE_TIMEOUT)
        .with_body_cap(PROBE_BODY_CAP);
    let response = http.send(request, policy).await.ok()?;
    if !response.is_success() || response.truncated {
        return None;
    }
    proves(print.proof, &response.body)
}

/// Fingerprints every known local port on this machine, skipping the host's
/// own ports. A runtime is reported only if it answers as itself; a port that
/// answers something else (the host's own bind on `8080`) is not reported.
pub async fn detect_local(
    http: &dyn Http,
    policy: &EndpointPolicy,
    options: &DetectOptions,
) -> Vec<ProviderDraft> {
    let mut drafts = Vec::new();
    for print in fingerprints() {
        if options.exclude_ports.contains(&print.port) {
            continue;
        }
        let root = format!("http://localhost:{}", print.port);
        if fingerprint_at(http, policy, &root, print).await.is_none() {
            continue;
        }
        let Some(descriptor) = catalogue::descriptor_for_runtime(print.runtime) else {
            continue;
        };
        let draft = ProviderDraft::new(descriptor.kind.as_str()).with_base_url(root);
        // Ollama and LM Studio have rows of their own. vLLM and llama.cpp share
        // the generic local-OpenAI kind, so they get a label that keeps their
        // slugs apart (and off the reserved `vllm` alias).
        drafts.push(if descriptor.slug() == "local-openai" {
            draft.with_label(format!("{} (port {})", print.runtime, print.port))
        } else {
            draft
        });
    }
    drafts
}

/// What [`Hub::local_status`](crate::Hub::local_status) found.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalRuntimeStatus {
    /// Something answered at the endpoint (its fingerprint, or a model list).
    pub reachable: bool,
    /// The runtime's own fingerprint answered.
    pub fingerprinted: Option<LocalRuntime>,
    /// The version the fingerprint reported, when it did.
    pub version: Option<String>,
    /// How many models the runtime lists, when it was readable.
    pub models: Option<usize>,
}
