//! OpenHuman's provider strings, read and written.
//!
//! OpenHuman names a route with a string: `openhuman`, `cloud`, `<slug>:<model>`
//! with an optional `@<temperature>` suffix, local prefixes (`ollama:`,
//! `lmstudio:`, `mlx:`, `omlx:`, `local-openai:` and their aliases),
//! `claude_agent_sdk[:model]`, `claude-code:<model>`, `ephemeral-route:<model>`,
//! `byok-inference:<model>` and the sentinel `__byok_incomplete__`. Several of
//! those readings are lossy (06-migration-mapping, cases 1-5, 7, 8); the parser
//! never guesses silently. Every reading that is not exact appends a
//! [`LossEntry`], and a string that cannot be resolved yields no route and a
//! `FailClosed` entry, so the turn fails instead of guessing.

use std::collections::BTreeMap;

use crate::ids::{ModelId, Slug};
use crate::import::{LossEntry, LossKind};
use crate::taxonomy::{CliKind, LocalRuntime};

use super::types::{ProviderRoute, RouteTarget, Temperature};

/// The sentinel OpenHuman writes when a BYOK setup is half finished.
pub const BYOK_INCOMPLETE: &str = "__byok_incomplete__";

/// What the parser needs to know about the surrounding OpenHuman config.
#[non_exhaustive]
#[derive(Clone, Debug, Default)]
pub struct OhContext {
    /// `local_ai.provider`: which runtime `ollama:` really means, because
    /// OpenHuman's UI writes every local reference as `ollama:` (case 2).
    pub local_ai_runtime: Option<LocalRuntime>,
    /// Each cloud row's `default_model`, used to replace an abstract
    /// `hint:<role>` model (case 5).
    pub default_models: BTreeMap<Slug, ModelId>,
}

impl OhContext {
    /// An empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the runtime `local_ai.provider` names.
    #[must_use]
    pub fn with_local_ai_runtime(mut self, runtime: LocalRuntime) -> Self {
        self.local_ai_runtime = Some(runtime);
        self
    }

    /// Records a row's default model.
    #[must_use]
    pub fn with_default_model(mut self, slug: Slug, model: ModelId) -> Self {
        self.default_models.insert(slug, model);
        self
    }
}

/// The result of reading one provider string.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct OhParsed {
    /// The route, or `None` when the string cannot be resolved (fail closed).
    pub route: Option<ProviderRoute>,
    /// Every lossy step, keyed by the string that was read.
    pub loss: Vec<LossEntry>,
}

fn local_runtime_of(prefix: &str) -> Option<LocalRuntime> {
    // `openai` is deliberately not here: the hub never treats a bare or
    // prefixed `openai` as a local runtime except through the explicit
    // ambiguity rule in `parse` (case 3).
    LocalRuntime::parse_loose(prefix)
}

fn prefix_of(runtime: LocalRuntime) -> &'static str {
    match runtime {
        LocalRuntime::Ollama => "ollama",
        LocalRuntime::LmStudio => "lmstudio",
        LocalRuntime::Mlx => "mlx",
        LocalRuntime::Omlx => "omlx",
        // OpenHuman reads `vllm` and `llamacpp` as its local-openai runtime, so
        // writing them keeps the hub's more specific runtime through a round trip.
        LocalRuntime::Vllm => "vllm",
        LocalRuntime::LlamaCpp => "llamacpp",
        _ => "local-openai",
    }
}

/// Splits `model@temp` at the **last** `@`, only when the tail is a finite
/// number (OpenHuman `routing.rs:196-210`).
fn split_temperature(rest: &str) -> (&str, Option<Temperature>) {
    if let Some((model, tail)) = rest.rsplit_once('@')
        && let Ok(value) = tail.trim().parse::<f64>()
        && value.is_finite()
        && let Some(temperature) = Temperature::new(value as f32)
    {
        return (model, Some(temperature));
    }
    (rest, None)
}

/// Reads an OpenHuman provider string.
pub fn parse(raw: &str, context: &OhContext) -> OhParsed {
    let key = raw.trim().to_string();
    let mut loss: Vec<LossEntry> = Vec::new();
    let fail = |loss: &mut Vec<LossEntry>, why: &str| {
        loss.push(LossEntry::new(key.clone(), LossKind::FailClosed, why));
    };
    let text = raw.trim();

    if text == BYOK_INCOMPLETE {
        fail(&mut loss, "an unfinished BYOK setup names no provider");
        return OhParsed { route: None, loss };
    }
    if text.is_empty() || text.eq_ignore_ascii_case("cloud") {
        // The primary cloud is resolved once at import from `primary_cloud`.
        return finish(key, ProviderRoute::default_route(), loss);
    }
    if text.eq_ignore_ascii_case("openhuman") {
        return finish(key, ProviderRoute::new(RouteTarget::Managed), loss);
    }

    let (head, rest) = match text.split_once(':') {
        Some((head, rest)) => (head.trim(), Some(rest.trim())),
        None => (text, None),
    };
    let head_lower = head.to_ascii_lowercase();

    // Bare strings: only a local alias resolves; `openai` is OpenHuman's trap.
    let Some(rest) = rest else {
        return parse_bare(key, &head_lower, loss);
    };

    let (model_text, temperature) = split_temperature(rest);
    if model_text.contains('@') {
        loss.push(LossEntry::new(
            key.clone(),
            LossKind::Ambiguous,
            "the model id contains an @ that is not a temperature suffix; read as part of the id",
        ));
    }

    let (target, mut model_text) = match head_lower.as_str() {
        "claude_agent_sdk" | "claude-code" => {
            if temperature.is_some() {
                loss.push(LossEntry::new(
                    key.clone(),
                    LossKind::Dropped,
                    "a CLI login takes no temperature",
                ));
            }
            (RouteTarget::Cli(CliKind::ClaudeCode), model_text)
        }
        "ephemeral-route" => (RouteTarget::Ephemeral, model_text),
        "byok-inference" => match Slug::parse("byok-inference") {
            Ok(slug) => (RouteTarget::Provider(slug), model_text),
            Err(_) => {
                fail(&mut loss, "the synthetic BYOK slug is invalid");
                return OhParsed { route: None, loss };
            }
        },
        other => {
            if let Some(runtime) = local_runtime_of(other) {
                let mut runtime = runtime;
                if runtime == LocalRuntime::Ollama
                    && let Some(actual) = context
                        .local_ai_runtime
                        .filter(|actual| *actual != LocalRuntime::Ollama)
                {
                    // The UI writes every local reference as `ollama:`.
                    runtime = actual;
                    loss.push(LossEntry::new(
                        key.clone(),
                        LossKind::Ambiguous,
                        "`ollama:` is what OpenHuman's UI writes for any local runtime; read as the configured local_ai runtime",
                    ));
                }
                (RouteTarget::Local(Some(runtime)), model_text)
            } else {
                // `openai:<model>` is the hosted OpenAI row in OpenHuman's prefix
                // form; only the *bare* string is the local trap (case 3).
                match Slug::parse(other) {
                    Ok(slug) => (RouteTarget::Provider(slug), model_text),
                    Err(_) => {
                        fail(&mut loss, "the provider prefix is not a valid slug");
                        return OhParsed { route: None, loss };
                    }
                }
            }
        }
    };

    // Abstract models: `hint:<role>` and legacy tiers become the row's default
    // model when there is one (case 5).
    let mut hint_replaced: Option<ModelId> = None;
    if model_text.starts_with("hint:") {
        if let RouteTarget::Provider(slug) = &target
            && let Some(default) = context.default_models.get(slug)
        {
            hint_replaced = Some(default.clone());
            loss.push(LossEntry::new(
                key.clone(),
                LossKind::Normalised,
                "an abstract hint model was replaced by the provider's default model",
            ));
        } else {
            loss.push(LossEntry::new(
                key.clone(),
                LossKind::FailClosed,
                "an abstract hint model has no default model to stand in for it",
            ));
            model_text = "";
        }
    }

    let mut route = ProviderRoute::new(target);
    route.temperature = temperature.filter(|_| !matches!(route.target, RouteTarget::Cli(_)));
    if let Some(model) = hint_replaced {
        route.model = Some(model);
    } else if !model_text.is_empty() {
        match ModelId::parse(model_text) {
            Ok(model) => route.model = Some(model),
            Err(_) => {
                fail(&mut loss, "the model id is not valid");
                return OhParsed { route: None, loss };
            }
        }
    }
    finish(key, route, loss)
}

fn parse_bare(key: String, head: &str, mut loss: Vec<LossEntry>) -> OhParsed {
    if head == "openai" {
        // OpenHuman's backend reads a bare `openai` as its local-openai runtime
        // (`profile.rs:255-265`); the hub follows what actually ran.
        loss.push(LossEntry::new(
            key.clone(),
            LossKind::Ambiguous,
            "a bare `openai` ran as OpenHuman's local OpenAI-compatible runtime, not as the hosted OpenAI row",
        ));
        return finish(
            key,
            ProviderRoute::new(RouteTarget::Local(Some(LocalRuntime::OpenAiCompatible))),
            loss,
        );
    }
    if let Some(runtime) = local_runtime_of(head) {
        return finish(
            key,
            ProviderRoute::new(RouteTarget::Local(Some(runtime))),
            loss,
        );
    }
    if head == "claude-code" || head == "claude_agent_sdk" {
        return finish(
            key,
            ProviderRoute::new(RouteTarget::Cli(CliKind::ClaudeCode)),
            loss,
        );
    }
    if head == "ephemeral-route" {
        return finish(key, ProviderRoute::new(RouteTarget::Ephemeral), loss);
    }
    if head == "byok-inference"
        && let Ok(slug) = Slug::parse("byok-inference")
    {
        return finish(key, ProviderRoute::provider(slug), loss);
    }
    loss.push(LossEntry::new(
        key,
        LossKind::FailClosed,
        "a bare string that names no local runtime cannot be resolved",
    ));
    OhParsed { route: None, loss }
}

/// Adds a `Normalised` entry when writing the route back does not reproduce the
/// input, so "parse then write" is the identity except where the report says
/// otherwise.
fn finish(key: String, route: ProviderRoute, mut loss: Vec<LossEntry>) -> OhParsed {
    let canonical = to_string(&route);
    let unexplained = canonical.as_deref() != Some(key.as_str())
        && !loss.iter().any(|e| {
            matches!(
                e.kind,
                LossKind::Ambiguous
                    | LossKind::Dropped
                    | LossKind::FailClosed
                    | LossKind::Normalised
            )
        });
    if unexplained {
        loss.push(LossEntry::new(
            key,
            LossKind::Normalised,
            "rewritten into the canonical spelling",
        ));
    }
    OhParsed {
        route: Some(route),
        loss,
    }
}

/// Writes a route as an OpenHuman provider string, or `None` for a route the
/// grammar cannot say: a default or managed route with a model, and a named
/// provider with no model (a bare provider name is OpenHuman's unresolvable
/// trap, so writing one would not read back as the same route).
///
/// Two routes are written in the spelling OpenHuman itself uses, and so do not
/// read back **identically**; both are deliberate and tested:
///
/// * [`RouteTarget::Local(None)`](RouteTarget::Local) ("whichever local runtime
///   is configured") is written as `ollama[:model]`, which is what OpenHuman's UI
///   writes for any local runtime and which OpenHuman resolves through
///   `local_ai.provider`. The grammar has no other way to say it. Read back with
///   no context it is `Local(Some(Ollama))`; read back with an [`OhContext`] that
///   names the configured runtime it is that runtime.
/// * a CLI login's temperature is **not written**: a CLI login takes none
///   (`parse` drops it with a `Dropped` entry), so the string that reads back to
///   the same route is the one without it.
pub fn to_string(route: &ProviderRoute) -> Option<String> {
    let model = route.model.as_ref().map(ModelId::as_str);
    let write = |head: &str, temperature: Option<Temperature>| -> String {
        let mut out = match model {
            Some(model) => format!("{head}:{model}"),
            None => head.to_string(),
        };
        if let Some(temperature) = temperature {
            out.push_str(&format!("@{}", temperature.get()));
        }
        out
    };
    let with_model = |head: &str| write(head, route.temperature);
    Some(match &route.target {
        RouteTarget::Default => {
            if model.is_some() || route.temperature.is_some() {
                return None;
            }
            String::new()
        }
        RouteTarget::Managed => {
            if model.is_some() || route.temperature.is_some() {
                return None;
            }
            "openhuman".to_string()
        }
        RouteTarget::Provider(slug) => {
            model?;
            // A slug that is a local runtime's prefix would read back as that runtime.
            if local_runtime_of(&slug.as_str().to_ascii_lowercase()).is_some() {
                return None;
            }
            with_model(slug.as_str())
        }
        RouteTarget::Local(Some(runtime)) => with_model(prefix_of(*runtime)),
        RouteTarget::Local(None) => with_model("ollama"),
        // A CLI login takes no temperature: writing one would read back dropped.
        RouteTarget::Cli(CliKind::ClaudeCode) => write("claude-code", None),
        // OpenHuman's grammar names one CLI; another would read back as it.
        RouteTarget::Cli(_) => return None,
        RouteTarget::Ephemeral => with_model("ephemeral-route"),
    })
}
