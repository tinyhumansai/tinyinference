//! OpenCompany's route strings, read and written.
//!
//! The grammar (`resolve.rs`, OpenCompany): `""` or `default` is the default
//! choice, `managed` is the managed provider, `local[:model]` is a local
//! runtime (a category, not a specific runtime), `claude-code[:model]` is the
//! Claude CLI login, and anything else is `<slug>[:model]` for a configured
//! provider. The model is everything after the **first** colon, so
//! `ollama:llama3:8b` names the model `llama3:8b`.

use crate::error::{InputField, InvalidInput};
use crate::ids::{ModelId, Slug};
use crate::taxonomy::CliKind;

use super::types::{ProviderRoute, RouteTarget};

/// Parses an OpenCompany route string.
///
/// # Errors
///
/// [`InvalidInput`] when the provider slug or the model id is not valid.
pub fn parse(raw: &str) -> Result<ProviderRoute, InvalidInput> {
    let raw = raw.trim();
    if raw.is_empty() || raw.eq_ignore_ascii_case("default") {
        return Ok(ProviderRoute::default_route());
    }
    let (head, model) = match raw.split_once(':') {
        Some((head, model)) => (head.trim(), Some(model.trim())),
        None => (raw, None),
    };
    let target = match head.to_ascii_lowercase().as_str() {
        "managed" => RouteTarget::Managed,
        "local" => RouteTarget::Local(None),
        "claude-code" => RouteTarget::Cli(CliKind::ClaudeCode),
        "" => return Err(InvalidInput::Empty(InputField::Slug)),
        other => RouteTarget::Provider(Slug::parse(other)?),
    };
    let mut route = ProviderRoute::new(target);
    if let Some(model) = model.filter(|m| !m.is_empty()) {
        route.model = Some(ModelId::parse(model)?);
    }
    Ok(route)
}

/// Writes a route as an OpenCompany route string, or `None` when the string
/// grammar cannot say it (a temperature, a specific local runtime, the Codex
/// CLI, an ephemeral route).
pub fn to_string(route: &ProviderRoute) -> Option<String> {
    if route.temperature.is_some() {
        return None;
    }
    let head = match &route.target {
        RouteTarget::Default => {
            return route.model.is_none().then(String::new);
        }
        RouteTarget::Managed => "managed".to_string(),
        RouteTarget::Local(None) => "local".to_string(),
        RouteTarget::Cli(CliKind::ClaudeCode) => "claude-code".to_string(),
        RouteTarget::Provider(slug) => slug.as_str().to_string(),
        _ => return None,
    };
    Some(match &route.model {
        Some(model) => format!("{head}:{model}"),
        None => head,
    })
}
