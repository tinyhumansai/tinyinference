//! Parsers for the listing shapes providers answer in.
//!
//! Tolerant per entry: an entry without a usable id is dropped and counted, an
//! optional field of the wrong type is treated as absent, and the rest of the
//! list survives. What is refused is a document that is not a listing at all
//! (not JSON, an error envelope, no `data`/`models` field).

use serde_json::Value;

use crate::descriptor::{CapSource, Sourced, Tri};
use crate::error::{ProviderFailure, ReasonCode, Retry};
use crate::ids::ModelId;

use super::types::ModelEntry;

/// A parsed listing and how much of it could not be used.
///
/// A listing that had rows but yielded **no** usable entry is not a healthy
/// empty catalog: see [`ParsedCatalog::into_usable`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedCatalog {
    /// The usable entries, deduplicated by id, in listing order.
    pub entries: Vec<ModelEntry>,
    /// Rows dropped: no usable id, an id that is not a valid [`ModelId`], or a
    /// duplicate.
    pub skipped: usize,
}

/// A listing that connected but did not parse says nothing about the
/// credential, so it is `Unknown`, never `Auth` (the only class that rolls a
/// key back). The one place that failure is built.
pub(crate) fn unreadable(text: impl AsRef<str>) -> ProviderFailure {
    ProviderFailure::new(ReasonCode::Unknown, Retry::Never).with_raw(text)
}

/// A listing (or one page of it) that ran past its size cap. Refused outright
/// rather than parsed truncated: a parser cannot tell "malformed" from "cut
/// off", and treating the second as the first is how a real, valid,
/// hundreds-of-models catalog once read as a healthy connection with zero
/// models. `Unknown`, not `Auth`, for the same reason as [`unreadable`].
pub(crate) fn too_large(what: &str) -> ProviderFailure {
    unreadable(format!("{what} is larger than the size cap")).with_truncated(true)
}

impl ParsedCatalog {
    /// The entries, unless the listing had rows and none were usable (every row
    /// lacked an id, or had one that is not a valid model id). That is a
    /// gateway answering in a shape this parser cannot read, not a provider with
    /// no models, and reporting it as a passing empty listing would mark a
    /// provider that cannot list anything as proven.
    ///
    /// # Errors
    ///
    /// A `ProviderFailure` (`unknown`, never retried) naming how many rows were
    /// unusable. This is a behaviour choice worth knowing: a provider whose
    /// every row is one the parser drops (including rows llm's envelope parser
    /// discards) reads as a broken listing, not as an empty one.
    pub fn into_usable(self) -> Result<Vec<ModelEntry>, ProviderFailure> {
        if self.entries.is_empty() && self.skipped > 0 {
            return Err(unreadable(format!(
                "the model list had {} rows and none produced a usable model id \
                 (missing, invalid or duplicate)",
                self.skipped
            )));
        }
        Ok(self.entries)
    }
}

fn json(body: &[u8]) -> Result<Value, ProviderFailure> {
    serde_json::from_slice(body)
        .map_err(|error| unreadable(format!("the model list was not JSON: {error}")))
}

fn finish(entries: impl IntoIterator<Item = Option<ModelEntry>>) -> ParsedCatalog {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    let mut skipped = 0;
    for entry in entries {
        match entry {
            Some(entry) if seen.insert(entry.id.clone()) => out.push(entry),
            _ => skipped += 1,
        }
    }
    ParsedCatalog {
        entries: out,
        skipped,
    }
}

/// Parses an OpenAI-shaped `GET /models` body.
///
/// Reuses llm's envelope handling (`data` or `models`; a `null` list on a
/// success envelope is an empty catalog, on an error envelope it is a failure;
/// an entry is an object with `id`, `slug` or `name`, or a bare string;
/// `context_length`/`context_window`; `pricing.inputPer1M`/`outputPer1M`) and
/// adds what a real fleet needs on top: a **bare top-level array** (Together's
/// listing) is accepted, and rows are deduplicated by id and validated as
/// [`ModelId`]s.
///
/// # Errors
///
/// A [`ProviderFailure`] (`unknown`, never retried) when the body is not JSON
/// or is not a listing.
pub fn parse_openai(body: &[u8]) -> Result<ParsedCatalog, ProviderFailure> {
    parse_openai_value(&json(body)?)
}

/// [`parse_openai`] for a body that is already parsed JSON (a driver that also
/// needs the envelope's paging fields parses once and reads both).
///
/// # Errors
///
/// A [`ProviderFailure`] (`unknown`, never retried) when the value is not a
/// listing.
pub fn parse_openai_value(value: &Value) -> Result<ParsedCatalog, ProviderFailure> {
    let wrapped;
    let value = if let Value::Array(items) = value {
        wrapped = serde_json::json!({ "data": items });
        &wrapped
    } else {
        value
    };
    // The rows llm's parser drops (no id) are invisible to it; count them from
    // the envelope so `skipped` is honest.
    let raw_rows = row_count(value);
    let infos = tinyinference_llm::catalog::parse_models_response(value)
        .map_err(|error| unreadable(error.to_string()))?;
    let dropped_by_llm = raw_rows.saturating_sub(infos.len());
    let mut parsed = finish(
        infos
            .into_iter()
            .map(|info| ModelEntry::try_from(info).ok()),
    );
    parsed.skipped += dropped_by_llm;
    Ok(parsed)
}

fn row_count(value: &Value) -> usize {
    value
        .get("data")
        .or_else(|| value.get("models"))
        .and_then(Value::as_array)
        .map_or(0, Vec::len)
}

fn text<'a>(row: &'a Value, key: &str) -> Option<&'a str> {
    row.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
}

/// Parses Ollama's `GET /api/tags`: `{"models":[{"name":"llama3:latest",
/// "model":"llama3:latest", "details":{...}}]}`. The id is `model`, else
/// `name`.
///
/// # Errors
///
/// A [`ProviderFailure`] (`unknown`) when the body is not JSON or has no
/// `models` list. `{"models":null}` is an empty catalog: Ollama with nothing
/// pulled.
pub fn parse_ollama_tags(body: &[u8]) -> Result<ParsedCatalog, ProviderFailure> {
    let value = json(body)?;
    let Some(field) = value.as_object().and_then(|o| o.get("models")) else {
        return Err(unreadable("the tags list had no `models` field"));
    };
    if field.is_null() {
        return Ok(finish(std::iter::empty()));
    }
    let Some(rows) = field.as_array() else {
        return Err(unreadable("the tags `models` field was not a list"));
    };
    Ok(finish(rows.iter().map(|row| {
        let id = text(row, "model").or_else(|| text(row, "name"))?;
        Some(ModelEntry::new(ModelId::parse(id).ok()?))
    })))
}

/// Parses LM Studio's `GET /api/v0/models`: `{"data":[{"id":..,"type":"llm",
/// "state":"loaded","max_context_length":..,"capabilities":["tool_use"]}]}`.
///
/// Embedding models are skipped (they cannot chat). What the runtime reports
/// about a model is tagged as a local probe, not as a provider claim.
///
/// # Errors
///
/// A [`ProviderFailure`] (`unknown`) when the body is not JSON or has no
/// `data` list.
pub fn parse_lmstudio_v0(body: &[u8]) -> Result<ParsedCatalog, ProviderFailure> {
    let value = json(body)?;
    let Some(rows) = value.get("data").and_then(Value::as_array) else {
        return Err(unreadable("the model list had no `data` list"));
    };
    // Embedding rows are expected (they cannot chat), so they are not damage and
    // are filtered out before rows are counted.
    let chat_rows = rows.iter().filter(|row| {
        !text(row, "type").is_some_and(|kind| kind.eq_ignore_ascii_case("embeddings"))
    });
    let parsed = finish(chat_rows.map(|row| {
        let id = text(row, "id")?;
        let kind = text(row, "type").unwrap_or("llm");
        let mut entry = ModelEntry::new(ModelId::parse(id).ok()?);
        if let Some(window) = row.get("max_context_length").and_then(Value::as_u64) {
            entry.capabilities.context_window = Sourced::new(Some(window), CapSource::LocalProbe);
        }
        if kind.eq_ignore_ascii_case("vlm") {
            entry.capabilities.vision = Sourced::new(Tri::Yes, CapSource::LocalProbe);
        }
        let caps = row.get("capabilities").and_then(Value::as_array);
        if caps.is_some_and(|c| c.iter().any(|v| v.as_str() == Some("tool_use"))) {
            entry.capabilities.tools = Sourced::new(Tri::Yes, CapSource::LocalProbe);
        }
        Some(entry)
    }));
    Ok(parsed)
}
