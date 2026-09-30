//! The TinyHumans proxy's paged model catalog:
//! `{"success":true,"data":{"object":"list","data":[...],"total":N,"limit":L,"offset":O}}`.
//!
//! Ported from OpenCompany's `paged_catalog.rs`. Pure; the reading loop lives
//! with the managed driver. Every id is kept exactly as given: nothing here
//! hardcodes, filters, prefers or rejects a model id by vendor or name.

use std::collections::HashSet;

use serde_json::Value;

use crate::descriptor::{CapSource, Sourced};
use crate::error::ProviderFailure;
use crate::ids::ModelId;

use super::parse::unreadable;
use super::types::ModelEntry;

/// Page size requested. The backend clamps `limit` to `[1, 500]`.
pub const PAGE_LIMIT: usize = 500;
/// Most pages one read follows, so a `total` never reached cannot loop.
pub const MAX_PAGES: usize = 20;

/// The path (with query) for one page at `offset`.
pub fn page_path(offset: usize) -> String {
    format!("/models?limit={PAGE_LIMIT}&offset={offset}")
}

/// One parsed page.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Page {
    /// Entries this page carried, after dropping malformed ones.
    pub entries: Vec<ModelEntry>,
    /// How many rows the page carried, usable or not. Paging advances by this,
    /// not by `entries.len()`, so a malformed row still moves the offset
    /// forward instead of being requested forever.
    pub raw_len: usize,
    /// The envelope's `total`, when it parses as a non-negative integer.
    pub total: Option<usize>,
}

/// What to do after one page.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextPage {
    /// Request the next page at this offset.
    At(usize),
    /// Stop: an empty page, `total` reached, or no `total` at all.
    Done,
    /// Stop: [`MAX_PAGES`] was reached before `total` was.
    Truncated {
        /// How many rows were read before stopping.
        read: usize,
        /// The envelope's own `total`.
        total: usize,
    },
}

fn field(row: &Value, key: &str) -> Option<String> {
    row.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

/// Parses one page.
///
/// # Errors
///
/// A [`ProviderFailure`] (`unknown`, never an empty page) on: not JSON;
/// `success: false`; no object `data`; no array `data.data`. A plain
/// `{"data":[...]}` body (the OpenAI shape) is an error, not a page with no
/// entries.
pub fn parse_page(body: &str) -> Result<Page, ProviderFailure> {
    let value: Value = serde_json::from_str(body)
        .map_err(|e| unreadable(format!("the model catalog was not JSON: {e}")))?;
    if value.get("success").and_then(Value::as_bool) == Some(false) {
        let reason = value
            .get("error")
            .or_else(|| value.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("no reason given");
        return Err(unreadable(format!(
            "the model catalog reported a failure: {reason}"
        )));
    }
    let Some(data) = value.get("data").filter(|d| d.is_object()) else {
        return Err(unreadable(
            "the model catalog was not in the `{success, data}` envelope",
        ));
    };
    let Some(rows) = data.get("data").and_then(Value::as_array) else {
        return Err(unreadable(
            "the model catalog envelope carried no `data` list",
        ));
    };
    let entries: Vec<ModelEntry> = rows
        .iter()
        .filter_map(|row| {
            let id = ModelId::parse(&field(row, "id")?).ok()?;
            let mut entry = ModelEntry::new(id);
            entry.display_name = field(row, "display_name").or_else(|| field(row, "name"));
            entry.owned_by = field(row, "owned_by");
            if let Some(window) = row
                .get("context_length")
                .or_else(|| row.get("context_window"))
                .and_then(Value::as_u64)
            {
                entry.capabilities.context_window =
                    Sourced::new(Some(window), CapSource::ProviderApi);
            }
            // `pricing` is already the charged price per 1M tokens.
            let price = |key: &str| {
                row.get("pricing")
                    .and_then(|p| p.get(key))
                    .and_then(Value::as_f64)
                    .filter(|n| n.is_finite() && *n >= 0.0)
            };
            entry.input_per_1m = price("inputPer1M");
            entry.output_per_1m = price("outputPer1M");
            Some(entry)
        })
        .collect();
    let total = data
        .get("total")
        .and_then(Value::as_u64)
        .and_then(|t| usize::try_from(t).ok());
    Ok(Page {
        entries,
        raw_len: rows.len(),
        total,
    })
}

/// Pages collected, deduplicated by id, in listing order.
#[derive(Debug, Default)]
pub struct Collector {
    seen: HashSet<String>,
    entries: Vec<ModelEntry>,
    offset: usize,
    pages: usize,
}

impl Collector {
    /// The offset the next page should request.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Folds one page in and says what to do next. Stops on: an empty page;
    /// reaching `total`; an empty page when there is no `total`; [`MAX_PAGES`].
    pub fn push(&mut self, page: Page) -> NextPage {
        self.pages += 1;
        for entry in page.entries {
            if self.seen.insert(entry.id.as_str().to_string()) {
                self.entries.push(entry);
            }
        }
        if page.raw_len == 0 {
            return NextPage::Done;
        }
        self.offset += page.raw_len;
        match page.total {
            Some(total) if self.offset < total && self.pages >= MAX_PAGES => NextPage::Truncated {
                read: self.offset,
                total,
            },
            Some(total) if self.offset < total => NextPage::At(self.offset),
            // No `total`: the server may clamp `limit` below what we asked for, so
            // the only end it announces is an empty page. Keep reading until one
            // (or the page cap) rather than presenting one page as the whole
            // catalog. (OpenCompany stopped here; the hub does not silently
            // truncate. The cost is one extra, empty request.)
            None if self.pages >= MAX_PAGES => NextPage::Truncated {
                read: self.offset,
                total: self.offset,
            },
            None => NextPage::At(self.offset),
            _ => NextPage::Done,
        }
    }

    /// The entries collected so far, consuming the collector.
    pub fn finish(self) -> Vec<ModelEntry> {
        self.entries
    }

    /// Whether rows were read but none produced a usable entry (every id missing
    /// or invalid). Judged over the **whole** read, never per page: one bad late
    /// page must not throw away the good pages before it.
    pub fn read_only_unusable_rows(&self) -> bool {
        self.entries.is_empty() && self.offset > 0
    }
}
