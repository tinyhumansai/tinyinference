//! [`LossReport`]: everything an importer could not carry over faithfully.
//!
//! An importer never guesses silently. Every step that drops, normalises,
//! synthesises or refuses something appends an entry, and the golden tests
//! assert the exact report. Entries name **keys and reasons**, never values, so
//! a report is safe to log once per scope per boot.

use serde::{Deserialize, Serialize};

/// What kind of loss an entry records.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LossKind {
    /// Something in the source has no home in the hub and was left out.
    Dropped,
    /// The source is ambiguous; the importer picked a reading and says which.
    Ambiguous,
    /// The source was rewritten into the hub's canonical spelling or shape.
    Normalised,
    /// The hub made something up that the source did not have (a record for
    /// OpenCompany's entry zero).
    Synthesised,
    /// The source names something that cannot be resolved; the hub refuses it
    /// at turn time instead of guessing.
    FailClosed,
}

/// One lossy step.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LossEntry {
    /// The source key the entry is about (`inference/routes`, `cloud_providers`,
    /// a role name). A name, never a value that could be a credential.
    pub source_key: String,
    /// The kind of loss.
    pub kind: LossKind,
    /// A fixed sentence saying what happened.
    pub detail: String,
}

impl LossEntry {
    /// An entry.
    pub fn new(source_key: impl Into<String>, kind: LossKind, detail: impl Into<String>) -> Self {
        Self {
            source_key: source_key.into(),
            kind,
            detail: detail.into(),
        }
    }
}

/// The ordered list of lossy steps an import took.
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LossReport {
    /// The entries, in the order the importer took the steps.
    pub entries: Vec<LossEntry>,
}

impl LossReport {
    /// An empty report.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends an entry.
    pub fn push(
        &mut self,
        source_key: impl Into<String>,
        kind: LossKind,
        detail: impl Into<String>,
    ) {
        self.entries.push(LossEntry::new(source_key, kind, detail));
    }

    /// Whether nothing was lost.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries of one kind.
    pub fn of_kind(&self, kind: LossKind) -> impl Iterator<Item = &LossEntry> {
        self.entries.iter().filter(move |e| e.kind == kind)
    }

    /// Whether some entry of `kind` is about `source_key`.
    pub fn has(&self, source_key: &str, kind: LossKind) -> bool {
        self.entries
            .iter()
            .any(|e| e.kind == kind && e.source_key == source_key)
    }

    /// Adds another report's entries after this one's.
    pub fn extend(&mut self, other: LossReport) {
        self.entries.extend(other.entries);
    }
}
