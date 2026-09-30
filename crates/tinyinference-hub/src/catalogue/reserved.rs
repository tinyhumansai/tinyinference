//! Reserved slugs: names a typed provider name may not take.

use crate::taxonomy::CliKind;

use super::descriptors;

/// Slugs reserved for the hub's and the hosts' own bookkeeping: the managed
/// kind and its legacy spellings (`managed`, OpenHuman's `openhuman`, `cloud`,
/// `pid`) and the two synthetic records the OpenHuman importer creates.
pub const INTERNAL_SLUGS: &[&str] = &[
    "tinyhumans",
    "managed",
    "openhuman",
    "cloud",
    "pid",
    "byok-inference",
    "ephemeral-route",
];

/// Whether `slug` belongs to a built-in provider and may not be taken by a
/// typed name.
///
/// The union of both hosts' rules: every catalogue kind and alias, both CLI
/// slugs (the option slug and the slug the login stores under, `openai`), and
/// [`INTERNAL_SLUGS`]. The route words `local` and `default` and the word `custom` are
/// *not* reserved. Compared case-insensitively.
pub fn is_reserved_slug(slug: &str) -> bool {
    let slug = slug.trim();
    if slug.is_empty() {
        return false;
    }
    let cli = [CliKind::ClaudeCode, CliKind::Codex].iter().any(|k| {
        k.option_slug().eq_ignore_ascii_case(slug) || k.stored_slug().eq_ignore_ascii_case(slug)
    });
    cli || INTERNAL_SLUGS.iter().any(|s| s.eq_ignore_ascii_case(slug))
        || descriptors().iter().any(|d| d.answers_to(slug))
}

/// Every reserved slug, sorted and de-duplicated. For a UI legend and for
/// tests.
pub fn reserved_slugs() -> Vec<String> {
    let mut out: Vec<String> = INTERNAL_SLUGS.iter().map(|s| (*s).to_string()).collect();
    for kind in [CliKind::ClaudeCode, CliKind::Codex] {
        out.push(kind.option_slug().to_string());
        out.push(kind.stored_slug().to_string());
    }
    for d in descriptors() {
        out.push(d.slug().to_string());
        out.extend(d.aliases.iter().map(|a| (*a).to_string()));
    }
    out.sort();
    out.dedup();
    out
}
