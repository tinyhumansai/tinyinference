//! Merging registry facts and operator overrides into a provider's listing.
//!
//! Precedence, per fact: an operator override beats the provider's own answer,
//! which beats the registry, which beats nothing. A registry only ever fills a
//! gap; it never replaces something the provider or a local probe said.

use crate::descriptor::{CapSource, Capabilities, Sourced, Tri};
use crate::ids::{KindId, ModelId};

use super::registry::ModelMetadataSource;
use super::types::{EntrySource, ModelEntry};

/// An operator's correction to what is known about one model. Also how a model
/// the provider does not list (an Azure deployment, a fine-tune) is added.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct ModelOverride {
    /// The model.
    pub model: ModelId,
    /// A human name.
    pub display_name: Option<String>,
    /// Context window in tokens.
    pub context_window: Option<u64>,
    /// Maximum output tokens.
    pub max_output: Option<u64>,
    /// Tool calling.
    pub tools: Option<Tri>,
    /// Image input.
    pub vision: Option<Tri>,
    /// Reasoning output.
    pub reasoning: Option<Tri>,
    /// Whether `temperature` is accepted.
    pub temperature: Option<Tri>,
    /// Structured output.
    pub structured_output: Option<Tri>,
    /// Price per million input tokens.
    pub input_per_1m: Option<f64>,
    /// Price per million output tokens.
    pub output_per_1m: Option<f64>,
}

impl ModelOverride {
    /// An override that changes nothing yet.
    pub fn new(model: ModelId) -> Self {
        Self {
            model,
            display_name: None,
            context_window: None,
            max_output: None,
            tools: None,
            vision: None,
            reasoning: None,
            temperature: None,
            structured_output: None,
            input_per_1m: None,
            output_per_1m: None,
        }
    }
}

/// Fills a tri-state from `incoming` only where nothing better is known.
fn fill_tri(slot: &mut Sourced<Tri>, incoming: Sourced<Tri>, source: CapSource) {
    if slot.source == CapSource::Default && incoming.value != Tri::Unknown {
        *slot = Sourced::new(incoming.value, source);
    }
}

fn fill_num(slot: &mut Sourced<Option<u64>>, incoming: Sourced<Option<u64>>, source: CapSource) {
    if slot.source == CapSource::Default && incoming.value.is_some() {
        *slot = Sourced::new(incoming.value, source);
    }
}

fn fill_capabilities(into: &mut Capabilities, from: &Capabilities) {
    let source = CapSource::Registry;
    fill_num(&mut into.context_window, from.context_window, source);
    fill_num(&mut into.max_output, from.max_output, source);
    fill_tri(&mut into.tools, from.tools, source);
    fill_tri(&mut into.vision, from.vision, source);
    fill_tri(&mut into.reasoning, from.reasoning, source);
    fill_tri(&mut into.temperature, from.temperature, source);
    fill_tri(&mut into.structured_output, from.structured_output, source);
}

fn apply_override(entry: &mut ModelEntry, over: &ModelOverride) {
    let user = CapSource::UserOverride;
    if let Some(name) = &over.display_name {
        entry.display_name = Some(name.clone());
    }
    if let Some(window) = over.context_window {
        entry.capabilities.context_window = Sourced::new(Some(window), user);
    }
    if let Some(max) = over.max_output {
        entry.capabilities.max_output = Sourced::new(Some(max), user);
    }
    if over.input_per_1m.is_some() {
        entry.input_per_1m = over.input_per_1m;
    }
    if over.output_per_1m.is_some() {
        entry.output_per_1m = over.output_per_1m;
    }
    for (slot, value) in [
        (&mut entry.capabilities.tools, over.tools),
        (&mut entry.capabilities.vision, over.vision),
        (&mut entry.capabilities.reasoning, over.reasoning),
        (&mut entry.capabilities.temperature, over.temperature),
        (
            &mut entry.capabilities.structured_output,
            over.structured_output,
        ),
    ] {
        if let Some(value) = value {
            *slot = Sourced::new(value, user);
        }
    }
}

/// Merges what a registry and the operator know into `entries`.
///
/// Overrides are applied first so a model an override adds (an Azure deployment)
/// is also enriched by the registry; precedence is unchanged because the registry
/// only fills slots nothing better supplied.
///
/// * a registry fills gaps only: facts the provider (or a local probe) already
///   supplied are kept, and everything it adds is tagged
///   [`CapSource::Registry`] whatever tag it arrived with;
/// * prices, display name, lifecycle and alias come from the registry only when
///   the provider gave none;
/// * an override replaces the value and is tagged [`CapSource::UserOverride`];
/// * an override naming a model the list lacks **adds** it, as
///   [`EntrySource::User`] (an Azure deployment name is never in `/models`).
pub fn merge_metadata(
    entries: &mut Vec<ModelEntry>,
    kind: &KindId,
    registry: Option<&dyn ModelMetadataSource>,
    overrides: &[ModelOverride],
) {
    for over in overrides {
        let position = entries.iter().position(|e| e.id == over.model);
        let index = position.unwrap_or_else(|| {
            let mut added = ModelEntry::new(over.model.clone());
            added.origin = EntrySource::User;
            entries.push(added);
            entries.len() - 1
        });
        apply_override(&mut entries[index], over);
    }
    if let Some(registry) = registry {
        for entry in entries.iter_mut() {
            let Some(meta) = registry.lookup(kind, entry.id.as_str()) else {
                continue;
            };
            fill_capabilities(&mut entry.capabilities, &meta.capabilities);
            if entry.display_name.is_none() {
                entry.display_name = meta.display_name;
            }
            if entry.input_per_1m.is_none() {
                entry.input_per_1m = meta.input_per_1m;
            }
            if entry.output_per_1m.is_none() {
                entry.output_per_1m = meta.output_per_1m;
            }
            if entry.lifecycle.is_none() {
                entry.lifecycle = meta.lifecycle;
            }
            if entry.alias_of.is_none() {
                entry.alias_of = meta.alias_of;
            }
        }
    }
}
