//! [`DriverRegistry`]: kind to driver.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::catalogue::{self, custom_descriptor};
use crate::ids::KindId;
use crate::taxonomy::ProviderGroup;

use super::{
    AnthropicDriver, CliDriver, KindDriver, LocalDriver, ManagedDriver, OpenAiCompatDriver,
};

/// The kind of the Anthropic catalogue row, which has a driver of its own.
const ANTHROPIC_KIND: &str = "anthropic";

/// Maps a [`KindId`] to the driver that serves it.
///
/// A stored record whose kind has no driver is a hard error at the hub (D14):
/// guessing "it is probably OpenAI-compatible" would send a key to a protocol
/// nobody checked.
#[derive(Default)]
pub struct DriverRegistry {
    by_kind: HashMap<KindId, Arc<dyn KindDriver>>,
}

impl DriverRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry with a driver for every catalogue kind, plus `custom`.
    ///
    /// The managed kind is registered as the paged envelope (OpenCompany's
    /// backend); a host on the OpenAI-shaped backend replaces it with
    /// [`ManagedDriver::openai_shaped`] through [`DriverRegistry::register`].
    pub fn with_builtin() -> Self {
        let mut registry = Self::new();
        for descriptor in catalogue::descriptors() {
            let descriptor = descriptor.clone();
            let driver: Arc<dyn KindDriver> = match descriptor.group {
                ProviderGroup::Managed => Arc::new(ManagedDriver::paged(descriptor)),
                ProviderGroup::Local => Arc::new(LocalDriver::for_descriptor(descriptor)),
                ProviderGroup::Cli => Arc::new(CliDriver::for_descriptor(descriptor)),
                _ if descriptor.kind.as_str() == ANTHROPIC_KIND => {
                    Arc::new(AnthropicDriver::for_descriptor(descriptor))
                }
                _ => Arc::new(OpenAiCompatDriver::for_descriptor(descriptor)),
            };
            registry.register(driver);
        }
        registry.register(Arc::new(OpenAiCompatDriver::for_descriptor(
            custom_descriptor(),
        )));
        registry
    }

    /// Adds (or replaces) the driver for its descriptor's kind, returning the
    /// one it replaced.
    pub fn register(&mut self, driver: Arc<dyn KindDriver>) -> Option<Arc<dyn KindDriver>> {
        let kind = driver.descriptor().kind.clone();
        self.by_kind.insert(kind, driver)
    }

    /// The driver for a kind.
    pub fn get(&self, kind: &KindId) -> Option<Arc<dyn KindDriver>> {
        self.by_kind.get(kind).cloned()
    }

    /// The driver for a name that may be an alias (`openhuman` for
    /// `tinyhumans`, `lm-studio` for `lmstudio`).
    pub fn resolve(&self, name: &str) -> Option<Arc<dyn KindDriver>> {
        let kind = catalogue::resolve_kind(name).unwrap_or_else(|| KindId::new(name));
        self.get(&kind)
    }

    /// The registered kinds, sorted.
    pub fn kinds(&self) -> Vec<KindId> {
        let mut kinds: Vec<KindId> = self.by_kind.keys().cloned().collect();
        kinds.sort();
        kinds
    }

    /// How many kinds are registered.
    pub fn len(&self) -> usize {
        self.by_kind.len()
    }

    /// Whether nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.by_kind.is_empty()
    }
}

impl fmt::Debug for DriverRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DriverRegistry")
            .field("kinds", &self.kinds())
            .finish()
    }
}
