//! [`ProviderDraft`]: a provider someone is about to add, before it is saved.

use crate::ids::{KindId, ModelId};
use crate::secret::Secret;

/// The input to adding or probing a provider, and the output of detection.
///
/// It carries the key as a [`Secret`] (redacted in `Debug`); once saved the key
/// lives in the credential store and the record never has one.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct ProviderDraft {
    /// The catalogue kind (or `custom`).
    pub kind: KindId,
    /// A display name; the kind's label when absent.
    pub label: Option<String>,
    /// The endpoint; the kind's preset when absent.
    pub base_url: Option<String>,
    /// The chosen model, if any.
    pub model: Option<ModelId>,
    /// The key to store.
    pub key: Option<Secret>,
}

impl ProviderDraft {
    /// A draft of `kind` with everything else unset.
    pub fn new(kind: impl AsRef<str>) -> Self {
        Self {
            kind: KindId::new(kind),
            label: None,
            base_url: None,
            model: None,
            key: None,
        }
    }

    /// Sets the endpoint.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the key.
    #[must_use]
    pub fn with_key(mut self, key: Secret) -> Self {
        self.key = Some(key);
        self
    }

    /// Sets the model.
    #[must_use]
    pub fn with_model(mut self, model: ModelId) -> Self {
        self.model = Some(model);
        self
    }

    /// Sets the display name.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}
