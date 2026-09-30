//! [`EventSink`] and [`HubEvent`]: what the hub tells its host about.

use std::fmt::Debug;

use crate::health::ProviderHealth;
use crate::ids::{ScopeKey, Slug};

/// Something the hub observed. Carries identifiers and counts only, never a
/// credential and never raw upstream text.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HubEvent {
    /// A provider's health changed.
    HealthChanged {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
        /// The previous health.
        from: ProviderHealth,
        /// The new health.
        to: ProviderHealth,
    },
    /// A provider was added.
    ProviderAdded {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
    },
    /// A provider's record changed (label, endpoint or model).
    ProviderEdited {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
    },
    /// A provider was removed (an add that was rolled back emits nothing).
    ProviderRemoved {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
    },
    /// A provider was enabled or disabled.
    EnabledChanged {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
        /// The new state.
        enabled: bool,
    },
    /// A provider's key was set, rotated or cleared. Its health and the
    /// scope's cached catalogs were dropped.
    KeyChanged {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
        /// Whether a key exists afterwards in the credential store.
        present: bool,
    },
    /// The default choice or an agent pin changed.
    DefaultChanged {
        /// The scope.
        scope: ScopeKey,
    },
    /// A model list was read from the provider.
    CatalogFetched {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
        /// How many models it listed.
        models: usize,
    },
    /// A model list was served from before because the provider failed. Carries
    /// the failure's reason code only.
    CatalogServedStale {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
        /// Why the fresh read failed.
        reason: crate::error::ReasonCode,
    },
    /// A scheduled re-test of a provider that was down on a rejected key or an
    /// exhausted account ran.
    Retested {
        /// The scope.
        scope: ScopeKey,
        /// The provider.
        slug: Slug,
        /// Whether it passed.
        ok: bool,
    },
}

/// Receives [`HubEvent`]s. The default drops them.
pub trait EventSink: Send + Sync + Debug {
    /// Delivers an event. Must not block: the hub calls it inline.
    fn emit(&self, event: HubEvent);
}
