//! [`HubConfig`]: everything the hub persists about one scope.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::descriptor::{LegacyFields, ProviderRecord};
use crate::error::{InputField, InvalidInput};
use crate::ids::{AgentKey, ModelId, Slug, WorkloadKey};
use crate::route::{ProviderRoute, RouteTarget};
use crate::secret::is_credential_name;

/// The schema version this build writes. A reader meeting a larger number knows
/// a newer hub wrote the document.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

/// A provider and a model together.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelChoice {
    /// The provider's slug.
    pub provider: Slug,
    /// The model id, unchanged from what the provider lists.
    pub model: ModelId,
}

impl ModelChoice {
    /// A choice of `model` on `provider`.
    pub fn new(provider: Slug, model: ModelId) -> Self {
        Self { provider, model }
    }
}

/// The scope's default provider (and model).
#[non_exhaustive]
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum DefaultChoice {
    /// Nothing chosen.
    #[default]
    Unset,
    /// A provider is chosen but no model. Fails closed on the turn path.
    ProviderOnly {
        /// The provider.
        provider: Slug,
    },
    /// A provider and a model.
    Full {
        /// The provider.
        provider: Slug,
        /// The model.
        model: ModelId,
    },
}

/// The persisted configuration of one scope.
///
/// **It never contains a credential.** Deserialising refuses a top-level field
/// whose name marks it as one; the records inside refuse the same (see
/// [`ProviderRecord`]). Fields a newer hub added are kept in
/// [`extra`](Self::extra) so an older hub saving the document does not silently
/// drop them.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(try_from = "HubConfigWire")]
pub struct HubConfig {
    /// Which schema wrote this document.
    pub schema_version: u32,
    /// The configured providers, in the order the operator sees them.
    pub providers: Vec<ProviderRecord>,
    /// The default choice.
    pub default: DefaultChoice,
    /// Per-agent pins, keyed by the host's agent key.
    pub agent_pins: BTreeMap<AgentKey, ModelChoice>,
    /// Per-workload routes, keyed by the host's opaque workload key (a tier or
    /// role; the hub never interprets it). Never holds a default route or an
    /// ephemeral one.
    pub workload_routes: BTreeMap<WorkloadKey, ProviderRoute>,
    /// Fields this build does not interpret, preserved on save.
    pub extra: LegacyFields,
}

#[derive(Deserialize)]
struct HubConfigWire {
    #[serde(default = "current_schema")]
    schema_version: u32,
    #[serde(default)]
    providers: Vec<ProviderRecord>,
    #[serde(default)]
    default: DefaultChoice,
    #[serde(default)]
    agent_pins: BTreeMap<AgentKey, ModelChoice>,
    #[serde(default)]
    workload_routes: BTreeMap<WorkloadKey, ProviderRoute>,
    #[serde(default, flatten)]
    extra: LegacyFields,
}

fn current_schema() -> u32 {
    CONFIG_SCHEMA_VERSION
}

/// `extra` is flattened on the way out too, so unknown fields round-trip. The
/// known fields are written last so an `extra` entry can never shadow one.
impl Serialize for HubConfig {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let mut map = serializer.serialize_map(None)?;
        for (key, value) in &self.extra {
            if !matches!(
                key.as_str(),
                "schema_version" | "providers" | "default" | "agent_pins" | "workload_routes"
            ) {
                map.serialize_entry(key, value)?;
            }
        }
        map.serialize_entry("schema_version", &self.schema_version)?;
        map.serialize_entry("providers", &self.providers)?;
        map.serialize_entry("default", &self.default)?;
        map.serialize_entry("agent_pins", &self.agent_pins)?;
        // Written only when there are routes, so a document that never used them
        // is byte-for-byte what earlier builds wrote.
        if !self.workload_routes.is_empty() {
            map.serialize_entry("workload_routes", &self.workload_routes)?;
        }
        map.end()
    }
}

impl TryFrom<HubConfigWire> for HubConfig {
    type Error = InvalidInput;

    fn try_from(wire: HubConfigWire) -> Result<Self, Self::Error> {
        let config = Self {
            schema_version: wire.schema_version,
            providers: wire.providers,
            default: wire.default,
            agent_pins: wire.agent_pins,
            workload_routes: wire.workload_routes,
            extra: wire.extra,
        };
        config.validate()?;
        Ok(config)
    }
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            schema_version: CONFIG_SCHEMA_VERSION,
            providers: Vec::new(),
            default: DefaultChoice::Unset,
            agent_pins: BTreeMap::new(),
            workload_routes: BTreeMap::new(),
            extra: LegacyFields::new(),
        }
    }
}

impl HubConfig {
    /// An empty configuration at the current schema.
    pub fn new() -> Self {
        Self::default()
    }

    /// The provider with this slug.
    pub fn provider(&self, slug: &Slug) -> Option<&ProviderRecord> {
        self.providers.iter().find(|p| &p.slug == slug)
    }

    /// The provider with this slug, mutably.
    pub fn provider_mut(&mut self, slug: &Slug) -> Option<&mut ProviderRecord> {
        self.providers.iter_mut().find(|p| &p.slug == slug)
    }

    /// Whether a provider with this slug exists.
    pub fn contains(&self, slug: &Slug) -> bool {
        self.provider(slug).is_some()
    }

    /// Checks the invariants a stored document must hold: no credential-shaped
    /// top-level field, and every record valid.
    ///
    /// # Errors
    ///
    /// [`InvalidInput::CredentialField`] for a credential-shaped field, or the
    /// record's own refusal.
    pub fn validate(&self) -> Result<(), InvalidInput> {
        // A document from a newer hub may mean something different by fields
        // this build reads; editing and saving it back would relabel that
        // reinterpretation as valid. Refusing to load it also stops the store
        // overwriting it: an unreadable document is never treated as empty.
        if self.schema_version > CONFIG_SCHEMA_VERSION {
            return Err(InvalidInput::Malformed {
                field: InputField::Config,
                reason: "written by a newer version of the hub",
            });
        }
        // A credential is refused at any depth, exactly as a record's own
        // fields are: nesting it under an unknown field must not get it saved.
        for (name, value) in &self.extra {
            let found = if is_credential_name(name) {
                Some(name.clone())
            } else {
                crate::descriptor::find_credential_field(value, 0)
            };
            if let Some(name) = found {
                return Err(InvalidInput::CredentialField { name });
            }
        }
        for route in self.workload_routes.values() {
            // A default route is the absence of a route, and an ephemeral one is
            // never persisted; either in a stored document is a bug upstream.
            if matches!(route.target, RouteTarget::Default | RouteTarget::Ephemeral) {
                return Err(InvalidInput::Malformed {
                    field: InputField::Config,
                    reason: "a workload route must name a provider, the managed provider, a local runtime or a CLI login",
                });
            }
        }
        let mut seen = std::collections::HashSet::new();
        for record in &self.providers {
            record.validate()?;
            if !seen.insert(&record.slug) {
                return Err(InvalidInput::Malformed {
                    field: InputField::Slug,
                    reason: "two providers share a slug",
                });
            }
        }
        Ok(())
    }
}
