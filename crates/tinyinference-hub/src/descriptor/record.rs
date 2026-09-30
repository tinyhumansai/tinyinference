//! [`ProviderRecord`], a configured instance of a kind.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::endpoint::{endpoint_has_credentials, endpoint_query_has_credential};
use crate::error::{InputField, InvalidInput};
use crate::ids::{KindId, ModelId, Slug};
use crate::secret::{Secret, is_credential_name};
use crate::taxonomy::AuthStyle;

/// Fields a record carries that the hub does not interpret (for example
/// OpenCompany's tier map). Preserved verbatim through a load and a save so an
/// adapter can round-trip its own data (`#[serde(flatten)]`).
pub type LegacyFields = BTreeMap<String, serde_json::Value>;

/// Where a record came from.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordOrigin {
    /// Created through the hub's operations.
    #[default]
    Indexed,
    /// OpenCompany's legacy "entry zero": read-only through list operations
    /// (guard G21).
    EntryZero,
    /// Produced by an `import` reader.
    Imported,
}

/// A configured provider instance.
///
/// **There is no credential field, ever** (invariant 1): a key lives in the
/// `CredentialStore` under [`Slug::key_slot`] and is read per request. The type
/// enforces it rather than only documenting it: deserialising (and
/// [`ProviderRecord::validate`]) refuse a `base_url` that carries userinfo and
/// any [`legacy`](ProviderRecord::legacy) field, at any depth, whose name marks
/// it as a credential, so plaintext cannot re-enter through a stored file.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "ProviderRecordWire")]
pub struct ProviderRecord {
    /// Opaque id (OpenCompany `prv_<32hex>`, OpenHuman `p_<slug>_<rand>`).
    pub id: String,
    /// The routing key.
    pub slug: Slug,
    /// The display name.
    pub label: String,
    /// The catalogue kind.
    pub kind: KindId,
    /// The normalised endpoint; never contains userinfo.
    pub base_url: String,
    /// The provider's chosen model, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelId>,
    /// Whether the record is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// An auth style overriding the kind's (custom rows only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_override: Option<AuthStyle>,
    /// A record the hub synthesises rather than the operator creating
    /// (`byok-inference`, `ephemeral-route`).
    #[serde(default)]
    pub synthetic: bool,
    /// Where the record came from.
    #[serde(default)]
    pub origin: RecordOrigin,
    /// Fields the hub does not interpret. Never credential-shaped.
    #[serde(default, flatten)]
    pub legacy: LegacyFields,
}

/// A credential pulled off a stored record by
/// [`ProviderRecord::extract_credentials`], to be written to the credential
/// store.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractedCredential {
    /// The field it was stored under (a name, not a value).
    pub field: String,
    /// The credential; redacts itself in `Debug`.
    pub value: Secret,
}

/// The deserialisation shape of [`ProviderRecord`]: the same fields, converted
/// through [`ProviderRecord::validate`].
#[derive(Deserialize)]
struct ProviderRecordWire {
    id: String,
    slug: Slug,
    label: String,
    kind: KindId,
    base_url: String,
    #[serde(default)]
    model: Option<ModelId>,
    #[serde(default = "default_true")]
    enabled: bool,
    #[serde(default)]
    auth_override: Option<AuthStyle>,
    #[serde(default)]
    synthetic: bool,
    #[serde(default)]
    origin: RecordOrigin,
    #[serde(default, flatten)]
    legacy: LegacyFields,
}

impl TryFrom<ProviderRecordWire> for ProviderRecord {
    type Error = InvalidInput;

    fn try_from(wire: ProviderRecordWire) -> Result<Self, Self::Error> {
        let record = Self {
            id: wire.id,
            slug: wire.slug,
            label: wire.label,
            kind: wire.kind,
            base_url: wire.base_url,
            model: wire.model,
            enabled: wire.enabled,
            auth_override: wire.auth_override,
            synthetic: wire.synthetic,
            origin: wire.origin,
            legacy: wire.legacy,
        };
        record.validate()?;
        Ok(record)
    }
}

/// The first credential-shaped field name anywhere in `value`, looking through
/// nested objects and arrays (a secret under `tiers` or `headers` is still a
/// secret on the record). Depth-bounded so a hostile file cannot recurse
/// without limit.
fn find_credential_field(value: &serde_json::Value, depth: usize) -> Option<String> {
    // Fail closed: a structure nested deeper than the bound is refused rather
    // than assumed clean, so a credential cannot hide under nine levels.
    if depth > 8 {
        return Some("(nested too deeply to check)".to_string());
    }
    match value {
        serde_json::Value::Object(map) => map.iter().find_map(|(name, inner)| {
            if is_credential_name(name) {
                Some(name.clone())
            } else {
                find_credential_field(inner, depth + 1)
            }
        }),
        serde_json::Value::Array(items) => items
            .iter()
            .find_map(|inner| find_credential_field(inner, depth + 1)),
        _ => None,
    }
}

fn default_true() -> bool {
    true
}

impl ProviderRecord {
    /// Checks the invariants the type promises: the endpoint carries no
    /// userinfo and no legacy field looks like a credential. Run automatically
    /// when a record is deserialised; call it after building or editing one by
    /// hand.
    ///
    /// # Errors
    ///
    /// [`InvalidInput::Malformed`] for an endpoint with userinfo, and
    /// [`InvalidInput::CredentialField`] naming the first credential-shaped
    /// legacy field (its name only, never its value).
    pub fn validate(&self) -> Result<(), InvalidInput> {
        if endpoint_has_credentials(&self.base_url) {
            return Err(InvalidInput::Malformed {
                field: InputField::Endpoint,
                reason: "the endpoint carries a username or password",
            });
        }
        // A record's endpoint is empty (a subprocess kind) or an http(s) URL
        // with a host; anything else is refused at load, not left for the
        // policy layer to catch at request time.
        if !self.base_url.trim().is_empty() {
            let usable = url::Url::parse(self.base_url.trim())
                .is_ok_and(|url| matches!(url.scheme(), "http" | "https") && url.host().is_some());
            if !usable {
                return Err(InvalidInput::Malformed {
                    field: InputField::Endpoint,
                    reason: "the endpoint must be an http or https URL with a host",
                });
            }
        }
        if endpoint_query_has_credential(&self.base_url) {
            return Err(InvalidInput::Malformed {
                field: InputField::Endpoint,
                reason: "the endpoint carries a credential in its query string",
            });
        }
        for (name, value) in &self.legacy {
            let found = if is_credential_name(name) {
                Some(name.clone())
            } else {
                find_credential_field(value, 0)
            };
            if let Some(name) = found {
                return Err(InvalidInput::CredentialField { name });
            }
        }
        Ok(())
    }

    /// The migration path for a stored record that predates the credential
    /// rule: removes every top-level credential-shaped string field from
    /// `value` and returns it beside the cleaned value, so an import reader can
    /// move each one into the credential store and then deserialise the rest.
    ///
    /// Only top-level string fields are extracted (a null or empty one is just
    /// dropped, so it cannot overwrite a real key with nothing). A credential
    /// of another type, one nested deeper, or a userinfo endpoint is left in place, so deserialising the cleaned
    /// value still fails loudly rather than persisting it.
    pub fn extract_credentials(
        value: serde_json::Value,
    ) -> (serde_json::Value, Vec<ExtractedCredential>) {
        let serde_json::Value::Object(mut map) = value else {
            return (value, Vec::new());
        };
        let names: Vec<String> = map
            .keys()
            .filter(|name| is_credential_name(name))
            .cloned()
            .collect();
        let mut extracted = Vec::new();
        for name in names {
            // A null, empty or non-string credential field carries nothing to
            // move, so it is dropped rather than left to fail the load; a
            // string is handed back so it can be written to the store.
            match map.get(&name) {
                Some(serde_json::Value::String(text)) if !text.trim().is_empty() => {
                    if let Some(serde_json::Value::String(text)) = map.remove(&name) {
                        extracted.push(ExtractedCredential {
                            field: name,
                            value: Secret::new(text),
                        });
                    }
                }
                Some(serde_json::Value::Null | serde_json::Value::String(_)) => {
                    map.remove(&name);
                }
                _ => {}
            }
        }
        (serde_json::Value::Object(map), extracted)
    }

    /// A new enabled, non-synthetic record.
    pub fn new(
        id: impl Into<String>,
        slug: Slug,
        label: impl Into<String>,
        kind: KindId,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            slug,
            label: label.into(),
            kind,
            base_url: base_url.into(),
            model: None,
            enabled: true,
            auth_override: None,
            synthetic: false,
            origin: RecordOrigin::Indexed,
            legacy: LegacyFields::new(),
        }
    }
}
