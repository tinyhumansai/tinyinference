//! The identifier types themselves.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{InputField, InvalidInput};

use super::validate::{MAX_PROVIDER_NAME_CHARS, check_model_id};

/// Declares an opaque, host-owned string key: no validation, because the host
/// decides what a scope, an agent or a workload is called (D7: the hub never
/// interprets workload vocabulary).
macro_rules! opaque_key {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(String);

        impl $name {
            /// Wraps a host-chosen key.
            pub fn new(value: impl Into<String>) -> Self {
                Self(value.into())
            }

            /// The key as written by the host.
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self::new(value)
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self::new(value)
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

opaque_key! {
    /// The tenancy scope every operation runs in, for example `company:acme` or
    /// `user:local`. Opaque to the hub; the catalog cache and the health map are
    /// partitioned by it.
    ScopeKey
}

opaque_key! {
    /// A host's agent identifier, used for per-agent model pins.
    AgentKey
}

opaque_key! {
    /// A host's workload identifier (D7: an opaque key, never a tier name the
    /// hub understands).
    WorkloadKey
}

/// A provider kind id such as `openai`, `custom` or `tinyhumans`: the key into
/// the catalogue. ASCII-lowercased and trimmed on construction so lookups are
/// case-insensitive; an unknown kind is a lookup miss, not a construction
/// error.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub struct KindId(String);

impl KindId {
    /// Normalises and wraps a kind id.
    pub fn new(value: impl AsRef<str>) -> Self {
        Self(value.as_ref().trim().to_ascii_lowercase())
    }

    /// The normalised kind id.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KindId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for KindId {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

/// Deserialisation goes through here, so a stored `"Custom"` still resolves.
impl From<String> for KindId {
    fn from(value: String) -> Self {
        Self::new(value)
    }
}

impl From<KindId> for String {
    fn from(value: KindId) -> Self {
        value.0
    }
}

/// A provider's routing key: what a route names and what a credential slot is
/// keyed on. Lowercase ASCII letters, digits, `-` and `_`, starting with a
/// letter or digit, at most 80 characters. Deserialisation validates, so a
/// stored record cannot smuggle in a slug the API would refuse.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Slug(String);

impl Slug {
    /// Parses and validates a slug (surrounding whitespace is trimmed).
    ///
    /// # Errors
    ///
    /// [`InvalidInput`] when the slug is empty, longer than 80 characters, or
    /// contains anything but lowercase ASCII letters, digits, `-` and `_`, or
    /// does not start with a letter or digit.
    pub fn parse(raw: &str) -> Result<Self, InvalidInput> {
        let slug = raw.trim();
        if slug.is_empty() {
            return Err(InvalidInput::Empty(InputField::Slug));
        }
        if slug.chars().count() > MAX_PROVIDER_NAME_CHARS {
            return Err(InvalidInput::TooLong {
                field: InputField::Slug,
                max: MAX_PROVIDER_NAME_CHARS,
            });
        }
        let allowed =
            |c: char| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_';
        if !slug.chars().all(allowed) || !slug.starts_with(|c: char| c.is_ascii_alphanumeric()) {
            return Err(InvalidInput::BadCharacters(InputField::Slug));
        }
        Ok(Self(slug.to_string()))
    }

    /// The slug as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The vault slot a provider's key lives in: `provider/<slug>/key`
    /// (OpenCompany's slot naming, kept so its adoption needs no key
    /// migration).
    pub fn key_slot(&self) -> String {
        format!("provider/{}/key", self.0)
    }
}

impl fmt::Display for Slug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for Slug {
    type Error = InvalidInput;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<Slug> for String {
    fn from(value: Slug) -> Self {
        value.0
    }
}

impl AsRef<str> for Slug {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// A model id: non-empty, no control characters or whitespace, at most 256
/// characters. Never checked against a catalogue, because catalogues go stale
/// and an Azure deployment name is never in `/models`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct ModelId(String);

impl ModelId {
    /// Parses a model id with no reserved words.
    ///
    /// # Errors
    ///
    /// See [`check_model_id`].
    pub fn parse(raw: &str) -> Result<Self, InvalidInput> {
        Self::parse_with_reserved(raw, &[])
    }

    /// Parses a model id, refusing any of the host-supplied reserved words
    /// (OpenCompany passes its tier names).
    ///
    /// # Errors
    ///
    /// See [`check_model_id`].
    pub fn parse_with_reserved(raw: &str, reserved: &[&str]) -> Result<Self, InvalidInput> {
        check_model_id(raw, reserved).map(Self)
    }

    /// The model id as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl TryFrom<String> for ModelId {
    type Error = InvalidInput;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::parse(&value)
    }
}

impl From<ModelId> for String {
    fn from(value: ModelId) -> Self {
        value.0
    }
}

impl AsRef<str> for ModelId {
    fn as_ref(&self) -> &str {
        &self.0
    }
}
