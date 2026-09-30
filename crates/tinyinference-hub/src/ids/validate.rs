//! Validators ported from OpenCompany's `company/inference/store.rs`.
//!
//! The slug is *derived* from a typed name ([`slugify`]), never typed, so the
//! name an operator chooses and the address a route names cannot disagree. The
//! bounds are checked here rather than only in a console, because a console is
//! not a security boundary and the slug is the address of a secret
//! ([`Slug::key_slot`](super::Slug::key_slot)).

use std::fmt;

use crate::error::{HubError, InputField, InvalidInput};

use super::types::Slug;

/// The longest provider name or slug accepted, counted in `char`s.
pub const MAX_PROVIDER_NAME_CHARS: usize = 80;

/// The longest model id accepted, counted in `char`s (not bytes).
pub const MAX_MODEL_ID_CHARS: usize = 256;

/// Why a slug cannot be used. Four named failures rather than a boolean,
/// because they need four different sentences.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlugError {
    /// Nothing was typed, or it normalised to nothing.
    Empty,
    /// The scope already has a provider with that slug.
    Taken,
    /// The catalogue (or the host) reserves that name.
    Reserved,
    /// Past [`MAX_PROVIDER_NAME_CHARS`].
    TooLong,
    /// Not made of lowercase letters, digits, `-` and `_` (the alphabet a
    /// [`Slug`](super::Slug) accepts). The slug is the address of a secret, so a
    /// path separator or space is refused here, not just by `Slug::parse`.
    Invalid,
}

impl fmt::Display for SlugError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "a provider needs a name"),
            Self::Taken => write!(f, "this scope already has a provider with that name"),
            Self::Reserved => write!(f, "that name belongs to a built-in provider"),
            Self::TooLong => write!(
                f,
                "a provider name may be at most {MAX_PROVIDER_NAME_CHARS} characters"
            ),
            Self::Invalid => write!(
                f,
                "a provider slug can only use lowercase letters, digits, - and _"
            ),
        }
    }
}

impl std::error::Error for SlugError {}

impl SlugError {
    /// Converts to the hub's error, naming `slug` where the variant needs it.
    pub fn into_hub_error(self, slug: &str) -> HubError {
        match self {
            Self::Empty => HubError::Invalid(InvalidInput::Empty(InputField::Slug)),
            Self::TooLong => HubError::Invalid(InvalidInput::TooLong {
                field: InputField::Slug,
                max: MAX_PROVIDER_NAME_CHARS,
            }),
            Self::Invalid => HubError::Invalid(InvalidInput::BadCharacters(InputField::Slug)),
            Self::Reserved => HubError::Invalid(InvalidInput::Reserved {
                field: InputField::Slug,
                value: slug.trim().to_string(),
            }),
            Self::Taken => match Slug::parse(slug) {
                Ok(slug) => HubError::AlreadyExists { slug },
                Err(invalid) => HubError::Invalid(invalid),
            },
        }
    }
}

/// Turns a typed label into a slug: ASCII alphanumerics lowercased, every run
/// of anything else collapsed to one `-`, no leading or trailing `-`.
///
/// The result may be empty (a label with no alphanumerics) and may be reserved;
/// [`check_slug`] decides whether it is usable.
pub fn slugify(label: &str) -> String {
    let mut out = String::new();
    let mut last_dash = true;
    for ch in label.trim().chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            last_dash = false;
        } else if !last_dash {
            out.push('-');
            last_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    out
}

/// Whether a typed provider **name** may be used at all, before any slug is
/// derived from it.
///
/// Separate from [`check_slug`] because the two bound different things: a name
/// can be long while its slug is short (`slugify` drops everything that is not
/// alphanumeric), so bounding only the slug would leave a page of prose in the
/// store under a three-character address.
///
/// # Errors
///
/// [`SlugError::Empty`] for a blank name and [`SlugError::TooLong`] past
/// [`MAX_PROVIDER_NAME_CHARS`] characters.
pub fn check_provider_name(label: &str) -> Result<(), SlugError> {
    let label = label.trim();
    if label.is_empty() {
        return Err(SlugError::Empty);
    }
    if label.chars().count() > MAX_PROVIDER_NAME_CHARS {
        return Err(SlugError::TooLong);
    }
    Ok(())
}

/// Whether `slug` may be used for a new provider in a scope that already holds
/// `existing`.
///
/// `is_reserved` is normally `catalogue::is_reserved_slug`: the reservation
/// applies to *typed* names only, because adding the catalogue's own `groq` row
/// should take the slug `groq` — that is the same provider, not a collision.
/// It is a typed name shadowing a built-in that has to be refused, because a
/// route saying `groq` would then mean two things.
///
/// # Errors
///
/// [`SlugError::Empty`], [`SlugError::TooLong`], [`SlugError::Invalid`],
/// [`SlugError::Taken`] or [`SlugError::Reserved`], checked in that order.
pub fn check_slug<I, S>(
    existing: I,
    slug: &str,
    is_reserved: impl Fn(&str) -> bool,
) -> Result<(), SlugError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let slug = slug.trim();
    if slug.is_empty() {
        return Err(SlugError::Empty);
    }
    if slug.chars().count() > MAX_PROVIDER_NAME_CHARS {
        return Err(SlugError::TooLong);
    }
    if Slug::parse(slug).is_err() {
        return Err(SlugError::Invalid);
    }
    if existing.into_iter().any(|p| p.as_ref() == slug) {
        return Err(SlugError::Taken);
    }
    if is_reserved(slug) {
        return Err(SlugError::Reserved);
    }
    Ok(())
}

/// The one validation every model-id write goes through. Returns the trimmed
/// id.
///
/// `reserved` holds host words that are not models (OpenCompany passes its
/// workload tier names); the hub itself reserves none (D7).
///
/// # Errors
///
/// [`InvalidInput`] when the id is empty, contains a control character or
/// whitespace, is longer than [`MAX_MODEL_ID_CHARS`] characters, or is one of
/// the `reserved` words.
pub fn check_model_id(raw: &str, reserved: &[&str]) -> Result<String, InvalidInput> {
    let id = raw.trim();
    if id.is_empty() {
        return Err(InvalidInput::Empty(InputField::ModelId));
    }
    if id.chars().any(char::is_control) {
        return Err(InvalidInput::ControlCharacters(InputField::ModelId));
    }
    if id.chars().any(char::is_whitespace) {
        return Err(InvalidInput::Whitespace(InputField::ModelId));
    }
    if id.chars().count() > MAX_MODEL_ID_CHARS {
        return Err(InvalidInput::TooLong {
            field: InputField::ModelId,
            max: MAX_MODEL_ID_CHARS,
        });
    }
    if reserved.contains(&id) {
        return Err(InvalidInput::Reserved {
            field: InputField::ModelId,
            value: id.to_string(),
        });
    }
    Ok(id.to_string())
}
