//! Media reference and output-shape standards shared by image and video
//! generation.
//!
//! Callers hand references to a generator in whatever form they hold them —
//! an HTTP(S) URL, a `data:` URL, raw bytes, or a local file path — and this
//! module normalizes each one into the content-part shape OpenRouter's media
//! APIs accept (`{"type": "image_url", "image_url": {"url": …}}`, and the
//! `video_url` / `audio_url` equivalents). Local files and bytes are inlined as
//! base64 `data:` URLs, bounded by a size cap, so a reference never depends on
//! the provider being able to reach the caller's filesystem.
//!
//! It also normalizes the loose spellings people and models use for output
//! shape — `"16x9"`, `"landscape"`, `"1080"`, `"full hd"`, `"1024×1024"` — into
//! the canonical values the wire format expects.

use std::path::{Path, PathBuf};

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use bytes::Bytes;
use serde::Serialize;

use crate::{Error, Result};

/// Default cap on one inlined reference (20 MiB of raw bytes).
pub const DEFAULT_MAX_REFERENCE_BYTES: usize = 20 * 1024 * 1024;

/// The modality a reference carries, which selects its wire content-part type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    /// A still image (`image_url`).
    Image,
    /// A video clip (`video_url`).
    Video,
    /// An audio clip (`audio_url`).
    Audio,
}

impl ReferenceKind {
    /// Infers the kind from a MIME type, defaulting to [`ReferenceKind::Image`].
    #[must_use]
    pub fn from_media_type(media_type: &str) -> Self {
        let media_type = media_type.trim().to_ascii_lowercase();
        if media_type.starts_with("video/") {
            Self::Video
        } else if media_type.starts_with("audio/") {
            Self::Audio
        } else {
            Self::Image
        }
    }
}

/// A caller-supplied reference asset.
#[derive(Clone, PartialEq, Eq)]
pub enum MediaReference {
    /// A publicly reachable HTTP(S) URL, forwarded as-is.
    Url(String),
    /// An inline `data:<media-type>;base64,<payload>` URL, forwarded as-is.
    DataUrl(String),
    /// Raw bytes with their media type, inlined as a `data:` URL.
    Bytes {
        /// MIME type such as `image/png` or `video/mp4`.
        media_type: String,
        /// The asset bytes.
        data: Bytes,
    },
    /// A local file, read and inlined as a `data:` URL when the request is built.
    Path(PathBuf),
    /// A URL with an explicitly-specified modality, useful for extensionless CDN URLs.
    Typed {
        /// The modality of this reference (image, video, or audio).
        kind: ReferenceKind,
        /// The URL, either HTTP(S) or a local file path.
        url: String,
    },
}

impl MediaReference {
    /// Classifies a free-form string: `http(s)://` is a URL, `data:` is a data
    /// URL, and anything else is treated as a local path.
    #[must_use]
    pub fn parse(value: &str) -> Self {
        let value = value.trim();
        let lower = value.to_ascii_lowercase();
        if lower.starts_with("http://") || lower.starts_with("https://") {
            Self::Url(value.to_owned())
        } else if lower.starts_with("data:") {
            Self::DataUrl(value.to_owned())
        } else {
            Self::Path(PathBuf::from(value))
        }
    }

    /// The reference's modality, from its media type, data-URL header, or
    /// file/URL extension.
    #[must_use]
    pub fn kind(&self) -> ReferenceKind {
        match self {
            Self::Typed { kind, .. } => *kind,
            Self::Bytes { media_type, .. } => ReferenceKind::from_media_type(media_type),
            Self::DataUrl(url) => ReferenceKind::from_media_type(
                url.get(5..)
                    .and_then(|rest| rest.split([';', ',']).next())
                    .unwrap_or_default(),
            ),
            Self::Url(url) => {
                let path = url.split(['?', '#']).next().unwrap_or(url);
                ReferenceKind::from_media_type(media_type_for_path(Path::new(path)))
            }
            Self::Path(path) => ReferenceKind::from_media_type(media_type_for_path(path)),
        }
    }

    /// Resolves the reference into the URL string sent on the wire, inlining
    /// bytes and local files as base64 `data:` URLs.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty or malformed reference,
    /// [`Error::TooLarge`] when the asset exceeds `max_bytes`, and
    /// [`Error::Io`] when a local file cannot be read.
    pub async fn resolve(&self, max_bytes: usize) -> Result<String> {
        match self {
            Self::Typed { url, .. } => {
                if url.trim().is_empty() {
                    return Err(Error::Validation("reference URL is empty".into()));
                }
                Ok(url.clone())
            }
            Self::Url(url) => {
                if url.trim().is_empty() {
                    return Err(Error::Validation("reference URL is empty".into()));
                }
                Ok(url.clone())
            }
            Self::DataUrl(url) => {
                let Some((header, payload)) = url.split_once(',') else {
                    return Err(Error::Validation("malformed data: URL reference".into()));
                };
                if !header.to_ascii_lowercase().starts_with("data:") || payload.is_empty() {
                    return Err(Error::Validation("malformed data: URL reference".into()));
                }
                // Decode and validate the actual payload size.
                let decoded = BASE64
                    .decode(payload)
                    .map_err(|_| Error::Validation("malformed base64 in data: URL".into()))?;
                if decoded.len() > max_bytes {
                    return Err(Error::TooLarge { limit: max_bytes });
                }
                Ok(url.clone())
            }
            Self::Bytes { media_type, data } => {
                if data.is_empty() {
                    return Err(Error::Validation("reference bytes are empty".into()));
                }
                if data.len() > max_bytes {
                    return Err(Error::TooLarge { limit: max_bytes });
                }
                Ok(data_url(media_type, data))
            }
            Self::Path(path) => {
                let metadata = tokio::fs::metadata(path).await?;
                if metadata.len() > max_bytes as u64 {
                    return Err(Error::TooLarge { limit: max_bytes });
                }
                let data = tokio::fs::read(path).await?;
                if data.is_empty() {
                    return Err(Error::Validation(format!(
                        "reference file {} is empty",
                        path.display()
                    )));
                }
                Ok(data_url(media_type_for_path(path), &data))
            }
        }
    }

    /// Resolves the reference into an OpenRouter content part.
    ///
    /// # Errors
    ///
    /// As for [`MediaReference::resolve`].
    pub async fn to_content_part(&self, max_bytes: usize) -> Result<serde_json::Value> {
        let url = self.resolve(max_bytes).await?;
        Ok(content_part(self.kind(), &url))
    }
}

impl std::fmt::Debug for MediaReference {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print inline payloads or signed-URL query strings.
        match self {
            Self::Typed { kind, url } => {
                let sanitized = url.split(['?', '#']).next().unwrap_or(url);
                let sanitized = sanitized.split('@').next_back().unwrap_or(sanitized);
                formatter
                    .debug_struct("Typed")
                    .field("kind", kind)
                    .field("url", &sanitized)
                    .finish()
            }
            Self::Url(url) => {
                let sanitized = url.split(['?', '#']).next().unwrap_or(url);
                let sanitized = sanitized.split('@').next_back().unwrap_or(sanitized);
                formatter.debug_tuple("Url").field(&sanitized).finish()
            }
            Self::DataUrl(_) => formatter
                .debug_struct("DataUrl")
                .field("data", &"<redacted>")
                .finish(),
            Self::Bytes { media_type, data } => formatter
                .debug_struct("Bytes")
                .field("media_type", media_type)
                .field("len", &data.len())
                .finish(),
            Self::Path(path) => formatter.debug_tuple("Path").field(path).finish(),
        }
    }
}

/// Builds an OpenRouter content part for a resolved URL.
#[must_use]
pub fn content_part(kind: ReferenceKind, url: &str) -> serde_json::Value {
    let key = match kind {
        ReferenceKind::Image => "image_url",
        ReferenceKind::Video => "video_url",
        ReferenceKind::Audio => "audio_url",
    };
    serde_json::json!({ "type": key, key: { "url": url } })
}

fn data_url(media_type: &str, data: &[u8]) -> String {
    format!("data:{media_type};base64,{}", BASE64.encode(data))
}

/// Guesses a MIME type from a file extension, defaulting to
/// `application/octet-stream`.
#[must_use]
pub fn media_type_for_path(path: &Path) -> &'static str {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "heic" => "image/heic",
        "avif" => "image/avif",
        "mp4" | "m4v" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "m4a" => "audio/mp4",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        _ => "application/octet-stream",
    }
}

/// Picks a file extension for a MIME type, falling back to `fallback`.
#[must_use]
pub fn extension_for_media_type<'a>(media_type: &str, fallback: &'a str) -> &'a str {
    let media_type = media_type
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    match media_type.as_str() {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/svg+xml" => "svg",
        "image/avif" => "avif",
        "video/mp4" => "mp4",
        "video/webm" => "webm",
        "video/quicktime" => "mov",
        "audio/mpeg" => "mp3",
        "audio/wav" | "audio/x-wav" => "wav",
        _ => fallback,
    }
}

/// Normalizes an aspect-ratio spelling to the canonical `W:H` form.
///
/// Accepts `16:9`, `16x9`, `16/9`, `16 × 9`, and the names `square`,
/// `landscape`, `portrait`, `widescreen`, `vertical`, `ultrawide`, `auto`.
/// Returns `None` when the value is not recognizable, in which case callers
/// should forward it unchanged and let the provider decide.
#[must_use]
pub fn normalize_aspect_ratio(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    let named = match value.as_str() {
        "auto" => Some("auto"),
        "square" => Some("1:1"),
        "landscape" | "widescreen" | "horizontal" => Some("16:9"),
        "portrait" | "vertical" | "story" | "reel" => Some("9:16"),
        "ultrawide" | "cinematic" => Some("21:9"),
        _ => None,
    };
    if let Some(named) = named {
        return Some(named.to_owned());
    }
    let separators: &[char] = &[':', 'x', '/', '×', '*'];
    let mut parts = value.split(separators).map(str::trim);
    let (width, height) = (parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let valid = |part: &str| !part.is_empty() && part.parse::<f64>().is_ok_and(|n| n > 0.0);
    (valid(width) && valid(height)).then(|| format!("{width}:{height}"))
}

/// Normalizes an image resolution tier (`512`, `1K`, `2K`, `4K`).
///
/// Accepts case and spelling variants (`1k`, `1024`, `2048`, `4096`, `hd`,
/// `4k uhd`). Returns `None` when unrecognized.
#[must_use]
pub fn normalize_image_resolution(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase().replace(' ', "");
    let tier = match value.as_str() {
        "512" | "0.5k" | "sd" => "512",
        "1k" | "1024" | "hd" => "1K",
        "2k" | "2048" | "qhd" => "2K",
        "4k" | "4096" | "uhd" | "4kuhd" => "4K",
        _ => return None,
    };
    Some(tier.to_owned())
}

/// Normalizes a video resolution (`360p` … `1080p`, `1K`, `2K`, `4K`).
///
/// Accepts `720`, `720P`, `hd`, `full hd`, `fhd`, `1080`, `4k`, `uhd`.
/// Returns `None` when unrecognized.
#[must_use]
pub fn normalize_video_resolution(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase().replace([' ', '-'], "");
    let resolution = match value.as_str() {
        "360" | "360p" => "360p",
        "480" | "480p" | "sd" => "480p",
        "720" | "720p" | "hd" => "720p",
        "768" | "768p" => "768p",
        "1080" | "1080p" | "fullhd" | "fhd" => "1080p",
        "1k" => "1K",
        "2k" | "1440p" | "qhd" => "2K",
        "4k" | "2160p" | "uhd" => "4K",
        _ => return None,
    };
    Some(resolution.to_owned())
}

/// Normalizes an explicit pixel size to `WIDTHxHEIGHT`, accepting `×`, `*`,
/// and surrounding whitespace. Tier sizes (`2K`) are returned uppercased.
/// Returns `None` when the value is neither.
#[must_use]
pub fn normalize_size(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if let Some(tier) =
        normalize_image_resolution(trimmed).filter(|_| trimmed.ends_with(['k', 'K']))
    {
        return Some(tier);
    }
    let lower = trimmed.to_ascii_lowercase();
    let mut parts = lower.split(['x', '×', '*']).map(str::trim);
    let (width, height) = (parts.next()?, parts.next()?);
    if parts.next().is_some() {
        return None;
    }
    let width: u32 = width.parse().ok().filter(|n| *n > 0)?;
    let height: u32 = height.parse().ok().filter(|n| *n > 0)?;
    Some(format!("{width}x{height}"))
}
