//! Generated media artifacts and their persistence.

use std::path::{Path, PathBuf};

use bytes::Bytes;

use crate::reference::extension_for_media_type;
use crate::{Error, Result};

/// One generated artifact, held in memory.
///
/// Providers return media either inline (OpenRouter images arrive as base64)
/// or behind an authenticated content endpoint (OpenRouter videos); generators
/// always download before returning, so a `GeneratedMedia` is self-contained
/// and never carries a URL that expires or needs a credential.
#[derive(Clone, PartialEq, Eq)]
pub struct GeneratedMedia {
    /// MIME type, for example `image/png` or `video/mp4`.
    pub media_type: String,
    /// The artifact bytes.
    pub data: Bytes,
}

impl GeneratedMedia {
    /// Creates an artifact from its media type and bytes.
    #[must_use]
    pub fn new(media_type: impl Into<String>, data: impl Into<Bytes>) -> Self {
        Self {
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    /// File extension for this artifact's media type, or `fallback`.
    #[must_use]
    pub fn extension<'a>(&self, fallback: &'a str) -> &'a str {
        extension_for_media_type(&self.media_type, fallback)
    }

    /// Writes the artifact to `dir/<stem>.<ext>`, creating `dir` if needed,
    /// and returns the written path. `fallback_extension` is used when the
    /// media type is unknown.
    ///
    /// The stem is sanitized to `[A-Za-z0-9_-]` so a provider id can never
    /// traverse out of `dir`.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for an empty artifact, [`Error::Io`] when the
    /// directory or file cannot be written.
    pub async fn persist(
        &self,
        dir: &Path,
        stem: &str,
        fallback_extension: &str,
    ) -> Result<PathBuf> {
        if self.data.is_empty() {
            return Err(Error::Validation("refusing to persist an empty artifact".into()));
        }
        tokio::fs::create_dir_all(dir).await?;
        let stem: String = stem
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
            .collect();
        let stem = if stem.is_empty() { "media".to_owned() } else { stem };
        let path = dir.join(format!("{stem}.{}", self.extension(fallback_extension)));
        tokio::fs::write(&path, &self.data).await?;
        tracing::debug!(
            path = %path.display(),
            bytes = self.data.len(),
            media_type = %self.media_type,
            "[tinyinference-image] persisted generated media"
        );
        Ok(path)
    }
}

impl std::fmt::Debug for GeneratedMedia {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("GeneratedMedia")
            .field("media_type", &self.media_type)
            .field("len", &self.data.len())
            .finish()
    }
}
