//! Public request, response and model-listing types for image generation.

use serde_json::Value;

use crate::capabilities::ModelCapabilities;
use crate::media::GeneratedMedia;
use crate::reference::MediaReference;
use crate::{Error, Result};

/// Maximum images per request accepted by OpenRouter's image API.
pub const MAX_IMAGES_PER_REQUEST: u32 = 10;

/// A provider-neutral image generation request.
///
/// Output-shape fields accept loose spellings (`"16x9"`, `"landscape"`,
/// `"2k"`, `"1024×1024"`); providers normalize them with the helpers in
/// [`reference`](mod@crate::reference) and forward anything unrecognized unchanged.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ImageRequest {
    /// Model id; `None` uses the generator's default. An `openrouter/` prefix
    /// is accepted and stripped.
    pub model: Option<String>,
    /// Text description of the desired image, or the edit instruction when
    /// references are supplied.
    pub prompt: String,
    /// Number of images (1–10); providers may return fewer.
    pub n: Option<u32>,
    /// Pixel size (`1536x1024`) or tier shorthand (`2K`).
    pub size: Option<String>,
    /// Resolution tier (`512`, `1K`, `2K`, `4K`).
    pub resolution: Option<String>,
    /// Aspect ratio (`16:9`, `auto`, …).
    pub aspect_ratio: Option<String>,
    /// Rendering quality (`auto`, `low`, `medium`, `high`).
    pub quality: Option<String>,
    /// Output encoding (`png`, `jpeg`, `webp`, `svg`).
    pub output_format: Option<String>,
    /// Background treatment (`auto`, `transparent`, `opaque`).
    pub background: Option<String>,
    /// Deterministic seed, where supported.
    pub seed: Option<i64>,
    /// Reference images for image-to-image generation and editing.
    pub references: Vec<MediaReference>,
    /// Stable end-user identifier for provider abuse detection.
    pub user: Option<String>,
    /// Observability grouping id (never sent to the upstream model provider).
    pub session_id: Option<String>,
    /// Provider-specific extra fields merged into the wire body.
    pub extra: serde_json::Map<String, Value>,
}

impl ImageRequest {
    /// Creates a request for `prompt` with every other field defaulted.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            ..Self::default()
        }
    }

    /// Sets the model id.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the number of images.
    #[must_use]
    pub fn with_n(mut self, n: u32) -> Self {
        self.n = Some(n);
        self
    }

    /// Sets the aspect ratio.
    #[must_use]
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Sets the resolution tier.
    #[must_use]
    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    /// Sets the pixel size or tier shorthand.
    #[must_use]
    pub fn with_size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }

    /// Sets the seed.
    #[must_use]
    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Adds a reference image.
    #[must_use]
    pub fn with_reference(mut self, reference: MediaReference) -> Self {
        self.references.push(reference);
        self
    }

    /// Checks the fields that are invalid for every model.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] for a blank prompt or an out-of-range `n`.
    pub fn validate(&self) -> Result<()> {
        if self.prompt.trim().is_empty() {
            return Err(Error::Validation("prompt is required".into()));
        }
        if let Some(n) = self.n
            && !(1..=MAX_IMAGES_PER_REQUEST).contains(&n)
        {
            return Err(Error::Validation(format!(
                "n must be between 1 and {MAX_IMAGES_PER_REQUEST}, got {n}"
            )));
        }
        Ok(())
    }
}

/// The result of a successful image generation. Always carries at least one
/// image: an accepted request that returns none fails with
/// [`Error::NoMedia`] instead.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageResponse {
    /// Wire model id that served the request.
    pub model: String,
    /// Generated images, in provider order.
    pub images: Vec<GeneratedMedia>,
    /// Provider-reported cost in USD, when available.
    pub cost_usd: Option<f64>,
    /// Provider creation timestamp (Unix seconds), when available.
    pub created: Option<i64>,
}

/// One entry from a media model listing.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MediaModel {
    /// Model slug to pass as `model`.
    pub id: String,
    /// Human-readable name, when provided.
    pub name: Option<String>,
    /// Short description, when provided.
    pub description: Option<String>,
    /// Advertised capabilities (fields are `None` when not advertised).
    pub capabilities: ModelCapabilities,
    /// The raw listing record, for fields this crate does not model.
    pub raw: Value,
}

impl MediaModel {
    /// Reads every record in a listing body. Accepts `{ "data": [...] }` (both
    /// OpenRouter and proxying backends) or a bare array.
    #[must_use]
    pub fn parse_listing(body: &Value, capabilities: fn(&Value) -> ModelCapabilities) -> Vec<Self> {
        let records = body
            .get("data")
            .and_then(Value::as_array)
            .or_else(|| body.as_array());
        records
            .map(|records| {
                records
                    .iter()
                    .filter_map(|record| {
                        let id = record.get("id").and_then(Value::as_str)?.to_owned();
                        let text =
                            |key: &str| record.get(key).and_then(Value::as_str).map(str::to_owned);
                        Some(Self {
                            id,
                            name: text("name").or_else(|| text("display_name")),
                            description: text("description"),
                            capabilities: capabilities(record),
                            raw: record.clone(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}
