//! OpenRouter image generation (`POST /images`).

use std::collections::HashMap;

use async_trait::async_trait;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tokio::sync::Mutex;

use crate::capabilities::ModelCapabilities;
use crate::media::GeneratedMedia;
use crate::reference::{
    DEFAULT_MAX_REFERENCE_BYTES, normalize_aspect_ratio, normalize_image_resolution, normalize_size,
};
use crate::transport::{MediaAuth, MediaTransport, wire_model_id};
use crate::types::{ImageRequest, ImageResponse, MediaModel};
use crate::{Error, ImageGenerator, Result};

/// Default image model: Seedream 5.0 Lite (flat per-image price,
/// image-to-image with up to 14 references, deterministic seed).
pub const DEFAULT_IMAGE_MODEL: &str = "bytedance-seed/seedream-5-0-lite";

/// Image generator for OpenRouter's image API, or any backend that proxies it
/// verbatim.
#[derive(Debug)]
pub struct OpenRouterImageGenerator {
    transport: MediaTransport,
    default_model: String,
    check_capabilities: bool,
    max_reference_bytes: usize,
    // Some(None) = listing unavailable; skip checks without refetching.
    capabilities: Mutex<Option<Option<HashMap<String, ModelCapabilities>>>>,
}

#[derive(Deserialize)]
struct WireResponse {
    #[serde(default)]
    created: Option<i64>,
    #[serde(default)]
    data: Vec<WireImage>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Deserialize)]
struct WireImage {
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    media_type: Option<String>,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    cost: Option<f64>,
}

impl OpenRouterImageGenerator {
    /// Creates a generator against OpenRouter's public API.
    #[must_use]
    pub fn new(auth: MediaAuth) -> Self {
        Self::with_transport(MediaTransport::new(auth))
    }

    /// Creates a generator from a pre-configured transport (base URL, client,
    /// headers, retry and size limits).
    #[must_use]
    pub fn with_transport(transport: MediaTransport) -> Self {
        Self {
            transport,
            default_model: DEFAULT_IMAGE_MODEL.to_owned(),
            check_capabilities: true,
            max_reference_bytes: DEFAULT_MAX_REFERENCE_BYTES,
            capabilities: Mutex::new(None),
        }
    }

    /// Creates a generator from `OPENROUTER_API_KEY`.
    ///
    /// # Errors
    ///
    /// [`Error::Auth`] when the variable is unset.
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(MediaAuth::from_env()?))
    }

    /// Sets the model used when a request names none.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Enables or disables pre-flight validation against the model listing
    /// (enabled by default; skipped silently when the listing is unavailable).
    #[must_use]
    pub fn with_capability_check(mut self, enabled: bool) -> Self {
        self.check_capabilities = enabled;
        self
    }

    /// Caps the size of each inlined reference (default 20 MiB).
    #[must_use]
    pub fn with_max_reference_bytes(mut self, max_reference_bytes: usize) -> Self {
        self.max_reference_bytes = max_reference_bytes;
        self
    }

    /// The underlying transport.
    #[must_use]
    pub fn transport(&self) -> &MediaTransport {
        &self.transport
    }

    async fn capabilities_for(&self, model: &str) -> Option<ModelCapabilities> {
        let mut cache = self.capabilities.lock().await;
        if cache.is_none() {
            match self.list_models().await {
                Ok(models) => {
                    *cache = Some(
                        models
                            .into_iter()
                            .map(|model| (wire_model_id(&model.id).to_owned(), model.capabilities))
                            .collect(),
                    );
                }
                Err(error) => {
                    tracing::debug!(
                        %error,
                        "[tinyinference-image] model listing unavailable; skipping capability check"
                    );
                    return None;
                }
            }
        }
        cache.as_ref()?.get(model).cloned()
    }

    fn validate_against(model: &str, request: &WireFields, caps: &ModelCapabilities) -> Result<()> {
        if let Some(value) = &request.aspect_ratio
            && value != "auto"
        {
            ModelCapabilities::check_one_of(
                model,
                "aspect_ratio",
                value,
                caps.aspect_ratios.as_deref(),
            )?;
        }
        if let Some(value) = &request.resolution {
            ModelCapabilities::check_one_of(
                model,
                "resolution",
                value,
                caps.resolutions.as_deref(),
            )?;
        }
        if let (Some(n), Some((min, max))) = (request.n, caps.n_range)
            && n > 1
            && !(min..=max).contains(&n)
        {
            return Err(Error::Unsupported {
                model: model.to_owned(),
                field: "n".into(),
                value: n.to_string(),
                allowed: vec![format!("{min}..={max}")],
            });
        }
        if let Some(max) = caps.max_references
            && request.references > max as usize
        {
            return Err(Error::Unsupported {
                model: model.to_owned(),
                field: "input_references".into(),
                value: request.references.to_string(),
                allowed: vec![format!("at most {max}")],
            });
        }
        if request.seed {
            ModelCapabilities::check_flag(model, "seed", caps.seed)?;
        }
        Ok(())
    }
}

/// Normalized fields the capability check reads.
struct WireFields {
    aspect_ratio: Option<String>,
    resolution: Option<String>,
    n: Option<u32>,
    references: usize,
    seed: bool,
}

/// Builds the wire body for `request`. Exposed for tests and for hosts that
/// need to audit exactly what leaves the process.
///
/// # Errors
///
/// Reference resolution errors from [`crate::MediaReference::resolve`].
pub async fn build_image_body(
    model: &str,
    request: &ImageRequest,
    max_reference_bytes: usize,
) -> Result<Value> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("prompt".into(), json!(request.prompt));
    if let Some(n) = request.n {
        body.insert("n".into(), json!(n));
    }
    let normalized = |value: &Option<String>, normalize: fn(&str) -> Option<String>| {
        value
            .as_deref()
            .map(|raw| normalize(raw).unwrap_or_else(|| raw.trim().to_owned()))
    };
    if let Some(size) = normalized(&request.size, normalize_size) {
        body.insert("size".into(), json!(size));
    }
    if let Some(resolution) = normalized(&request.resolution, normalize_image_resolution) {
        body.insert("resolution".into(), json!(resolution));
    }
    if let Some(aspect_ratio) = normalized(&request.aspect_ratio, normalize_aspect_ratio) {
        body.insert("aspect_ratio".into(), json!(aspect_ratio));
    }
    for (key, value) in [
        ("quality", &request.quality),
        ("output_format", &request.output_format),
        ("background", &request.background),
        ("user", &request.user),
        ("session_id", &request.session_id),
    ] {
        if let Some(value) = value {
            body.insert(key.into(), json!(value.trim()));
        }
    }
    if let Some(seed) = request.seed {
        body.insert("seed".into(), json!(seed));
    }
    if !request.references.is_empty() {
        let mut parts = Vec::with_capacity(request.references.len());
        for reference in &request.references {
            parts.push(reference.to_content_part(max_reference_bytes).await?);
        }
        body.insert("input_references".into(), Value::Array(parts));
    }
    for (key, value) in &request.extra {
        body.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Ok(Value::Object(body))
}

/// Sniffs a media type from magic bytes, for responses that omit it.
fn sniff_image_type(data: &[u8]) -> &'static str {
    match data {
        [0x89, b'P', b'N', b'G', ..] => "image/png",
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg",
        [
            b'R',
            b'I',
            b'F',
            b'F',
            _,
            _,
            _,
            _,
            b'W',
            b'E',
            b'B',
            b'P',
            ..,
        ] => "image/webp",
        [b'G', b'I', b'F', b'8', ..] => "image/gif",
        [b'<', ..] => "image/svg+xml",
        _ => "image/png",
    }
}

#[async_trait]
impl ImageGenerator for OpenRouterImageGenerator {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    async fn generate(&self, request: ImageRequest) -> Result<ImageResponse> {
        request.validate()?;
        let model =
            wire_model_id(request.model.as_deref().unwrap_or(&self.default_model)).to_owned();
        let body = build_image_body(&model, &request, self.max_reference_bytes).await?;

        if self.check_capabilities
            && let Some(caps) = self.capabilities_for(&model).await
        {
            let field = |key: &str| body.get(key).and_then(Value::as_str).map(str::to_owned);
            Self::validate_against(
                &model,
                &WireFields {
                    aspect_ratio: field("aspect_ratio"),
                    resolution: field("resolution"),
                    n: request.n,
                    references: request.references.len(),
                    seed: request.seed.is_some(),
                },
                &caps,
            )?;
        }

        tracing::info!(
            model = %model,
            n = request.n.unwrap_or(1),
            references = request.references.len(),
            base_url = %self.transport.base_url(),
            "[tinyinference-image] generating image"
        );
        let response: WireResponse = self.transport.post_json("images", &body).await?;
        let cost_usd = response.usage.as_ref().and_then(|usage| usage.cost);

        let mut images = Vec::with_capacity(response.data.len());
        for (index, image) in response.data.into_iter().enumerate() {
            let Some(encoded) = image.b64_json.filter(|value| !value.is_empty()) else {
                tracing::warn!(index, "[tinyinference-image] image entry without b64_json");
                continue;
            };
            if encoded.len() / 4 * 3 > self.transport.max_media_bytes() {
                return Err(Error::TooLarge {
                    limit: self.transport.max_media_bytes(),
                });
            }
            let data = BASE64.decode(encoded.as_bytes()).map_err(|error| {
                Error::Decode(format!("image {index} is not valid base64: {error}"))
            })?;
            let media_type = image
                .media_type
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| sniff_image_type(&data).to_owned());
            images.push(GeneratedMedia::new(media_type, data));
        }
        if images.is_empty() {
            tracing::warn!(
                model = %model,
                cost_usd,
                "[tinyinference-image] provider accepted the request but returned no images"
            );
            return Err(Error::NoMedia { request_id: None });
        }
        tracing::info!(
            model = %model,
            images = images.len(),
            cost_usd,
            "[tinyinference-image] image generation complete"
        );
        Ok(ImageResponse {
            model,
            images,
            cost_usd,
            created: response.created,
        })
    }

    async fn list_models(&self) -> Result<Vec<MediaModel>> {
        let body: Value = self.transport.get_json("images/models").await?;
        Ok(MediaModel::parse_listing(
            &body,
            ModelCapabilities::from_image_model,
        ))
    }
}
