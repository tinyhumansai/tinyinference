//! OpenRouter video generation (`POST /videos`, `GET /videos/{id}`,
//! `GET /videos/{id}/content`).

use std::collections::HashMap;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use tinyinference_image::reference::{
    DEFAULT_MAX_REFERENCE_BYTES, normalize_aspect_ratio, normalize_size, normalize_video_resolution,
};
use tinyinference_image::transport::wire_model_id;
use tinyinference_image::{GeneratedMedia, MediaAuth, MediaModel, MediaTransport, ModelCapabilities};
use tokio::sync::Mutex;

use crate::types::{JobState, VideoJob, VideoJobStatus, VideoRequest};
use crate::{Error, Result, VideoGenerator};

/// Default video model: Seedance 2.0 Mini (text/image-to-video, first and last
/// frame control, 4–15 s, 480p/720p, optional audio).
pub const DEFAULT_VIDEO_MODEL: &str = "bytedance/seedance-2.0-mini";

/// Video generator for OpenRouter's video API, or any backend that proxies it
/// verbatim.
#[derive(Debug)]
pub struct OpenRouterVideoGenerator {
    transport: MediaTransport,
    default_model: String,
    check_capabilities: bool,
    max_reference_bytes: usize,
    capabilities: Mutex<Option<HashMap<String, ModelCapabilities>>>,
}

#[derive(Deserialize)]
struct WireJob {
    id: String,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    unsigned_urls: Vec<String>,
    #[serde(default)]
    usage: Option<WireUsage>,
    #[serde(default)]
    error: Option<Value>,
}

#[derive(Deserialize)]
struct WireUsage {
    #[serde(default)]
    cost: Option<f64>,
}

impl OpenRouterVideoGenerator {
    /// Creates a generator against OpenRouter's public API.
    #[must_use]
    pub fn new(auth: MediaAuth) -> Self {
        Self::with_transport(MediaTransport::new(auth))
    }

    /// Creates a generator from a pre-configured transport.
    #[must_use]
    pub fn with_transport(transport: MediaTransport) -> Self {
        Self {
            transport,
            default_model: DEFAULT_VIDEO_MODEL.to_owned(),
            check_capabilities: true,
            max_reference_bytes: DEFAULT_MAX_REFERENCE_BYTES,
            capabilities: Mutex::new(None),
        }
    }

    /// Creates a generator from `OPENROUTER_API_KEY`.
    ///
    /// # Errors
    ///
    /// [`Error::Media`] wrapping an auth error when the variable is unset.
    pub fn from_env() -> Result<Self> {
        Ok(Self::new(MediaAuth::from_env()?))
    }

    /// Sets the model used when a request names none.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Enables or disables pre-flight validation against the model listing.
    #[must_use]
    pub fn with_capability_check(mut self, enabled: bool) -> Self {
        self.check_capabilities = enabled;
        self
    }

    /// Caps the size of each inlined reference or frame (default 20 MiB).
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
                        "[tinyinference-video] model listing unavailable; skipping capability check"
                    );
                    return None;
                }
            }
        }
        cache.as_ref()?.get(model).cloned()
    }
}

fn validate_against(
    model: &str,
    body: &Value,
    request: &VideoRequest,
    caps: &ModelCapabilities,
) -> tinyinference_image::Result<()> {
    let field = |key: &str| body.get(key).and_then(Value::as_str);
    if let Some(value) = field("resolution") {
        ModelCapabilities::check_one_of(model, "resolution", value, caps.resolutions.as_deref())?;
    }
    if let Some(value) = field("aspect_ratio") {
        ModelCapabilities::check_one_of(model, "aspect_ratio", value, caps.aspect_ratios.as_deref())?;
    }
    if let (Some(duration), Some(allowed)) = (request.duration_s, caps.durations.as_ref()) {
        let allowed: Vec<String> = allowed.iter().map(u32::to_string).collect();
        ModelCapabilities::check_one_of(model, "duration", &duration.to_string(), Some(&allowed))?;
    }
    for (role, present) in [
        ("first_frame", request.first_frame.is_some()),
        ("last_frame", request.last_frame.is_some()),
    ] {
        if present {
            ModelCapabilities::check_one_of(model, "frame_images", role, caps.frame_images.as_deref())?;
        }
    }
    if request.generate_audio == Some(true) {
        ModelCapabilities::check_flag(model, "generate_audio", caps.generate_audio)?;
    }
    if request.seed.is_some() {
        ModelCapabilities::check_flag(model, "seed", caps.seed)?;
    }
    Ok(())
}

/// Builds the wire body for `request`.
///
/// # Errors
///
/// Reference resolution errors from [`tinyinference_image::MediaReference::resolve`].
pub async fn build_video_body(
    model: &str,
    request: &VideoRequest,
    max_reference_bytes: usize,
) -> tinyinference_image::Result<Value> {
    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    if let Some(prompt) = request.prompt.as_deref().filter(|p| !p.trim().is_empty()) {
        body.insert("prompt".into(), json!(prompt));
    }
    if let Some(duration) = request.duration_s {
        body.insert("duration".into(), json!(duration));
    }
    let normalized = |value: &Option<String>, normalize: fn(&str) -> Option<String>| {
        value
            .as_deref()
            .map(|raw| normalize(raw).unwrap_or_else(|| raw.trim().to_owned()))
    };
    if let Some(resolution) = normalized(&request.resolution, normalize_video_resolution) {
        body.insert("resolution".into(), json!(resolution));
    }
    if let Some(aspect_ratio) = normalized(&request.aspect_ratio, normalize_aspect_ratio) {
        body.insert("aspect_ratio".into(), json!(aspect_ratio));
    }
    if let Some(size) = normalized(&request.size, normalize_size) {
        body.insert("size".into(), json!(size));
    }
    if let Some(generate_audio) = request.generate_audio {
        body.insert("generate_audio".into(), json!(generate_audio));
    }
    if let Some(seed) = request.seed {
        body.insert("seed".into(), json!(seed));
    }
    let mut frames = Vec::new();
    for (role, frame) in [
        ("first_frame", &request.first_frame),
        ("last_frame", &request.last_frame),
    ] {
        if let Some(frame) = frame {
            let url = frame.resolve(max_reference_bytes).await?;
            frames.push(json!({
                "type": "image_url",
                "image_url": { "url": url },
                "frame_type": role,
            }));
        }
    }
    if !frames.is_empty() {
        body.insert("frame_images".into(), Value::Array(frames));
    }
    if !request.references.is_empty() {
        let mut parts = Vec::with_capacity(request.references.len());
        for reference in &request.references {
            parts.push(reference.to_content_part(max_reference_bytes).await?);
        }
        body.insert("input_references".into(), Value::Array(parts));
    }
    for (key, value) in [
        ("previous_job_id", &request.previous_job_id),
        ("user", &request.user),
        ("session_id", &request.session_id),
    ] {
        if let Some(value) = value {
            body.insert(key.into(), json!(value));
        }
    }
    for (key, value) in &request.extra {
        body.entry(key.clone()).or_insert_with(|| value.clone());
    }
    Ok(Value::Object(body))
}

/// Refuses job ids that could escape the `videos/{id}` path segment.
fn checked_job_id(job_id: &str) -> tinyinference_image::Result<&str> {
    let valid = !job_id.is_empty()
        && job_id.len() <= 256
        && job_id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_');
    if valid {
        Ok(job_id)
    } else {
        Err(tinyinference_image::Error::Validation(format!(
            "invalid video job id {job_id:?}"
        )))
    }
}

fn error_text(error: Option<Value>) -> Option<String> {
    match error? {
        Value::String(text) => Some(text),
        Value::Null => None,
        other => other
            .pointer("/message")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| Some(other.to_string())),
    }
}

#[async_trait]
impl VideoGenerator for OpenRouterVideoGenerator {
    fn name(&self) -> &str {
        "openrouter"
    }

    fn default_model(&self) -> &str {
        &self.default_model
    }

    async fn submit(&self, request: VideoRequest) -> Result<VideoJob> {
        request.validate()?;
        let model = wire_model_id(request.model.as_deref().unwrap_or(&self.default_model)).to_owned();
        let body = build_video_body(&model, &request, self.max_reference_bytes).await?;
        if self.check_capabilities
            && let Some(caps) = self.capabilities_for(&model).await
        {
            validate_against(&model, &body, &request, &caps)?;
        }
        tracing::info!(
            model = %model,
            duration_s = request.duration_s,
            first_frame = request.first_frame.is_some(),
            last_frame = request.last_frame.is_some(),
            references = request.references.len(),
            base_url = %self.transport.base_url(),
            "[tinyinference-video] submitting job"
        );
        let job: WireJob = self.transport.post_json("videos", &body).await?;
        checked_job_id(&job.id)?;
        Ok(VideoJob {
            id: job.id,
            model,
            state: JobState::parse(job.status.as_deref().unwrap_or("pending")),
        })
    }

    async fn poll(&self, job_id: &str) -> Result<VideoJobStatus> {
        let job_id = checked_job_id(job_id)?;
        let job: WireJob = self.transport.get_json(&format!("videos/{job_id}")).await?;
        Ok(VideoJobStatus {
            id: job.id,
            state: JobState::parse(job.status.as_deref().unwrap_or("pending")),
            outputs: job.unsigned_urls.iter().filter(|url| !url.is_empty()).count(),
            cost_usd: job.usage.and_then(|usage| usage.cost),
            error: error_text(job.error),
        })
    }

    async fn content(&self, job_id: &str, index: usize) -> Result<GeneratedMedia> {
        let job_id = checked_job_id(job_id)?;
        let (data, content_type) = self
            .transport
            .get_bytes(&format!("videos/{job_id}/content?index={index}"))
            .await?;
        if data.is_empty() {
            return Err(Error::Media(tinyinference_image::Error::NoMedia {
                request_id: Some(job_id.to_owned()),
            }));
        }
        let media_type = content_type
            .filter(|value| !value.is_empty() && !value.starts_with("application/json"))
            .unwrap_or_else(|| "video/mp4".to_owned());
        Ok(GeneratedMedia::new(media_type, data))
    }

    async fn list_models(&self) -> Result<Vec<MediaModel>> {
        let body: Value = self.transport.get_json("videos/models").await?;
        Ok(MediaModel::parse_listing(&body, ModelCapabilities::from_video_model))
    }
}
