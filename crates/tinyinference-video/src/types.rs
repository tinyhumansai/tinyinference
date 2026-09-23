//! Public request, job, and response types for video generation.

use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tinyinference_image::{GeneratedMedia, MediaReference};

use crate::Result;

/// A provider-neutral video generation request.
///
/// Output-shape fields accept loose spellings (`"720"`, `"full hd"`,
/// `"landscape"`); providers normalize them with
/// [`tinyinference_image::reference`] and forward anything unrecognized.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VideoRequest {
    /// Model id; `None` uses the generator's default. An `openrouter/` prefix
    /// is accepted and stripped.
    pub model: Option<String>,
    /// Text prompt. Optional only when a frame image or reference drives the
    /// generation on its own.
    pub prompt: Option<String>,
    /// Clip duration in seconds.
    pub duration_s: Option<u32>,
    /// Output resolution (`480p`, `720p`, `1080p`, `4K`, …).
    pub resolution: Option<String>,
    /// Aspect ratio (`16:9`, `9:16`, …).
    pub aspect_ratio: Option<String>,
    /// Exact pixel size (`1280x720`); interchangeable with resolution +
    /// aspect ratio.
    pub size: Option<String>,
    /// Whether to generate an audio track, where supported.
    pub generate_audio: Option<bool>,
    /// Deterministic seed, where supported.
    pub seed: Option<i64>,
    /// Image to use as the first frame (image-to-video).
    pub first_frame: Option<MediaReference>,
    /// Image to use as the last frame.
    pub last_frame: Option<MediaReference>,
    /// Reference assets (image, video or audio) guiding subject or style.
    pub references: Vec<MediaReference>,
    /// A completed job to edit or extend, for models that support it.
    pub previous_job_id: Option<String>,
    /// Stable end-user identifier for provider abuse detection.
    pub user: Option<String>,
    /// Observability grouping id (never sent to the upstream model provider).
    pub session_id: Option<String>,
    /// Provider-specific extra fields merged into the wire body.
    pub extra: serde_json::Map<String, Value>,
}

impl VideoRequest {
    /// Creates a text-to-video request.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: Some(prompt.into()),
            ..Self::default()
        }
    }

    /// Sets the model id.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Sets the duration in seconds.
    #[must_use]
    pub fn with_duration(mut self, seconds: u32) -> Self {
        self.duration_s = Some(seconds);
        self
    }

    /// Sets the resolution.
    #[must_use]
    pub fn with_resolution(mut self, resolution: impl Into<String>) -> Self {
        self.resolution = Some(resolution.into());
        self
    }

    /// Sets the aspect ratio.
    #[must_use]
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Sets audio generation.
    #[must_use]
    pub fn with_audio(mut self, generate_audio: bool) -> Self {
        self.generate_audio = Some(generate_audio);
        self
    }

    /// Sets the first frame.
    #[must_use]
    pub fn with_first_frame(mut self, frame: MediaReference) -> Self {
        self.first_frame = Some(frame);
        self
    }

    /// Sets the last frame.
    #[must_use]
    pub fn with_last_frame(mut self, frame: MediaReference) -> Self {
        self.last_frame = Some(frame);
        self
    }

    /// Adds a reference asset.
    #[must_use]
    pub fn with_reference(mut self, reference: MediaReference) -> Self {
        self.references.push(reference);
        self
    }

    /// Checks the fields that are invalid for every model.
    ///
    /// # Errors
    ///
    /// [`tinyinference_image::Error::Validation`] when there is neither a
    /// prompt nor an image input, or the duration is zero.
    pub fn validate(&self) -> Result<()> {
        let has_prompt = self.prompt.as_deref().is_some_and(|p| !p.trim().is_empty());
        let has_image_input =
            self.first_frame.is_some() || self.last_frame.is_some() || !self.references.is_empty();
        if !has_prompt && !has_image_input {
            return Err(validation("a prompt or an input image is required"));
        }
        if self.duration_s == Some(0) {
            return Err(validation("duration must be at least 1 second"));
        }
        Ok(())
    }
}

fn validation(message: &str) -> crate::Error {
    crate::Error::Media(tinyinference_image::Error::Validation(message.to_owned()))
}

/// Lifecycle state of a video job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    /// Queued, not started.
    Pending,
    /// Generating.
    InProgress,
    /// Finished; output should be downloadable.
    Completed,
    /// Terminal failure.
    Failed,
    /// Cancelled before completion.
    Cancelled,
    /// Output expired before it was collected.
    Expired,
    /// A state this crate does not know; treated as in-flight.
    Other(String),
}

impl JobState {
    /// Parses a provider status string (case-insensitive).
    #[must_use]
    pub fn parse(value: &str) -> Self {
        match value.trim().to_ascii_lowercase().as_str() {
            "pending" | "queued" => Self::Pending,
            "in_progress" | "processing" | "running" => Self::InProgress,
            "completed" | "succeeded" | "success" => Self::Completed,
            "failed" | "error" => Self::Failed,
            "cancelled" | "canceled" => Self::Cancelled,
            "expired" => Self::Expired,
            other => Self::Other(other.to_owned()),
        }
    }

    /// Whether the job can no longer produce output.
    #[must_use]
    pub fn is_terminal_failure(&self) -> bool {
        matches!(self, Self::Failed | Self::Cancelled | Self::Expired)
    }

    /// The canonical lowercase name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Expired => "expired",
            Self::Other(other) => other,
        }
    }
}

impl std::fmt::Display for JobState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A submitted job.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoJob {
    /// Provider job id; use it to poll, download, or resume.
    pub id: String,
    /// Wire model id the job runs on.
    pub model: String,
    /// State reported at submit.
    pub state: JobState,
}

/// One poll result.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoJobStatus {
    /// Provider job id.
    pub id: String,
    /// Current state.
    pub state: JobState,
    /// How many outputs the provider reports as ready (`unsigned_urls`).
    pub outputs: usize,
    /// Provider-reported cost in USD, once known.
    pub cost_usd: Option<f64>,
    /// Provider-reported error, for failed jobs.
    pub error: Option<String>,
}

impl VideoJobStatus {
    /// Whether the job finished *and* has output to download.
    ///
    /// A `completed` status with no outputs is not delivered: providers can
    /// flip the status before the artifact is materialized, and treating that
    /// as terminal turns a paid, about-to-deliver job into a false failure.
    #[must_use]
    pub fn is_delivered(&self) -> bool {
        self.state == JobState::Completed && self.outputs > 0
    }
}

/// Called with every poll result, for host progress reporting.
pub type ProgressFn = Arc<dyn Fn(&VideoJobStatus) + Send + Sync>;

/// How long and how often to wait for a job.
#[derive(Clone)]
pub struct WaitPolicy {
    /// Delay between polls.
    pub interval: Duration,
    /// Total wait budget measured from the first poll.
    pub timeout: Duration,
    /// Optional progress observer.
    pub progress: Option<ProgressFn>,
}

impl WaitPolicy {
    /// Polls every `interval` for at most `timeout`.
    #[must_use]
    pub fn new(interval: Duration, timeout: Duration) -> Self {
        Self {
            interval,
            timeout,
            progress: None,
        }
    }

    /// Installs a progress observer.
    #[must_use]
    pub fn with_progress(mut self, progress: ProgressFn) -> Self {
        self.progress = Some(progress);
        self
    }
}

impl Default for WaitPolicy {
    /// Polls every 5 seconds for up to 10 minutes.
    fn default() -> Self {
        Self::new(Duration::from_secs(5), Duration::from_secs(600))
    }
}

impl std::fmt::Debug for WaitPolicy {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WaitPolicy")
            .field("interval", &self.interval)
            .field("timeout", &self.timeout)
            .field("progress", &self.progress.is_some())
            .finish()
    }
}

/// A delivered video generation. Always carries at least one video.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoResponse {
    /// Provider job id.
    pub job_id: String,
    /// Wire model id.
    pub model: String,
    /// Downloaded videos, in provider order.
    pub videos: Vec<GeneratedMedia>,
    /// Provider-reported cost in USD, when available.
    pub cost_usd: Option<f64>,
}
