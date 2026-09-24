//! Provider-neutral asynchronous video generation for TinyInference.
//!
//! Video generation is a job: **submit** (billed), **poll** until the job
//! delivers, then **download** each output. [`VideoGenerator`] exposes the three
//! steps, and [`VideoGenerator::generate`] runs them end to end through
//! [`wait_for_job`], which is also callable on its own to resume a job by id.
//!
//! The wait loop returns only on a *delivered* outcome — `completed` **with**
//! outputs — or a terminal failure. A `completed` status that reports no outputs
//! yet keeps polling instead of ending the job, because providers can flip the
//! status before the artifact exists and a caller told "failed" will resubmit
//! and pay again. Every failure after a successful submit names the job id and
//! says not to resubmit.
//!
//! Reference assets, output-shape normalization and the OpenRouter transport
//! come from [`tinyinference_image`], so image and video generation share one
//! vocabulary and one credential model.

mod error;
mod mock;
pub mod openrouter;
mod types;

pub use error::{Error, Result};
pub use mock::{MockVideoGenerator, MockVideoScript};
pub use openrouter::{DEFAULT_VIDEO_MODEL, OpenRouterVideoGenerator};
pub use tinyinference_image::{
    GeneratedMedia, MediaAuth, MediaModel, MediaReference, MediaTransport, ModelCapabilities,
};
pub use types::{
    JobState, ProgressFn, VideoJob, VideoJobStatus, VideoRequest, VideoResponse, WaitPolicy,
};

use std::time::Instant;

use async_trait::async_trait;

/// A video generation provider.
#[async_trait]
pub trait VideoGenerator: Send + Sync {
    /// Short provider name for logs and diagnostics (`"openrouter"`).
    fn name(&self) -> &str;

    /// Model used when a request names none.
    fn default_model(&self) -> &str;

    /// Submits a job. This is the billed step.
    ///
    /// # Errors
    ///
    /// [`Error::Media`] for validation, capability, auth or provider failures.
    async fn submit(&self, request: VideoRequest) -> Result<VideoJob>;

    /// Reads a job's current state.
    ///
    /// # Errors
    ///
    /// [`Error::Media`] for provider or decode failures.
    async fn poll(&self, job_id: &str) -> Result<VideoJobStatus>;

    /// Downloads output `index` of a completed job.
    ///
    /// # Errors
    ///
    /// [`Error::Media`] for provider, size-cap or decode failures.
    async fn content(&self, job_id: &str, index: usize) -> Result<GeneratedMedia>;

    /// Lists the models this provider can generate with.
    ///
    /// # Errors
    ///
    /// Provider or decode errors from the listing endpoint.
    async fn list_models(&self) -> Result<Vec<MediaModel>>;

    /// Submits `request` and waits for delivery under `wait`.
    ///
    /// # Errors
    ///
    /// As for [`VideoGenerator::submit`] and [`wait_for_job`].
    async fn generate(&self, request: VideoRequest, wait: &WaitPolicy) -> Result<VideoResponse> {
        let job = self.submit(request).await?;
        tracing::info!(
            provider = self.name(),
            job_id = %job.id,
            model = %job.model,
            state = %job.state,
            "[tinyinference-video] job submitted"
        );
        wait_for_job(self, &job.id, &job.model, wait).await
    }
}

/// Waits for job `job_id` to deliver and downloads every output.
///
/// Use this directly to resume a job that an earlier call submitted and then
/// timed out on, without paying for a new generation.
///
/// Transient poll failures (rate limits, 5xx, transport) are retried until the
/// deadline. A `completed` job with no outputs keeps polling; at the deadline
/// one direct download of output 0 is attempted before giving up, so a
/// provider that never lists outputs still delivers.
///
/// # Errors
///
/// [`Error::JobFailed`] for a terminal failure, [`Error::Timeout`] when the
/// budget elapses, and [`Error::Job`] for a non-transient poll or download
/// failure. All carry the job id.
pub async fn wait_for_job<G: VideoGenerator + ?Sized>(
    generator: &G,
    job_id: &str,
    model: &str,
    wait: &WaitPolicy,
) -> Result<VideoResponse> {
    let started = Instant::now();
    let mut last_state = JobState::Pending;
    let mut last_poll_error: Option<String> = None;
    loop {
        let elapsed = started.elapsed();
        if elapsed >= wait.timeout {
            if last_state == JobState::Completed {
                let remaining = wait.timeout.saturating_sub(elapsed);
                if let Ok(Ok(video)) =
                    tokio::time::timeout(remaining, generator.content(job_id, 0)).await
                {
                    tracing::info!(
                        job_id,
                        "[tinyinference-video] completed job without listed outputs delivered on direct download"
                    );
                    return Ok(VideoResponse {
                        job_id: job_id.to_owned(),
                        model: model.to_owned(),
                        videos: vec![video],
                        cost_usd: None,
                    });
                }
            }
            tracing::warn!(
                job_id,
                last_state = %last_state,
                last_poll_error = ?last_poll_error,
                waited_secs = elapsed.as_secs(),
                "[tinyinference-video] wait budget elapsed"
            );
            return Err(Error::Timeout {
                job_id: job_id.to_owned(),
                waited_secs: elapsed.as_secs(),
                last_state: last_state.to_string(),
            });
        }

        let remaining = wait.timeout - elapsed;
        match tokio::time::timeout(remaining, generator.poll(job_id)).await {
            Ok(Ok(status)) => {
                if let Some(progress) = &wait.progress {
                    progress(&status);
                }
                tracing::debug!(
                    job_id,
                    state = %status.state,
                    outputs = status.outputs,
                    "[tinyinference-video] poll"
                );
                last_poll_error = None;
                if status.is_delivered() {
                    return download_all(generator, job_id, model, status.outputs, status.cost_usd, started, wait)
                        .await;
                }
                if status.state.is_terminal_failure() {
                    tracing::warn!(job_id, state = %status.state, "[tinyinference-video] job failed");
                    return Err(Error::JobFailed {
                        job_id: job_id.to_owned(),
                        state: status.state.to_string(),
                        message: status
                            .error
                            .unwrap_or_else(|| "no reason given by the provider".into()),
                    });
                }
                if status.state == JobState::Completed {
                    tracing::warn!(
                        job_id,
                        "[tinyinference-video] job reports completed with no outputs yet; still polling"
                    );
                }
                last_state = status.state;
            }
            Ok(Err(Error::Media(error))) if error.is_retryable() => {
                tracing::warn!(job_id, %error, "[tinyinference-video] transient poll failure");
                last_poll_error = Some(error.to_string());
            }
            Ok(Err(Error::Media(error))) => {
                return Err(Error::Job {
                    job_id: job_id.to_owned(),
                    stage: "polling".into(),
                    source: Box::new(error),
                });
            }
            Ok(Err(other)) => return Err(other),
            Err(_) => {
                tracing::warn!(job_id, "[tinyinference-video] poll request timeout");
                last_poll_error = Some("poll request timed out".into());
            }
        }

        let elapsed = started.elapsed();
        if elapsed >= wait.timeout {
            if last_state == JobState::Completed {
                let remaining = wait.timeout.saturating_sub(elapsed);
                if let Ok(Ok(video)) =
                    tokio::time::timeout(remaining, generator.content(job_id, 0)).await
                {
                    tracing::info!(
                        job_id,
                        "[tinyinference-video] completed job without listed outputs delivered on direct download"
                    );
                    return Ok(VideoResponse {
                        job_id: job_id.to_owned(),
                        model: model.to_owned(),
                        videos: vec![video],
                        cost_usd: None,
                    });
                }
            }
            tracing::warn!(
                job_id,
                last_state = %last_state,
                last_poll_error = ?last_poll_error,
                waited_secs = elapsed.as_secs(),
                "[tinyinference-video] wait budget elapsed"
            );
            return Err(Error::Timeout {
                job_id: job_id.to_owned(),
                waited_secs: elapsed.as_secs(),
                last_state: last_state.to_string(),
            });
        }
        tokio::time::sleep(wait.interval.min(wait.timeout - elapsed)).await;
    }
}

async fn download_all<G: VideoGenerator + ?Sized>(
    generator: &G,
    job_id: &str,
    model: &str,
    outputs: usize,
    cost_usd: Option<f64>,
    started: Instant,
    wait: &WaitPolicy,
) -> Result<VideoResponse> {
    let mut videos = Vec::with_capacity(outputs);
    for index in 0..outputs {
        let elapsed = started.elapsed();
        if elapsed >= wait.timeout {
            return Err(Error::Timeout {
                job_id: job_id.to_owned(),
                waited_secs: elapsed.as_secs(),
                last_state: "downloading".to_owned(),
            });
        }
        let remaining = wait.timeout - elapsed;
        match tokio::time::timeout(remaining, generator.content(job_id, index)).await {
            Ok(Ok(video)) => videos.push(video),
            Ok(Err(Error::Media(source))) => {
                return Err(Error::Job {
                    job_id: job_id.to_owned(),
                    stage: format!("downloading output {index}"),
                    source: Box::new(source),
                });
            }
            Ok(Err(other)) => return Err(other),
            Err(_) => {
                return Err(Error::Timeout {
                    job_id: job_id.to_owned(),
                    waited_secs: started.elapsed().as_secs(),
                    last_state: format!("downloading output {index}"),
                });
            }
        }
    }
    tracing::info!(
        job_id,
        videos = videos.len(),
        cost_usd,
        "[tinyinference-video] job delivered"
    );
    Ok(VideoResponse {
        job_id: job_id.to_owned(),
        model: model.to_owned(),
        videos,
        cost_usd,
    })
}

#[cfg(test)]
#[path = "job_test.rs"]
mod job_test;

#[cfg(test)]
#[path = "openrouter_test.rs"]
mod openrouter_test;
