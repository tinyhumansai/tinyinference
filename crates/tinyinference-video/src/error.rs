//! Error type for video generation.

use thiserror::Error;

/// Result returned by TinyInference video APIs.
pub type Result<T> = std::result::Result<T, Error>;

/// A normalized video-generation failure.
///
/// Video jobs are billed on submit. Every failure that happens *after* a
/// successful submit names the job id and says not to resubmit, because a new
/// submit is a new, separately billed generation; a job that is still running
/// can be resumed with [`crate::wait_for_job`] instead.
#[derive(Debug, Error)]
pub enum Error {
    /// A failure before or during submit (validation, capability, auth,
    /// transport). Nothing was billed unless the variant says so.
    #[error(transparent)]
    Media(#[from] tinyinference_image::Error),
    /// A failure after the job was accepted (polling or download).
    #[error(
        "video job {job_id} was accepted and billed, but {stage} failed: {source}; do not resubmit — resume by job id or report this to the user"
    )]
    Job {
        /// Provider job id.
        job_id: String,
        /// What was being done (`polling`, `downloading output 0`).
        stage: String,
        /// The underlying failure (boxed to keep `Result` small).
        #[source]
        source: Box<tinyinference_image::Error>,
    },
    /// The provider reported a terminal failure for the job.
    #[error("video job {job_id} ended as {state}: {message}; it was accepted and billed and cannot be resubmitted — resume by job id or report this to the user")]
    JobFailed {
        /// Provider job id.
        job_id: String,
        /// Terminal state (`failed`, `cancelled`, `expired`).
        state: String,
        /// Provider-reported reason, or a placeholder.
        message: String,
    },
    /// The wait budget elapsed before the job delivered a video.
    #[error(
        "video job {job_id} did not deliver within {waited_secs}s (last state: {last_state}); it was accepted and billed and may still be running — do not resubmit; resume by job id or report this to the user"
    )]
    Timeout {
        /// Provider job id.
        job_id: String,
        /// Seconds waited.
        waited_secs: u64,
        /// Last observed state.
        last_state: String,
    },
}

impl Error {
    /// The job id, when the failure happened after a billed submit.
    #[must_use]
    pub fn job_id(&self) -> Option<&str> {
        match self {
            Self::Job { job_id, .. }
            | Self::JobFailed { job_id, .. }
            | Self::Timeout { job_id, .. } => Some(job_id),
            Self::Media(_) => None,
        }
    }
}
