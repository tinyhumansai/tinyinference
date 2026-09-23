//! Tests for the submit → poll → download job loop.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use tinyinference_image::{GeneratedMedia, MediaModel};

use crate::{
    Error, JobState, MockVideoGenerator, MockVideoScript, Result, VideoGenerator, VideoJob,
    VideoJobStatus, VideoRequest, WaitPolicy, wait_for_job,
};

fn fast(timeout_ms: u64) -> WaitPolicy {
    WaitPolicy::new(Duration::from_millis(1), Duration::from_millis(timeout_ms))
}

#[tokio::test]
async fn delivers_after_pending_and_in_progress() {
    let generator = MockVideoGenerator::new(MockVideoScript::delivers());
    let seen = Arc::new(AtomicUsize::new(0));
    let counter = seen.clone();
    let wait = fast(5_000).with_progress(Arc::new(move |_status: &VideoJobStatus| {
        counter.fetch_add(1, Ordering::SeqCst);
    }));
    let response = generator
        .generate(VideoRequest::new("a cat surfing"), &wait)
        .await
        .unwrap();
    assert_eq!(response.job_id, "mock-job");
    assert_eq!(response.videos.len(), 1);
    assert_eq!(response.videos[0].media_type, "video/mp4");
    assert_eq!(seen.load(Ordering::SeqCst), 3);
}

/// Regression (R1): a `completed` status that reports no outputs yet is not a
/// terminal result — the loop keeps polling until outputs appear.
#[tokio::test]
async fn completed_without_outputs_keeps_polling_until_outputs_appear() {
    let generator = MockVideoGenerator::new(MockVideoScript {
        polls: vec![
            (JobState::Completed, 0),
            (JobState::Completed, 0),
            (JobState::Completed, 2),
        ],
        error: None,
    });
    let response = generator
        .generate(VideoRequest::new("x"), &fast(5_000))
        .await
        .unwrap();
    assert_eq!(response.videos.len(), 2);
}

/// A generator whose poll script is fixed and whose downloads can be failed.
struct Scripted {
    status: (JobState, usize),
    content_ok: bool,
    poll_error: Option<fn() -> tinyinference_image::Error>,
    polls: AtomicUsize,
}

impl Scripted {
    fn new(state: JobState, outputs: usize, content_ok: bool) -> Self {
        Self {
            status: (state, outputs),
            content_ok,
            poll_error: None,
            polls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl VideoGenerator for Scripted {
    fn name(&self) -> &str {
        "scripted"
    }
    fn default_model(&self) -> &str {
        "scripted/video"
    }
    async fn submit(&self, _request: VideoRequest) -> Result<VideoJob> {
        Ok(VideoJob {
            id: "job-1".into(),
            model: "scripted/video".into(),
            state: JobState::Pending,
        })
    }
    async fn poll(&self, job_id: &str) -> Result<VideoJobStatus> {
        let call = self.polls.fetch_add(1, Ordering::SeqCst);
        if let Some(make) = self.poll_error
            && call == 0
        {
            return Err(Error::Media(make()));
        }
        Ok(VideoJobStatus {
            id: job_id.into(),
            state: self.status.0.clone(),
            outputs: self.status.1,
            cost_usd: None,
            error: Some("content policy".into()),
        })
    }
    async fn content(&self, job_id: &str, _index: usize) -> Result<GeneratedMedia> {
        if self.content_ok {
            Ok(GeneratedMedia::new("video/mp4", &b"mp4"[..]))
        } else {
            Err(Error::Media(tinyinference_image::Error::Http {
                status: 404,
                message: format!("no content for {job_id}"),
            }))
        }
    }
    async fn list_models(&self) -> Result<Vec<MediaModel>> {
        Ok(Vec::new())
    }
}

/// Regression (R1): a provider that says `completed` but never lists outputs
/// still delivers through one direct download at the deadline.
#[tokio::test]
async fn completed_without_listed_outputs_falls_back_to_direct_download() {
    let generator = Scripted::new(JobState::Completed, 0, true);
    let response = wait_for_job(&generator, "job-1", "m", &fast(20)).await.unwrap();
    assert_eq!(response.videos.len(), 1);
}

/// Regression (R1/R2): when nothing is ever delivered, the result is a timeout
/// that names the billed job and says not to resubmit — never a success.
#[tokio::test]
async fn completed_without_any_output_times_out_naming_the_job() {
    let generator = Scripted::new(JobState::Completed, 0, false);
    let error = wait_for_job(&generator, "job-1", "m", &fast(20)).await.unwrap_err();
    assert!(matches!(error, Error::Timeout { .. }), "{error:?}");
    assert_eq!(error.job_id(), Some("job-1"));
    let message = error.to_string();
    assert!(message.contains("job-1") && message.contains("do not resubmit"), "{message}");
}

#[tokio::test]
async fn in_progress_past_the_deadline_times_out() {
    let generator = Scripted::new(JobState::InProgress, 0, true);
    let error = wait_for_job(&generator, "job-1", "m", &fast(20)).await.unwrap_err();
    match error {
        Error::Timeout { last_state, .. } => assert_eq!(last_state, "in_progress"),
        other => panic!("expected timeout, got {other:?}"),
    }
}

#[tokio::test]
async fn terminal_failure_reports_state_and_reason() {
    let generator = Scripted::new(JobState::Failed, 0, true);
    let error = wait_for_job(&generator, "job-1", "m", &fast(5_000)).await.unwrap_err();
    match &error {
        Error::JobFailed { job_id, state, message } => {
            assert_eq!((job_id.as_str(), state.as_str()), ("job-1", "failed"));
            assert_eq!(message, "content policy");
        }
        other => panic!("expected JobFailed, got {other:?}"),
    }
}

#[tokio::test]
async fn transient_poll_errors_are_retried() {
    let mut generator = Scripted::new(JobState::Completed, 1, true);
    generator.poll_error = Some(|| tinyinference_image::Error::Http {
        status: 503,
        message: "busy".into(),
    });
    let response = wait_for_job(&generator, "job-1", "m", &fast(5_000)).await.unwrap();
    assert_eq!(response.videos.len(), 1);
    assert_eq!(generator.polls.load(Ordering::SeqCst), 2);
}

/// Regression (R2): a hard failure after submit carries the job id.
#[tokio::test]
async fn hard_poll_errors_name_the_billed_job() {
    let mut generator = Scripted::new(JobState::Completed, 1, true);
    generator.poll_error = Some(|| tinyinference_image::Error::Http {
        status: 404,
        message: "unknown job".into(),
    });
    let error = wait_for_job(&generator, "job-1", "m", &fast(5_000)).await.unwrap_err();
    assert!(matches!(error, Error::Job { ref stage, .. } if stage == "polling"), "{error:?}");
    assert_eq!(error.job_id(), Some("job-1"));
    assert!(error.to_string().contains("do not resubmit"));
}

#[tokio::test]
async fn download_failures_name_the_output() {
    let generator = Scripted::new(JobState::Completed, 1, false);
    let error = wait_for_job(&generator, "job-1", "m", &fast(5_000)).await.unwrap_err();
    assert!(
        matches!(error, Error::Job { ref stage, .. } if stage == "downloading output 0"),
        "{error:?}"
    );
}

#[tokio::test]
async fn requests_need_a_prompt_or_an_image() {
    let generator = MockVideoGenerator::new(MockVideoScript::delivers());
    let empty = VideoRequest::default();
    assert!(matches!(
        generator.generate(empty, &fast(10)).await,
        Err(Error::Media(tinyinference_image::Error::Validation(_)))
    ));
    let image_only = VideoRequest::default()
        .with_first_frame(tinyinference_image::MediaReference::Url("https://x.test/f.png".into()));
    generator.generate(image_only, &fast(5_000)).await.unwrap();
    assert!(matches!(
        generator.generate(VideoRequest::new("x").with_duration(0), &fast(10)).await,
        Err(Error::Media(tinyinference_image::Error::Validation(_)))
    ));
}

#[test]
fn job_states_parse_provider_spellings() {
    assert_eq!(JobState::parse("in_progress"), JobState::InProgress);
    assert_eq!(JobState::parse("QUEUED"), JobState::Pending);
    assert_eq!(JobState::parse("succeeded"), JobState::Completed);
    assert_eq!(JobState::parse("canceled"), JobState::Cancelled);
    assert!(JobState::parse("expired").is_terminal_failure());
    assert_eq!(JobState::parse("warming"), JobState::Other("warming".into()));
    assert!(!JobState::parse("warming").is_terminal_failure());
}
