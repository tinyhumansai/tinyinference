//! Scripted, offline video generator for tests.

use std::collections::VecDeque;
use std::sync::Mutex;

use async_trait::async_trait;
use tinyinference_image::{GeneratedMedia, MediaModel, ModelCapabilities};

use crate::types::{JobState, VideoJob, VideoJobStatus, VideoRequest};
use crate::{Result, VideoGenerator};

/// Four bytes of an MP4 `ftyp` box — enough for type sniffing in tests.
const FAKE_MP4: &[u8] = b"\x00\x00\x00\x18ftypmp42";

/// A poll script: each poll pops the next state; the last one repeats.
#[derive(Debug, Clone)]
pub struct MockVideoScript {
    /// `(state, outputs)` returned by successive polls.
    pub polls: Vec<(JobState, usize)>,
    /// Error message reported with a terminal failure.
    pub error: Option<String>,
}

impl MockVideoScript {
    /// A job that goes pending → in progress → completed with one output.
    #[must_use]
    pub fn delivers() -> Self {
        Self {
            polls: vec![
                (JobState::Pending, 0),
                (JobState::InProgress, 0),
                (JobState::Completed, 1),
            ],
            error: None,
        }
    }
}

/// Replays a [`MockVideoScript`] and records submitted requests.
#[derive(Debug)]
pub struct MockVideoGenerator {
    polls: Mutex<VecDeque<(JobState, usize)>>,
    last: Mutex<(JobState, usize)>,
    error: Option<String>,
    requests: Mutex<Vec<VideoRequest>>,
}

impl MockVideoGenerator {
    /// Creates a mock that replays `script`.
    #[must_use]
    pub fn new(script: MockVideoScript) -> Self {
        let last = script
            .polls
            .last()
            .cloned()
            .unwrap_or((JobState::Completed, 1));
        Self {
            polls: Mutex::new(script.polls.into()),
            last: Mutex::new(last),
            error: script.error,
            requests: Mutex::new(Vec::new()),
        }
    }

    /// Requests submitted so far.
    ///
    /// # Panics
    ///
    /// If a previous holder of the internal lock panicked.
    #[must_use]
    pub fn requests(&self) -> Vec<VideoRequest> {
        self.requests.lock().expect("mock lock poisoned").clone()
    }
}

#[async_trait]
impl VideoGenerator for MockVideoGenerator {
    fn name(&self) -> &str {
        "mock"
    }

    fn default_model(&self) -> &str {
        "mock/video"
    }

    async fn submit(&self, request: VideoRequest) -> Result<VideoJob> {
        request.validate()?;
        let model = request.model.clone().unwrap_or_else(|| self.default_model().to_owned());
        self.requests.lock().expect("mock lock poisoned").push(request);
        Ok(VideoJob {
            id: "mock-job".into(),
            model,
            state: JobState::Pending,
        })
    }

    async fn poll(&self, job_id: &str) -> Result<VideoJobStatus> {
        let next = self.polls.lock().expect("mock lock poisoned").pop_front();
        let (state, outputs) = match next {
            Some(step) => {
                *self.last.lock().expect("mock lock poisoned") = step.clone();
                step
            }
            None => self.last.lock().expect("mock lock poisoned").clone(),
        };
        let error = state.is_terminal_failure().then(|| self.error.clone()).flatten();
        Ok(VideoJobStatus {
            id: job_id.to_owned(),
            state,
            outputs,
            cost_usd: Some(0.0),
            error,
        })
    }

    async fn content(&self, _job_id: &str, _index: usize) -> Result<GeneratedMedia> {
        Ok(GeneratedMedia::new("video/mp4", FAKE_MP4))
    }

    async fn list_models(&self) -> Result<Vec<MediaModel>> {
        Ok(vec![MediaModel {
            id: self.default_model().to_owned(),
            name: Some("Mock video".into()),
            description: None,
            capabilities: ModelCapabilities::default(),
            raw: serde_json::Value::Null,
        }])
    }
}
