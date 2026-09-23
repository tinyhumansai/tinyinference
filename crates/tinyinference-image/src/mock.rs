//! Deterministic, offline image generator for tests.

use std::sync::Mutex;

use async_trait::async_trait;

use crate::media::GeneratedMedia;
use crate::types::{ImageRequest, ImageResponse, MediaModel};
use crate::{Error, ImageGenerator, ModelCapabilities, Result};

/// The smallest valid PNG (1×1, transparent), returned by the mock.
pub(crate) const TINY_PNG: &[u8] = &[
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4,
    0x89, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9C, 0x63, 0x00, 0x01, 0x00, 0x00,
    0x05, 0x00, 0x01, 0x0D, 0x0A, 0x2D, 0xB4, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE,
    0x42, 0x60, 0x82,
];

/// Returns one tiny PNG per requested image, records every request, and can
/// be told to simulate a billed non-delivery.
#[derive(Debug, Default)]
pub struct MockImageGenerator {
    requests: Mutex<Vec<ImageRequest>>,
    return_no_media: bool,
}

impl MockImageGenerator {
    /// Creates a mock that always succeeds.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a mock whose every call fails with [`Error::NoMedia`].
    #[must_use]
    pub fn returning_no_media() -> Self {
        Self {
            return_no_media: true,
            ..Self::default()
        }
    }

    /// Requests received so far, in order.
    ///
    /// # Panics
    ///
    /// If a previous holder of the internal lock panicked.
    #[must_use]
    pub fn requests(&self) -> Vec<ImageRequest> {
        self.requests.lock().expect("mock lock poisoned").clone()
    }
}

#[async_trait]
impl ImageGenerator for MockImageGenerator {
    fn name(&self) -> &str {
        "mock"
    }

    fn default_model(&self) -> &str {
        "mock/image"
    }

    async fn generate(&self, request: ImageRequest) -> Result<ImageResponse> {
        request.validate()?;
        let n = request.n.unwrap_or(1);
        let model = request.model.clone().unwrap_or_else(|| self.default_model().to_owned());
        self.requests.lock().expect("mock lock poisoned").push(request);
        if self.return_no_media {
            return Err(Error::NoMedia {
                request_id: Some("mock-request".into()),
            });
        }
        Ok(ImageResponse {
            model,
            images: (0..n).map(|_| GeneratedMedia::new("image/png", TINY_PNG)).collect(),
            cost_usd: Some(0.0),
            created: None,
        })
    }

    async fn list_models(&self) -> Result<Vec<MediaModel>> {
        Ok(vec![MediaModel {
            id: self.default_model().to_owned(),
            name: Some("Mock image".into()),
            description: None,
            capabilities: ModelCapabilities::default(),
            raw: serde_json::Value::Null,
        }])
    }
}
