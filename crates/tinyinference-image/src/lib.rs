//! Provider-neutral image generation for TinyInference.
//!
//! This crate owns three things:
//!
//! - **Media standards** ([`reference`]) — how a reference asset is described
//!   (URL, `data:` URL, bytes, local path) and inlined, and how loose
//!   output-shape spellings (`"16x9"`, `"landscape"`, `"full hd"`) normalize to
//!   canonical wire values. The video crate reuses these, so image and video
//!   generation agree on one vocabulary.
//! - **The OpenRouter media transport** ([`transport`]) — authenticated HTTP
//!   with billing-aware retries, shared by image and video generation. The same
//!   transport serves OpenRouter directly (API key) and any host backend that
//!   proxies OpenRouter's media routes verbatim (host-resolved bearer).
//! - **Image generation** — the [`ImageGenerator`] trait,
//!   [`OpenRouterImageGenerator`], and [`MockImageGenerator`] for offline
//!   tests.
//!
//! A generator either returns at least one image or fails: an accepted request
//! that yields nothing is [`Error::NoMedia`], never an empty success, because
//! the request was already billed and a caller told "success" would report an
//! image that does not exist.
//!
//! # Example
//! ```
//! use tinyinference_image::{ImageGenerator, ImageRequest, MockImageGenerator};
//!
//! # tokio::runtime::Runtime::new().unwrap().block_on(async {
//! let generator = MockImageGenerator::new();
//! let response = generator
//!     .generate(ImageRequest::new("a red panda astronaut").with_aspect_ratio("landscape"))
//!     .await
//!     .unwrap();
//! assert_eq!(response.images.len(), 1);
//! # });
//! ```

pub mod capabilities;
mod error;
pub mod media;
mod mock;
pub mod openrouter;
pub mod reference;
pub mod transport;
mod types;

pub use capabilities::ModelCapabilities;
pub use error::{Error, Result};
pub use media::GeneratedMedia;
pub use mock::MockImageGenerator;
pub use openrouter::{DEFAULT_IMAGE_MODEL, OpenRouterImageGenerator};
pub use reference::{MediaReference, ReferenceKind};
pub use transport::{BearerResolver, MediaAuth, MediaTransport};
pub use types::{ImageRequest, ImageResponse, MAX_IMAGES_PER_REQUEST, MediaModel};

use async_trait::async_trait;

/// An image generation provider.
///
/// Implementations must be `Send + Sync` so hosts can share one generator
/// across concurrent tool calls.
#[async_trait]
pub trait ImageGenerator: Send + Sync {
    /// Short provider name for logs and diagnostics (`"openrouter"`).
    fn name(&self) -> &str;

    /// Model used when a request names none.
    fn default_model(&self) -> &str;

    /// Generates one or more images.
    ///
    /// # Errors
    ///
    /// [`Error::Validation`] / [`Error::Unsupported`] before any request is
    /// sent; [`Error::Auth`], [`Error::Http`], [`Error::Transport`] from the
    /// provider; [`Error::NoMedia`] when the provider accepted and billed the
    /// request but returned no images.
    async fn generate(&self, request: ImageRequest) -> Result<ImageResponse>;

    /// Lists the models this provider can generate with.
    ///
    /// # Errors
    ///
    /// Provider or decode errors from the listing endpoint.
    async fn list_models(&self) -> Result<Vec<MediaModel>>;
}

#[cfg(test)]
#[path = "reference_test.rs"]
mod reference_test;

#[cfg(test)]
#[path = "openrouter_test.rs"]
mod openrouter_test;
