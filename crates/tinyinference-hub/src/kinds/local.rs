//! [`LocalDriver`]: runtimes on this machine.

use async_trait::async_trait;

use crate::catalog::{Fetched, ParsedCatalog, parse_lmstudio_v0, parse_ollama_tags, too_large};
use crate::descriptor::ProviderDescriptor;
use crate::error::{HubError, ProviderFailure, ReasonCode};
use crate::taxonomy::LocalRuntime;

use super::openai_compat::read_listing;
use super::{DriverContext, KindDriver, Target};

/// A local runtime (Ollama, LM Studio, OMLX, MLX, or any OpenAI-compatible
/// server).
///
/// Every runtime answers the OpenAI-compatible `/v1/models`; two also have a
/// richer native listing that is tried first where it exists:
///
/// * LM Studio's `/api/v0/models` reports context windows, vision and tool use
///   (tagged as local-probe facts);
/// * Ollama's `/api/tags` is the fallback when `/v1/models` is missing on an
///   older build.
///
/// A native listing that is absent (`404`) or unreadable falls back to the
/// OpenAI one; a runtime that is down does not (a second request to a refused
/// connection only doubles the wait). An **empty** listing is success: a runtime
/// with nothing pulled is healthy.
#[derive(Clone, Debug)]
pub struct LocalDriver {
    descriptor: ProviderDescriptor,
}

impl LocalDriver {
    /// A driver for one local-runtime catalogue row.
    pub fn for_descriptor(descriptor: ProviderDescriptor) -> Self {
        Self { descriptor }
    }

    fn runtime(&self) -> Option<LocalRuntime> {
        self.descriptor.local_runtime
    }
}

/// The runtime's own origin: an endpoint like `http://host:11434/v1` without
/// its trailing `/v1`. Native APIs (`/api/tags`, `/api/v0/models`) live there.
fn origin_of(base: &str) -> String {
    let base = base.trim_end_matches('/');
    base.strip_suffix("/v1").unwrap_or(base).to_string()
}

/// A body that was answered but is not a listing: `unknown` with no status and
/// not cut at a cap. Only meaningful for a failure that came from *parsing* an
/// answer, never from the transport (whose `Other` failures look the same).
fn is_unreadable(failure: &ProviderFailure) -> bool {
    failure.status.is_none() && failure.reason == ReasonCode::Unknown && !failure.truncated
}

/// Whether a failed request means "this listing is not here" (so the next one
/// is worth trying) rather than "nothing is listening" or "the key was
/// refused". Only a `404` does: a transport failure of any kind is not retried
/// against a second path.
fn is_missing(failure: &ProviderFailure) -> bool {
    failure.status == Some(404)
}

/// Whether an answer from LM Studio's native `/api/v0/*` means "not usable
/// here" (a proxy or an older build answering 400, 404, 405 or a 5xx for a path
/// it does not know) rather than a verdict on the key or a transport failure.
/// The OpenAI-compatible listing is then worth trying.
fn native_listing_unavailable(failure: &ProviderFailure) -> bool {
    failure
        .status
        .is_some_and(|status| !matches!(status, 401 | 403))
}

#[async_trait]
impl KindDriver for LocalDriver {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    async fn list_models(
        &self,
        cx: &DriverContext<'_>,
        target: &Target<'_>,
    ) -> Result<Fetched, HubError> {
        let base = target.base();
        let origin = origin_of(base);
        let started = cx.clock.now();
        let read = |url: String| cx.list_request(&self.descriptor, target, url, started);
        if self.runtime() == Some(LocalRuntime::LmStudio) {
            match cx
                .call(self, read(format!("{origin}/api/v0/models"))?)
                .await
            {
                Ok(response) if !response.truncated => {
                    if let Ok(entries) =
                        parse_lmstudio_v0(&response.body).and_then(ParsedCatalog::into_usable)
                    {
                        return Ok(Fetched::new(entries));
                    }
                }
                Ok(_) => {}
                Err(HubError::Provider(failure)) if native_listing_unavailable(&failure) => {}
                Err(other) => return Err(other),
            }
        }
        // The OpenAI-compatible listing, read and parsed as one step so that a
        // body that is not a listing (an older build, a proxy's landing page)
        // can fall back exactly like a missing path.
        let (openai, answered) = match cx.call(self, read(target.join("/models"))?).await {
            Ok(response) => (read_listing(&response), true),
            Err(error) => (Err(error), false),
        };
        match openai {
            Err(HubError::Provider(failure))
                if self.runtime() == Some(LocalRuntime::Ollama)
                    && (is_missing(&failure) || (answered && is_unreadable(&failure))) =>
            {
                let response = cx.call(self, read(format!("{origin}/api/tags"))?).await?;
                if response.truncated {
                    return Err(HubError::Provider(too_large("the tags list")));
                }
                let parsed = parse_ollama_tags(&response.body).map_err(HubError::Provider)?;
                Ok(Fetched::new(
                    parsed.into_usable().map_err(HubError::Provider)?,
                ))
            }
            other => other,
        }
    }
}
