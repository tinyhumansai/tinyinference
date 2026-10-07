//! Native Perplexity Agent API provider.
//!
//! Select a provider-qualified model or a dynamic preset with one explicit key.
//! Perplexity executes hosted tools; this crate returns local function requests
//! to the caller. Full ordered output lives in [`ModelResponse::output`], while
//! [`ModelResponse::message`] remains the text/reasoning/function-call projection.
//! Stream progress and unknown events survive in execution metadata.
//!
//! ```no_run
//! use tinyinference_llm::{ChatModel, Message, ModelRequest, PerplexityModel, PerplexitySelection};
//! # async fn example(key: String) -> tinyinference_llm::Result<()> {
//! let model = PerplexityModel::new(key, PerplexitySelection::preset("low"))?;
//! let response = model.invoke(&(), ModelRequest::new(vec![Message::user(
//!     "Explain the latest changes to this project with sources."
//! )])).await?;
//! println!("{}", response.text());
//! for item in &response.output {
//!     // Inspect typed citations, hosted results, or generated-file notices.
//!     let _ = &item.kind;
//! }
//! # Ok(())
//! # }
//! ```
//!
//! To replay a signed custom call, use [`crate::message::ToolMessage::for_call`]
//! and pin the returned model. Include the preceding conversation, original
//! assistant call and matching result; leave [`ModelRequest::continuation_id`]
//! unset. This is the verified Google function-continuation path. For ordinary
//! chat follow-up, set the continuation ID and send only the new user turn. `store: false`
//! hides retrieval; it does not promise that Perplexity avoids persistence.
//!
//! [`PerplexityModel::submit_background`] returns a scoped handle. Retrieve or
//! resume it instead of retrying creation after a disconnect. Remote cancellation
//! is explicit; dropping a local stream only stops local work. Creation retries
//! cover explicit rate limits only, never ambiguous transport/5xx failures.
//!
//! Defaults: 600-second total budget, 30-second connection budget, 60-second
//! stream inactivity budget, 32 MiB input/body/stream traffic, 4 MiB per event,
//! and 64 MiB per explicit file download. All are configurable. No local file
//! reads, file writes, tool execution, or automatic response caching are performed.

use crate::{
    Error, Result,
    model::{ChatModel, ModelRequest, ModelResponse},
};
use async_trait::async_trait;
use std::sync::Arc;

mod config;
mod lifecycle;
mod request;
mod response;
mod stream;
mod transport;
mod types;

pub use config::*;

/// Perplexity Agent model. The key is supplied explicitly and never read from the environment.
#[derive(Clone, Debug)]
pub struct PerplexityModel {
    inner: Arc<transport::Transport>,
}

impl PerplexityModel {
    /// Constructs a model/preset client with bounded defaults.
    ///
    /// # Errors
    /// Returns validation errors for invalid credentials, selection, or limits.
    pub fn new(key: impl Into<String>, selection: PerplexitySelection) -> Result<Self> {
        Self::with_config(key, PerplexityConfig::new(selection))
    }

    /// Constructs a native provider using validated immutable configuration.
    ///
    /// # Errors
    /// Returns validation or HTTP-client setup errors; makes no provider call.
    pub fn with_config(key: impl Into<String>, config: PerplexityConfig) -> Result<Self> {
        Ok(Self {
            inner: Arc::new(transport::Transport::new(key.into(), config)?),
        })
    }

    fn body(
        &self,
        request: &ModelRequest,
        streaming: bool,
        submit: bool,
    ) -> Result<serde_json::Value> {
        self.inner.check_input_size(request)?;
        let mut body = request::build(&self.inner.config, request, streaming, submit)?;
        if let Some(hook) = &self.inner.config.on_payload {
            let original = body.clone();
            hook(&mut body);
            request::validate_hook(&original, &body)?;
        }
        Ok(body)
    }

    fn create_path(&self) -> &'static str {
        if self.inner.config.responses_alias {
            "/responses"
        } else {
            "/agent"
        }
    }
}

/// Constructs the provider behind the existing object-safe chat interface.
///
/// # Errors
/// Returns the same configuration errors as [`PerplexityModel::with_config`].
pub fn build_perplexity_model(
    key: impl Into<String>,
    config: PerplexityConfig,
) -> Result<Arc<dyn ChatModel<()>>> {
    Ok(Arc::new(PerplexityModel::with_config(key, config)?))
}

#[async_trait]
impl<State: Send + Sync> ChatModel<State> for PerplexityModel {
    async fn invoke(&self, _state: &State, request: ModelRequest) -> Result<ModelResponse> {
        crate::network_guard::ensure_network_models_allowed()?;
        let deadline = self.inner.deadline(request.timeout_ms)?;
        let body = self.body(&request, false, false)?;
        let outgoing =
            self.inner
                .request(reqwest::Method::POST, self.create_path(), Some(&body))?;
        let incoming = self
            .inner
            .send(outgoing, transport::Operation::Create, deadline)
            .await?;
        let result =
            response::completed(response::parse(self.inner.json(incoming, deadline).await?)?)?;
        if tokio::time::Instant::now() >= deadline {
            return Err(response::with_partial(transport::timeout(), result));
        }
        Ok(result.inherit_correlation(request.correlation))
    }

    async fn stream(
        &self,
        _state: &State,
        request: ModelRequest,
    ) -> Result<crate::model::ModelStream> {
        crate::network_guard::ensure_network_models_allowed()?;
        let deadline = self.inner.deadline(request.timeout_ms)?;
        let body = self.body(&request, true, false)?;
        let outgoing =
            self.inner
                .request(reqwest::Method::POST, self.create_path(), Some(&body))?;
        let incoming = self
            .inner
            .send(outgoing, transport::Operation::Create, deadline)
            .await?;
        let mut stream = stream::open(self.inner.clone(), incoming, deadline, None, None)?;
        if let Some(correlation) = request.correlation {
            stream = stream.with_correlation(correlation);
        }
        Ok(stream)
    }

    async fn fetch_deferred(
        &self,
        handle: &crate::model::DeferredHandle,
    ) -> Result<crate::model::DeferredStatus> {
        use crate::model::{DeferredStatus, ExecutionStatus};
        let snapshot = self.retrieve_response(handle).await?;
        match snapshot.execution.as_ref().map(|e| &e.status) {
            Some(
                ExecutionStatus::Queued | ExecutionStatus::InProgress | ExecutionStatus::Cancelling,
            ) => Ok(DeferredStatus::Pending),
            Some(ExecutionStatus::Cancelled) => Ok(DeferredStatus::Cancelled {
                response: Some(Box::new(snapshot)),
            }),
            Some(ExecutionStatus::Completed | ExecutionStatus::Incomplete) => Ok(
                DeferredStatus::Completed(Box::new(response::completed(snapshot)?)),
            ),
            _ => match response::completed(snapshot) {
                Err(Error::Provider(error)) => Ok(DeferredStatus::ProviderFailed(error)),
                Err(error) => Err(error),
                Ok(_) => Err(response::malformed("unknown deferred state")),
            },
        }
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "stream_tests.rs"]
mod stream_test;

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod lifecycle_test;
