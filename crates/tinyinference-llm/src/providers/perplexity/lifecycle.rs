use super::{
    PerplexityModel, response, stream,
    transport::{self, Operation},
    types::FileList,
};
use crate::model::{
    DeferredHandle, ExecutionStatus, ModelFile, ModelFileContent, ModelRequest, ModelResponse,
    ModelStream, ModelStreamItem,
};
use crate::{Error, Result};
use serde_json::json;
use std::collections::HashSet;
use tokio::time::Instant;

impl PerplexityModel {
    /// Submits one durable run without waiting for its result.
    ///
    /// Persist the returned handle in the caller. Submission is billable;
    /// interrupted acknowledgement is not automatically resubmitted.
    ///
    /// # Errors
    /// Returns validation, network, timeout, or provider errors. Explicit
    /// `store: false` is rejected because durable recovery requires retrieval.
    pub async fn submit_background(&self, request: ModelRequest) -> Result<DeferredHandle> {
        crate::network_guard::ensure_network_models_allowed()?;
        let deadline = self.inner.deadline(request.timeout_ms)?;
        let body = self.body(&request, false, true)?;
        let outgoing =
            self.inner
                .request(reqwest::Method::POST, self.create_path(), Some(&body))?;
        let incoming = self
            .inner
            .send(outgoing, Operation::Create, deadline)
            .await?;
        let snapshot = response::parse(self.inner.json(incoming, deadline).await?)?
            .inherit_correlation(request.correlation);
        match snapshot.execution.as_ref().map(|e| &e.status) {
            Some(
                ExecutionStatus::Queued
                | ExecutionStatus::InProgress
                | ExecutionStatus::Completed
                | ExecutionStatus::Incomplete,
            ) => {
                let mut handle = self.response_handle(&snapshot)?;
                handle.kind = Some("background".into());
                Ok(handle)
            }
            _ => response::completed(snapshot)
                .and_then(|_| Err(response::malformed("invalid background acknowledgement"))),
        }
    }

    /// Creates a serializable, endpoint-scoped handle for an existing response.
    ///
    /// # Errors
    /// Rejects responses without Perplexity origin or a nonblank execution ID.
    pub fn response_handle(&self, response: &ModelResponse) -> Result<DeferredHandle> {
        let execution = response
            .execution
            .as_ref()
            .ok_or_else(|| Error::Validation("response has no execution identity".into()))?;
        if !response
            .message
            .origin
            .as_ref()
            .is_some_and(|o| o.provider == "perplexity" && o.api == "agent")
        {
            return Err(Error::Validation(
                "response belongs to another provider".into(),
            ));
        }
        validate_id(&execution.id)?;
        let mut handle = DeferredHandle::new("perplexity", &execution.id);
        handle.kind = Some("response".into());
        handle
            .metadata
            .insert("base_url".into(), json!(self.inner.config.base_url));
        if let Some(correlation) = &response.correlation {
            handle
                .metadata
                .insert("correlation".into(), serde_json::to_value(correlation)?);
        }
        Ok(handle)
    }

    /// Retrieves one snapshot without polling or creating another run.
    ///
    /// # Errors
    /// Rejects foreign handles and propagates retrieval/decoding failures.
    /// A failed or incomplete run remains visible as an explicit snapshot status.
    pub async fn retrieve_response(&self, handle: &DeferredHandle) -> Result<ModelResponse> {
        self.retrieve_at(handle, self.inner.deadline(None)?).await
    }

    async fn retrieve_at(
        &self,
        handle: &DeferredHandle,
        deadline: Instant,
    ) -> Result<ModelResponse> {
        self.validate_handle(handle)?;
        let outgoing =
            self.inner
                .request(reqwest::Method::GET, &path(&["agent", &handle.id])?, None)?;
        let incoming = self.inner.send(outgoing, Operation::Read, deadline).await?;
        let mut snapshot = response::parse(self.inner.json(incoming, deadline).await?)?;
        if snapshot
            .execution
            .as_ref()
            .is_none_or(|execution| execution.id != handle.id)
        {
            return Err(response::with_partial(
                response::malformed("retrieval returned a different response id"),
                snapshot,
            ));
        }
        if let Some(value) = handle.metadata.get("correlation") {
            snapshot.correlation = Some(
                serde_json::from_value(value.clone())
                    .map_err(|_| Error::Validation("invalid handle correlation".into()))?,
            );
        }
        Ok(snapshot)
    }

    /// Reconnects to a durable stream after a previously observed sequence number.
    ///
    /// Only new events are emitted; the final response snapshot is complete.
    /// An already-terminal run is returned directly. An expired reconnect window
    /// produces an error; call [`Self::retrieve_response`] to inspect the run.
    ///
    /// # Errors
    /// Rejects non-background/foreign handles and propagates retrieval/reconnect
    /// errors. Never submits a replacement run.
    pub async fn resume_background(
        &self,
        handle: &DeferredHandle,
        after: Option<u64>,
    ) -> Result<ModelStream> {
        self.require_background(handle)?;
        let deadline = self.inner.deadline(None)?;
        let snapshot = self.retrieve_at(handle, deadline).await?;
        let execution = snapshot
            .execution
            .clone()
            .ok_or_else(|| response::malformed("response has no execution metadata"))?;
        if execution.status.is_terminal() {
            let terminal = match response::completed(snapshot) {
                Ok(result) => ModelStreamItem::Completed(result),
                Err(Error::Provider(error)) => ModelStreamItem::ProviderFailed(*error),
                Err(error) => return Err(error),
            };
            return Ok(ModelStream::new(Box::pin(futures::stream::iter([
                ModelStreamItem::Started,
                terminal,
            ]))));
        }
        if !matches!(
            execution.status,
            ExecutionStatus::Queued | ExecutionStatus::InProgress | ExecutionStatus::Cancelling
        ) {
            return Err(response::with_partial(
                response::malformed("unknown background state"),
                snapshot,
            ));
        }
        let mut outgoing =
            self.inner
                .request(reqwest::Method::GET, &path(&["agent", &handle.id])?, None)?;
        outgoing
            .url_mut()
            .query_pairs_mut()
            .append_pair("stream", "true");
        if let Some(after) = after {
            outgoing
                .url_mut()
                .query_pairs_mut()
                .append_pair("starting_after", &after.to_string());
        }
        let incoming = match self.inner.send(outgoing, Operation::Read, deadline).await {
            Ok(incoming) => incoming,
            Err(Error::Provider(mut error)) if error.status == Some(400) => {
                error.code = Some("reconnect_unavailable".into());
                error.message =
                    "reconnect unavailable; retrieve the existing response instead of resubmitting"
                        .into();
                return Err(response::with_partial(Error::Provider(error), snapshot));
            }
            Err(error) => return Err(response::with_partial(error, snapshot)),
        };
        let mut result = stream::open(
            self.inner.clone(),
            incoming,
            deadline,
            Some(execution),
            after,
        )?;
        if let Some(correlation) = snapshot.correlation {
            result = result.with_correlation(correlation);
        }
        Ok(result)
    }

    /// Requests remote cancellation; `Cancelling` is acknowledgement, not completion.
    ///
    /// Poll afterward to observe `Cancelled`. Dropping a local stream does not
    /// implicitly call this endpoint.
    ///
    /// # Errors
    /// Rejects foreign/non-background handles and propagates cancel errors,
    /// including already-terminal (400) and unknown (404) responses. No auto retry.
    pub async fn cancel_background(&self, handle: &DeferredHandle) -> Result<ExecutionStatus> {
        self.require_background(handle)?;
        let deadline = self.inner.deadline(None)?;
        let outgoing = self.inner.request(
            reqwest::Method::POST,
            &path(&["agent", &handle.id, "cancel"])?,
            None,
        )?;
        let incoming = self
            .inner
            .send(outgoing, Operation::Cancel, deadline)
            .await?;
        let value = self.inner.json(incoming, deadline).await?.value;
        if value.get("response_id").and_then(serde_json::Value::as_str) != Some(&handle.id)
            || value.get("status").and_then(serde_json::Value::as_str) != Some("cancelling")
        {
            return Err(response::malformed("invalid cancellation acknowledgement"));
        }
        Ok(ExecutionStatus::Cancelling)
    }

    /// Lists authoritative file descriptors for this response without downloading them.
    ///
    /// # Errors
    /// Rejects foreign handles, invalid descriptors, duplicate IDs or provider errors.
    pub async fn list_response_files(&self, handle: &DeferredHandle) -> Result<Vec<ModelFile>> {
        self.validate_handle(handle)?;
        let deadline = self.inner.deadline(None)?;
        let outgoing = self.inner.request(
            reqwest::Method::GET,
            &path(&["agent", &handle.id, "files"])?,
            None,
        )?;
        let incoming = self.inner.send(outgoing, Operation::Read, deadline).await?;
        let list: FileList =
            serde_json::from_value(self.inner.json(incoming, deadline).await?.value)
                .map_err(|_| response::malformed("invalid file list"))?;
        let mut ids = HashSet::new();
        list.data
            .into_iter()
            .map(|file| {
                validate_id(&file.id)?;
                if !ids.insert(file.id.clone()) {
                    return Err(response::malformed("duplicate file id"));
                }
                Ok(ModelFile {
                    response_id: handle.id.clone(),
                    id: file.id,
                    filename: file.filename,
                    bytes: file.bytes,
                    created_at: file.created_at,
                })
            })
            .collect()
    }

    /// Downloads one existing file into bounded memory. Filenames are never used as paths.
    ///
    /// # Errors
    /// Rejects foreign handles or invalid file IDs; returns provider, size, or
    /// deadline errors without regenerating the original response.
    pub async fn download_response_file(
        &self,
        handle: &DeferredHandle,
        file_id: &str,
    ) -> Result<ModelFileContent> {
        self.validate_handle(handle)?;
        validate_id(file_id)?;
        let deadline = self.inner.deadline(None)?;
        let outgoing = self.inner.request(
            reqwest::Method::GET,
            &path(&["agent", &handle.id, "files", file_id, "content"])?,
            None,
        )?;
        let incoming = self.inner.send(outgoing, Operation::Read, deadline).await?;
        let media_type = incoming
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(str::to_owned);
        let bytes = tokio::time::timeout_at(
            deadline,
            transport::read_limited(incoming, self.inner.config.max_file_bytes),
        )
        .await
        .map_err(|_| transport::timeout())??;
        Ok(ModelFileContent {
            response_id: handle.id.clone(),
            file_id: file_id.into(),
            media_type,
            bytes,
        })
    }

    fn validate_handle(&self, handle: &DeferredHandle) -> Result<()> {
        if handle.provider != "perplexity"
            || !matches!(handle.kind.as_deref(), Some("response" | "background"))
            || handle
                .metadata
                .get("base_url")
                .and_then(serde_json::Value::as_str)
                != Some(&self.inner.config.base_url)
        {
            return Err(Error::Validation(
                "handle belongs to a different provider or endpoint".into(),
            ));
        }
        validate_id(&handle.id)
    }

    fn require_background(&self, handle: &DeferredHandle) -> Result<()> {
        self.validate_handle(handle)?;
        if handle.kind.as_deref() != Some("background") {
            return Err(Error::Unsupported(
                "operation requires a background handle".into(),
            ));
        }
        Ok(())
    }
}

fn validate_id(id: &str) -> Result<()> {
    if id.trim().is_empty() || id.len() > 512 || matches!(id, "." | "..") {
        Err(Error::Validation("invalid response/file identifier".into()))
    } else {
        Ok(())
    }
}

fn path(segments: &[&str]) -> Result<String> {
    let mut url = reqwest::Url::parse("https://path.invalid/")
        .map_err(|_| Error::Validation("invalid path base".into()))?;
    {
        let mut path = url
            .path_segments_mut()
            .map_err(|_| Error::Validation("invalid path base".into()))?;
        path.clear();
        for segment in segments {
            validate_id(segment)?;
            path.push(segment);
        }
    }
    Ok(url.path().into())
}
