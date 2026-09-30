//! The chat model the runner's hub builds: every request goes through the
//! scripted transport (so it is recorded and checked), never a socket.

use std::sync::Arc;

use async_trait::async_trait;
use tinyinference_llm::MockModel;
use tinyinference_llm::error::{Error, Result as LlmResult};
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse, ProviderError};

use crate::client::{ModelFactory, ModelSpec};
use crate::error::HubError;
use crate::policy::EndpointPolicy;
use crate::ports::{Http, HubRequest};
use crate::testkit::ScriptedHttp;

/// Builds [`SimModel`]s over the runner's scripted transport.
#[derive(Debug)]
pub(crate) struct SimFactory {
    pub(crate) http: Arc<ScriptedHttp>,
    pub(crate) policy: EndpointPolicy,
}

impl ModelFactory for SimFactory {
    fn build(&self, spec: &ModelSpec<'_>) -> Result<Arc<dyn ChatModel<()>>, HubError> {
        Ok(Arc::new(SimModel {
            http: self.http.clone(),
            policy: self.policy.clone(),
            base: spec.turn.base_url.clone(),
            key: spec.key.map(str::to_string),
        }))
    }
}

/// One built model: it remembers the endpoint and key it was built with and
/// sends exactly that pair.
struct SimModel {
    http: Arc<ScriptedHttp>,
    policy: EndpointPolicy,
    base: String,
    key: Option<String>,
}

#[async_trait]
impl ChatModel<()> for SimModel {
    async fn invoke(&self, _: &(), _: ModelRequest) -> LlmResult<ModelResponse> {
        let url = format!("{}/chat/completions", self.base.trim_end_matches('/'));
        let mut request = HubRequest::post_json(url, &serde_json::json!({"messages": []}));
        if let Some(key) = &self.key {
            request = request
                .with_header("authorization", format!("Bearer {key}"))
                .with_credentialed(true);
        }
        let response = self
            .http
            .send(request, &self.policy)
            .await
            .map_err(|e| Error::Model(e.to_string()))?;
        if response.is_success() {
            return Ok(MockModel::text_response("ok"));
        }
        Err(Error::Provider(Box::new(ProviderError {
            provider: "sim".into(),
            status: Some(response.status),
            message: response.text(),
            ..ProviderError::default()
        })))
    }
}
