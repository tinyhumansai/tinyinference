//! Public Perplexity operations must respect the host's network policy.

use tinyinference_llm::{
    ChatModel, ExecutionStatus, Message, ModelExecution, ModelRequest, ModelResponse,
    PerplexityModel, PerplexitySelection, allow_network_models, deny_network_models,
    network_models_denied,
};

/// Restores the process-wide network policy even if an assertion panics.
struct NetworkDenied {
    previously_denied: bool,
}

impl NetworkDenied {
    fn install() -> Self {
        let previously_denied = network_models_denied();
        deny_network_models();
        Self { previously_denied }
    }
}

impl Drop for NetworkDenied {
    fn drop(&mut self) {
        if !self.previously_denied {
            allow_network_models();
        }
    }
}

#[tokio::test]
async fn every_public_network_operation_obeys_the_host_network_guard() {
    let model =
        PerplexityModel::new("test-key", PerplexitySelection::Model("openai/test".into())).unwrap();
    let mut snapshot = ModelResponse::assistant("answer");
    snapshot.message.origin = Some(tinyinference_llm::message::MessageOrigin {
        provider: "perplexity".into(),
        api: "agent".into(),
        model: "openai/test".into(),
    });
    snapshot.execution = Some(ModelExecution {
        id: "resp_1".into(),
        model: "openai/test".into(),
        status: ExecutionStatus::Completed,
        incomplete_reason: None,
        sequence_number: None,
        cost: None,
        tool_usage: Default::default(),
        progress: Vec::new(),
    });
    let mut handle = model.response_handle(&snapshot).unwrap();
    handle.kind = Some("background".into());
    let _denied = NetworkDenied::install();
    let request = || ModelRequest::new(vec![Message::user("question")]);
    let errors = [
        model.invoke(&(), request()).await.unwrap_err(),
        model.stream(&(), request()).await.unwrap_err(),
        model.submit_background(request()).await.unwrap_err(),
        model.retrieve_response(&handle).await.unwrap_err(),
        model.resume_background(&handle, Some(1)).await.unwrap_err(),
        model.cancel_background(&handle).await.unwrap_err(),
        model.list_response_files(&handle).await.unwrap_err(),
        model
            .download_response_file(&handle, "file_1")
            .await
            .unwrap_err(),
        <PerplexityModel as ChatModel<()>>::fetch_deferred(&model, &handle)
            .await
            .unwrap_err(),
    ];
    for error in errors {
        assert!(
            error
                .to_string()
                .contains("network-backed model calls are denied"),
            "{error}"
        );
    }
}
