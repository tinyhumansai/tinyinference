//! Frozen-surface guard (compat plan 05 §6.3).
//!
//! Downstream crates match these enums exhaustively and build these structs
//! with full literals, so adding a variant or a field to any of them is a
//! breaking change for tinyagents, OpenHuman and OpenCompany. The hub adds none;
//! this file asserts *current reality* so a future upstream addition fails here
//! first, with the downstream site named. It changes no llm code.
//!
//! Every `match` below has **no wildcard arm** on purpose.

use serde_json::json;
use tinyinference_llm::model::{ProviderError, ToolChoice};
use tinyinference_llm::providers::anthropic::AnthropicConfig;
use tinyinference_llm::providers::openai::{AuthStyle, OpenAiConfig};
use tinyinference_llm::{
    AssistantMessage, ContentBlock, Error, Message, ModelResponse, ModelStreamItem,
    ProviderFailureClass, ToolCall, ToolFormat, ToolSchema,
};

/// llm `Error` (6 variants). Downstream: tinyagents-harness
/// `error.rs:339-348`.
fn llm_error(e: &Error) -> u8 {
    match e {
        Error::Model(_) => 1,
        Error::Provider(_) => 2,
        Error::Validation(_) => 3,
        Error::Serialization(_) => 4,
        Error::Catalog(_) => 5,
        Error::Unsupported(_) => 6,
    }
}

/// embeddings `Error` (4). Downstream: tinyagents-harness `error.rs:352-360`.
fn embeddings_error(e: &tinyinference_embeddings::Error) -> u8 {
    match e {
        tinyinference_embeddings::Error::Validation(_) => 1,
        tinyinference_embeddings::Error::Serialization(_) => 2,
        tinyinference_embeddings::Error::Embedding(_) => 3,
        tinyinference_embeddings::Error::Cancelled => 4,
    }
}

/// llm `Message` (5). Downstream: opencompany-core
/// `harness/built_in/provider.rs:531-561`.
fn message(m: &Message) -> u8 {
    match m {
        Message::System(_) => 1,
        Message::User(_) => 2,
        Message::Assistant(_) => 3,
        Message::Tool(_) => 4,
        Message::Custom(_) => 5,
    }
}

/// llm `ContentBlock` (9). Downstream: tinyagents-harness
/// `summarization/render.rs:71-84`.
fn content_block(b: &ContentBlock) -> u8 {
    match b {
        ContentBlock::Text(_) => 1,
        ContentBlock::Json(_) => 2,
        ContentBlock::Image(_) => 3,
        ContentBlock::Thinking { .. } => 4,
        ContentBlock::RedactedThinking { .. } => 5,
        ContentBlock::ProviderExtension(_) => 6,
        ContentBlock::Audio(_) => 7,
        ContentBlock::Video(_) => 8,
        ContentBlock::Document(_) => 9,
    }
}

/// llm `ModelStreamItem` (11). Downstream: tinyagents-harness
/// `stream/frame.rs:135-206`.
fn stream_item(i: &ModelStreamItem) -> u8 {
    match i {
        ModelStreamItem::Started => 1,
        ModelStreamItem::MessageDelta(_) => 2,
        ModelStreamItem::ToolCallDelta(_) => 3,
        ModelStreamItem::UsageDelta(_) => 4,
        ModelStreamItem::BlockStart { .. } => 5,
        ModelStreamItem::BlockDelta { .. } => 6,
        ModelStreamItem::BlockEnd { .. } => 7,
        ModelStreamItem::Completed(_) => 8,
        ModelStreamItem::Failed(_) => 9,
        ModelStreamItem::ProviderFailed(_) => 10,
        ModelStreamItem::Deferred(_) => 11,
    }
}

/// llm `ToolChoice` (4). Downstream: opencompany-core
/// `harness/built_in/provider.rs:604-607`.
fn tool_choice(c: &ToolChoice) -> u8 {
    match c {
        ToolChoice::Auto => 1,
        ToolChoice::None => 2,
        ToolChoice::Required => 3,
        ToolChoice::Tool(_) => 4,
    }
}

/// llm `ProviderFailureClass` (5). Downstream: opencompany-core
/// `server/operator.rs:4037-4051`.
fn failure_class(c: ProviderFailureClass) -> u8 {
    match c {
        ProviderFailureClass::Retryable => 1,
        ProviderFailureClass::NonRetryable => 2,
        ProviderFailureClass::RateLimited => 3,
        ProviderFailureClass::NonRetryableRateLimit => 4,
        ProviderFailureClass::UpstreamUnhealthy => 5,
    }
}

#[test]
fn the_frozen_enums_still_have_exactly_the_variants_downstream_matches() {
    // The exhaustive matches above are the guard (they fail to compile on a
    // change); calling them keeps them from being dead code.
    assert_eq!(llm_error(&Error::Model(String::new())), 1);
    assert_eq!(llm_error(&Error::Unsupported(String::new())), 6);
    assert_eq!(
        embeddings_error(&tinyinference_embeddings::Error::Cancelled),
        4
    );
    assert_eq!(message(&Message::user("hi")), 2);
    assert_eq!(content_block(&ContentBlock::Text(String::new())), 1);
    assert_eq!(stream_item(&ModelStreamItem::Started), 1);
    assert_eq!(tool_choice(&ToolChoice::Auto), 1);
    assert_eq!(tool_choice(&ToolChoice::Tool("t".into())), 4);
    assert_eq!(failure_class(ProviderFailureClass::UpstreamUnhealthy), 5);
}

#[test]
fn the_frozen_structs_still_accept_the_full_literals_downstream_writes() {
    // Full literals (no `..Default::default()`): a new field breaks these.
    let message = AssistantMessage {
        id: None,
        content: Vec::new(),
        tool_calls: Vec::new(),
        usage: None,
        origin: None,
    };
    let response = ModelResponse {
        message: message.clone(),
        usage: None,
        finish_reason: None,
        raw: None,
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    };
    assert!(response.message.content.is_empty());
    let call = ToolCall {
        id: "c1".into(),
        name: "t".into(),
        arguments: json!({}),
        invalid: None,
    };
    assert_eq!(call.name, "t");
    let schema = ToolSchema {
        name: "t".into(),
        description: "d".into(),
        parameters: json!({}),
        format: ToolFormat::default(),
    };
    assert_eq!(schema.name, "t");
    let error = ProviderError {
        provider: "p".into(),
        model: None,
        status: None,
        code: None,
        message: "m".into(),
        retryable: false,
        retry_after_ms: None,
        raw: None,
        partial_message: None,
        stop_reason: None,
    };
    assert_eq!(error.provider, "p");
}

#[test]
fn the_frozen_provider_configs_still_accept_their_full_literals() {
    let openai = OpenAiConfig {
        provider_name: "openai",
        endpoint: "https://api.openai.com/v1",
        api_key: "sk-not-a-real-key",
        auth_style: AuthStyle::Bearer,
        model: "m",
        temperature_unsupported_models: &[],
        temperature_override: None,
        merge_system_into_user: false,
        extra_headers: &[],
        native_tool_calling: None,
        vision: None,
        default_provider_options: None,
        responses_api_primary: false,
        responses_omit_max_output_tokens: false,
        extra_query_params: &[],
        user_agent: None,
        explicit_cache_control: false,
    };
    assert_eq!(openai.model, "m");
    let anthropic = AnthropicConfig {
        endpoint: "https://api.anthropic.com/v1",
        api_key: "sk-not-a-real-key",
        model: "m",
        temperature_override: None,
        temperature_unsupported_models: &[],
        extra_headers: &[],
    };
    assert_eq!(anthropic.model, "m");
}

#[test]
fn hub_errors_do_not_alter_llms_error_display_or_debug() {
    // The hub maps *from* llm's error and never changes it.
    let e = Error::Unsupported("tools".into());
    assert_eq!(e.to_string(), "unsupported operation: tools");
}
