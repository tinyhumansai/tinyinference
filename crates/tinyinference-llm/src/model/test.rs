//! Tests for the harness model layer: `ModelRequest`/`ModelResponse` builders
//! and accessors, `ModelProfile`/`CapabilitySet` matching, and
//! `StreamAccumulator`/`collect_model_stream` folding of streamed
//! items (deltas, usage, terminal `Completed`/`Failed`) into a response.

use super::*;
use crate::message::Message;
use crate::usage::Usage;
use serde_json::json;

#[test]
fn request_builder_sets_fields() {
    let req = ModelRequest::new(vec![Message::user("hi")])
        .with_model("gpt")
        .with_model_hint(ModelHint {
            model: "fast".into(),
            priority: 10,
            reason: Some("latency".into()),
        })
        .with_reuse_previous_model(true)
        .with_temperature(0.5)
        .with_top_p(0.9)
        .with_max_tokens(128)
        .with_stop_sequences(["END", "STOP"])
        .with_seed(42)
        .with_timeout_ms(1000)
        .with_tool_choice(ToolChoice::Required)
        .with_tag("t");
    assert_eq!(req.model.as_deref(), Some("gpt"));
    assert_eq!(req.temperature, Some(0.5));
    assert_eq!(req.top_p, Some(0.9));
    assert_eq!(req.max_tokens, Some(128));
    assert_eq!(req.stop_sequences, vec!["END", "STOP"]);
    assert_eq!(req.seed, Some(42));
    assert_eq!(req.timeout_ms, Some(1000));
    assert_eq!(req.tool_choice, ToolChoice::Required);
    assert_eq!(req.tags, vec!["t".to_string()]);
    assert_eq!(req.model_hints[0].model, "fast");
    assert!(req.reuse_previous_model);
}

#[test]
fn tool_choice_defaults_to_auto() {
    assert_eq!(ModelRequest::default().tool_choice, ToolChoice::Auto);
}

#[test]
fn lifecycle_helpers_gate_retired_and_deprecated_models() {
    use crate::model::ModelStatus;

    let stable = ModelProfile::default();
    assert!(stable.is_usable());
    assert!(!stable.is_deprecated());

    let deprecated = ModelProfile {
        status: ModelStatus::Deprecated,
        ..ModelProfile::default()
    };
    assert!(deprecated.is_usable()); // still callable, but flagged
    assert!(deprecated.is_deprecated());

    let retired = ModelProfile {
        status: ModelStatus::Retired,
        ..ModelProfile::default()
    };
    assert!(!retired.is_usable());
    assert!(retired.is_deprecated());
}

#[test]
fn context_window_patterns_cover_common_provider_families() {
    assert_eq!(context_window_for_model_id("gpt-4.1"), Some(1_047_576));
    assert_eq!(
        context_window_for_model_id("openai/gpt-4o-mini"),
        Some(128_000)
    );
    assert_eq!(
        context_window_for_model_id("github_copilot/claude-haiku-4.5"),
        Some(200_000)
    );
    assert_eq!(context_window_for_model_id("deepseek-chat"), Some(128_000));
    assert_eq!(context_window_for_model_id("gemma3:4b"), Some(8_192));
    assert_eq!(context_window_for_model_id("llama3:8b"), Some(128_000));
    assert_eq!(context_window_for_model_id("totally-unknown-model"), None);
    assert_eq!(context_window_for_model_id("   "), None);
}

#[test]
fn o1_o3_context_patterns_require_segment_boundaries() {
    assert_eq!(context_window_for_model_id("o1"), Some(200_000));
    assert_eq!(context_window_for_model_id("o1-mini"), Some(200_000));
    assert_eq!(context_window_for_model_id("o3-mini"), Some(200_000));
    assert_eq!(
        context_window_for_model_id("openai/o1-preview"),
        Some(200_000)
    );

    assert_eq!(context_window_for_model_id("solo1-7b"), None);
    assert_eq!(context_window_for_model_id("proto3-chat"), None);
    assert_eq!(context_window_for_model_id("octo3thing"), None);
    assert_eq!(
        context_window_for_model_id("ollama/mistral-for-o1-benchmark"),
        Some(200_000)
    );
}

#[test]
fn model_id_vision_capability_is_conservative() {
    for model in [
        "moondream:1.8b-v2-q4_K_S",
        "llava:7b",
        "llama3.2-vision:11b",
        "qwen2.5vl:7b",
        "gemma3:4b-it-qat",
        "gemma3:latest",
        "gemma4:e4b-it-q8_0",
        "hf.co/user/llava-v1.6-mistral-7b",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "hf.co/user/gemma3:4b",
    ] {
        assert!(model_id_supports_vision(model), "{model}");
    }
    for model in [
        "",
        "gemma3:270m-it-qat",
        "gemma3:1b",
        "gemma3n:e4b-it-q8_0",
        "llama3.1:8b",
        "qwen2.5:14b",
        "bge-m3",
        "acme-visionary-text",
    ] {
        assert!(!model_id_supports_vision(model), "{model}");
    }
}

#[test]
fn model_id_globs_and_temperature_policy_are_provider_neutral() {
    assert!(model_id_glob_match("o1*", "O1-preview"));
    assert!(model_id_glob_match("*turbo", "gpt-4-turbo"));
    assert!(model_id_glob_match("*mid*", "a-middle-b"));
    assert!(model_id_glob_match("*a", "aa"));
    assert!(model_id_glob_match("foo*bar", "foobarbar"));
    assert!(!model_id_glob_match("gpt-4o", "gpt-4o-mini"));
    assert!(!model_id_glob_match("foo*foo", "foo"));

    let unsupported = vec!["o1*".to_string(), "gpt-5*".to_string()];
    assert_eq!(
        effective_temperature("o1-preview", Some(0.7), Some(0.2), &unsupported),
        None
    );
    assert_eq!(
        effective_temperature("gpt-4o", Some(0.7), Some(0.2), &unsupported),
        Some(0.2)
    );
}

#[test]
fn cacheable_prefix_ids_in_order() {
    let req = ModelRequest::new(vec![]).with_cache_segments(vec![
        PromptSegment {
            id: "sys".into(),
            role: SegmentRole::System,
            cacheable: true,
        },
        PromptSegment {
            id: "tools".into(),
            role: SegmentRole::Tools,
            cacheable: true,
        },
        PromptSegment {
            id: "tail".into(),
            role: SegmentRole::Volatile,
            cacheable: false,
        },
    ]);
    assert_eq!(req.cacheable_prefix_ids(), vec!["sys", "tools"]);
}

#[test]
fn response_format_json_schema() {
    let fmt = ResponseFormat::json_schema("person", json!({"type": "object"}));
    match fmt {
        ResponseFormat::JsonSchema { name, .. } => assert_eq!(name, "person"),
        _ => panic!("expected json schema"),
    }
}

#[test]
fn response_format_auto_constructor() {
    let fmt = ResponseFormat::auto("person", json!({"type": "object"}));
    match fmt {
        ResponseFormat::Auto { name, .. } => assert_eq!(name, "person"),
        _ => panic!("expected auto"),
    }
}

#[test]
fn default_profile_is_conservative() {
    let profile = ModelProfile::default();
    assert!(!profile.tool_calling);
    assert!(!profile.native_structured_output);
    assert!(!profile.streaming);
    assert_eq!(profile.status, ModelStatus::Stable);
    // Default modalities are text-only.
    assert!(profile.modalities.text_in && profile.modalities.text_out);
    assert!(!profile.modalities.image_in);
}

#[test]
fn empty_capability_set_is_always_satisfied() {
    let profile = ModelProfile::default();
    assert!(profile.satisfies(&CapabilitySet::default()));
}

#[test]
fn profile_satisfies_matching_capabilities() {
    let profile = ModelProfile {
        tool_calling: true,
        streaming: true,
        json_schema: true,
        native_structured_output: true,
        max_input_tokens: Some(128_000),
        max_output_tokens: Some(8_000),
        ..ModelProfile::default()
    };
    let required = CapabilitySet {
        tool_calling: true,
        json_schema: true,
        native_structured_output: true,
        min_input_tokens: Some(100_000),
        min_output_tokens: Some(4_000),
        ..CapabilitySet::default()
    };
    assert!(profile.satisfies(&required));
}

#[test]
fn profile_does_not_satisfy_missing_capability() {
    let profile = ModelProfile {
        tool_calling: true,
        ..ModelProfile::default()
    };
    // Requires reasoning, which the profile does not advertise.
    let required = CapabilitySet {
        tool_calling: true,
        reasoning: true,
        ..CapabilitySet::default()
    };
    assert!(!profile.satisfies(&required));
}

#[test]
fn profile_token_requirement_fails_when_capacity_unknown_or_too_small() {
    // Unknown capacity fails a token requirement.
    let unknown = ModelProfile::default();
    assert!(!unknown.satisfies(&CapabilitySet {
        min_input_tokens: Some(1_000),
        ..CapabilitySet::default()
    }));

    // Known-but-too-small capacity also fails.
    let small = ModelProfile {
        max_output_tokens: Some(512),
        ..ModelProfile::default()
    };
    assert!(!small.satisfies(&CapabilitySet {
        min_output_tokens: Some(4_096),
        ..CapabilitySet::default()
    }));
}

#[test]
fn permissive_profile_satisfies_demanding_capability_set() {
    let profile = ModelProfile::permissive();
    let required = CapabilitySet {
        tool_calling: true,
        parallel_tool_calls: true,
        streaming: true,
        streaming_tool_chunks: true,
        native_structured_output: true,
        json_schema: true,
        reasoning: true,
        image_in: true,
        image_out: true,
        audio_in: true,
        audio_out: true,
        ..CapabilitySet::default()
    };
    assert!(profile.satisfies(&required));
}

#[test]
fn model_request_capability_and_provider_option_builders() {
    let caps = CapabilitySet {
        tool_calling: true,
        ..CapabilitySet::default()
    };
    let req = ModelRequest::new(vec![])
        .with_required_capabilities(caps.clone())
        .with_provider_options(json!({"top_k": 5}))
        .with_provider_option("hotness", json!("high"))
        .with_continuation_id("resp-1");
    assert_eq!(req.required_capabilities, Some(caps));
    assert_eq!(req.provider_options, json!({"top_k": 5, "hotness": "high"}));
    assert_eq!(req.continuation_id.as_deref(), Some("resp-1"));
    // Defaults stay null/None so existing builders/tests are unaffected.
    assert!(ModelRequest::default().provider_options.is_null());
    assert!(ModelRequest::default().required_capabilities.is_none());
}

#[test]
fn response_helpers() {
    let resolved = ResolvedModel {
        name: "fast".into(),
        requested: Some("fast".into()),
        source: ModelResolutionSource::Hint,
    };
    let resp = ModelResponse::assistant("hi")
        .with_finish_reason("stop")
        .with_resolved_model(resolved.clone());
    assert_eq!(resp.text(), "hi");
    assert!(resp.tool_calls().is_empty());
    assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
    assert_eq!(resp.resolved_model, Some(resolved));
}

#[test]
fn stream_accumulator_collects_reasoning_side_channel() {
    use crate::message::MessageDelta;

    let mut acc = StreamAccumulator::new();
    acc.push(&ModelStreamItem::Started);
    acc.push(&ModelStreamItem::MessageDelta(MessageDelta::reasoning(
        "thinking...",
    )));
    acc.push(&ModelStreamItem::MessageDelta(MessageDelta::text(
        "visible",
    )));
    acc.push(&ModelStreamItem::MessageDelta(MessageDelta {
        text: " answer".into(),
        reasoning: " more".into(),
        tool_call: None,
    }));
    acc.push(&ModelStreamItem::Completed(ModelResponse::assistant(
        "visible answer",
    )));

    // Reasoning is a side channel, kept out of the final message text.
    assert_eq!(acc.reasoning(), "thinking... more");
    let response = acc.finish().unwrap();
    assert_eq!(response.text(), "visible answer");
}

#[test]
fn finish_preserves_message_usage_from_completed_response() {
    // A completed response that carries usage only on the message (not the
    // top-level field) and no streamed UsageDelta. finish must not clobber the
    // message usage with None; it should promote it to the response too.
    let mut response = ModelResponse::assistant("hi");
    response.usage = None;
    response.message.usage = Some(Usage::new(10, 20));

    let mut acc = StreamAccumulator::new();
    acc.push(&ModelStreamItem::Started);
    acc.push(&ModelStreamItem::Completed(response));

    let finished = acc.finish().unwrap();
    assert_eq!(finished.message.usage, Some(Usage::new(10, 20)));
    assert_eq!(finished.usage, Some(Usage::new(10, 20)));
}

#[test]
fn finish_backfills_usage_from_stream_delta_when_completed_lacks_it() {
    // No usage anywhere on the completed response, but a UsageDelta arrived. Both
    // the response and its message pick up the streamed usage.
    let mut response = ModelResponse::assistant("hi");
    response.usage = None;
    response.message.usage = None;

    let mut acc = StreamAccumulator::new();
    acc.push(&ModelStreamItem::Started);
    acc.push(&ModelStreamItem::UsageDelta(Usage::new(4, 6)));
    acc.push(&ModelStreamItem::Completed(response));

    let finished = acc.finish().unwrap();
    assert_eq!(finished.usage, Some(Usage::new(4, 6)));
    assert_eq!(finished.message.usage, Some(Usage::new(4, 6)));
}

#[test]
fn finish_preserves_provider_error_classification_from_provider_failed() {
    // A streamed `ProviderFailed` carrying a non-retryable provider error (401
    // auth) must surface as `Error::Provider` with the struct intact —
    // not stringified into `Model` — so `is_retryable` classifies it as permanent
    // instead of retrying + fallback-churning it as transient.
    use crate::Error;
    use crate::failure::classify_provider_error;

    let mut acc = StreamAccumulator::new();
    acc.push(&ModelStreamItem::Started);
    acc.push(&ModelStreamItem::ProviderFailed(ProviderError {
        provider: "openai".into(),
        status: Some(401),
        code: Some("invalid_api_key".into()),
        message: "Incorrect API key provided".into(),
        retryable: false,
        ..ProviderError::default()
    }));

    assert!(acc.is_terminal());
    let err = acc.finish().unwrap_err();
    match &err {
        Error::Provider(error) => {
            assert_eq!(error.status, Some(401));
            assert_eq!(error.code.as_deref(), Some("invalid_api_key"));
            assert!(!error.retryable);
        }
        other => panic!("expected Provider error, got {other:?}"),
    }
    assert!(
        !match &err {
            Error::Provider(error) => classify_provider_error(error).is_retryable(),
            _ => true,
        },
        "a permanent streamed provider failure must not be retried as transient"
    );
}

#[test]
fn finish_maps_unstructured_failed_to_model_error() {
    // The unstructured `Failed(String)` path stays `Error::Model` — no
    // structured detail to classify from, so the retry layer treats it as a
    // transient transport/parse failure.
    use crate::Error;

    let mut acc = StreamAccumulator::new();
    acc.push(&ModelStreamItem::Failed("stream broke".into()));

    assert!(acc.is_terminal());
    match acc.finish().unwrap_err() {
        Error::Model(message) => assert_eq!(message, "stream broke"),
        other => panic!("expected Model error, got {other:?}"),
    }
}

#[test]
fn finish_names_reconstructed_tool_call_from_the_call_opening_delta_name() {
    // A call-opening delta carries the tool name (no args yet); subsequent
    // argument fragments carry only content. With no authoritative `Completed`
    // response, the accumulator must still name the reconstructed tool call from
    // the first non-empty `tool_name` it saw for that call id.
    use crate::tool::ToolDelta;

    let mut acc = StreamAccumulator::new();
    acc.push(&ModelStreamItem::ToolCallDelta(ToolDelta {
        call_id: "call-1".into(),
        content: String::new(),
        tool_name: Some("search".into()),
        ..Default::default()
    }));
    acc.push(&ModelStreamItem::ToolCallDelta(ToolDelta {
        call_id: "call-1".into(),
        content: r#"{"q":"rust"}"#.into(),
        tool_name: None,
        ..Default::default()
    }));

    let finished = acc.finish().unwrap();
    assert_eq!(finished.message.tool_calls.len(), 1);
    let call = &finished.message.tool_calls[0];
    assert_eq!(call.id, "call-1");
    assert_eq!(call.name, "search", "name carried from the opening delta");
    assert_eq!(call.arguments, serde_json::json!({ "q": "rust" }));
}

#[test]
fn finish_marks_malformed_reconstructed_tool_arguments_invalid() {
    let mut accumulator = StreamAccumulator::new();
    accumulator.push(&ModelStreamItem::ToolCallDelta(crate::tool::ToolDelta {
        call_id: "call-1".into(),
        content: "{broken".into(),
        tool_name: Some("search".into()),
        ..Default::default()
    }));
    let response = accumulator.finish().unwrap();
    let call = &response.message.tool_calls[0];
    assert!(call.is_invalid());
    assert_eq!(call.arguments, serde_json::Value::String("{broken".into()));
}

/// Round-trips a [`ModelStreamItem`] through JSON and asserts the re-serialized
/// form is byte-for-byte stable, proving every variant survives serde.
fn roundtrip_stream_item(item: ModelStreamItem) {
    let value = serde_json::to_value(&item).expect("serialize ModelStreamItem");
    let back: ModelStreamItem =
        serde_json::from_value(value.clone()).expect("deserialize ModelStreamItem");
    let reserialized = serde_json::to_value(&back).expect("re-serialize ModelStreamItem");
    assert_eq!(value, reserialized, "round-trip differs for {value}");
}

#[test]
fn model_stream_item_roundtrips_every_variant() {
    roundtrip_stream_item(ModelStreamItem::Started);
    roundtrip_stream_item(ModelStreamItem::MessageDelta(
        crate::message::MessageDelta::text("hi"),
    ));
    roundtrip_stream_item(ModelStreamItem::ToolCallDelta(crate::tool::ToolDelta {
        call_id: "call-1".into(),
        content: "{\"q\":1}".into(),
        tool_name: None,
        ..Default::default()
    }));
    roundtrip_stream_item(ModelStreamItem::UsageDelta(Usage::new(3, 5)));
    roundtrip_stream_item(ModelStreamItem::BlockStart {
        index: 0,
        kind: crate::model::BlockKind::Text,
    });
    roundtrip_stream_item(ModelStreamItem::BlockStart {
        index: 1,
        kind: crate::model::BlockKind::ToolCall {
            id: "call-1".into(),
            name: "search".into(),
        },
    });
    roundtrip_stream_item(ModelStreamItem::BlockDelta {
        index: 0,
        delta: crate::model::BlockDelta::ToolArgs("{}".into()),
    });
    roundtrip_stream_item(ModelStreamItem::BlockEnd {
        index: 0,
        block: crate::message::ContentBlock::Text("done".into()),
    });
    roundtrip_stream_item(ModelStreamItem::Completed(ModelResponse::assistant("done")));
    // The scalar-carrying variant an internally tagged enum could not encode.
    roundtrip_stream_item(ModelStreamItem::Failed("boom".to_string()));
    roundtrip_stream_item(ModelStreamItem::ProviderFailed(ProviderError {
        provider: "openai".into(),
        message: "nope".into(),
        ..ProviderError::default()
    }));
    roundtrip_stream_item(ModelStreamItem::ProviderFailed(ProviderError {
        provider: "anthropic".into(),
        message: "overloaded".into(),
        stop_reason: Some("pause_turn".into()),
        partial_message: Some(crate::message::AssistantMessage {
            id: Some("msg_1".into()),
            content: vec![crate::message::ContentBlock::Text("partial".into())],
            tool_calls: Vec::new(),
            usage: None,
        }),
        ..ProviderError::default()
    }));
}

#[test]
fn block_delta_to_message_delta_maps_each_channel() {
    use crate::model::{BlockDelta, block_delta_to_message_delta};

    let text = block_delta_to_message_delta(&BlockDelta::Text("hi".into()), "", None);
    assert_eq!(text.text, "hi");
    assert!(text.reasoning.is_empty());
    assert!(text.tool_call.is_none());

    let thinking = block_delta_to_message_delta(&BlockDelta::Thinking("plan".into()), "", None);
    assert_eq!(thinking.reasoning, "plan");
    assert!(thinking.text.is_empty());

    let args =
        block_delta_to_message_delta(&BlockDelta::ToolArgs("{}".into()), "call-1", Some("s"));
    let tool_call = args.tool_call.expect("tool_call fragment");
    assert_eq!(tool_call.call_id, "call-1");
    assert_eq!(tool_call.content, "{}");
    assert_eq!(tool_call.tool_name.as_deref(), Some("s"));
}

#[test]
fn model_stream_item_failed_serializes_without_panicking() {
    // Under internal tagging this call errored; adjacent tagging encodes the
    // string payload under `content`.
    let value = serde_json::to_value(ModelStreamItem::Failed("boom".into())).unwrap();
    assert_eq!(value["type"], json!("failed"));
    assert_eq!(value["content"], json!("boom"));
}

#[test]
fn stream_accumulator_reconstruct_preserves_reasoning_as_thinking_block() {
    use crate::message::{ContentBlock, MessageDelta};

    // No `Completed` item: `finish` reconstructs the message from deltas. The
    // accumulated reasoning must survive as a leading `Thinking` block rather
    // than being dropped.
    let mut acc = StreamAccumulator::new();
    acc.push(&ModelStreamItem::Started);
    acc.push(&ModelStreamItem::MessageDelta(MessageDelta::reasoning(
        "let me think",
    )));
    acc.push(&ModelStreamItem::MessageDelta(MessageDelta::text("42")));

    let response = acc.finish().unwrap();
    // Visible text excludes reasoning.
    assert_eq!(response.text(), "42");
    // Leading block is the preserved thinking; text follows.
    let content = &response.message.content;
    assert_eq!(content.len(), 2);
    assert_eq!(
        content[0],
        ContentBlock::Thinking {
            text: "let me think".into(),
            signature: None,
        }
    );
    assert_eq!(content[1], ContentBlock::Text("42".into()));
}

#[test]
fn stream_accumulator_reconstruct_without_reasoning_has_no_thinking_block() {
    use crate::message::{ContentBlock, MessageDelta};

    let mut acc = StreamAccumulator::new();
    acc.push(&ModelStreamItem::MessageDelta(MessageDelta::text("hi")));
    let response = acc.finish().unwrap();
    assert_eq!(
        response.message.content,
        vec![ContentBlock::Text("hi".into())]
    );
}
