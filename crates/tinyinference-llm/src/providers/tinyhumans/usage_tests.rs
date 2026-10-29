use super::*;
use crate::message::AssistantMessage;
use crate::model::ModelResponse;
use crate::usage::{Usage, recover_usage};

fn response(usage: Usage, raw: serde_json::Value) -> ModelResponse {
    ModelResponse {
        output: Vec::new(),
        execution: None,
        message: AssistantMessage {
            id: None,
            content: vec![],
            tool_calls: vec![],
            usage: None,
            origin: None,
        },
        usage: Some(usage),
        finish_reason: None,
        raw: Some(raw),
        resolved_model: None,
        continue_turn: None,
        served_from_cache: false,
        correlation: None,
        resolved_route: None,
    }
}

#[test]
fn projects_managed_usage_without_fabricating_a_charge() {
    let projected = project_tinyhumans_envelope(response(
        Usage {
            input_tokens: 1000,
            output_tokens: 50,
            cache_read_tokens: 17,
            ..Usage::default()
        },
        serde_json::json!({
            "openhuman": {
                "usage": { "cached_input_tokens": 128, "context_window": 200000 }
            }
        }),
    ));

    let recovered = recover_usage(&projected).expect("usage recovered");
    assert_eq!(recovered.usage.input_tokens, 1000);
    assert_eq!(recovered.usage.output_tokens, 50);
    assert_eq!(recovered.usage.cache_read_tokens, 17);
    assert_eq!(recovered.context_window, 200_000);
    assert_eq!(recovered.charged_amount_usd, None);
}

#[test]
fn leaves_non_managed_usage_untouched() {
    let projected = project_tinyhumans_envelope(response(
        Usage {
            input_tokens: 10,
            output_tokens: 5,
            cache_read_tokens: 3,
            ..Usage::default()
        },
        serde_json::json!({ "id": "resp_1" }),
    ));

    assert!(
        projected
            .raw
            .as_ref()
            .unwrap()
            .get("openhuman_usage_meta")
            .is_none()
    );
    let recovered = recover_usage(&projected).expect("usage recovered");
    assert_eq!(recovered.charged_amount_usd, None);
    assert_eq!(recovered.usage.cache_read_tokens, 3);
}

#[test]
fn projects_managed_cached_tokens_when_provider_usage_has_none() {
    let projected = project_tinyhumans_envelope(response(
        Usage {
            input_tokens: 1000,
            output_tokens: 50,
            ..Usage::default()
        },
        serde_json::json!({
            "openhuman": {
                "usage": { "cached_input_tokens": 128, "context_window": 200000 },
                "billing": { "charged_amount_usd": 0.0042 }
            }
        }),
    ));

    let recovered = recover_usage(&projected).expect("usage recovered");
    assert_eq!(recovered.usage.cache_read_tokens, 128);
    assert_eq!(recovered.context_window, 200_000);
    assert_eq!(recovered.usage.input_tokens, 1000);
    assert_eq!(recovered.usage.output_tokens, 50);
    assert_eq!(recovered.charged_amount_usd, Some(0.0042));
}

#[test]
fn distinguishes_a_reported_zero_charge_from_missing_billing() {
    let free_call = project_tinyhumans_envelope(response(
        Usage::default(),
        serde_json::json!({ "openhuman": { "billing": { "charged_amount_usd": 0.0 } } }),
    ));
    let no_billing = project_tinyhumans_envelope(response(
        Usage::default(),
        serde_json::json!({ "openhuman": { "usage": { "context_window": 8192 } } }),
    ));
    let billing_without_amount = project_tinyhumans_envelope(response(
        Usage::default(),
        serde_json::json!({ "openhuman": { "billing": {} } }),
    ));

    assert_eq!(
        recover_usage(&free_call)
            .expect("free call has usage")
            .charged_amount_usd,
        Some(0.0)
    );
    assert_eq!(
        recover_usage(&no_billing)
            .expect("unbilled call has usage")
            .charged_amount_usd,
        None
    );
    assert_eq!(
        recover_usage(&billing_without_amount)
            .expect("billing block has usage")
            .charged_amount_usd,
        Some(0.0)
    );
}
