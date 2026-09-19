//! Tests for token usage accounting.
//!
//! Cover [`Usage::new`] totalling, `+`/`+=` accumulation across the detailed
//! token fields, and [`UsageTotals`] call-count tracking.

use super::*;

#[test]
fn new_sets_total() {
    let usage = Usage::new(10, 5);
    assert_eq!(usage.input_tokens, 10);
    assert_eq!(usage.output_tokens, 5);
    assert_eq!(usage.total_tokens, 15);
}

#[test]
fn add_accumulates_all_fields() {
    let a = Usage {
        input_tokens: 1,
        output_tokens: 2,
        total_tokens: 3,
        cache_read_tokens: 4,
        cache_creation_tokens: 5,
        reasoning_tokens: 6,
        charged_amount: Some(ChargedAmount::usd_micros(42)),
        context_window_tokens: Some(8_192),
    };
    let b = a;
    let sum = a + b;
    assert_eq!(sum.input_tokens, 2);
    assert_eq!(sum.cache_read_tokens, 8);
    assert_eq!(sum.reasoning_tokens, 12);
    assert_eq!(sum.charged_amount, Some(ChargedAmount::usd_micros(84)));
    assert_eq!(sum.context_window_tokens, Some(8_192));
}

#[test]
fn usage_accumulation_preserves_optional_amounts_and_latest_context_window() {
    let with_amount = Usage {
        charged_amount: Some(ChargedAmount::usd_micros(17)),
        context_window_tokens: Some(4_096),
        ..Usage::default()
    };
    let without_amount = Usage {
        context_window_tokens: Some(8_192),
        ..Usage::default()
    };

    let sum = with_amount + without_amount;
    assert_eq!(sum.charged_amount, Some(ChargedAmount::usd_micros(17)));
    assert_eq!(sum.context_window_tokens, Some(8_192));

    let reverse = without_amount + with_amount;
    assert_eq!(reverse.charged_amount, Some(ChargedAmount::usd_micros(17)));
    assert_eq!(reverse.context_window_tokens, Some(4_096));
}

#[test]
fn usage_amount_accumulation_saturates_and_add_assign_preserves_context_window() {
    let mut usage = Usage {
        charged_amount: Some(ChargedAmount::usd_micros(i64::MAX - 1)),
        context_window_tokens: Some(2_048),
        ..Usage::default()
    };
    usage += Usage {
        charged_amount: Some(ChargedAmount::usd_micros(2)),
        context_window_tokens: None,
        ..Usage::default()
    };

    assert_eq!(
        usage.charged_amount,
        Some(ChargedAmount::usd_micros(i64::MAX))
    );
    assert_eq!(usage.context_window_tokens, Some(2_048));
}

#[test]
fn usage_serialization_preserves_optional_metadata_and_accepts_legacy_payloads() {
    let usage = Usage {
        input_tokens: 11,
        output_tokens: 13,
        charged_amount: Some(ChargedAmount::usd_micros(42)),
        context_window_tokens: Some(128_000),
        ..Usage::default()
    };
    let encoded = serde_json::to_value(usage).expect("usage serializes");
    assert_eq!(
        encoded["charged_amount"],
        serde_json::json!({ "micros": 42 })
    );
    assert_eq!(encoded["context_window_tokens"], serde_json::json!(128_000));
    assert_eq!(serde_json::from_value::<Usage>(encoded).unwrap(), usage);

    let absent = serde_json::to_value(Usage::default()).expect("default usage serializes");
    assert!(absent.get("charged_amount").is_none());
    assert!(absent.get("context_window_tokens").is_none());

    let legacy: Usage = serde_json::from_value(serde_json::json!({
        "input_tokens": 3,
        "output_tokens": 5,
        "total_tokens": 8
    }))
    .expect("legacy usage without new metadata deserializes");
    assert_eq!(legacy.charged_amount, None);
    assert_eq!(legacy.context_window_tokens, None);
}

#[test]
fn add_assign_works() {
    let mut total = Usage::default();
    total += Usage::new(3, 4);
    total += Usage::new(1, 1);
    assert_eq!(total.total_tokens, 9);
}

#[test]
fn effective_total_falls_back() {
    let usage = Usage {
        input_tokens: 4,
        output_tokens: 6,
        total_tokens: 0,
        ..Usage::default()
    };
    assert_eq!(usage.effective_total(), 10);
}

#[test]
fn add_assign_does_not_lose_totals_when_one_side_omits_total_tokens() {
    // A record whose provider omitted `total_tokens` (so it's a real `0`)
    // combined with a normal record must still sum both records' effective
    // totals, not just the raw `total_tokens` fields.
    let mut a = Usage {
        input_tokens: 100,
        output_tokens: 50,
        total_tokens: 0,
        ..Usage::default()
    };
    let b = Usage::new(10, 5);
    a += b;
    assert_eq!(a.effective_total(), 165);
    assert_eq!(a.total_tokens, 165);
}

#[test]
fn usage_totals_count_calls() {
    let mut totals = UsageTotals::new();
    totals.record(Usage::new(10, 10));
    totals += Usage::new(5, 5);
    assert_eq!(totals.calls, 2);
    assert_eq!(totals.usage.total_tokens, 30);
}
