//! Tests for the paged envelope, ported from OpenCompany's
//! `paged_catalog_tests.rs` and extended.

use serde_json::json;

use super::*;
use crate::error::{ReasonCode, Retry};

fn page_body(ids: &[&str], total: Option<u64>) -> String {
    let rows: Vec<_> = ids.iter().map(|i| json!({"id": i})).collect();
    let mut data = json!({"object": "list", "data": rows});
    if let Some(total) = total {
        data["total"] = json!(total);
    }
    json!({"success": true, "data": data}).to_string()
}

fn page(ids: &[&str], total: Option<u64>) -> Page {
    parse_page(&page_body(ids, total)).unwrap()
}

#[test]
fn paged_the_page_path_carries_the_limit_and_offset() {
    assert_eq!(page_path(0), "/models?limit=500&offset=0");
    assert_eq!(page_path(1500), "/models?limit=500&offset=1500");
}

#[test]
fn paged_a_page_parses_ids_names_windows_and_prices() {
    let body = json!({"success": true, "data": {"object": "list", "total": 2, "data": [
        {"id": "anthropic/claude-x", "display_name": "Claude X", "context_length": 200000,
         "pricing": {"inputPer1M": 3.0, "outputPer1M": 15.0}, "owned_by": "anthropic"},
        {"id": "meta/llama", "name": "Llama", "context_window": 8192}
    ]}})
    .to_string();
    let page = parse_page(&body).unwrap();
    assert_eq!((page.raw_len, page.total), (2, Some(2)));
    let first = &page.entries[0];
    assert_eq!(first.display_name.as_deref(), Some("Claude X"));
    assert_eq!(first.capabilities.context_window.value, Some(200_000));
    assert_eq!(
        (first.input_per_1m, first.output_per_1m),
        (Some(3.0), Some(15.0))
    );
    assert_eq!(first.owned_by.as_deref(), Some("anthropic"));
    assert_eq!(
        page.entries[1].display_name.as_deref(),
        Some("Llama"),
        "name is the fallback"
    );
    assert_eq!(
        page.entries[1].capabilities.context_window.value,
        Some(8192)
    );
}

#[test]
fn paged_every_id_is_kept_exactly_as_given() {
    let page = page(
        &["openai/gpt-5", "x-ai/grok-9", "some-vendor/tier-fast:free"],
        None,
    );
    let ids: Vec<&str> = page.entries.iter().map(|e| e.id.as_str()).collect();
    assert_eq!(
        ids,
        ["openai/gpt-5", "x-ai/grok-9", "some-vendor/tier-fast:free"]
    );
}

#[test]
fn paged_a_malformed_row_is_dropped_but_still_advances_the_offset() {
    let body = json!({"success": true, "data": {"total": 3, "data": [
        {"id": "a"}, {"id": 5}, {"nope": true}
    ]}})
    .to_string();
    let page = parse_page(&body).unwrap();
    assert_eq!(page.entries.len(), 1);
    assert_eq!(page.raw_len, 3);
}

#[test]
fn paged_a_body_that_is_not_the_envelope_is_unknown_never_auth() {
    let bodies = [
        "not json".to_string(),
        json!({"data": [{"id": "a"}]}).to_string(),
        json!({"success": true, "data": []}).to_string(),
        json!({"success": true, "data": {"object": "list"}}).to_string(),
        json!({"success": true}).to_string(),
        json!({"success": false, "error": "boom"}).to_string(),
        json!({"success": false}).to_string(),
        json!({"success": false, "message": "nope"}).to_string(),
    ];
    for body in bodies {
        let failure = parse_page(&body).unwrap_err();
        assert_eq!(failure.reason, ReasonCode::Unknown, "{body}");
        assert_eq!(failure.retry, Retry::Never);
    }
}

#[test]
fn paged_the_collector_walks_to_total_and_stops() {
    let mut collector = Collector::default();
    assert_eq!(collector.offset(), 0);
    assert_eq!(collector.push(page(&["a", "b"], Some(5))), NextPage::At(2));
    assert_eq!(collector.offset(), 2);
    assert_eq!(collector.push(page(&["c", "d"], Some(5))), NextPage::At(4));
    assert_eq!(collector.push(page(&["e"], Some(5))), NextPage::Done);
    let ids: Vec<_> = collector
        .finish()
        .into_iter()
        .map(|e| e.id.to_string())
        .collect();
    assert_eq!(ids, ["a", "b", "c", "d", "e"]);
}

#[test]
fn paged_only_an_empty_page_ends_a_read_with_no_total() {
    assert_eq!(
        Collector::default().push(page(&["a"], None)),
        NextPage::At(1),
        "no total: ask again"
    );
    assert_eq!(Collector::default().push(page(&[], None)), NextPage::Done);
    let mut collector = Collector::default();
    assert_eq!(collector.push(page(&[], Some(100))), NextPage::Done);
}

#[test]
fn paged_duplicates_across_pages_are_dropped() {
    let mut collector = Collector::default();
    collector.push(page(&["a", "b"], Some(4)));
    collector.push(page(&["b", "c"], Some(4)));
    let ids: Vec<_> = collector
        .finish()
        .into_iter()
        .map(|e| e.id.to_string())
        .collect();
    assert_eq!(ids, ["a", "b", "c"]);
}

#[test]
fn paged_a_total_that_is_never_reached_stops_at_the_page_cap() {
    let mut collector = Collector::default();
    let mut last = NextPage::Done;
    for n in 0..MAX_PAGES {
        let id = format!("m{n}");
        last = collector.push(page(&[id.as_str()], Some(1_000_000)));
        if n + 1 < MAX_PAGES {
            assert!(matches!(last, NextPage::At(_)), "page {n}");
        }
    }
    assert_eq!(
        last,
        NextPage::Truncated {
            read: MAX_PAGES,
            total: 1_000_000
        }
    );
    assert_eq!(collector.finish().len(), MAX_PAGES);
}

#[test]
fn paged_reaching_total_exactly_at_the_cap_is_done_not_truncated() {
    let mut collector = Collector::default();
    let mut last = NextPage::Done;
    for n in 0..MAX_PAGES {
        let id = format!("m{n}");
        last = collector.push(page(&[id.as_str()], Some(MAX_PAGES as u64)));
    }
    assert_eq!(last, NextPage::Done);
}

#[test]
fn paged_a_total_that_is_not_a_number_is_no_total() {
    let body =
        json!({"success": true, "data": {"total": "many", "data": [{"id": "a"}]}}).to_string();
    assert_eq!(parse_page(&body).unwrap().total, None);
    let negative =
        json!({"success": true, "data": {"total": -3, "data": [{"id": "a"}]}}).to_string();
    assert_eq!(parse_page(&negative).unwrap().total, None);
}

#[test]
fn paged_no_total_keeps_reading_whatever_the_page_size_until_an_empty_page() {
    // The server clamped `limit` to 100 and never sends a total.
    let mut collector = Collector::default();
    for n in 0..3 {
        let ids: Vec<String> = (0..100).map(|i| format!("m{n}-{i}")).collect();
        let refs: Vec<&str> = ids.iter().map(String::as_str).collect();
        assert_eq!(
            collector.push(page(&refs, None)),
            NextPage::At((n + 1) * 100)
        );
    }
    assert_eq!(collector.push(page(&[], None)), NextPage::Done);
    assert_eq!(collector.finish().len(), 300);
}

#[test]
fn paged_no_total_and_pages_that_never_end_stop_at_the_page_cap_and_say_so() {
    let mut collector = Collector::default();
    let mut last = NextPage::Done;
    for n in 0..MAX_PAGES {
        let id = format!("m{n}");
        last = collector.push(page(&[id.as_str()], None));
    }
    assert_eq!(
        last,
        NextPage::Truncated {
            read: MAX_PAGES,
            total: MAX_PAGES
        }
    );
}

#[test]
fn paged_a_page_of_unusable_rows_still_moves_the_offset_and_is_judged_over_the_whole_read() {
    let bad = json!({"success": true, "data": {"total": 4, "data": [{"nope": 1}, {"id": 7}]}})
        .to_string();
    let page_bad = parse_page(&bad).unwrap();
    assert!(page_bad.entries.is_empty() && page_bad.raw_len == 2);
    // Bad page first, good page after: the good rows are kept.
    let mut collector = Collector::default();
    assert_eq!(collector.push(page_bad.clone()), NextPage::At(2));
    assert!(collector.read_only_unusable_rows(), "so far nothing usable");
    collector.push(page(&["good"], Some(4)));
    assert!(!collector.read_only_unusable_rows());
    assert_eq!(collector.finish().len(), 1);
    // Nothing usable anywhere is what is reported.
    let mut all_bad = Collector::default();
    all_bad.push(page_bad);
    assert!(all_bad.read_only_unusable_rows());
    // A genuinely empty read is not "unusable rows".
    let mut empty = Collector::default();
    empty.push(page(&[], Some(0)));
    assert!(!empty.read_only_unusable_rows());
}
