//! Tests for the health fold and the tracker.

use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::error::{ProviderFailure, ReasonCode, Retry};
use crate::ids::{ScopeKey, Slug};
use crate::ports::memory::{MemoryEvents, MemoryHealth};
use crate::ports::{HubEvent, PortError};
use crate::taxonomy::TestDepth;
use crate::testkit::FakeClock;

fn fail(reason: ReasonCode) -> Option<(ReasonCode, Option<u16>)> {
    Some((reason, None))
}

fn slug() -> Slug {
    Slug::parse("openai").unwrap()
}

fn scope() -> ScopeKey {
    ScopeKey::new("company:acme")
}

// ---- the fold, as a table --------------------------------------------------

#[derive(Clone, Copy)]
enum Step {
    Probe(TestDepth, Option<ReasonCode>),
    Turn(Option<ReasonCode>),
}

fn run(steps: &[Step]) -> ProviderHealth {
    let mut snapshot = HealthSnapshot::default();
    for (n, step) in steps.iter().enumerate() {
        let now = 1_000 + n as u64;
        match *step {
            Step::Probe(depth, reason) => {
                snapshot.record_probe(
                    depth,
                    reason.map(|r| (r, None)),
                    Some(5),
                    depth != Catalog,
                    now,
                );
            }
            Step::Turn(reason) => {
                snapshot.record_turn(reason.map(|r| (r, None)), None, now);
            }
        }
    }
    snapshot.health
}

use Step::{Probe, Turn};
use TestDepth::{Catalog, Completion, KeyOnly};

#[test]
fn health_the_fold_table() {
    let ok = ProviderHealth::Ok;
    let cases: Vec<(&str, Vec<Step>, ProviderHealth)> = vec![
        ("nothing recorded", vec![], ProviderHealth::Unknown),
        ("a passing catalog", vec![Probe(Catalog, None)], ok),
        ("a passing turn", vec![Turn(None)], ok),
        // The turn lane is latest-wins: a turn that got an answer shows the
        // credential was accepted *now*, so it ends a Down that only turns caused;
        // a rate limit is an answer too. (A failing completion **probe** is the
        // deliberate finding a passive turn does not hide; see the rows above.)
        (
            "a turn that works after a rejected turn ends the rejection",
            vec![Turn(Some(ReasonCode::Auth)), Turn(None)],
            ok,
        ),
        (
            "a rate-limited turn after a rejected one shows the key was accepted",
            vec![
                Turn(Some(ReasonCode::Auth)),
                Turn(Some(ReasonCode::RateLimited)),
            ],
            ProviderHealth::Degraded(ReasonCode::RateLimited),
        ),
        (
            "a rejected key is down at once",
            vec![Probe(Catalog, Some(ReasonCode::Auth))],
            ProviderHealth::Down(ReasonCode::Auth),
        ),
        (
            "an exhausted account is down at once",
            vec![Turn(Some(ReasonCode::Quota))],
            ProviderHealth::Down(ReasonCode::Quota),
        ),
        (
            "auth is down even when another lane passes",
            vec![
                Probe(Catalog, None),
                Probe(Completion, Some(ReasonCode::Auth)),
            ],
            ProviderHealth::Down(ReasonCode::Auth),
        ),
        (
            "an unreachable endpoint is down",
            vec![Probe(Catalog, Some(ReasonCode::Endpoint))],
            ProviderHealth::Down(ReasonCode::Endpoint),
        ),
        (
            "one timeout is only degraded",
            vec![Probe(Catalog, Some(ReasonCode::Timeout))],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "an unknown model is degraded",
            vec![Turn(Some(ReasonCode::Model))],
            ProviderHealth::Degraded(ReasonCode::Model),
        ),
        (
            "a rate limit is degraded",
            vec![Turn(Some(ReasonCode::RateLimited))],
            ProviderHealth::Degraded(ReasonCode::RateLimited),
        ),
        (
            "the partial outage: catalog fails, completion works",
            vec![
                Probe(Completion, None),
                Probe(Catalog, Some(ReasonCode::Unknown)),
            ],
            ProviderHealth::Degraded(ReasonCode::Unknown),
        ),
        (
            "two lanes failing is down",
            vec![
                Probe(Catalog, Some(ReasonCode::Timeout)),
                Probe(Completion, Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Down(ReasonCode::Timeout),
        ),
        (
            "three failed turns in a row is down",
            vec![
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Down(ReasonCode::Timeout),
        ),
        (
            "two failed turns are still degraded",
            vec![
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "a success resets the run of failures",
            vec![
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
                Turn(None),
                Turn(Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "the worst failure names the reason",
            vec![
                Probe(Catalog, Some(ReasonCode::RateLimited)),
                Probe(Completion, Some(ReasonCode::Endpoint)),
            ],
            ProviderHealth::Down(ReasonCode::Endpoint),
        ),
        (
            "a later completion pass does not hide an earlier catalog failure",
            vec![
                Probe(Catalog, Some(ReasonCode::Timeout)),
                Probe(Completion, None),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "a later catalog failure is not superseded by an earlier completion pass",
            vec![
                Probe(Completion, None),
                Probe(Catalog, Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "a real turn that worked supersedes an older completion probe failure",
            vec![Probe(Completion, Some(ReasonCode::Model)), Turn(None)],
            ok,
        ),
        (
            "a real turn that worked does not hide a broken catalog",
            vec![Probe(Catalog, Some(ReasonCode::Endpoint)), Turn(None)],
            ProviderHealth::Degraded(ReasonCode::Endpoint),
        ),
        (
            "a rejected key on the catalog stays down even if a turn worked before it",
            vec![Turn(None), Probe(Catalog, Some(ReasonCode::Auth))],
            ProviderHealth::Down(ReasonCode::Auth),
        ),
        (
            "a run of failed turns is down even though a probe once passed",
            vec![
                Probe(Catalog, None),
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Down(ReasonCode::Timeout),
        ),
        (
            "two failed turns beside a passing probe are still degraded",
            vec![
                Probe(Catalog, None),
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "a rejected key is not hidden by a later successful turn",
            vec![Probe(Completion, Some(ReasonCode::Auth)), Turn(None)],
            ProviderHealth::Down(ReasonCode::Auth),
        ),
        (
            "a re-test that passes clears a turn's rejected key (the operator fixed it)",
            vec![Turn(Some(ReasonCode::Auth)), Probe(Completion, None)],
            ok,
        ),
        (
            "a re-test that passes clears an exhausted account",
            vec![
                Probe(Completion, Some(ReasonCode::Quota)),
                Probe(Completion, None),
            ],
            ok,
        ),
        (
            "a passing catalog does not clear an exhausted account",
            vec![Turn(Some(ReasonCode::Quota)), Probe(Catalog, None)],
            ProviderHealth::Down(ReasonCode::Quota),
        ),
        (
            "one blip after a recovery is degraded, not the fourth failure in a row",
            vec![
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
                Probe(Completion, None),
                Turn(Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "three rate-limited turns are throttling, not an outage",
            vec![
                Probe(Catalog, None),
                Turn(Some(ReasonCode::RateLimited)),
                Turn(Some(ReasonCode::RateLimited)),
                Turn(Some(ReasonCode::RateLimited)),
            ],
            ProviderHealth::Degraded(ReasonCode::RateLimited),
        ),
        (
            "three turns naming an unknown model are a model problem, not an outage",
            vec![
                Turn(Some(ReasonCode::Model)),
                Turn(Some(ReasonCode::Model)),
                Turn(Some(ReasonCode::Model)),
            ],
            ProviderHealth::Degraded(ReasonCode::Model),
        ),
        (
            "a cleared failure stays cleared when the clearing lane fails later",
            vec![
                Probe(Completion, Some(ReasonCode::Timeout)),
                Turn(None),
                Turn(Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "and the same the other way round",
            vec![
                Turn(Some(ReasonCode::Timeout)),
                Probe(Completion, None),
                Probe(Completion, Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "a passing key-only probe does not hide a chat lane's rejected key",
            vec![Turn(Some(ReasonCode::Auth)), Probe(KeyOnly, None)],
            ProviderHealth::Down(ReasonCode::Auth),
        ),
        (
            "nor does a passing catalog hide a completion probe's rejected key",
            vec![
                Probe(Completion, Some(ReasonCode::Auth)),
                Probe(Catalog, None),
            ],
            ProviderHealth::Down(ReasonCode::Auth),
        ),
        (
            "and a working completion clears an exhausted account on the shallow lanes",
            vec![
                Probe(KeyOnly, Some(ReasonCode::Quota)),
                Probe(Completion, None),
            ],
            ok,
        ),
        (
            "a working completion clears a rejected key on the catalog lane",
            vec![
                Probe(Catalog, Some(ReasonCode::Auth)),
                Probe(Completion, None),
            ],
            ok,
        ),
        (
            "but a working completion does not hide a catalog outage",
            vec![
                Probe(Catalog, Some(ReasonCode::Timeout)),
                Probe(Completion, None),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "rate-limited turns are not made Down by an unrelated timing-out catalog",
            vec![
                Probe(KeyOnly, None),
                Probe(Catalog, Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::RateLimited)),
                Turn(Some(ReasonCode::RateLimited)),
                Turn(Some(ReasonCode::RateLimited)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "rate-limited turns beside an older timed-out probe with nothing passing are throttling",
            vec![
                Probe(Catalog, Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::RateLimited)),
            ],
            ProviderHealth::Degraded(ReasonCode::Timeout),
        ),
        (
            "but the same shape with a real outage on both lanes is down",
            vec![
                Probe(Catalog, Some(ReasonCode::Timeout)),
                Turn(Some(ReasonCode::Timeout)),
            ],
            ProviderHealth::Down(ReasonCode::Timeout),
        ),
        (
            "but not an exhausted account",
            vec![Turn(Some(ReasonCode::Quota)), Probe(KeyOnly, None)],
            ProviderHealth::Down(ReasonCode::Quota),
        ),
        (
            "a failed turn is not superseded by an older completion pass",
            vec![Probe(Completion, None), Turn(Some(ReasonCode::Endpoint))],
            ProviderHealth::Degraded(ReasonCode::Endpoint),
        ),
        (
            "a passing catalog does not clear failing turns",
            vec![Turn(Some(ReasonCode::Endpoint)), Probe(Catalog, None)],
            ProviderHealth::Degraded(ReasonCode::Endpoint),
        ),
        (
            "a passing completion clears failing turns",
            vec![Turn(Some(ReasonCode::Endpoint)), Probe(Completion, None)],
            ok,
        ),
        (
            "key-only pass does not clear a failed completion",
            vec![
                Probe(Completion, Some(ReasonCode::Model)),
                Probe(KeyOnly, None),
            ],
            ProviderHealth::Degraded(ReasonCode::Model),
        ),
        (
            "an unclassified failure is degraded",
            vec![Turn(Some(ReasonCode::Unknown))],
            ProviderHealth::Degraded(ReasonCode::Unknown),
        ),
    ];
    for (name, steps, expected) in cases {
        assert_eq!(run(&steps), expected, "{name}");
    }
}

#[test]
fn health_a_status_change_reports_true_and_a_repeat_reports_false() {
    let mut snapshot = HealthSnapshot::default();
    assert!(
        snapshot.record_probe(Catalog, None, None, false, 10),
        "Unknown to Ok"
    );
    assert!(
        !snapshot.record_probe(Catalog, None, None, false, 20),
        "Ok to Ok"
    );
    assert_eq!(
        snapshot.changed_at_ms, 10,
        "the change time is when the status changed"
    );
    assert!(snapshot.record_turn(fail(ReasonCode::Auth), None, 30));
    assert_eq!(snapshot.changed_at_ms, 30);
    assert_eq!(snapshot.last_ok_ms, Some(20));
    let note = snapshot.last_failure.unwrap();
    assert_eq!((note.reason, note.at_ms), (ReasonCode::Auth, 30));
}

#[test]
fn health_signed_out_is_its_own_state_and_survives_until_something_new_is_heard() {
    let mut snapshot = HealthSnapshot::default();
    snapshot.record_probe(Catalog, None, None, false, 1);
    assert!(snapshot.record_signed_out(2));
    assert_eq!(snapshot.health, ProviderHealth::SignedOut);
    assert!(snapshot.probes.is_empty() && snapshot.turn.is_none());
    assert!(!snapshot.record_signed_out(3), "already signed out");
    snapshot.record_probe(Catalog, None, None, false, 4);
    assert_eq!(
        snapshot.health,
        ProviderHealth::Ok,
        "signing back in recovers"
    );
}

#[test]
fn health_usable_states() {
    for (health, usable) in [
        (ProviderHealth::Unknown, true),
        (ProviderHealth::Ok, true),
        (ProviderHealth::Degraded(ReasonCode::Timeout), true),
        (ProviderHealth::Down(ReasonCode::Auth), false),
        (ProviderHealth::SignedOut, false),
        (ProviderHealth::Disabled, false),
    ] {
        assert_eq!(health.is_usable(), usable, "{health:?}");
    }
}

#[test]
fn health_wire_forms_are_stable() {
    let json = |h: ProviderHealth| serde_json::to_string(&h).unwrap();
    assert_eq!(json(ProviderHealth::Ok), r#"{"state":"ok"}"#);
    assert_eq!(
        json(ProviderHealth::Down(ReasonCode::Auth)),
        r#"{"state":"down","reason":"auth"}"#
    );
    assert_eq!(json(ProviderHealth::SignedOut), r#"{"state":"signed_out"}"#);
    let mut snapshot = HealthSnapshot::default();
    snapshot.record_probe(Completion, fail(ReasonCode::Model), Some(9), true, 5);
    snapshot.record_turn(None, None, 6);
    let back: HealthSnapshot =
        serde_json::from_str(&serde_json::to_string(&snapshot).unwrap()).unwrap();
    assert_eq!(back, snapshot);
}

// ---- the tracker -----------------------------------------------------------

struct Bed {
    tracker: HealthTracker,
    store: Arc<MemoryHealth>,
    events: Arc<MemoryEvents>,
    clock: FakeClock,
}

fn bed() -> Bed {
    let store = Arc::new(MemoryHealth::new());
    let events = Arc::new(MemoryEvents::new());
    let clock = FakeClock::new();
    Bed {
        tracker: HealthTracker::new(store.clone(), Arc::new(clock.clone()), events.clone()),
        store,
        events,
        clock,
    }
}

fn failure(reason: ReasonCode) -> Outcome {
    Outcome::Failed(ProviderFailure::new(reason, Retry::Never).with_status(500))
}

#[tokio::test]
async fn health_a_provider_nobody_has_heard_from_is_unknown() {
    let bed = bed();
    assert_eq!(
        bed.tracker.health(&scope(), &slug()).await.unwrap(),
        ProviderHealth::Unknown
    );
    assert_eq!(
        bed.tracker.snapshot(&scope(), &slug()).await.unwrap(),
        HealthSnapshot::default()
    );
}

#[tokio::test]
async fn health_turns_move_the_status_and_each_change_emits_one_event() {
    let bed = bed();
    let (s, p) = (scope(), slug());
    let ok = Outcome::Ok {
        latency: Duration::from_millis(40),
    };
    assert_eq!(
        bed.tracker.record_outcome(&s, &p, &ok).await.unwrap(),
        ProviderHealth::Ok
    );
    assert_eq!(
        bed.tracker.record_outcome(&s, &p, &ok).await.unwrap(),
        ProviderHealth::Ok
    );
    let status = bed
        .tracker
        .record_outcome(&s, &p, &failure(ReasonCode::Auth))
        .await
        .unwrap();
    assert_eq!(status, ProviderHealth::Down(ReasonCode::Auth));
    let events = bed.events.events();
    assert_eq!(
        events.len(),
        2,
        "Unknown to Ok, Ok to Down; the repeat says nothing"
    );
    assert!(matches!(
        &events[1],
        HubEvent::HealthChanged {
            from: ProviderHealth::Ok,
            to: ProviderHealth::Down(ReasonCode::Auth),
            ..
        }
    ));
    let snapshot = bed.tracker.snapshot(&s, &p).await.unwrap();
    assert_eq!(snapshot.last_failure.unwrap().status, Some(500));
}

#[tokio::test]
async fn health_the_wall_clock_stamps_the_signals() {
    let bed = bed();
    let (s, p) = (scope(), slug());
    bed.tracker
        .record_outcome(
            &s,
            &p,
            &Outcome::Ok {
                latency: Duration::ZERO,
            },
        )
        .await
        .unwrap();
    bed.clock.advance(Duration::from_secs(120));
    bed.tracker
        .record_outcome(&s, &p, &failure(ReasonCode::Endpoint))
        .await
        .unwrap();
    let snapshot = bed.tracker.snapshot(&s, &p).await.unwrap();
    assert_eq!(snapshot.last_ok_ms, Some(FakeClock::START_WALL_MS));
    assert_eq!(snapshot.changed_at_ms, FakeClock::START_WALL_MS + 120_000);
}

#[tokio::test]
async fn health_a_signed_out_failure_is_the_signed_out_state_not_a_red_error() {
    let bed = bed();
    let (s, p) = (scope(), slug());
    let status = bed
        .tracker
        .record_outcome(&s, &p, &failure(ReasonCode::SignedOut))
        .await
        .unwrap();
    assert_eq!(status, ProviderHealth::SignedOut);
    assert_eq!(
        bed.tracker.mark_signed_out(&s, &p).await.unwrap(),
        ProviderHealth::SignedOut
    );
    assert_eq!(
        bed.events.events().len(),
        1,
        "no event for staying signed out"
    );
}

#[tokio::test]
async fn health_forgetting_a_provider_returns_it_to_unknown() {
    let bed = bed();
    let (s, p) = (scope(), slug());
    bed.tracker
        .record_outcome(&s, &p, &failure(ReasonCode::Auth))
        .await
        .unwrap();
    bed.tracker.forget(&s, &p).await.unwrap();
    assert_eq!(
        bed.tracker.health(&s, &p).await.unwrap(),
        ProviderHealth::Unknown
    );
}

#[tokio::test]
async fn health_scopes_and_providers_are_independent() {
    let bed = bed();
    bed.tracker
        .record_outcome(&scope(), &slug(), &failure(ReasonCode::Auth))
        .await
        .unwrap();
    assert_eq!(
        bed.tracker
            .health(&ScopeKey::new("other"), &slug())
            .await
            .unwrap(),
        ProviderHealth::Unknown
    );
    assert_eq!(
        bed.tracker
            .health(&scope(), &Slug::parse("groq").unwrap())
            .await
            .unwrap(),
        ProviderHealth::Unknown
    );
}

#[tokio::test]
async fn health_a_failing_store_is_a_typed_error_not_a_silent_unknown() {
    let bed = bed();
    bed.store.set_unavailable(true);
    let error = bed.tracker.health(&scope(), &slug()).await.unwrap_err();
    assert!(matches!(
        error,
        crate::HubError::StoreUnreadable {
            port: crate::error::PortName::Health,
            ..
        }
    ));
    assert!(
        bed.tracker
            .record_outcome(&scope(), &slug(), &failure(ReasonCode::Auth))
            .await
            .is_err()
    );
    assert!(bed.tracker.forget(&scope(), &slug()).await.is_err());
    assert!(
        bed.tracker
            .mark_signed_out(&scope(), &slug())
            .await
            .is_err()
    );
    assert!(
        bed.events.events().is_empty(),
        "nothing is announced that was not stored"
    );
    let _ = PortError::Conflict;
}

#[tokio::test]
async fn health_concurrent_outcomes_do_not_lose_each_other() {
    let bed = Arc::new(bed());
    let calls: Vec<_> = (0..30)
        .map(|_| {
            let bed = bed.clone();
            async move {
                bed.tracker
                    .record_outcome(&scope(), &slug(), &failure(ReasonCode::Timeout))
                    .await
                    .unwrap()
            }
        })
        .collect();
    futures::future::join_all(calls).await;
    let snapshot = bed.tracker.snapshot(&scope(), &slug()).await.unwrap();
    assert_eq!(snapshot.consecutive_failures, 30, "every failure counted");
    assert_eq!(snapshot.health, ProviderHealth::Down(ReasonCode::Timeout));
    assert!(format!("{:?}", bed.tracker).contains("HealthTracker"));
}

#[tokio::test]
async fn health_a_probe_report_feeds_the_tracker_including_its_failure_and_latency() {
    use crate::probe::ProbeReport;
    let bed = bed();
    let (s, p) = (scope(), slug());
    let mut report = ProbeReport {
        depth: TestDepth::Catalog,
        failure: None,
        refusal: None,
        latency: Duration::from_millis(120),
        started_ms: 0,
        models: Vec::new(),
        proves_key: true,
        notes: Vec::new(),
    };
    assert_eq!(
        bed.tracker.record_probe(&s, &p, &report).await.unwrap(),
        ProviderHealth::Ok
    );
    let snapshot = bed.tracker.snapshot(&s, &p).await.unwrap();
    assert_eq!(snapshot.probes[&TestDepth::Catalog].latency_ms, Some(120));
    report.failure = Some(ProviderFailure::new(ReasonCode::Auth, Retry::Never).with_status(401));
    assert_eq!(
        bed.tracker.record_probe(&s, &p, &report).await.unwrap(),
        ProviderHealth::Down(ReasonCode::Auth)
    );
    let snapshot = bed.tracker.snapshot(&s, &p).await.unwrap();
    assert_eq!(snapshot.last_failure.unwrap().status, Some(401));
    assert!(!snapshot.probes[&TestDepth::Catalog].ok);
    assert_eq!(bed.events.events().len(), 2);
}

#[test]
fn health_a_pass_never_clears_a_failure_recorded_while_the_probe_was_in_flight() {
    // The recorded limitation of run 2: a slow completion ping that started at
    // t=1000 and finished at t=2000 must not erase a turn that failed at t=1500,
    // which is newer than anything the probe saw.
    let mut snapshot = HealthSnapshot::default();
    snapshot.record_turn(Some((ReasonCode::Timeout, None)), None, 1_500);
    snapshot.record_probe_started(TestDepth::Completion, None, Some(5), true, 1_000, 2_000);
    assert!(
        !snapshot.turn.unwrap().superseded,
        "the newer failure is kept"
    );
    assert_eq!(
        snapshot.health,
        ProviderHealth::Degraded(ReasonCode::Timeout)
    );
    assert_eq!(
        snapshot.probes[&TestDepth::Completion].started_ms,
        Some(1_000)
    );
    // The same failure recorded before the probe started is cleared.
    let mut snapshot = HealthSnapshot::default();
    snapshot.record_turn(Some((ReasonCode::Timeout, None)), None, 900);
    snapshot.record_probe_started(TestDepth::Completion, None, Some(5), true, 1_000, 2_000);
    assert!(snapshot.turn.unwrap().superseded);
    assert_eq!(snapshot.health, ProviderHealth::Ok);
}

#[test]
fn health_a_shallower_pass_keeps_a_deeper_rejection_recorded_during_the_probe() {
    let mut snapshot = HealthSnapshot::default();
    snapshot.record_probe(
        TestDepth::KeyOnly,
        Some((ReasonCode::Auth, Some(401))),
        None,
        false,
        1_500,
    );
    // A catalog pass that proves the key, started before the failure landed.
    snapshot.record_probe_started(TestDepth::Catalog, None, Some(5), true, 1_000, 2_000);
    assert_eq!(snapshot.health, ProviderHealth::Down(ReasonCode::Auth));
    // Started after it: the rejection is cleared.
    snapshot.record_probe_started(TestDepth::Catalog, None, Some(5), true, 2_100, 2_200);
    assert_eq!(snapshot.health, ProviderHealth::Ok);
}

#[test]
fn health_a_snapshot_without_a_start_time_still_loads_and_compares() {
    let json =
        r#"{"probes":{"completion":{"ok":false,"reason":"auth","at_ms":10,"latency_ms":null}}}"#;
    let mut snapshot: HealthSnapshot = serde_json::from_str(json).unwrap();
    assert_eq!(snapshot.probes[&TestDepth::Completion].started_ms, None);
    // A record_probe (started == recorded) clears an older failure.
    snapshot.record_probe(TestDepth::Completion, None, None, true, 50);
    assert_eq!(snapshot.health, ProviderHealth::Ok);
}

#[test]
fn health_a_stored_failure_with_no_reason_is_degraded_not_ok() {
    // Data written by another build can carry a failed signal with no reason.
    let json = r#"{"health":{"state":"unknown"},"changed_at_ms":0,
        "probes":{"catalog":{"ok":false,"reason":null,"at_ms":1,"latency_ms":null}}}"#;
    let mut snapshot: HealthSnapshot = serde_json::from_str(json).unwrap();
    snapshot.record_probe(TestDepth::KeyOnly, None, None, true, 2);
    assert_eq!(
        snapshot.health,
        ProviderHealth::Degraded(ReasonCode::Unknown)
    );
}

mod health_props {
    use proptest::prelude::*;

    use super::*;

    fn step() -> impl Strategy<Value = (u8, Option<u8>)> {
        (0u8..4, prop::option::of(0u8..6))
    }

    fn reason(n: u8) -> ReasonCode {
        [
            ReasonCode::Auth,
            ReasonCode::Quota,
            ReasonCode::Endpoint,
            ReasonCode::Timeout,
            ReasonCode::Model,
            ReasonCode::RateLimited,
        ][usize::from(n) % 6]
    }

    fn apply(snapshot: &mut HealthSnapshot, lane: u8, outcome: Option<u8>, now: u64) {
        let failure = outcome.map(|r| (reason(r), None));
        match lane {
            0 => snapshot.record_probe(TestDepth::KeyOnly, failure, None, true, now),
            1 => snapshot.record_probe(TestDepth::Catalog, failure, None, false, now),
            2 => snapshot.record_probe(TestDepth::Completion, failure, None, true, now),
            _ => snapshot.record_turn(failure, None, now),
        };
    }

    proptest! {
        /// Whatever came before, one success in every lane is `Ok`, and an
        /// account-level failure heard last is `Down`.
        #[test]
        fn health_prop_all_lanes_passing_is_ok_and_a_terminal_failure_heard_last_is_down(
            steps in proptest::collection::vec(step(), 0..40),
            terminal in 0u8..2,
            lane in 0u8..4,
        ) {
            let mut snapshot = HealthSnapshot::default();
            let mut now = 1;
            for (l, outcome) in &steps {
                apply(&mut snapshot, *l, *outcome, now);
                now += 1;
            }
            let mut healed = snapshot.clone();
            for l in 0..4 {
                apply(&mut healed, l, None, now);
                now += 1;
            }
            prop_assert_eq!(healed.health, ProviderHealth::Ok);
            prop_assert_eq!(healed.consecutive_failures, 0);

            apply(&mut snapshot, lane, Some(terminal), now);
            // Auth outranks Quota, so an older unresolved rejection elsewhere wins
            // over a newer quota failure; either way it is Down for an account reason.
            match (terminal, snapshot.health) {
                (0, health) => prop_assert_eq!(health, ProviderHealth::Down(ReasonCode::Auth)),
                (_, ProviderHealth::Down(ReasonCode::Auth | ReasonCode::Quota)) => {}
                (_, other) => prop_assert!(false, "{other:?}"),
            }
            // The snapshot always survives a JSON round trip.
            let back: HealthSnapshot = serde_json::from_str(&serde_json::to_string(&snapshot).unwrap()).unwrap();
            prop_assert_eq!(back, snapshot);
        }

        /// `Unknown` only ever describes a provider nobody has heard from.
        #[test]
        fn health_prop_a_heard_provider_is_never_unknown(steps in proptest::collection::vec(step(), 1..30)) {
            let mut snapshot = HealthSnapshot::default();
            for (n, (l, outcome)) in steps.iter().enumerate() {
                apply(&mut snapshot, *l, *outcome, n as u64 + 1);
            }
            prop_assert_ne!(snapshot.health, ProviderHealth::Unknown);
        }
    }
}

#[tokio::test]
async fn health_a_slow_store_for_one_provider_does_not_hold_up_another() {
    use std::sync::atomic::{AtomicBool, Ordering};

    use async_trait::async_trait;

    use crate::ports::HealthStore;

    /// A store whose `put` for one slug never finishes until released.
    #[derive(Debug)]
    struct Gated {
        inner: MemoryHealth,
        slow: Slug,
        released: AtomicBool,
        gate: tokio::sync::Notify,
    }

    #[async_trait]
    impl HealthStore for Gated {
        async fn get(&self, s: &ScopeKey, p: &Slug) -> Result<Option<HealthSnapshot>, PortError> {
            self.inner.get(s, p).await
        }
        async fn put(&self, s: &ScopeKey, p: &Slug, h: HealthSnapshot) -> Result<(), PortError> {
            if *p == self.slow && !self.released.load(Ordering::SeqCst) {
                self.gate.notified().await;
            }
            self.inner.put(s, p, h).await
        }
        async fn forget(&self, s: &ScopeKey, p: &Slug) -> Result<(), PortError> {
            self.inner.forget(s, p).await
        }
    }

    let slow = slug();
    let fast = Slug::parse("groq").unwrap();
    let store = Arc::new(Gated {
        inner: MemoryHealth::new(),
        slow: slow.clone(),
        released: AtomicBool::new(false),
        gate: tokio::sync::Notify::new(),
    });
    let clock = FakeClock::new();
    let tracker = Arc::new(HealthTracker::new(
        store.clone(),
        Arc::new(clock),
        Arc::new(MemoryEvents::new()),
    ));
    let slow_call = {
        let (tracker, slow) = (tracker.clone(), slow.clone());
        tokio::spawn(async move {
            tracker
                .record_outcome(&scope(), &slow, &failure(ReasonCode::Timeout))
                .await
        })
    };
    tokio::task::yield_now().await;
    // The other provider completes while the first is stuck in its store.
    let status = tracker
        .record_outcome(
            &scope(),
            &fast,
            &Outcome::Ok {
                latency: Duration::ZERO,
            },
        )
        .await
        .unwrap();
    assert_eq!(status, ProviderHealth::Ok);
    store.released.store(true, Ordering::SeqCst);
    store.gate.notify_one();
    slow_call.await.unwrap().unwrap();
    tracker.forget(&scope(), &slow).await.unwrap();
    assert_eq!(
        tracker.health(&scope(), &slow).await.unwrap(),
        ProviderHealth::Unknown
    );
}

#[tokio::test]
async fn health_forgetting_never_drops_a_lock_somebody_still_holds() {
    let bed = bed();
    let (s, p) = (scope(), slug());
    bed.tracker
        .record_outcome(&s, &p, &failure(ReasonCode::Auth))
        .await
        .unwrap();
    // A caller queued behind the forget holds the provider's lock.
    let held = bed.tracker.lease(&s, &p);
    bed.tracker.forget(&s, &p).await.unwrap();
    assert!(
        Arc::ptr_eq(&held.lock, &bed.tracker.lease(&s, &p).lock),
        "the next caller must meet the same lock, or two updates could run at once"
    );
    drop(held);
    bed.tracker.forget(&s, &p).await.unwrap();
    // Idle again: the entry went.
    assert_eq!(bed.tracker.kept_locks(), 0);
}

#[test]
fn health_a_pass_hours_older_than_the_failure_does_not_keep_a_dead_endpoint_degraded() {
    let mut recent = HealthSnapshot::default();
    recent.record_probe(TestDepth::Catalog, None, None, false, 0);
    recent.record_probe(
        TestDepth::Completion,
        Some((ReasonCode::Endpoint, None)),
        None,
        true,
        5 * 60 * 1000,
    );
    // Endpoint is severe on its own only with nothing passing; a fresh pass keeps it partial.
    assert_eq!(
        recent.health,
        ProviderHealth::Degraded(ReasonCode::Endpoint)
    );

    let mut old = HealthSnapshot::default();
    old.record_probe(TestDepth::Catalog, None, None, false, 0);
    old.record_probe(
        TestDepth::Completion,
        Some((ReasonCode::Endpoint, None)),
        None,
        true,
        31 * 60 * 1000,
    );
    assert_eq!(
        old.health,
        ProviderHealth::Down(ReasonCode::Endpoint),
        "the pass predates the failure by over 30 minutes"
    );

    let mut timeout = HealthSnapshot::default();
    timeout.record_probe(TestDepth::Catalog, None, None, false, 0);
    timeout.record_probe(
        TestDepth::Completion,
        Some((ReasonCode::Timeout, None)),
        None,
        true,
        31 * 60 * 1000,
    );
    timeout.record_probe(
        TestDepth::KeyOnly,
        Some((ReasonCode::Timeout, None)),
        None,
        true,
        31 * 60 * 1000,
    );
    assert_eq!(
        timeout.health,
        ProviderHealth::Down(ReasonCode::Timeout),
        "two lanes failing, nothing fresh passing"
    );
}

#[test]
fn health_the_superseded_flag_survives_storage_and_is_absent_when_false() {
    let mut snapshot = HealthSnapshot::default();
    snapshot.record_turn(Some((ReasonCode::Timeout, None)), None, 1);
    snapshot.record_probe(TestDepth::Completion, None, None, true, 2);
    let text = serde_json::to_string(&snapshot).unwrap();
    assert!(text.contains("\"superseded\":true"), "{text}");
    let back: HealthSnapshot = serde_json::from_str(&text).unwrap();
    assert_eq!(back, snapshot);
    let plain = serde_json::to_string(&HealthSnapshot::default()).unwrap();
    assert!(!plain.contains("superseded"));
}

#[tokio::test]
async fn health_idle_locks_are_dropped_so_the_map_does_not_grow_with_every_provider_ever_seen() {
    let bed = bed();
    for n in 0..50 {
        let p = Slug::parse(&format!("p{n}")).unwrap();
        bed.tracker
            .record_outcome(&scope(), &p, &failure(ReasonCode::Timeout))
            .await
            .unwrap();
        // Idle after the update: nothing is kept.
        assert_eq!(bed.tracker.kept_locks(), 0, "after {n} providers");
    }
}

#[test]
fn health_a_key_proving_pass_clears_rejected_keys_only_in_the_lanes_shallower_than_itself() {
    let mut snapshot = HealthSnapshot::default();
    snapshot.record_turn(Some((ReasonCode::Auth, None)), None, 1);
    snapshot.record_probe(
        TestDepth::KeyOnly,
        Some((ReasonCode::Auth, None)),
        None,
        false,
        2,
    );
    // A catalog pass on a public listing proves nothing about the key.
    snapshot.record_probe(TestDepth::Catalog, None, None, false, 3);
    assert_eq!(snapshot.health, ProviderHealth::Down(ReasonCode::Auth));
    // One that does clears the key-only lane's rejection, but the chat lane's
    // (deeper) rejection stands: a key that lists can still be refused a chat.
    snapshot.record_probe(TestDepth::Catalog, None, None, true, 4);
    assert!(snapshot.probes[&TestDepth::KeyOnly].superseded);
    assert!(!snapshot.turn.unwrap().superseded);
    assert_eq!(snapshot.health, ProviderHealth::Down(ReasonCode::Auth));
    // Only a completion, the deepest check, clears it, and ends the run of failures.
    snapshot.record_probe(TestDepth::Completion, None, None, true, 5);
    assert_eq!(snapshot.health, ProviderHealth::Ok);
    assert_eq!(snapshot.consecutive_failures, 0);
}

#[tokio::test]
async fn health_a_turns_latency_is_kept_in_the_snapshot() {
    let bed = bed();
    let (s, p) = (scope(), slug());
    bed.tracker
        .record_outcome(
            &s,
            &p,
            &Outcome::Ok {
                latency: Duration::from_millis(1234),
            },
        )
        .await
        .unwrap();
    let snapshot = bed.tracker.snapshot(&s, &p).await.unwrap();
    assert_eq!(snapshot.turn.unwrap().latency_ms, Some(1234));
    let text = serde_json::to_string(&snapshot).unwrap();
    assert!(text.contains("\"latency_ms\":1234"), "{text}");
    // A failed turn has none, and none is not written.
    bed.tracker
        .record_outcome(&s, &p, &failure(ReasonCode::Timeout))
        .await
        .unwrap();
    let after = serde_json::to_string(&bed.tracker.snapshot(&s, &p).await.unwrap().turn).unwrap();
    assert!(!after.contains("latency_ms"), "{after}");
}

#[tokio::test]
async fn health_a_cancelled_waiter_does_not_leak_the_providers_lock() {
    use std::sync::atomic::{AtomicBool, Ordering};

    use async_trait::async_trait;

    use crate::ports::HealthStore;

    /// A store whose first `put` waits for a release.
    #[derive(Debug)]
    struct Gated {
        inner: MemoryHealth,
        released: AtomicBool,
        gate: tokio::sync::Notify,
    }

    #[async_trait]
    impl HealthStore for Gated {
        async fn get(&self, s: &ScopeKey, p: &Slug) -> Result<Option<HealthSnapshot>, PortError> {
            self.inner.get(s, p).await
        }
        async fn put(&self, s: &ScopeKey, p: &Slug, h: HealthSnapshot) -> Result<(), PortError> {
            if !self.released.load(Ordering::SeqCst) {
                self.gate.notified().await;
            }
            self.inner.put(s, p, h).await
        }
        async fn forget(&self, s: &ScopeKey, p: &Slug) -> Result<(), PortError> {
            self.inner.forget(s, p).await
        }
    }

    let store = Arc::new(Gated {
        inner: MemoryHealth::new(),
        released: AtomicBool::new(false),
        gate: tokio::sync::Notify::new(),
    });
    let tracker = Arc::new(HealthTracker::new(
        store.clone(),
        Arc::new(FakeClock::new()),
        Arc::new(MemoryEvents::new()),
    ));
    let spawn = |tracker: Arc<HealthTracker>| {
        tokio::spawn(async move {
            tracker
                .record_outcome(&scope(), &slug(), &failure(ReasonCode::Timeout))
                .await
        })
    };
    let first = spawn(tracker.clone());
    tokio::task::yield_now().await;
    // A second update queues on the provider's lock behind the first...
    let second = spawn(tracker.clone());
    tokio::task::yield_now().await;
    // ...and is cancelled (a timed-out request) while it waits.
    second.abort();
    let _ = second.await;
    store.released.store(true, Ordering::SeqCst);
    store.gate.notify_one();
    first.await.unwrap().unwrap();
    assert_eq!(
        tracker.kept_locks(),
        0,
        "the cancelled waiter's lease still gave the lock back"
    );
}

#[tokio::test]
async fn health_a_result_measured_before_a_forget_is_dropped_and_one_measured_after_is_kept() {
    use crate::probe::ProbeReport;
    let bed = bed();
    let (s, p) = (scope(), slug());
    let report = |failure: Option<ProviderFailure>| ProbeReport {
        depth: TestDepth::Catalog,
        failure,
        refusal: None,
        latency: Duration::from_millis(5),
        started_ms: 0,
        models: Vec::new(),
        proves_key: true,
        notes: Vec::new(),
    };
    let stale = bed.tracker.epoch(&s, &p);
    bed.tracker.forget(&s, &p).await.unwrap();
    assert_ne!(
        bed.tracker.epoch(&s, &p),
        stale,
        "a forget changes the epoch"
    );
    let bad = report(Some(ProviderFailure::new(ReasonCode::Auth, Retry::Never)));
    assert_eq!(
        bed.tracker
            .record_probe_at(&s, &p, &bad, Some(stale))
            .await
            .unwrap(),
        None
    );
    let dropped = bed
        .tracker
        .record_outcome_at(&s, &p, &failure(ReasonCode::Auth), Some(stale))
        .await
        .unwrap();
    assert_eq!(dropped, None);
    assert_eq!(
        bed.tracker
            .mark_signed_out_at(&s, &p, Some(stale))
            .await
            .unwrap(),
        None
    );
    assert_eq!(
        bed.tracker.health(&s, &p).await.unwrap(),
        ProviderHealth::Unknown,
        "nothing was recorded"
    );
    // Measured after: recorded. And no epoch at all means "unconditional".
    let fresh = bed.tracker.epoch(&s, &p);
    assert_eq!(
        bed.tracker
            .record_probe_at(&s, &p, &bad, Some(fresh))
            .await
            .unwrap(),
        Some(ProviderHealth::Down(ReasonCode::Auth))
    );
    assert!(
        bed.tracker
            .record_outcome_at(
                &s,
                &p,
                &Outcome::Ok {
                    latency: Duration::ZERO
                },
                None
            )
            .await
            .unwrap()
            .is_some()
    );
    // Another provider's epoch is its own.
    let other = Slug::parse("groq").unwrap();
    assert_eq!(bed.tracker.epoch(&s, &other), 0);
}

#[tokio::test]
async fn health_the_forget_marks_are_bounded_and_a_reset_errs_on_dropping() {
    let bed = bed();
    let s = scope();
    let early = bed.tracker.epoch(&s, &Slug::parse("p0").unwrap());
    for n in 0..4200 {
        bed.tracker
            .forget(&s, &Slug::parse(&format!("p{n}")).unwrap())
            .await
            .unwrap();
    }
    let epochs = bed.tracker.epochs_len();
    assert!(epochs <= 4096, "{epochs}");
    // Something captured before the reset is now older than the floor: stale.
    assert!(bed.tracker.epoch(&s, &Slug::parse("p0").unwrap()) > early);
    let untouched = Slug::parse("never-forgotten").unwrap();
    assert!(
        bed.tracker.epoch(&s, &untouched) > 0,
        "the floor applies to every provider after a reset"
    );
}
