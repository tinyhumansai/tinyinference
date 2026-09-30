//! The seeded random scenario runner: fresh seeds every round, plus every seed
//! that ever failed (`tests/golden/sim_seeds.txt`). No sockets, no wall clock.
#![cfg(feature = "testing")]

use tinyinference_hub::EndpointPolicy;
use tinyinference_hub::testkit::{FaultPlan, ScenarioRunner};

const REGRESSIONS: &str = include_str!("golden/sim_seeds.txt");

/// The first fresh seed of the current round. Bumped by hand between rounds of
/// the test loop, never read from the environment.
const FRESH_BASE: u64 = 0x5EED_0000;
const FRESH_SEEDS: u64 = 64;
const STEPS: usize = 200;

async fn run(seed: u64, flaky: bool, policy: EndpointPolicy) {
    let faults = if flaky {
        FaultPlan::flaky(seed)
    } else {
        FaultPlan::none(seed)
    };
    let mut runner = ScenarioRunner::with(seed, policy, faults);
    if let Err(failure) = runner.run_random(STEPS).await {
        panic!("{failure}");
    }
}

#[tokio::test]
async fn sim_random_fresh_seeds_hold_every_invariant() {
    for n in 0..FRESH_SEEDS {
        run(FRESH_BASE + n, false, EndpointPolicy::desktop()).await;
    }
}

#[tokio::test]
async fn sim_random_fresh_seeds_with_flaky_stores_hold_every_invariant() {
    for n in 0..FRESH_SEEDS {
        run(FRESH_BASE + 0x1000 + n, true, EndpointPolicy::desktop()).await;
    }
}

#[tokio::test]
async fn sim_random_fresh_seeds_under_the_hosted_policy() {
    for n in 0..FRESH_SEEDS / 2 {
        run(FRESH_BASE + 0x2000 + n, false, EndpointPolicy::hosted()).await;
    }
}

#[tokio::test]
async fn sim_random_regressions() {
    let mut replayed = 0;
    for line in REGRESSIONS.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let seed: u64 = parts
            .next()
            .and_then(|s| {
                s.strip_prefix("0x")
                    .map_or_else(|| s.parse().ok(), |hex| u64::from_str_radix(hex, 16).ok())
            })
            .unwrap_or_else(|| panic!("bad seed line: {line}"));
        let flaky = parts.next() == Some("flaky");
        run(seed, flaky, EndpointPolicy::desktop()).await;
        replayed += 1;
    }
    eprintln!("replayed {replayed} regression seeds");
}

#[tokio::test]
async fn sim_random_a_run_is_reproducible_from_its_seed() {
    let trace_of = |seed| async move {
        let mut runner = ScenarioRunner::new(seed);
        runner.run_random(60).await.unwrap();
        runner.trace().to_vec()
    };
    assert_eq!(trace_of(42).await, trace_of(42).await);
    assert_ne!(trace_of(42).await, trace_of(43).await);
}
