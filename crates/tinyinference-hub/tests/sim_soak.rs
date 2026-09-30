//! The long randomized soak: tens of thousands of seeds through the scenario
//! runner, in process (no sockets, no ports, no wall clock), every invariant
//! checked after every step.
//!
//! Ignored by default because it takes minutes; run it with
//!
//! ```text
//! cargo test -p tinyinference-hub --all-features --release --test sim_soak -- --ignored
//! ```
//!
//! It is split into shards so the test harness runs them on separate threads.
//! A failing seed is printed with its trace; add it (with `flaky` when it was the
//! flaky-store run) to `tests/golden/sim_seeds.txt` in the commit that fixes it,
//! and `sim_random_regressions` replays it forever.
#![cfg(feature = "testing")]

use tinyinference_hub::EndpointPolicy;
use tinyinference_hub::testkit::{FaultPlan, ScenarioRunner};

/// Where the soak's seeds start: far from the fresh seeds of `sim_random`.
const SOAK_BASE: u64 = 0x50AC_0000_0000;
const SEEDS_PER_SHARD: u64 = 2_500;
const STEPS: usize = 200;

async fn shard(index: u64) {
    let start = SOAK_BASE + index * SEEDS_PER_SHARD;
    for seed in start..start + SEEDS_PER_SHARD {
        for flaky in [false, true] {
            let faults = if flaky {
                FaultPlan::flaky(seed)
            } else {
                FaultPlan::none(seed)
            };
            let mut runner = ScenarioRunner::with(seed, EndpointPolicy::desktop(), faults);
            if let Err(failure) = runner.run_random(STEPS).await {
                panic!("SOAK FAILURE flaky={flaky}\n{failure}");
            }
        }
    }
}

macro_rules! shards {
    ($($name:ident = $index:expr),* $(,)?) => {$(
        #[tokio::test]
        #[ignore = "long soak: run with --ignored (see the module docs)"]
        async fn $name() {
            shard($index).await;
        }
    )*};
}

// 8 shards x 2,500 seeds x 2 modes (plain and flaky stores) = 40,000 runs of
// 200 steps.
shards!(
    soak_shard_0 = 0,
    soak_shard_1 = 1,
    soak_shard_2 = 2,
    soak_shard_3 = 3,
    soak_shard_4 = 4,
    soak_shard_5 = 5,
    soak_shard_6 = 6,
    soak_shard_7 = 7,
);
