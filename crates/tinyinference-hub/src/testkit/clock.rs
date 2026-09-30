//! [`FakeClock`]: time that moves only when told to.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::ports::Clock;

/// A manual clock. Clones share one timeline.
///
/// The monotonic side is a real [`Instant`] captured once as an anchor plus a
/// millisecond offset; the wall side starts at a fixed epoch value, so a
/// snapshot written by a test is reproducible.
#[derive(Clone)]
pub struct FakeClock {
    anchor: Instant,
    offset_ns: Arc<AtomicU64>,
    wall_start_ms: u64,
}

impl FakeClock {
    /// The wall-clock reading at construction: 2026-01-01T00:00:00Z.
    pub const START_WALL_MS: u64 = 1_767_225_600_000;

    /// A clock at [`FakeClock::START_WALL_MS`].
    pub fn new() -> Self {
        Self {
            anchor: Instant::now(),
            offset_ns: Arc::new(AtomicU64::new(0)),
            wall_start_ms: Self::START_WALL_MS,
        }
    }

    /// Moves time forward. There is no way to move it back.
    pub fn advance(&self, by: Duration) {
        // Nanoseconds, so sub-millisecond latencies accumulate instead of
        // vanishing.
        let ns = u64::try_from(by.as_nanos()).unwrap_or(u64::MAX);
        self.offset_ns.fetch_add(ns, Ordering::SeqCst);
    }

    /// How far the clock has moved since construction.
    pub fn elapsed(&self) -> Duration {
        Duration::from_nanos(self.offset_ns.load(Ordering::SeqCst))
    }
}

impl Default for FakeClock {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for FakeClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FakeClock")
            .field("elapsed", &self.elapsed())
            .finish()
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Instant {
        self.anchor + self.elapsed()
    }

    fn wall_ms(&self) -> u64 {
        self.wall_start_ms + self.offset_ns.load(Ordering::SeqCst) / 1_000_000
    }
}
