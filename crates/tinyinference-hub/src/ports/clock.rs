//! [`Clock`]: the only source of time the hub reads.

use std::fmt::Debug;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// A monotonic and a wall clock.
///
/// The hub never calls `Instant::now()` itself. Cache freshness, health ages
/// and probe latency all read this port, which is what lets a simulation drive
/// an hour of cache expiry without sleeping.
pub trait Clock: Send + Sync + Debug {
    /// A monotonic instant, for measuring elapsed time.
    fn now(&self) -> Instant;
    /// Milliseconds since the Unix epoch, for values that are persisted.
    fn wall_ms(&self) -> u64;
}

/// The real clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn wall_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }
}
