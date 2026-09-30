//! Deterministic interleaving for the in-memory stores.
//!
//! The in-memory ports complete every call without yielding, so two hub
//! operations run on one task never overlap and a race between them cannot be
//! staged. An [`Interleave`] fixes that: a test **holds** the `n`-th matching
//! call of a store at a chosen point (before it takes effect, or after it has
//! but before it returns), runs whatever it wants while the operation is
//! parked, then releases it. The schedule is written in the test, not found by
//! chance, so the race it reproduces reproduces every time.
//!
//! A hold can instead be a scripted **fault** ([`Hold::fail`]): the matching call
//! returns an outage error at that point (after taking effect, for
//! [`Hold::after`]: a store that commits and then times out), with no test
//! choreography needed.
//!
//! Several armed holds count the calls they match independently, in the order
//! they were armed; the first whose count is exhausted fires and is spent, and
//! the holds armed after it do not see that call. Two identical `skip(n)` holds
//! therefore fire on consecutive calls.
//!
//! ```
//! use tinyinference_hub::ports::CredentialStore;
//! use tinyinference_hub::ports::memory::{Call, Hold, MemoryCredentials};
//! use tinyinference_hub::{ScopeKey, Secret};
//!
//! # #[tokio::main(flavor = "current_thread")]
//! # async fn main() {
//! let store = MemoryCredentials::new();
//! let scope = ScopeKey::new("a");
//! let mut held = store.hold(Hold::before(Call::Set));
//! let write = store.set(&scope, "slot", Secret::new("v"));
//! let observe = async {
//!     held.reached().await;
//!     // The write is parked before it took effect.
//!     assert!(store.get(&scope, "slot").await.unwrap().is_none());
//!     held.release();
//! };
//! let (written, ()) = tokio::join!(write, observe);
//! written.unwrap();
//! assert!(store.get(&scope, "slot").await.unwrap().is_some());
//! # }
//! ```

use std::fmt;
use std::sync::Mutex;

use tokio::sync::oneshot;

use super::PortError;

/// A store call that a [`Hold`] can park or fail.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Call {
    /// `CredentialStore::get`.
    Get,
    /// `CredentialStore::set`.
    Set,
    /// `CredentialStore::delete`.
    Delete,
    /// `ConfigStore::load`.
    Load,
    /// `ConfigStore::save`.
    Save,
    /// `HealthStore::forget`.
    Forget,
}

/// Where in a call it is held.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Before the call reads or changes anything.
    Before,
    /// After the call has taken effect (and computed its answer), before it
    /// returns: the caller has not yet seen the result.
    After,
}

/// Which call to hold: the `skip`-th (from zero) call matching `call`, `phase`
/// and, when set, `slot`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hold {
    call: Call,
    phase: Phase,
    slot: Option<String>,
    skip: usize,
    fail: bool,
}

impl Hold {
    /// The call this hold names.
    pub(crate) fn call(&self) -> Call {
        self.call
    }

    /// Holds the next `call` before it takes effect.
    pub fn before(call: Call) -> Self {
        Self {
            call,
            phase: Phase::Before,
            slot: None,
            skip: 0,
            fail: false,
        }
    }

    /// Holds the next `call` after it took effect, before it returns.
    pub fn after(call: Call) -> Self {
        Self {
            phase: Phase::After,
            ..Self::before(call)
        }
    }

    /// Only calls on this credential slot count (ignored by config calls).
    #[must_use]
    pub fn slot(mut self, slot: impl Into<String>) -> Self {
        self.slot = Some(slot.into());
        self
    }

    /// Lets `n` matching calls through first.
    #[must_use]
    pub fn skip(mut self, n: usize) -> Self {
        self.skip = n;
        self
    }

    /// Makes the matching call fail with an outage at that point instead of
    /// parking it.
    #[must_use]
    pub fn fail(mut self) -> Self {
        self.fail = true;
        self
    }
}

/// A held call, seen from the test.
pub struct Held {
    reached: Option<oneshot::Receiver<()>>,
    release: Option<oneshot::Sender<()>>,
}

impl Held {
    /// Resolves once the operation under test is parked at the hold.
    ///
    /// **It waits forever if the operation never makes the call the hold names**
    /// (the store cannot tell "not yet" from "never"): when the schedule might not
    /// be reached, race this against the operation finishing, as this crate's own
    /// tests do with `tokio::select!`.
    ///
    /// # Panics
    ///
    /// When the store was dropped without the hold being reached.
    pub async fn reached(&mut self) {
        let reached = self.reached.take().expect("reached() was already awaited");
        reached
            .await
            .expect("the operation finished without reaching its hold");
    }

    /// Lets the parked call continue.
    pub fn release(&mut self) {
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        // A test that forgot to release must not hang the operation.
        self.release();
    }
}

impl fmt::Debug for Held {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Held").finish_non_exhaustive()
    }
}

struct Armed {
    hold: Hold,
    reached: Option<oneshot::Sender<()>>,
    release: Option<oneshot::Receiver<()>>,
}

/// The armed holds of one store (each memory store owns one).
#[derive(Default)]
pub(crate) struct Interleave {
    armed: Mutex<Vec<Armed>>,
}

impl Interleave {
    /// Arms `hold` and returns the test's end of it.
    pub(crate) fn hold(&self, hold: Hold, allowed: &[Call]) -> Held {
        // A hold on a call the store never makes would never fire: refuse it now
        // rather than let a test hang.
        assert!(
            allowed.contains(&hold.call()),
            "this store has no interleave point for {:?} (it has {allowed:?})",
            hold.call()
        );
        let (reached_tx, reached_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        self.armed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(Armed {
                hold,
                reached: Some(reached_tx),
                release: Some(release_rx),
            });
        Held {
            reached: Some(reached_rx),
            release: Some(release_tx),
        }
    }

    /// Called by a store at each point a call can be held. Parks the call when
    /// an armed hold matches and returns once the test releases it.
    ///
    /// # Errors
    ///
    /// An outage, when the matching hold is a scripted fault.
    pub(crate) async fn point(
        &self,
        call: Call,
        phase: Phase,
        slot: Option<&str>,
    ) -> Result<(), PortError> {
        let parked = {
            let mut armed = self
                .armed
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let position = armed.iter_mut().position(|a| {
                let h = &mut a.hold;
                let matches = h.call == call
                    && h.phase == phase
                    && match (h.slot.as_deref(), slot) {
                        (Some(wanted), Some(actual)) => wanted == actual,
                        // No filter, or a call that has no slot (a config call):
                        // the filter does not apply.
                        _ => true,
                    };
                if matches && h.skip > 0 {
                    h.skip -= 1;
                    return false;
                }
                matches
            });
            position.map(|i| armed.remove(i))
        };
        if let Some(mut hold) = parked {
            if let Some(reached) = hold.reached.take() {
                let _ = reached.send(());
            }
            if hold.hold.fail {
                return Err(PortError::unavailable("injected fault"));
            }
            if let Some(release) = hold.release.take() {
                let _ = release.await;
            }
        }
        Ok(())
    }
}

impl fmt::Debug for Interleave {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Interleave").finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "interleave_test.rs"]
mod tests;
