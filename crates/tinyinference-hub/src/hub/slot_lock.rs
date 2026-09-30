//! The per-provider lock that orders every key-slot operation of one hub.
//!
//! The configuration is compare-and-swapped, but the credential slot is a
//! second store with no version, and a record-and-key sequence spans both. Two
//! operations on one slug can therefore interleave between their two steps:
//! an add that lost its provider to a removal and a second add of the same slug
//! (the three-way race of finding 4.6) leave the first add writing, then
//! deleting, the key the second add owns. Holding one lock per `(scope, slug)`
//! across "check the record, then touch the slot" makes those two steps atomic
//! against every other operation of the same hub.
//!
//! This orders the operations of **one** hub value (and its clones). Two hubs
//! over one store are ordered only by the stores' own guarantees, as before.
//! The lock is never held across a probe or a catalog read. It is held across
//! the few store calls of a key change or an origin move; across one
//! credential-chain read for an origin move **without** a key (the guard that
//! refuses to send a chain credential to a new origin); and across one chain
//! read (after a record read) by a probe or listing of a provider whose endpoint
//! can be edited (`credential_checked`). A chain source that hangs blocks that
//! one provider's key operations and nothing else, and a source must not call back
//! into the hub from `resolve`: the lock is not reentrant.

use std::collections::HashMap;
use std::sync::{Arc, Weak};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::hub::Hub;
use crate::ids::{ScopeKey, Slug};

/// The lock table: weak, so an idle provider costs nothing.
#[derive(Default)]
pub(crate) struct SlotLocks {
    locks: HashMap<(ScopeKey, Slug), Weak<AsyncMutex<()>>>,
}

impl Hub {
    /// Waits for and takes the lock of `(scope, slug)`'s key slot. Drop the
    /// guard to release it. Never call another operation that takes it while
    /// holding it.
    pub(crate) async fn slot_lock(&self, scope: &ScopeKey, slug: &Slug) -> OwnedMutexGuard<()> {
        let lock = {
            let mut table = self
                .inner
                .slot_locks
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let key = (scope.clone(), slug.clone());
            match table.locks.get(&key).and_then(Weak::upgrade) {
                Some(lock) => lock,
                None => {
                    // Idle entries die with their last guard; sweep them so the
                    // table does not grow with every slug ever touched.
                    table.locks.retain(|_, weak| weak.strong_count() > 0);
                    let lock = Arc::new(AsyncMutex::new(()));
                    table.locks.insert(key, Arc::downgrade(&lock));
                    lock
                }
            }
        };
        lock.lock_owned().await
    }
}
