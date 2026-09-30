//! [`HealthTracker`]: health fed by probes and by real turns.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::Mutex;

use crate::error::{HubError, PortName, ProviderFailure, ReasonCode};
use crate::ids::{ScopeKey, Slug};
use crate::ports::{Clock, EventSink, HealthStore, HubEvent};
use crate::probe::ProbeReport;

use super::types::{HealthSnapshot, ProviderHealth};

/// One async lock per `(scope, provider)`.
type LockMap = HashMap<(ScopeKey, Slug), Arc<Mutex<()>>>;

/// How many providers' forget marks are kept before the map is reset.
const MAX_EPOCHS: usize = 4096;

/// Remembers when each provider's health was last forgotten, so a result that
/// was measured **before** that moment (a probe or a turn that was in flight
/// when its key changed) is dropped instead of recorded against the new key.
#[derive(Default)]
struct Epochs {
    counter: u64,
    /// Marks older than this are gone: an operation that captured an epoch below
    /// it is treated as stale, which errs on the side of dropping a result.
    floor: u64,
    forgotten: HashMap<(ScopeKey, Slug), u64>,
}

/// How a real turn went, reported by the host after every turn so a provider
/// that passes probes but fails turns does not look green.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The turn succeeded.
    Ok {
        /// How long it took.
        latency: Duration,
    },
    /// The turn failed, already classified.
    Failed(ProviderFailure),
}

/// Reads and updates health snapshots and tells the host when a status changes.
///
/// Updates are serialised in-process **per provider** so two turns finishing
/// together cannot lose one another's signal, while a slow store round trip for
/// one provider never delays another's.
pub struct HealthTracker {
    store: Arc<dyn HealthStore>,
    clock: Arc<dyn Clock>,
    events: Arc<dyn EventSink>,
    locks: std::sync::Mutex<LockMap>,
    epochs: std::sync::Mutex<Epochs>,
}

impl HealthTracker {
    /// A tracker over `store`.
    pub fn new(
        store: Arc<dyn HealthStore>,
        clock: Arc<dyn Clock>,
        events: Arc<dyn EventSink>,
    ) -> Self {
        Self {
            store,
            clock,
            events,
            locks: std::sync::Mutex::new(HashMap::new()),
            epochs: std::sync::Mutex::new(Epochs::default()),
        }
    }

    /// The lock for one provider, created on first use.
    fn lock_for(&self, scope: &ScopeKey, slug: &Slug) -> Arc<Mutex<()>> {
        let mut locks = self
            .locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(locks.entry((scope.clone(), slug.clone())).or_default())
    }

    /// Takes the provider's lock as a lease that gives it back on drop. Without
    /// giving it back the map would keep one entry per provider ever seen;
    /// dropping the entry under a waiter would hand the next caller a fresh lock
    /// and let two updates run at once. Doing it in `Drop` means a cancelled
    /// future (a timed-out request) cannot skip it.
    pub(super) fn lease(&self, scope: &ScopeKey, slug: &Slug) -> LockLease<'_> {
        LockLease {
            tracker: self,
            key: (scope.clone(), slug.clone()),
            lock: self.lock_for(scope, slug),
        }
    }

    /// How many provider locks are currently kept (tests only).
    #[cfg(test)]
    pub(super) fn epochs_len(&self) -> usize {
        self.epochs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .forgotten
            .len()
    }

    #[cfg(test)]
    pub(super) fn kept_locks(&self) -> usize {
        self.locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    /// The snapshot for a provider (an empty one when nothing was recorded).
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store cannot be read.
    pub async fn snapshot(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
    ) -> Result<HealthSnapshot, HubError> {
        Ok(self
            .store
            .get(scope, slug)
            .await
            .map_err(|e| e.into_hub(PortName::Health))?
            .unwrap_or_default())
    }

    /// The folded status for a provider; `Unknown` when nothing was recorded.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store cannot be read.
    pub async fn health(&self, scope: &ScopeKey, slug: &Slug) -> Result<ProviderHealth, HubError> {
        Ok(self.snapshot(scope, slug).await?.health)
    }

    async fn update<F>(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        apply: F,
    ) -> Result<ProviderHealth, HubError>
    where
        F: FnOnce(&mut HealthSnapshot, u64) -> bool,
    {
        match self.update_at(scope, slug, None, apply).await? {
            Some(health) => Ok(health),
            None => self.health(scope, slug).await,
        }
    }

    /// Like [`update`](Self::update), but does nothing (and returns `None`) when
    /// the provider's health was forgotten since `expect` was read from
    /// [`epoch`](Self::epoch). The check runs under the provider's lock, so it
    /// cannot interleave with the forget.
    async fn update_at<F>(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        expect: Option<u64>,
        apply: F,
    ) -> Result<Option<ProviderHealth>, HubError>
    where
        F: FnOnce(&mut HealthSnapshot, u64) -> bool,
    {
        let lease = self.lease(scope, slug);
        let _serial = lease.lock.lock().await;
        if expect.is_some_and(|epoch| epoch != self.epoch(scope, slug)) {
            return Ok(None);
        }
        let mut snapshot = self.snapshot(scope, slug).await?;
        let from = snapshot.health;
        let changed = apply(&mut snapshot, self.clock.wall_ms());
        let to = snapshot.health;
        self.store
            .put(scope, slug, snapshot)
            .await
            .map_err(|e| e.into_hub(PortName::Health))?;
        if changed {
            self.events.emit(HubEvent::HealthChanged {
                scope: scope.clone(),
                slug: slug.clone(),
                from,
                to,
            });
        }
        Ok(Some(to))
    }

    /// The provider's current epoch: read it **before** measuring something that
    /// will be recorded later (a probe, a turn) and hand it to
    /// [`record_probe_at`](Self::record_probe_at) or
    /// [`record_outcome_at`](Self::record_outcome_at). A [`forget`](Self::forget)
    /// in between changes it, and the stale result is dropped.
    pub fn epoch(&self, scope: &ScopeKey, slug: &Slug) -> u64 {
        let epochs = self
            .epochs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        epochs
            .forgotten
            .get(&(scope.clone(), slug.clone()))
            .copied()
            .unwrap_or(0)
            .max(epochs.floor)
    }

    fn mark_forgotten(&self, scope: &ScopeKey, slug: &Slug) {
        let mut epochs = self
            .epochs
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        epochs.counter += 1;
        let mark = epochs.counter;
        if epochs.forgotten.len() >= MAX_EPOCHS {
            epochs.floor = mark;
            epochs.forgotten.clear();
        }
        epochs.forgotten.insert((scope.clone(), slug.clone()), mark);
    }

    /// Records a probe's result. Returns the new status.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store fails.
    pub async fn record_probe(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        report: &ProbeReport,
    ) -> Result<ProviderHealth, HubError> {
        match self.record_probe_at(scope, slug, report, None).await? {
            Some(health) => Ok(health),
            None => self.health(scope, slug).await,
        }
    }

    /// [`record_probe`](Self::record_probe) that is dropped (`None`) when the
    /// provider's health was forgotten after `epoch` was read.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store fails.
    pub async fn record_probe_at(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        report: &ProbeReport,
        epoch: Option<u64>,
    ) -> Result<Option<ProviderHealth>, HubError> {
        let failure = report.failure.as_ref().map(|f| (f.reason, f.status));
        let latency = u64::try_from(report.latency.as_millis()).ok();
        let depth = report.depth;
        let proves_key = report.proves_key;
        let started_ms = report.started_ms;
        self.update_at(scope, slug, epoch, move |snapshot, now| {
            snapshot.record_probe_started(depth, failure, latency, proves_key, started_ms, now)
        })
        .await
    }

    /// Records how a real turn went. A failure whose reason is `signed_out`
    /// marks the provider signed out. Returns the new status.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store fails.
    pub async fn record_outcome(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        outcome: &Outcome,
    ) -> Result<ProviderHealth, HubError> {
        match self.record_outcome_at(scope, slug, outcome, None).await? {
            Some(health) => Ok(health),
            None => self.health(scope, slug).await,
        }
    }

    /// [`record_outcome`](Self::record_outcome) that is dropped (`None`) when the
    /// provider's health was forgotten after `epoch` was read.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store fails.
    pub async fn record_outcome_at(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        outcome: &Outcome,
        epoch: Option<u64>,
    ) -> Result<Option<ProviderHealth>, HubError> {
        match outcome {
            Outcome::Ok { latency } => {
                let latency_ms = u64::try_from(latency.as_millis()).ok();
                self.update_at(scope, slug, epoch, move |snapshot, now| {
                    snapshot.record_turn(None, latency_ms, now)
                })
                .await
            }
            Outcome::Failed(failure) if failure.reason == ReasonCode::SignedOut => {
                self.update_at(scope, slug, epoch, |snapshot, now| {
                    snapshot.record_signed_out(now)
                })
                .await
            }
            Outcome::Failed(failure) => {
                let note = (failure.reason, failure.status);
                self.update_at(scope, slug, epoch, move |snapshot, now| {
                    snapshot.record_turn(Some(note), None, now)
                })
                .await
            }
        }
    }

    /// Marks a provider signed out (the managed credential chain is empty).
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store fails.
    pub async fn mark_signed_out(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
    ) -> Result<ProviderHealth, HubError> {
        self.update(scope, slug, |snapshot, now| snapshot.record_signed_out(now))
            .await
    }

    /// [`mark_signed_out`](Self::mark_signed_out) that is dropped (`None`) when
    /// the provider's health was forgotten after `epoch` was read.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store fails.
    pub async fn mark_signed_out_at(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        epoch: Option<u64>,
    ) -> Result<Option<ProviderHealth>, HubError> {
        self.update_at(scope, slug, epoch, |snapshot, now| {
            snapshot.record_signed_out(now)
        })
        .await
    }

    /// Forgets a provider's health: it was removed, or its key changed and what
    /// was learned with the old key no longer applies.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the health store fails.
    pub async fn forget(&self, scope: &ScopeKey, slug: &Slug) -> Result<(), HubError> {
        let lease = self.lease(scope, slug);
        let _serial = lease.lock.lock().await;
        // Marked even when the store cannot forget: whatever was in flight was
        // measured against the old credential either way.
        self.mark_forgotten(scope, slug);
        self.store
            .forget(scope, slug)
            .await
            .map_err(|e| e.into_hub(PortName::Health))
    }
}

/// A provider's lock, given back to the tracker's map when dropped (even if the
/// future holding it is cancelled) if nobody else holds or waits on it: the
/// map's copy and the lease's own are the two references when idle.
pub(super) struct LockLease<'a> {
    tracker: &'a HealthTracker,
    key: (ScopeKey, Slug),
    pub(super) lock: Arc<Mutex<()>>,
}

impl Drop for LockLease<'_> {
    fn drop(&mut self) {
        let mut locks = self
            .tracker
            .locks
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if Arc::strong_count(&self.lock) <= 2
            && locks
                .get(&self.key)
                .is_some_and(|held| Arc::ptr_eq(held, &self.lock))
        {
            locks.remove(&self.key);
        }
    }
}

impl fmt::Debug for HealthTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HealthTracker").finish_non_exhaustive()
    }
}
