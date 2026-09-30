//! In-memory implementations of the ports: the defaults for a one-shot CLI, an
//! embedded host that persists nothing, and every test.
//!
//! They behave like real stores where it matters: [`MemoryConfig`] keeps real
//! compare-and-swap versions and round-trips the document through JSON (so a
//! type that cannot be stored is caught here), and both stores can be told to
//! fail so the unreadable-is-not-absent and lost-CAS paths are testable.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use async_trait::async_trait;

use crate::config::HubConfig;
use crate::health::HealthSnapshot;
use crate::ids::{ScopeKey, Slug};
use crate::secret::Secret;

use super::interleave::Interleave;
pub use super::interleave::{Call, Held, Hold, Phase};
use super::{
    ConfigStore, CredentialStore, EnvSource, EventSink, HealthStore, HubEvent, PortError, Version,
};

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    // A panicking test thread must not poison every later assertion.
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// A document that cannot be stored is reported as an unavailable store: the
/// hub never writes what it cannot read back.
fn not_storable<E>(_: E) -> PortError {
    PortError::unavailable("the configuration is not storable")
}

/// Which credential-store operations an injected outage breaks.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialFault {
    /// `get` fails.
    Read,
    /// `set` and `delete` fail.
    Write,
    /// Everything fails.
    All,
}

/// A [`CredentialStore`] held in memory.
#[derive(Default)]
pub struct MemoryCredentials {
    slots: Mutex<HashMap<(ScopeKey, String), Secret>>,
    fault: Mutex<Option<CredentialFault>>,
    interleave: Interleave,
}

impl MemoryCredentials {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Injects an outage until [`MemoryCredentials::heal`].
    pub fn inject(&self, fault: CredentialFault) {
        *lock(&self.fault) = Some(fault);
    }

    /// Ends an injected outage.
    pub fn heal(&self) {
        *lock(&self.fault) = None;
    }

    /// Parks the matching call until the returned [`Held`] is released, so a
    /// test can run another operation in the middle of it.
    ///
    /// # Panics
    ///
    /// When `hold` names a call this store never makes (only get, set and delete
    /// have points), rather than letting the test hang.
    pub fn hold(&self, hold: Hold) -> Held {
        self.interleave
            .hold(hold, &[Call::Get, Call::Set, Call::Delete])
    }

    /// How many slots are stored, across every scope.
    pub fn len(&self) -> usize {
        lock(&self.slots).len()
    }

    /// Whether nothing is stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The slot names stored for a scope, sorted. Names only, never values.
    pub fn slots_in(&self, scope: &ScopeKey) -> Vec<String> {
        let mut names: Vec<String> = lock(&self.slots)
            .keys()
            .filter(|(s, _)| s == scope)
            .map(|(_, slot)| slot.clone())
            .collect();
        names.sort();
        names
    }

    fn check(&self, write: bool) -> Result<(), PortError> {
        match *lock(&self.fault) {
            Some(CredentialFault::All) => Err(PortError::unavailable("injected outage")),
            Some(CredentialFault::Read) if !write => Err(PortError::unavailable("injected outage")),
            Some(CredentialFault::Write) if write => Err(PortError::unavailable("injected outage")),
            _ => Ok(()),
        }
    }
}

impl fmt::Debug for MemoryCredentials {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryCredentials")
            .field("slots", &self.len())
            .finish()
    }
}

#[async_trait]
impl CredentialStore for MemoryCredentials {
    async fn get(&self, scope: &ScopeKey, slot: &str) -> Result<Option<Secret>, PortError> {
        self.interleave
            .point(Call::Get, Phase::Before, Some(slot))
            .await?;
        self.check(false)?;
        let found = lock(&self.slots)
            .get(&(scope.clone(), slot.to_string()))
            .cloned();
        self.interleave
            .point(Call::Get, Phase::After, Some(slot))
            .await?;
        Ok(found)
    }

    async fn set(&self, scope: &ScopeKey, slot: &str, value: Secret) -> Result<(), PortError> {
        self.interleave
            .point(Call::Set, Phase::Before, Some(slot))
            .await?;
        self.check(true)?;
        lock(&self.slots).insert((scope.clone(), slot.to_string()), value);
        self.interleave
            .point(Call::Set, Phase::After, Some(slot))
            .await?;
        Ok(())
    }

    async fn delete(&self, scope: &ScopeKey, slot: &str) -> Result<(), PortError> {
        self.interleave
            .point(Call::Delete, Phase::Before, Some(slot))
            .await?;
        self.check(true)?;
        lock(&self.slots).remove(&(scope.clone(), slot.to_string()));
        self.interleave
            .point(Call::Delete, Phase::After, Some(slot))
            .await?;
        Ok(())
    }
}

/// A [`ConfigStore`] held in memory, with real compare-and-swap versions.
///
/// The first save is version 1 and each later save adds one. The document is
/// stored as JSON, so what a load returns is what a real store would give back.
#[derive(Default)]
pub struct MemoryConfig {
    docs: Mutex<HashMap<ScopeKey, (String, u64)>>,
    conflicts: AtomicU32,
    unavailable: AtomicBool,
    interleave: Interleave,
}

impl MemoryConfig {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// The next `n` saves lose their compare-and-swap, as if another writer got
    /// there first. The stored document is not changed by a lost save.
    pub fn conflict_next(&self, n: u32) {
        self.conflicts.store(n, Ordering::SeqCst);
    }

    /// While `true`, every load and save fails as an outage.
    pub fn set_unavailable(&self, unavailable: bool) {
        self.unavailable.store(unavailable, Ordering::SeqCst);
    }

    /// Parks the matching `load` or `save` until the returned [`Held`] is
    /// released, so a test can run another operation in the middle of one.
    ///
    /// # Panics
    ///
    /// When `hold` names a call this store never makes (only load and save have
    /// points), rather than letting the test hang.
    pub fn hold(&self, hold: Hold) -> Held {
        self.interleave.hold(hold, &[Call::Load, Call::Save])
    }

    /// Replaces a scope's stored JSON directly, bumping the version, to stage a
    /// corrupt or foreign document. Not a hub operation.
    pub fn put_raw(&self, scope: &ScopeKey, json: impl Into<String>) {
        let mut docs = lock(&self.docs);
        let version = docs.get(scope).map_or(1, |(_, v)| v + 1);
        docs.insert(scope.clone(), (json.into(), version));
    }

    /// The stored JSON for a scope, to assert nothing secret was written.
    pub fn raw(&self, scope: &ScopeKey) -> Option<String> {
        lock(&self.docs).get(scope).map(|(json, _)| json.clone())
    }

    fn check(&self) -> Result<(), PortError> {
        if self.unavailable.load(Ordering::SeqCst) {
            Err(PortError::unavailable("injected outage"))
        } else {
            Ok(())
        }
    }
}

impl fmt::Debug for MemoryConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryConfig")
            .field("scopes", &lock(&self.docs).len())
            .finish()
    }
}

#[async_trait]
impl ConfigStore for MemoryConfig {
    async fn load(&self, scope: &ScopeKey) -> Result<Option<(HubConfig, Version)>, PortError> {
        self.interleave
            .point(Call::Load, Phase::Before, None)
            .await?;
        self.check()?;
        let Some((json, version)) = lock(&self.docs).get(scope).cloned() else {
            self.interleave
                .point(Call::Load, Phase::After, None)
                .await?;
            return Ok(None);
        };
        let config: HubConfig = serde_json::from_str(&json)
            .map_err(|_| PortError::unavailable("the stored configuration is not readable"))?;
        self.interleave
            .point(Call::Load, Phase::After, None)
            .await?;
        Ok(Some((config, Version::new(version))))
    }

    async fn save(
        &self,
        scope: &ScopeKey,
        config: &HubConfig,
        expect: Option<Version>,
    ) -> Result<Version, PortError> {
        self.interleave
            .point(Call::Save, Phase::Before, None)
            .await?;
        self.check()?;
        if self
            .conflicts
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            return Err(PortError::Conflict);
        }
        config.validate().map_err(not_storable)?;
        let json = serde_json::to_string(config).map_err(not_storable)?;
        let next = {
            let mut docs = lock(&self.docs);
            let current = docs.get(scope).map(|(_, v)| Version::new(*v));
            if current != expect {
                return Err(PortError::Conflict);
            }
            let next = current.map_or(1, |v| v.get() + 1);
            docs.insert(scope.clone(), (json, next));
            next
        };
        self.interleave
            .point(Call::Save, Phase::After, None)
            .await?;
        Ok(Version::new(next))
    }
}

/// A [`HealthStore`] held in memory: the default.
#[derive(Default)]
pub struct MemoryHealth {
    map: Mutex<HashMap<(ScopeKey, Slug), HealthSnapshot>>,
    unavailable: AtomicBool,
    interleave: Interleave,
}

impl MemoryHealth {
    /// An empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// While `true`, every call fails as an outage.
    pub fn set_unavailable(&self, unavailable: bool) {
        self.unavailable.store(unavailable, Ordering::SeqCst);
    }

    /// Parks (or fails) the matching `forget` call, the one every credential or
    /// endpoint change ends with. It runs after the change is committed; for
    /// `edit` and `add` that is after the provider's lock is released, so a test
    /// can act on the same provider in the gap, but `set_key`, `clear_key`,
    /// `remove` and an undone add still hold it (another operation on that
    /// provider would wait for the parked call). While it is parked the health
    /// tracker's own lock for the provider is held.
    ///
    /// # Panics
    ///
    /// When `hold` names a call this store never makes (only [`Call::Forget`]
    /// has a point), rather than letting the test hang.
    pub fn hold(&self, hold: Hold) -> Held {
        self.interleave.hold(hold, &[Call::Forget])
    }

    fn check(&self) -> Result<(), PortError> {
        if self.unavailable.load(Ordering::SeqCst) {
            Err(PortError::unavailable("injected outage"))
        } else {
            Ok(())
        }
    }
}

impl fmt::Debug for MemoryHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryHealth")
            .field("entries", &lock(&self.map).len())
            .finish()
    }
}

#[async_trait]
impl HealthStore for MemoryHealth {
    async fn get(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
    ) -> Result<Option<HealthSnapshot>, PortError> {
        self.check()?;
        Ok(lock(&self.map).get(&(scope.clone(), slug.clone())).cloned())
    }

    async fn put(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        snapshot: HealthSnapshot,
    ) -> Result<(), PortError> {
        self.check()?;
        lock(&self.map).insert((scope.clone(), slug.clone()), snapshot);
        Ok(())
    }

    async fn forget(&self, scope: &ScopeKey, slug: &Slug) -> Result<(), PortError> {
        self.interleave
            .point(Call::Forget, Phase::Before, None)
            .await?;
        self.check()?;
        lock(&self.map).remove(&(scope.clone(), slug.clone()));
        self.interleave
            .point(Call::Forget, Phase::After, None)
            .await?;
        Ok(())
    }
}

/// An [`EventSink`] that drops everything: the default.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoopEvents;

impl EventSink for NoopEvents {
    fn emit(&self, _event: HubEvent) {}
}

/// An [`EventSink`] that keeps what it was sent, for tests and for hosts that
/// poll.
#[derive(Default)]
pub struct MemoryEvents {
    events: Mutex<Vec<HubEvent>>,
}

impl MemoryEvents {
    /// An empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Everything received so far.
    pub fn events(&self) -> Vec<HubEvent> {
        lock(&self.events).clone()
    }

    /// Delivers an event as if the hub had emitted it (for tests of what
    /// consumes events).
    pub fn emit_test(&self, event: HubEvent) {
        lock(&self.events).push(event);
    }

    /// Takes everything received so far, leaving the sink empty.
    pub fn drain(&self) -> Vec<HubEvent> {
        std::mem::take(&mut *lock(&self.events))
    }
}

impl fmt::Debug for MemoryEvents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryEvents")
            .field("events", &lock(&self.events).len())
            .finish()
    }
}

impl EventSink for MemoryEvents {
    fn emit(&self, event: HubEvent) {
        lock(&self.events).push(event);
    }
}

/// An [`EnvSource`] over a fixed map, so no test reads the process environment.
/// `Debug` prints the variable names only.
#[derive(Clone, Default)]
pub struct MapEnv {
    vars: HashMap<String, String>,
}

impl MapEnv {
    /// An empty environment.
    pub fn new() -> Self {
        Self::default()
    }

    /// This environment plus one variable.
    #[must_use]
    pub fn with(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.vars.insert(name.into(), value.into());
        self
    }
}

impl fmt::Debug for MapEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut names: Vec<&String> = self.vars.keys().collect();
        names.sort();
        f.debug_struct("MapEnv").field("names", &names).finish()
    }
}

impl EnvSource for MapEnv {
    fn var(&self, name: &str) -> Option<String> {
        self.vars.get(name).cloned()
    }
}
