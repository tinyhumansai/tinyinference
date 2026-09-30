//! The model-list cache, ported from OpenCompany's `inference_models.rs` with
//! the semantics it earned in production:
//!
//! * **keyed on the endpoint, never on the credential.** A credential must not
//!   become a map key: hashing one to key a cache puts a derivative of it in
//!   process memory next to the data it guards, and would split one endpoint's
//!   single flight per key;
//!   (The key is the endpoint, the shape, and, when a credential was sent, the
//!   scope and the provider.) **An entry does remember which credential it was
//!   read with**, as an opaque per-process [`SecretId`] (a randomly keyed 64-bit
//!   hash, never printed or persisted; see its docs), so a listing read with a
//!   key that has since been replaced is never served to the new key, not even
//!   in the moment between the replacement and the eviction (finding 3.7). It
//!   is entry metadata checked at hit time, not a map key, so single flight is
//!   unchanged. A source that rotates its token by itself (the managed platform
//!   token) passes no identity: the account, not the token, owns the list;
//! * **partitioned by scope and provider whenever a credential was sent.** An
//!   endpoint may publish an entitlement-scoped listing, so a base-URL-only key
//!   would hand one tenant's list to the next, and two providers of one tenant
//!   on one endpoint (two accounts) must not share one either. A keyless read
//!   is a public property of the endpoint and stays shared;
//! * **a rejected credential (`401`, or reason `auth`) and a `403` are never
//!   remembered.** They are facts about the key presented, not about the
//!   endpoint; memoising one would make a second tenant read the first's
//!   rejection and make a tenant that just rotated a bad key wait out the memo.
//!   Callers already queued behind the request that was rejected share that one
//!   answer; nobody who arrives later does. A rejection also drops the list read
//!   with that key ("a bad key must show"); a bare `403` (a WAF, a geo block)
//!   does not, and the older list is served stale;
//! * success is fresh for an hour, a failure is remembered for a minute (so an
//!   unreachable provider costs one attempt a minute, not one per request);
//! * **single flight**: callers for one endpoint queue on its lock and re-check
//!   after acquiring it, so a hundred concurrent callers make one request;
//! * **stale on error**: when a refresh fails and an older list exists, the
//!   older list is served with a typed warning instead of an error, unless the
//!   failure was a rejected credential. Neither an empty list nor one older than
//!   [`STALE_RETENTION`] is served stale.
//!
//! Time comes from the [`Clock`] port, so an hour of expiry is a
//! `FakeClock::advance`, not a sleep.

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use tokio::sync::Mutex as AsyncMutex;

use crate::endpoint::redact_endpoint;
use crate::error::{HubError, ProviderFailure};
use crate::ids::{ScopeKey, Slug};
use crate::ports::Clock;
use crate::secret::SecretId;
use crate::taxonomy::CatalogShape;

use super::types::{Freshness, ModelEntry, ModelList};

/// How long a successful listing stays fresh.
pub const CATALOG_TTL: Duration = Duration::from_secs(60 * 60);

/// How long a *failed* read is remembered.
pub const FAILURE_TTL: Duration = Duration::from_secs(60);

/// How long an **empty** listing stays fresh. Short, because an empty listing
/// is usually a local runtime with nothing pulled yet, and the operator is
/// about to fix that.
pub const EMPTY_CATALOG_TTL: Duration = Duration::from_secs(60);

/// How long an expired listing is kept to serve as the stale fallback.
pub const STALE_RETENTION: Duration = Duration::from_secs(24 * 60 * 60);

/// The most endpoint slots kept; beyond this the least recently used go.
pub const MAX_SLOTS: usize = 1024;

/// A boxed fetch: what a caller's closure is turned into so the cache's logic
/// is not duplicated per closure type.
type FetchFn<'a> = Box<dyn FnOnce() -> FetchFuture<'a> + Send + 'a>;
type FetchFuture<'a> =
    std::pin::Pin<Box<dyn Future<Output = Result<Fetched, HubError>> + Send + 'a>>;

/// What one cache slot is keyed on. Never contains a credential.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct CatalogKey {
    scope: Option<ScopeKey>,
    provider: Option<Slug>,
    endpoint: String,
    shape: CatalogShape,
}

impl CatalogKey {
    /// The key for reading `endpoint` in `shape`.
    ///
    /// `scope` and `provider` are used **only when `credentialed`**: a read that
    /// presented nothing has nothing tenant-specific to leak, and sharing it
    /// keeps one fetch serving every scope on a public endpoint. A credentialed
    /// read is partitioned by provider as well as scope, because two providers
    /// in one scope may point at one endpoint with different keys (a work and a
    /// personal account) and an entitlement-scoped listing must not cross
    /// between them.
    /// Trailing slashes and surrounding space do not make a different endpoint.
    /// The shape is part of the key because one URL can be read two ways and
    /// must not serve a paged envelope to an OpenAI-shaped reader.
    pub fn new(
        scope: &ScopeKey,
        provider: &Slug,
        credentialed: bool,
        endpoint: &str,
        shape: CatalogShape,
    ) -> Self {
        Self {
            scope: credentialed.then(|| scope.clone()),
            provider: credentialed.then(|| provider.clone()),
            endpoint: endpoint.trim().trim_end_matches('/').to_string(),
            shape,
        }
    }

    /// The scope this slot is partitioned by, if any.
    pub fn scope(&self) -> Option<&ScopeKey> {
        self.scope.as_ref()
    }
}

impl fmt::Debug for CatalogKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CatalogKey")
            .field("scope", &self.scope)
            .field("provider", &self.provider)
            .field("endpoint", &redact_endpoint(&self.endpoint))
            .field("shape", &self.shape)
            .finish()
    }
}

/// What a fetch closure returns.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub struct Fetched {
    /// The models, in the provider's order.
    pub models: Vec<ModelEntry>,
    /// The listing had more pages than one read follows.
    pub truncated: bool,
    /// The credential-scoped listing was unavailable and the public one was read
    /// instead (OpenRouter's `/models/user` answering 404). The list is real but
    /// says nothing about the key, and is not filtered by its permissions.
    pub public_fallback: bool,
}

impl Fetched {
    /// A complete listing.
    pub fn new(models: Vec<ModelEntry>) -> Self {
        Self {
            models,
            truncated: false,
            public_fallback: false,
        }
    }
}

struct Entry {
    at: Instant,
    ttl: Duration,
    /// Shared, so a read clones a pointer under the lock and the list outside it.
    models: Arc<Vec<ModelEntry>>,
    truncated: bool,
    /// The credential the list was read with (`None`: a keyless read, or a
    /// source that rotates its own token).
    credential: Option<SecretId>,
}

#[derive(Default)]
struct SlotState {
    entry: Option<Entry>,
    failure: Option<(Instant, ProviderFailure)>,
    /// The last failure that is deliberately **not remembered** (a rejected
    /// credential, or a bare 403), the generation it completed in, and whether
    /// an older list may still be served beside it. Never replayed to a later
    /// caller; only to callers that were already queued behind the fetch that
    /// produced it.
    unremembered: Option<(u64, ProviderFailure, bool, Option<SecretId>)>,
    last_used: Option<Instant>,
}

struct Slot {
    state: Mutex<SlotState>,
    /// Serialises fetches for this endpoint. Held across the whole fetch.
    fetch: AsyncMutex<()>,
    /// Bumped whenever a fetch completes, success or failure, so a caller that
    /// queued behind one can tell it need not fetch again.
    generation: AtomicU64,
}

impl Slot {
    fn new() -> Self {
        Self {
            state: Mutex::new(SlotState::default()),
            fetch: AsyncMutex::new(()),
            generation: AtomicU64::new(0),
        }
    }

    fn state(&self) -> MutexGuard<'_, SlotState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn fresh(&self, now: Instant, credential: Option<SecretId>) -> Option<ModelList> {
        let (models, truncated) = {
            let state = self.state();
            let entry = state.entry.as_ref()?;
            if now.saturating_duration_since(entry.at) >= entry.ttl
                || entry.credential != credential
            {
                return None;
            }
            (Arc::clone(&entry.models), entry.truncated)
        };
        Some(ModelList {
            models,
            freshness: Freshness::Cached,
            truncated,
        })
    }

    /// The unremembered failure of `generation`, only for a caller holding the
    /// credential it was about: a rejection of one key says nothing about another.
    fn unremembered_at(
        &self,
        generation: u64,
        credential: Option<SecretId>,
    ) -> Option<(ProviderFailure, bool)> {
        let state = self.state();
        let (at, failure, soft, rejected) = state.unremembered.as_ref()?;
        (*at == generation && *rejected == credential).then(|| (failure.clone(), *soft))
    }

    fn fresh_failure(&self, now: Instant) -> Option<ProviderFailure> {
        let state = self.state();
        let (at, failure) = state.failure.as_ref()?;
        (now.saturating_duration_since(*at) < FAILURE_TTL).then(|| failure.clone())
    }

    /// The remembered list as a stale answer, or the failure.
    ///
    /// An **empty** remembered list is not an answer worth serving ("no models,
    /// as of some time ago" beside a warning hides the failure that matters),
    /// and neither is one older than [`STALE_RETENTION`]: past a day the picker
    /// would be offering models that may have been retired.
    ///
    /// A list read with another credential is not served either: it is what a
    /// key that has since been replaced was entitled to.
    fn stale_or(
        &self,
        failure: ProviderFailure,
        now: Instant,
        credential: Option<SecretId>,
    ) -> Result<ModelList, HubError> {
        let kept = {
            let state = self.state();
            state
                .entry
                .as_ref()
                .filter(|entry| {
                    !entry.models.is_empty()
                        && entry.credential == credential
                        && now.saturating_duration_since(entry.at) < STALE_RETENTION
                })
                .map(|entry| (Arc::clone(&entry.models), entry.truncated))
        };
        match kept {
            Some((models, truncated)) => Ok(ModelList {
                models,
                freshness: Freshness::Stale { failure },
                truncated,
            }),
            None => Err(HubError::Provider(failure)),
        }
    }

    fn touch(&self, now: Instant) {
        self.state().last_used = Some(now);
    }
}

/// A `403` that is not a rejection (the classifier has no bare-403 rule: a WAF,
/// a geo block, an entitlement wall all read `unknown`). It is still about the
/// presented key rather than the endpoint, so it is never remembered, but it
/// proves nothing about the older list, which is served stale.
fn is_forbidden(failure: &ProviderFailure) -> bool {
    failure.status == Some(403) && !failure.is_rejection()
}

/// The catalog cache: one slot per [`CatalogKey`].
pub struct CatalogCache {
    clock: Arc<dyn Clock>,
    slots: Mutex<HashMap<CatalogKey, Arc<Slot>>>,
}

impl CatalogCache {
    /// An empty cache reading time from `clock`.
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            slots: Mutex::new(HashMap::new()),
        }
    }

    fn slots(&self) -> MutexGuard<'_, HashMap<CatalogKey, Arc<Slot>>> {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn slot(&self, key: CatalogKey) -> Arc<Slot> {
        let now = self.clock.now();
        let mut slots = self.slots();
        if !slots.contains_key(&key) && slots.len() >= MAX_SLOTS {
            Self::prune(&mut slots, now);
        }
        let slot = slots.entry(key).or_insert_with(|| Arc::new(Slot::new()));
        slot.touch(now);
        Arc::clone(slot)
    }

    /// Makes room: drops slots that were not used for the stale retention, then,
    /// if still full, the least recently used tenth.
    ///
    /// A slot somebody holds (a read in flight, possibly mid-fetch) is never
    /// dropped: removing it would let the next caller create a fresh slot and
    /// lose the single-flight guarantee. A slot that only ever saw rejections has
    /// no data, so it ages by when it was last used.
    fn prune(slots: &mut HashMap<CatalogKey, Arc<Slot>>, now: Instant) {
        slots.retain(|_, slot| {
            if Arc::strong_count(slot) > 1 {
                return true;
            }
            let state = slot.state();
            let newest = state
                .entry
                .as_ref()
                .map(|e| e.at)
                .into_iter()
                .chain(state.failure.as_ref().map(|(at, _)| *at))
                .chain(state.last_used)
                .max();
            newest.is_none_or(|at| now.saturating_duration_since(at) < STALE_RETENTION)
        });
        if slots.len() >= MAX_SLOTS {
            let mut by_use: Vec<(CatalogKey, Option<Instant>)> = slots
                .iter()
                .filter(|(_, slot)| Arc::strong_count(slot) == 1)
                .map(|(key, slot)| (key.clone(), slot.state().last_used))
                .collect();
            by_use.sort_by_key(|(_, used)| *used);
            for (key, _) in by_use.into_iter().take(MAX_SLOTS / 10 + 1) {
                slots.remove(&key);
            }
        }
    }

    /// How many endpoint slots exist.
    pub fn len(&self) -> usize {
        self.slots().len()
    }

    /// Whether the cache holds nothing.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drops every slot read on behalf of `scope`, because its credential
    /// changed. A rotation changes what the endpoint will answer without
    /// changing anything in the (non-secret) key, so without this a tenant that
    /// rotated to a key with different entitlements would keep reading the
    /// previous credential's list for the rest of the hour. Keyless slots are
    /// shared and untouched: no credential change can alter them.
    pub fn evict_scope(&self, scope: &ScopeKey) {
        self.slots().retain(|key, _| key.scope() != Some(scope));
    }

    /// Drops every slot for `endpoint`, whatever the scope or shape (the row's
    /// endpoint was edited).
    pub fn evict_endpoint(&self, endpoint: &str) {
        let endpoint = endpoint.trim().trim_end_matches('/');
        self.slots().retain(|key, _| key.endpoint != endpoint);
    }

    /// Drops everything.
    pub fn clear(&self) {
        self.slots().clear();
    }

    /// Returns the list for `key`, calling `fetch` on a miss.
    ///
    /// `refresh` bypasses a fresh entry and a remembered failure (the
    /// operator's Refresh button); callers that queued behind another fetch
    /// still reuse its result rather than each making their own.
    ///
    /// # Errors
    ///
    /// Whatever `fetch` returned, when there is nothing older to serve or the
    /// failure was a rejected credential; the remembered failure when one is
    /// fresh and there is nothing older to serve.
    pub async fn read<F, Fut>(
        &self,
        key: CatalogKey,
        refresh: bool,
        fetch: F,
    ) -> Result<ModelList, HubError>
    where
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<Fetched, HubError>> + Send,
    {
        self.read_as(key, refresh, None, fetch).await
    }

    /// [`CatalogCache::read`] for a caller that knows which credential it reads
    /// with: an entry (or a shared rejection) read with a different credential
    /// is not served to it, and its own fetch replaces that entry.
    ///
    /// # Errors
    ///
    /// As [`CatalogCache::read`].
    pub async fn read_as<F, Fut>(
        &self,
        key: CatalogKey,
        refresh: bool,
        credential: Option<SecretId>,
        fetch: F,
    ) -> Result<ModelList, HubError>
    where
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = Result<Fetched, HubError>> + Send,
    {
        // The body is compiled once, not once per caller's closure type.
        self.read_boxed(
            key,
            refresh,
            credential,
            Box::new(move || Box::pin(fetch())),
        )
        .await
    }

    async fn read_boxed(
        &self,
        key: CatalogKey,
        refresh: bool,
        credential: Option<SecretId>,
        fetch: FetchFn<'_>,
    ) -> Result<ModelList, HubError> {
        // A rejection is a fact about the *key presented*. A keyless read
        // presented none, so a 401 there is a fact about the endpoint (an auth
        // proxy in front of a public listing) and is remembered like any other.
        let credentialed = key.scope.is_some();
        let slot = self.slot(key);
        let generation_seen = slot.generation.load(Ordering::SeqCst);
        if !refresh {
            let now = self.clock.now();
            if let Some(list) = slot.fresh(now, credential) {
                return Ok(list);
            }
            if let Some(failure) = slot.fresh_failure(now) {
                return slot.stale_or(failure, self.clock.now(), credential);
            }
        }
        let _flight = slot.fetch.lock().await;
        // A fetch completed while this caller waited for the lock: use its
        // answer instead of asking the provider again.
        let generation_now = slot.generation.load(Ordering::SeqCst);
        if generation_now != generation_seen {
            // The fetch they queued behind was rejected: they share that answer
            // (one request, not one per waiter), but it is never replayed to a
            // caller that arrives later.
            if let Some((failure, soft)) = slot.unremembered_at(generation_now, credential) {
                return if soft {
                    slot.stale_or(failure, self.clock.now(), credential)
                } else {
                    Err(HubError::Provider(failure))
                };
            }
            let now = self.clock.now();
            if let Some(mut list) = slot.fresh(now, credential) {
                list.freshness = Freshness::Cached;
                return Ok(list);
            }
            if let Some(failure) = slot.fresh_failure(now) {
                return slot.stale_or(failure, self.clock.now(), credential);
            }
        }
        let outcome = fetch().await;
        let now = self.clock.now();
        match outcome {
            Ok(fetched) => {
                let ttl = if fetched.models.is_empty() {
                    EMPTY_CATALOG_TTL
                } else {
                    CATALOG_TTL
                };
                let models = Arc::new(fetched.models);
                let list = ModelList {
                    models: Arc::clone(&models),
                    freshness: Freshness::Fresh,
                    truncated: fetched.truncated,
                };
                {
                    let mut state = slot.state();
                    state.entry = Some(Entry {
                        at: now,
                        ttl,
                        models,
                        truncated: fetched.truncated,
                        credential,
                    });
                    // The endpoint answers again; a stale "unreachable" would
                    // keep reporting it.
                    state.failure = None;
                }
                slot.generation.fetch_add(1, Ordering::SeqCst);
                Ok(list)
            }
            // About the presented key: reported to this caller, never
            // remembered, and never answered with an older list (a bad key must
            // show).
            Err(HubError::Provider(failure)) if credentialed && failure.is_rejection() => {
                let generation = slot.generation.fetch_add(1, Ordering::SeqCst) + 1;
                {
                    let mut state = slot.state();
                    state.unremembered = Some((generation, failure.clone(), false, credential));
                    // "A bad key must show": the list read with a key the provider
                    // now refuses is not served as if nothing happened, and an
                    // earlier endpoint failure must not answer instead of the
                    // rejection this request actually got.
                    state.entry = None;
                    state.failure = None;
                }
                Err(HubError::Provider(failure))
            }
            Err(HubError::Provider(failure)) if credentialed && is_forbidden(&failure) => {
                // Shared with the callers queued behind this request (one 403,
                // not one per waiter), never remembered for later ones.
                let generation = slot.generation.fetch_add(1, Ordering::SeqCst) + 1;
                slot.state().unremembered = Some((generation, failure.clone(), true, credential));
                slot.stale_or(failure, now, credential)
            }
            Err(HubError::Provider(failure)) => {
                slot.state().failure = Some((now, failure.clone()));
                slot.generation.fetch_add(1, Ordering::SeqCst);
                slot.stale_or(failure, now, credential)
            }
            // A policy refusal, an unreadable store, a bad input: deterministic
            // or about something other than the endpoint, so not memoised.
            Err(other) => Err(other),
        }
    }
}

impl fmt::Debug for CatalogCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CatalogCache")
            .field("slots", &self.len())
            .finish()
    }
}
