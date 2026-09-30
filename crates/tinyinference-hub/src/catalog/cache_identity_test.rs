//! Credential identity on cache entries (finding 3.7): a listing read with one
//! credential is never served to another, though both share a slot and one
//! endpoint's single flight.

use std::sync::Arc;
use std::time::Duration;

use super::*;
use crate::error::{HubError, ProviderFailure, ReasonCode, Retry};
use crate::ids::{ModelId, ScopeKey, Slug};
use crate::secret::Secret;
use crate::taxonomy::CatalogShape;
use crate::testkit::FakeClock;

const ENDPOINT: &str = "https://api.acme.test/v1";

fn key() -> CatalogKey {
    CatalogKey::new(
        &ScopeKey::new("a"),
        &Slug::parse("openai").unwrap(),
        true,
        ENDPOINT,
        CatalogShape::OpenAi,
    )
}

fn cache() -> (CatalogCache, FakeClock) {
    let clock = FakeClock::new();
    (CatalogCache::new(Arc::new(clock.clone())), clock)
}

fn listing(name: &str) -> Result<Fetched, HubError> {
    Ok(Fetched::new(vec![ModelEntry::new(
        ModelId::parse(name).unwrap(),
    )]))
}

fn rejected() -> HubError {
    let mut failure = ProviderFailure::new(ReasonCode::Auth, Retry::Never);
    failure.status = Some(401);
    HubError::Provider(failure)
}

fn outage() -> HubError {
    HubError::Provider(ProviderFailure::new(
        ReasonCode::Endpoint,
        Retry::Later(None),
    ))
}

fn id(text: &str) -> Option<crate::secret::SecretId> {
    Some(Secret::new(text).id())
}

#[tokio::test]
async fn cache_identity_a_list_read_with_one_key_is_not_served_to_another() {
    let (cache, _clock) = cache();
    let first = cache
        .read_as(key(), false, id("old"), || async { listing("old-list") })
        .await
        .unwrap();
    assert_eq!(first.freshness, Freshness::Fresh);
    // The same key hits.
    let same = cache
        .read_as(key(), false, id("old"), || async {
            Err(HubError::Conflict)
        })
        .await
        .unwrap();
    assert_eq!(same.freshness, Freshness::Cached);
    // A replacement key does not, and its own read replaces the entry.
    let new = cache
        .read_as(key(), false, id("new"), || async { listing("new-list") })
        .await
        .unwrap();
    assert_eq!(new.freshness, Freshness::Fresh);
    assert_eq!(new.ids(), ["new-list"]);
    // The old key now misses.
    let old = cache
        .read_as(key(), false, id("old"), || async { listing("old-again") })
        .await
        .unwrap();
    assert_eq!(old.ids(), ["old-again"]);
}

#[tokio::test]
async fn cache_identity_a_failure_never_serves_another_keys_list_as_stale() {
    let (cache, clock) = cache();
    cache
        .read_as(key(), false, id("old"), || async { listing("old-list") })
        .await
        .unwrap();
    clock.advance(CATALOG_TTL + Duration::from_secs(1));
    // The endpoint is down; the new key must see the outage, not the old key's list.
    let error = cache
        .read_as(key(), false, id("new"), || async { Err(outage()) })
        .await
        .unwrap_err();
    assert!(matches!(error, HubError::Provider(_)), "{error:?}");
    // The old key still gets its own list as the stale fallback.
    let stale = cache
        .read_as(key(), true, id("old"), || async { Err(outage()) })
        .await
        .unwrap();
    assert!(matches!(stale.freshness, Freshness::Stale { .. }));
    assert_eq!(stale.ids(), ["old-list"]);
}

#[tokio::test]
async fn cache_identity_a_rejection_is_not_shared_with_a_caller_holding_another_key() {
    let (cache, _clock) = cache();
    let (a, b) = futures::join!(
        cache.read_as(key(), false, id("bad"), || async {
            tokio::task::yield_now().await;
            Err(rejected())
        }),
        cache.read_as(key(), false, id("good"), || async { listing("good-list") }),
    );
    assert!(a.is_err());
    assert_eq!(b.unwrap().ids(), ["good-list"]);
}

#[tokio::test]
async fn cache_identity_a_token_source_without_identity_shares_across_rotations() {
    // The managed platform token rotates every minute; the list is the
    // account's, so no identity means one fetch, not one per token.
    let (cache, _clock) = cache();
    cache
        .read_as(key(), false, None, || async { listing("managed") })
        .await
        .unwrap();
    let again = cache
        .read_as(key(), false, None, || async { Err(HubError::Conflict) })
        .await
        .unwrap();
    assert_eq!(again.freshness, Freshness::Cached);
}
