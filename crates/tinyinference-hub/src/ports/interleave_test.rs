//! The interleaving hook: it parks exactly the call it names and nothing else.

use crate::HubConfig;
use crate::ids::ScopeKey;
use crate::ports::memory::{Call, Hold, MemoryConfig, MemoryCredentials};
use crate::ports::{ConfigStore, CredentialStore};
use crate::secret::Secret;

fn scope() -> ScopeKey {
    ScopeKey::new("a")
}

#[tokio::test]
async fn interleave_a_call_held_before_has_not_taken_effect_and_one_held_after_has() {
    let store = MemoryCredentials::new();
    let scope = scope();
    let mut before = store.hold(Hold::before(Call::Set));
    let write = store.set(&scope, "s", Secret::new("v"));
    let look = async {
        before.reached().await;
        assert!(store.get(&scope, "s").await.unwrap().is_none());
        before.release();
    };
    let (done, ()) = tokio::join!(write, look);
    done.unwrap();

    let mut after = store.hold(Hold::after(Call::Set));
    let write = store.set(&scope, "s", Secret::new("w"));
    let look = async {
        after.reached().await;
        let seen = store.get(&scope, "s").await.unwrap().unwrap();
        assert_eq!(seen.expose(), "w", "it took effect before it returned");
        after.release();
    };
    let (done, ()) = tokio::join!(write, look);
    done.unwrap();
}

#[tokio::test]
async fn interleave_skip_and_slot_pick_the_call_to_hold() {
    let store = MemoryCredentials::new();
    let scope = scope();
    let mut held = store.hold(Hold::before(Call::Set).slot("wanted").skip(1));
    let writes = async {
        store.set(&scope, "other", Secret::new("1")).await.unwrap();
        store.set(&scope, "wanted", Secret::new("2")).await.unwrap();
        store.set(&scope, "wanted", Secret::new("3")).await.unwrap();
    };
    let look = async {
        held.reached().await;
        // Only the second write to "wanted" is parked.
        assert_eq!(
            store.get(&scope, "wanted").await.unwrap().unwrap().expose(),
            "2"
        );
        assert!(store.get(&scope, "other").await.unwrap().is_some());
        held.release();
    };
    tokio::join!(writes, look);
    assert_eq!(
        store.get(&scope, "wanted").await.unwrap().unwrap().expose(),
        "3"
    );
}

#[tokio::test]
async fn interleave_a_dropped_handle_releases_the_call() {
    let store = MemoryCredentials::new();
    let held = store.hold(Hold::before(Call::Delete));
    drop(held);
    store.delete(&scope(), "s").await.unwrap();
}

#[tokio::test]
async fn interleave_config_saves_and_loads_can_be_held() {
    let config = MemoryConfig::new();
    let scope = scope();
    let mut held = config.hold(Hold::after(Call::Save));
    let doc = HubConfig::new();
    let save = config.save(&scope, &doc, None);
    let look = async {
        held.reached().await;
        assert!(config.raw(&scope).is_some());
        held.release();
    };
    let (saved, ()) = tokio::join!(save, look);
    saved.unwrap();
    let mut held = config.hold(Hold::before(Call::Load));
    let load = config.load(&scope);
    let look = async {
        held.reached().await;
        held.release();
    };
    let (loaded, ()) = tokio::join!(load, look);
    assert!(loaded.unwrap().is_some());
}

#[tokio::test]
async fn interleave_a_scripted_fault_fails_the_matching_call_only() {
    let store = MemoryCredentials::new();
    let scope = scope();
    let _fault = store.hold(Hold::before(Call::Get).skip(1).fail());
    assert!(store.get(&scope, "s").await.is_ok());
    assert!(
        store.get(&scope, "s").await.is_err(),
        "the second get fails"
    );
    assert!(store.get(&scope, "s").await.is_ok(), "and only that one");
    // After a commit, before the caller hears of it: the write happened.
    let _late = store.hold(Hold::after(Call::Set).fail());
    assert!(store.set(&scope, "s", Secret::new("v")).await.is_err());
    assert!(store.get(&scope, "s").await.unwrap().is_some());
}

#[tokio::test]
async fn interleave_a_slot_filter_does_not_apply_to_a_config_call() {
    // `Hold::slot` names a credential slot; a config call has none, so the filter
    // is ignored rather than making the hold unmatchable.
    let config = MemoryConfig::new();
    let scope = scope();
    let _fault = config.hold(Hold::before(Call::Load).slot("anything").fail());
    assert!(config.load(&scope).await.is_err());
    assert!(config.load(&scope).await.is_ok());
}

#[tokio::test]
async fn interleave_a_health_forget_can_be_held_and_failed() {
    use crate::ids::Slug;
    use crate::ports::HealthStore;
    use crate::ports::memory::MemoryHealth;
    let health = MemoryHealth::new();
    let scope = scope();
    let slug = Slug::parse("acme").unwrap();
    let mut held = health.hold(Hold::before(Call::Forget));
    let forget = health.forget(&scope, &slug);
    let look = async {
        held.reached().await;
        held.release();
    };
    let (done, ()) = tokio::join!(forget, look);
    done.unwrap();
    let _fault = health.hold(Hold::after(Call::Forget).fail());
    assert!(health.forget(&scope, &slug).await.is_err());
}

#[test]
#[should_panic(expected = "no interleave point")]
fn interleave_a_hold_on_a_call_the_store_never_makes_is_refused_not_hung() {
    let health = crate::ports::memory::MemoryHealth::new();
    let _ = health.hold(Hold::before(Call::Get));
}
