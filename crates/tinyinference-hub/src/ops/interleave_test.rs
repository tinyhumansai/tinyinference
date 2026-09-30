//! Races staged with the interleaving hook (`Hold`): the schedules the plain
//! in-memory ports cannot produce, written down instead of hoped for.
//!
//! * finding 5.4: an origin move must never let a credential entered for one
//!   origin be used against the other, at **any** point of the edit;
//! * finding 4.6: the three-way same-slug race must not delete the key the
//!   winning add wrote;
//! * findings 4.4 and 4.5: an edit and a key change on one provider are ordered,
//!   and a failing config store at any two consecutive reads never leaves an
//!   orphaned key behind an undone add.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tinyinference_llm::MockModel;
use tinyinference_llm::error::Result as LlmResult;
use tinyinference_llm::model::{ChatModel, ModelRequest, ModelResponse, ModelStream};

use crate::client::{ModelFactory, ModelSpec};
use crate::config::ProviderDraft;
use crate::error::HubError;
use crate::hub::fixtures::{Bed, model, slug};
use crate::hub::{Confirm, ConnectOptions, Hub, ProviderPatch};
use crate::ids::{ModelId, ScopeKey};
use crate::ports::memory::{Call, Hold};
use crate::route::TurnQuery;
use crate::secret::Secret;

const OLD: &str = "https://llm.acme.test/v1";
const NEW: &str = "https://llm.acme-two.test/v1";
const K_OLD: &str = "sk-not-a-real-key-old";
const K_NEW: &str = "sk-not-a-real-key-new";
/// A credential the host supplies through another source of the chain.
const K_HOST: &str = "sk-not-a-real-key-host";

/// A model that answers and remembers nothing; what matters is what the
/// factory was asked to build it with.
struct Echo;

impl std::fmt::Debug for Echo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Echo")
    }
}

#[async_trait]
impl ChatModel<()> for Echo {
    async fn invoke(&self, _: &(), _: ModelRequest) -> LlmResult<ModelResponse> {
        Ok(MockModel::text_response("ok"))
    }

    async fn stream(&self, state: &(), request: ModelRequest) -> LlmResult<ModelStream> {
        let response = self.invoke(state, request).await?;
        Ok(ModelStream::new(Box::pin(futures::stream::iter(vec![
            tinyinference_llm::model::ModelStreamItem::Completed(response),
        ]))))
    }
}

/// Records `(endpoint, key)` for every model the hub builds: the pair a request
/// would carry.
#[derive(Debug, Default)]
struct Spy {
    built: Mutex<Vec<(String, Option<String>)>>,
}

impl ModelFactory for Spy {
    fn build(&self, spec: &ModelSpec<'_>) -> Result<Arc<dyn ChatModel<()>>, HubError> {
        self.built
            .lock()
            .unwrap()
            .push((spec.turn.base_url.clone(), spec.key.map(str::to_string)));
        Ok(Arc::new(Echo))
    }
}

async fn acme_bed() -> (Bed, Arc<Spy>) {
    acme_bed_with(true).await
}

/// `stored`: the provider has a stored key (`K_OLD`). Otherwise it has none and
/// the only credential is the host's own source (`K_HOST`), which answers
/// whatever the endpoint is.
async fn acme_bed_with(stored: bool) -> (Bed, Arc<Spy>) {
    let spy = Arc::new(Spy::default());
    let bed = Bed::with(|b| {
        let b = b.model_factory(spy.clone());
        if stored {
            b
        } else {
            b.credential_source(
                "custom",
                crate::credential::StaticSource::new(Secret::new(K_HOST)),
            )
        }
    });
    let mut draft = ProviderDraft::new("custom")
        .with_label("Acme")
        .with_base_url(OLD)
        .with_model(model("m"));
    if stored {
        draft = draft.with_key(Secret::new(K_OLD));
    }
    bed.hub.add(&bed.scope, draft).await.unwrap();
    (bed, spy)
}

/// Uses the provider the two ways a host does: through a model it kept from
/// before, and through a fresh resolve. Errors are fine (a fail-closed refusal
/// is the point); only what gets *built* matters.
async fn use_it(hub: &Hub, scope: &ScopeKey, kept: &Arc<dyn ChatModel<()>>) {
    let _ = kept.invoke(&(), ModelRequest::default()).await;
    if let Ok(turn) = hub.resolve_for_turn(scope, &TurnQuery::new()).await
        && let Ok(fresh) = hub.chat_model(scope, &turn).await
    {
        let _ = fresh.invoke(&(), ModelRequest::default()).await;
    }
}

fn assert_no_cross_origin_credential(spy: &Spy, at: &str) {
    for (endpoint, key) in spy.built.lock().unwrap().iter() {
        let allowed = match (endpoint.as_str(), key.as_deref()) {
            // A host credential belongs to the host's provider, not to an origin:
            // it is only ever offered where the provider was when it was added.
            (OLD, Some(K_OLD | K_HOST)) | (NEW, Some(K_NEW)) => true,
            // No key at all is fail-closed or keyless: never a leak.
            (_, None) => true,
            _ => false,
        };
        assert!(
            allowed,
            "{at}: a request would carry {key:?} to {endpoint}, an origin it was not entered for"
        );
    }
}

/// Runs `edit` (an origin move with a key entered for the new origin) holding
/// it at `hold`, using the provider from another task while it is parked.
async fn move_origin_holding(hold: Option<(bool, Hold)>, label: &str, stored: bool) -> bool {
    let (bed, spy) = acme_bed_with(stored).await;
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    let kept = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
    kept.invoke(&(), ModelRequest::default()).await.unwrap();
    let mut held = hold.map(|(on_config, hold)| {
        if on_config {
            bed.ports.config.hold(hold)
        } else {
            bed.ports.credentials.hold(hold)
        }
    });
    let reached = std::cell::Cell::new(false);
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
    let edit = async {
        let result = bed
            .hub
            .edit(
                &bed.scope,
                &slug("acme"),
                ProviderPatch::new().base_url(NEW).key(Secret::new(K_NEW)),
            )
            .await;
        let _ = done_tx.send(());
        result
    };
    let probe = async {
        if let Some(held) = held.as_mut() {
            tokio::select! {
                () = held.reached() => {
                    reached.set(true);
                    use_it(&bed.hub, &bed.scope, &kept).await;
                    held.release();
                }
                _ = done_rx => {}
            }
        }
    };
    let (edited, ()) = tokio::join!(edit, probe);
    edited.unwrap();
    // A hold that was never reached must not park the calls made from here on.
    drop(held);
    use_it(&bed.hub, &bed.scope, &kept).await;
    assert_no_cross_origin_credential(&spy, label);
    // And the move worked: the new origin with the new key.
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(turn.base_url, NEW);
    assert_eq!(bed.key_of("acme").await.as_deref(), Some(K_NEW));
    reached.get()
}

#[tokio::test]
async fn ops_race_an_origin_move_never_pairs_a_key_with_an_origin_it_was_not_entered_for() {
    // Every point of the edit at which a host can run: the config's first
    // load/save (before and after taking effect) and the credential store's
    // first and second get/set/delete. `skip` reaches the second call of a kind.
    let slot = slug("acme").key_slot();
    let cfg = |hold: Hold| (true, hold);
    let cred = |hold: Hold| (false, hold);
    let holds: Vec<(&str, (bool, Hold))> = vec![
        ("config save before", cfg(Hold::before(Call::Save))),
        ("config save after", cfg(Hold::after(Call::Save))),
        (
            "config load before #1",
            cfg(Hold::before(Call::Load).skip(1)),
        ),
        (
            "config load before #2",
            cfg(Hold::before(Call::Load).skip(2)),
        ),
        ("config load after #2", cfg(Hold::after(Call::Load).skip(2))),
        ("config load after #3", cfg(Hold::after(Call::Load).skip(3))),
        ("cred set before", cred(Hold::before(Call::Set).slot(&slot))),
        ("cred set after", cred(Hold::after(Call::Set).slot(&slot))),
        (
            "config save before #2",
            cfg(Hold::before(Call::Save).skip(1)),
        ),
        ("config save after #2", cfg(Hold::after(Call::Save).skip(1))),
        ("cred get before", cred(Hold::before(Call::Get).slot(&slot))),
        ("cred get after", cred(Hold::after(Call::Get).slot(&slot))),
        (
            "cred get before #2",
            cred(Hold::before(Call::Get).slot(&slot).skip(1)),
        ),
        (
            "cred get after #2",
            cred(Hold::after(Call::Get).slot(&slot).skip(1)),
        ),
    ];
    let mut missed = Vec::new();
    // Twice: once with the key in the provider's own slot, once with the only
    // credential coming from another source of the chain (the host's).
    for stored in [true, false] {
        for (label, hold) in holds.clone() {
            if !move_origin_holding(Some(hold), label, stored).await {
                missed.push(label);
            }
        }
    }
    // The schedule must actually have happened: a hold the edit never reached
    // proves nothing about that point.
    assert!(missed.is_empty(), "holds never reached: {missed:?}");
    // And the untouched run is fine too.
    assert!(!move_origin_holding(None, "no hold", true).await);
    assert!(!move_origin_holding(None, "no hold", false).await);
}

#[tokio::test]
async fn ops_race_the_three_way_same_slug_add_does_not_delete_the_winning_adds_key() {
    // A1 adds `acme` and is parked right after its record is committed. Meanwhile
    // the provider is removed and added again (A2, with its own key). When A1
    // resumes it must not write its key over A2's, and must not delete it.
    let bed = Bed::new();
    let draft = |key: &str| {
        ProviderDraft::new("custom")
            .with_label("Acme")
            .with_base_url(OLD)
            .with_key(Secret::new(key))
            .with_model(model("m"))
    };
    let mut held = bed.ports.config.hold(Hold::after(Call::Save));
    let a1 = bed.hub.add(&bed.scope, draft("sk-not-a-real-key-a1"));
    let others = async {
        held.reached().await;
        bed.hub
            .remove(&bed.scope, &slug("acme"), Confirm::in_use())
            .await
            .unwrap();
        bed.hub
            .add(&bed.scope, draft("sk-not-a-real-key-a2"))
            .await
            .unwrap();
        held.release();
    };
    let (first, ()) = tokio::join!(a1, others);
    assert!(
        matches!(first, Err(HubError::NotFound(_))),
        "the first add lost its provider: {first:?}"
    );
    assert_eq!(
        bed.key_of("acme").await.as_deref(),
        Some("sk-not-a-real-key-a2"),
        "the winning add's key is untouched"
    );
    assert_eq!(bed.ports.credentials.len(), 1);
}

#[tokio::test]
async fn ops_race_a_key_set_while_an_edit_moves_the_origin_is_ordered_not_interleaved() {
    // set_key on a provider whose edit is parked mid-move waits for the edit
    // (they share the provider's lock), then lands on the provider as it is.
    let (bed, _spy) = acme_bed().await;
    let acme = slug("acme");
    let mut held = bed.ports.config.hold(Hold::before(Call::Save));
    let edit = bed.hub.edit(
        &bed.scope,
        &acme,
        ProviderPatch::new().base_url(NEW).key(Secret::new(K_NEW)),
    );
    let set = async {
        held.reached().await;
        // Started while the edit is parked; it cannot finish before the edit does.
        let mut set = std::pin::pin!(bed.hub.set_key(
            &bed.scope,
            &acme,
            Secret::new("sk-not-a-real-key-late")
        ));
        let polled = futures::poll!(set.as_mut());
        assert!(polled.is_pending(), "set_key waits for the edit's lock");
        held.release();
        set.await.unwrap();
    };
    let (edited, ()) = tokio::join!(edit, set);
    edited.unwrap();
    assert_eq!(
        bed.key_of("acme").await.as_deref(),
        Some("sk-not-a-real-key-late")
    );
    let config = bed.hub.status(&bed.scope).await.unwrap();
    let acme = config
        .providers
        .iter()
        .find(|p| p.view.record.slug == slug("acme"))
        .unwrap();
    assert_eq!(acme.view.record.base_url, NEW);
}

#[tokio::test]
async fn ops_race_a_config_store_failing_at_any_two_reads_never_leaves_a_key_behind_an_undone_add()
{
    // Findings 4.5 / 5.3: `undo_add` read the config twice before deciding what
    // to restore. Rather than guess which read, fail every consecutive pair in
    // turn and check the one thing that must always hold: when the add did not
    // stick, no key belongs to a provider that does not exist.
    for first in 0..14usize {
        let bed = Bed::new();
        bed.openai_rejects_key();
        // The slot held a leftover key from an earlier provider of that slug.
        bed.store_key("openai", "sk-not-a-real-key-earlier").await;
        // Two scripted faults on consecutive loads (identical holds fire on
        // consecutive calls), starting at the `first`-th load of the connect.
        let _a = bed
            .ports
            .config
            .hold(Hold::before(Call::Load).skip(first).fail());
        let _b = bed
            .ports
            .config
            .hold(Hold::before(Call::Load).skip(first).fail());
        let result = bed
            .hub
            .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
            .await;
        let stored = bed.ports.config.raw(&bed.scope).unwrap_or_default();
        let record_exists = stored.contains("\"slug\":\"openai\"");
        let key = bed.key_of("openai").await;
        if !record_exists {
            assert_eq!(
                key.as_deref(),
                Some("sk-not-a-real-key-earlier"),
                "pair {first}: the add is gone, so the slot is what it was before it ({result:?})"
            );
        }
    }
}

async fn state_of(bed: &Bed) -> (String, Option<String>) {
    let status = bed.hub.status(&bed.scope).await.unwrap();
    let acme = status
        .providers
        .iter()
        .find(|p| p.view.record.slug == slug("acme"))
        .unwrap();
    (acme.view.record.base_url.clone(), bed.key_of("acme").await)
}

fn move_patch() -> ProviderPatch {
    ProviderPatch::new().base_url(NEW).key(Secret::new(K_NEW))
}

#[tokio::test]
async fn ops_an_origin_move_whose_record_cannot_be_saved_keeps_the_old_origin_and_key() {
    let (bed, _spy) = acme_bed().await;
    let _fault = bed.ports.config.hold(Hold::before(Call::Save).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    assert_eq!(state_of(&bed).await, (OLD.to_string(), Some(K_OLD.into())));
}

#[tokio::test]
async fn ops_an_origin_move_whose_key_cannot_be_written_goes_back_to_the_old_origin_and_key() {
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _fault = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    // The record was moved and moved back; the old key is back with it.
    assert_eq!(state_of(&bed).await, (OLD.to_string(), Some(K_OLD.into())));
}

#[tokio::test]
async fn ops_an_origin_move_that_cannot_be_undone_fails_closed_with_no_key_at_all() {
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _write = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    // The commit is the first save, the move back the second.
    let _back = bed
        .ports
        .config
        .hold(Hold::before(Call::Save).skip(1).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    // At the new origin with no key: neither key is usable anywhere.
    assert_eq!(state_of(&bed).await, (NEW.to_string(), None));
}

#[tokio::test]
async fn ops_an_origin_move_validated_against_an_endpoint_that_moved_meanwhile_is_a_conflict() {
    // Finding 4.4: another writer (another hub over this store) moves the
    // endpoint after this edit validated its move and before it saves. Keyless,
    // so that the move itself is allowed.
    let bed = Bed::new();
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("custom")
                .with_label("Acme")
                .with_base_url(OLD)
                .with_model(model("m")),
        )
        .await
        .unwrap();
    let acme = slug("acme");
    // The edit's loads: the first read, the re-read under the lock, then the
    // transaction's own.
    let mut held = bed.ports.config.hold(Hold::before(Call::Load).skip(2));
    let edit = bed
        .hub
        .edit(&bed.scope, &acme, ProviderPatch::new().base_url(NEW));
    let other_writer = async {
        held.reached().await;
        let mut doc: serde_json::Value =
            serde_json::from_str(&bed.ports.config.raw(&bed.scope).unwrap()).unwrap();
        for row in doc["providers"].as_array_mut().unwrap() {
            if row["slug"] == "acme" {
                row["base_url"] = "https://llm.third.test/v1".into();
            }
        }
        bed.ports.config.put_raw(&bed.scope, doc.to_string());
        held.release();
    };
    let (result, ()) = tokio::join!(edit, other_writer);
    assert!(matches!(result, Err(HubError::Conflict)), "{result:?}");
    let (base, _) = state_of(&bed).await;
    assert_eq!(
        base, "https://llm.third.test/v1",
        "the other writer's move stands"
    );
}

#[tokio::test]
async fn ops_race_a_model_parked_at_its_key_read_never_pairs_the_new_key_with_the_old_origin() {
    // The mirror of the origin-move race: the kept model has checked the record
    // (old origin) and is about to read the key while the whole edit runs. The
    // key it then reads is the new one, for the new origin: it must notice.
    for after in [false, true] {
        let (bed, spy) = acme_bed().await;
        let turn = bed
            .hub
            .resolve_for_turn(&bed.scope, &TurnQuery::new())
            .await
            .unwrap();
        let kept = bed.hub.chat_model(&bed.scope, &turn).await.unwrap();
        kept.invoke(&(), ModelRequest::default()).await.unwrap();
        let slot = slug("acme").key_slot();
        let hold = if after {
            Hold::after(Call::Get)
        } else {
            Hold::before(Call::Get)
        };
        let mut held = bed.ports.credentials.hold(hold.slot(&slot));
        let send = kept.invoke(&(), ModelRequest::default());
        let edit = async {
            held.reached().await;
            bed.hub
                .edit(&bed.scope, &slug("acme"), move_patch())
                .await
                .unwrap();
            held.release();
        };
        let (sent, ()) = tokio::join!(send, edit);
        drop(held);
        // Read before the move (after = true) is still the old pair; read after it
        // is refused. Never the new key at the old origin.
        if !after {
            assert!(sent.is_err(), "a stale model refuses instead of sending");
        }
        assert_no_cross_origin_credential(&spy, "model parked at its key read");
    }
}

#[tokio::test]
async fn ops_race_a_key_clear_that_waited_for_a_removal_reports_the_provider_gone() {
    // clear_key checked the provider, then waited for the lock while a removal
    // ran. Deciding again once it holds the lock, it finds nothing to clear: it
    // must not report "no key to remove" for a provider that no longer exists
    // (nor, had a new add landed, delete that add's key).
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let acme = slug("acme");
    let mut held = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Delete).slot(&slot));
    let removal = bed.hub.remove(&bed.scope, &acme, Confirm::in_use());
    let clear = async {
        held.reached().await;
        let mut clear = std::pin::pin!(bed.hub.clear_key(&bed.scope, &acme, Confirm::in_use()));
        assert!(
            futures::poll!(clear.as_mut()).is_pending(),
            "clear_key waits for the removal's lock"
        );
        held.release();
        clear.await
    };
    let (removed, cleared) = tokio::join!(removal, clear);
    removed.unwrap();
    assert!(matches!(cleared, Err(HubError::NotFound(_))), "{cleared:?}");
}

#[tokio::test]
async fn ops_an_origin_move_whose_old_key_cannot_be_deleted_changes_nothing() {
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _fault = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Delete).slot(&slot).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    assert_eq!(state_of(&bed).await, (OLD.to_string(), Some(K_OLD.into())));
    let turn = bed
        .hub
        .resolve_for_turn(&bed.scope, &TurnQuery::new())
        .await
        .unwrap();
    assert_eq!(turn.base_url, OLD, "still enabled and usable");
}

#[tokio::test]
async fn ops_an_origin_move_that_loses_both_the_key_write_and_the_slot_cleanup_never_leaves_the_old_key_at_the_new_origin()
 {
    // Seed 1592594484 (flaky): the credential store stopped answering writes
    // mid-move, so the new key could not be written and the slot could not be
    // emptied either. The old key was already out of the slot (deleted first), so
    // the disabled record at the new origin has nothing to send; `test` of a
    // disabled provider still sends a credential, so it is checked too.
    let (bed, spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _write = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    // Delete #0 is the move's first step; #1 is the undo's.
    let _cleanup = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Delete).slot(&slot).skip(1).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    let (base, key) = state_of(&bed).await;
    assert_eq!((base.as_str(), key), (NEW, None));
    let record_enabled = bed
        .hub
        .status(&bed.scope)
        .await
        .unwrap()
        .providers
        .iter()
        .find(|p| p.view.record.slug == slug("acme"))
        .unwrap()
        .view
        .record
        .enabled;
    assert!(!record_enabled, "left disabled");
    bed.ports.http.route(
        crate::testkit::Match::prefix(NEW),
        crate::testkit::Scripted::json(200, &crate::hub::fixtures::models_body(&["m"])),
    );
    bed.hub
        .test(
            &bed.scope,
            &slug("acme"),
            crate::taxonomy::TestDepth::Catalog,
            None,
        )
        .await
        .unwrap();
    assert!(
        bed.ports.http.request_count() > 0,
        "the test of the disabled provider did send a request"
    );
    for request in bed.ports.http.requests_from(0) {
        assert!(
            !request.carried(&Secret::new(K_OLD)),
            "the old key was sent to {}",
            request.url
        );
    }
    assert_no_cross_origin_credential(&spy, "double failure");
}

async fn enabled_of(bed: &Bed) -> bool {
    bed.hub
        .status(&bed.scope)
        .await
        .unwrap()
        .providers
        .iter()
        .find(|p| p.view.record.slug == slug("acme"))
        .unwrap()
        .view
        .record
        .enabled
}

#[tokio::test]
async fn ops_an_origin_move_that_cannot_switch_the_provider_back_on_leaves_it_disabled_with_its_new_key()
 {
    let (bed, _spy) = acme_bed().await;
    // Saves: the move (#0), then switching back on (#1).
    let _fault = bed
        .ports
        .config
        .hold(Hold::before(Call::Save).skip(1).fail());
    let mutation = bed
        .hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap();
    // The edit went through; only the flag could not be restored: a warning on a
    // successful edit, not an error a caller would retry.
    assert_eq!(
        mutation.status,
        crate::hub::MutationStatus::SavedWithWarning
    );
    assert_eq!(state_of(&bed).await, (NEW.to_string(), Some(K_NEW.into())));
    assert!(!enabled_of(&bed).await, "unusable rather than half usable");
}

#[tokio::test]
async fn ops_an_origin_move_whose_record_cannot_be_saved_and_whose_old_key_cannot_come_back_ends_with_no_key()
 {
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _save = bed.ports.config.hold(Hold::before(Call::Save).fail());
    let _restore = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    // The old origin, enabled, with no key: fail closed, never the wrong pair.
    assert_eq!(state_of(&bed).await, (OLD.to_string(), None));
    assert!(enabled_of(&bed).await);
}

#[tokio::test]
async fn ops_an_undone_origin_move_whose_old_key_cannot_come_back_ends_with_no_key() {
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    // The new key's write fails (Set #0) and so does putting the old one back (#1).
    let _write = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    let _restore = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).skip(0).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    assert_eq!(state_of(&bed).await, (OLD.to_string(), None));
    assert!(enabled_of(&bed).await);
}

#[tokio::test]
async fn ops_a_probe_that_read_the_record_before_an_origin_move_follows_the_move_never_mixes_it() {
    // A probe reads the record (old origin) and is parked; the whole move runs;
    // the probe then resolves its credential. It notices the endpoint changed
    // since it read the record and starts again from the new one: what it sends
    // is the new key to the new origin, never the new key (or a host credential)
    // to the old endpoint. Both credential shapes.
    for stored in [true, false] {
        let (bed, spy) = acme_bed_with(stored).await;
        bed.ports.http.route(
            crate::testkit::Match::prefix(NEW),
            crate::testkit::Scripted::json(200, &crate::hub::fixtures::models_body(&["m"])),
        );
        let acme = slug("acme");
        let mut held = bed.ports.config.hold(Hold::after(Call::Load));
        let test = bed
            .hub
            .test(&bed.scope, &acme, crate::taxonomy::TestDepth::Catalog, None);
        let moving = async {
            held.reached().await;
            bed.hub.edit(&bed.scope, &acme, move_patch()).await.unwrap();
            held.release();
        };
        let (tested, ()) = tokio::join!(test, moving);
        tested.unwrap();
        assert!(bed.ports.http.request_count() > 0);
        for request in bed.ports.http.requests_from(0) {
            assert!(request.url.starts_with(NEW), "{}", request.url);
            assert!(request.carried(&Secret::new(K_NEW)));
        }
        assert_no_cross_origin_credential(&spy, "probe read before a move");
    }
}

#[tokio::test]
async fn ops_a_probe_started_during_an_origin_move_waits_for_it_and_sends_the_new_pair() {
    for stored in [true, false] {
        let (bed, _spy) = acme_bed_with(stored).await;
        bed.ports.http.route(
            crate::testkit::Match::prefix(NEW),
            crate::testkit::Scripted::json(200, &crate::hub::fixtures::models_body(&["m"])),
        );
        let acme = slug("acme");
        let mut held = bed.ports.config.hold(Hold::after(Call::Save));
        let edit = bed.hub.edit(&bed.scope, &acme, move_patch());
        let others = async {
            held.reached().await;
            // The move has committed (disabled, at the new origin) and is parked.
            let mut test = std::pin::pin!(bed.hub.test(
                &bed.scope,
                &acme,
                crate::taxonomy::TestDepth::Catalog,
                None
            ));
            assert!(
                futures::poll!(test.as_mut()).is_pending(),
                "waits for the move"
            );
            held.release();
            test.await
        };
        let (edited, tested) = tokio::join!(edit, others);
        edited.unwrap();
        tested.unwrap();
        for request in bed.ports.http.requests_from(0) {
            assert!(request.url.starts_with(NEW), "{}", request.url);
            assert!(
                !request.carried(&Secret::new(K_OLD)) && !request.carried(&Secret::new(K_HOST)),
                "an old credential reached {}",
                request.url
            );
            assert!(request.carried(&Secret::new(K_NEW)));
        }
        assert!(bed.ports.http.request_count() > 0);
    }
}

#[tokio::test]
async fn ops_a_store_that_commits_the_move_and_then_fails_is_read_back_not_assumed_away() {
    // The record commit reaches the store and the answer is lost: the record is
    // at the new origin. The edit must carry on from there (it did move), not
    // put the old key back beside the new origin.
    let (bed, _spy) = acme_bed().await;
    let _fault = bed.ports.config.hold(Hold::after(Call::Save).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap();
    assert_eq!(state_of(&bed).await, (NEW.to_string(), Some(K_NEW.into())));
    assert!(enabled_of(&bed).await);
}

#[tokio::test]
async fn ops_a_move_whose_outcome_cannot_be_read_back_leaves_the_slot_empty() {
    // The commit's answer is lost and so is the read that would say where the
    // record is: the old key must not be restored beside a record that may be at
    // the new origin.
    let (bed, _spy) = acme_bed().await;
    let _lost = bed.ports.config.hold(Hold::after(Call::Save).fail());
    // Loads: the first read (#0), the re-read under the lock (#1), the
    // transaction's (#2), then the read-back (#3).
    let _blind = bed
        .ports
        .config
        .hold(Hold::before(Call::Load).skip(3).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    let (base, key) = state_of(&bed).await;
    assert_eq!((base.as_str(), key), (NEW, None));
}

#[tokio::test]
async fn ops_a_delete_that_commits_and_then_fails_still_gets_the_old_key_back() {
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _fault = bed
        .ports
        .credentials
        .hold(Hold::after(Call::Delete).slot(&slot).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    assert_eq!(state_of(&bed).await, (OLD.to_string(), Some(K_OLD.into())));
}

#[tokio::test]
async fn ops_a_disable_made_during_an_origin_move_is_applied_after_it_not_overwritten() {
    let (bed, _spy) = acme_bed().await;
    let acme = slug("acme");
    let mut held = bed.ports.config.hold(Hold::after(Call::Save));
    let edit = bed.hub.edit(&bed.scope, &acme, move_patch());
    let disable = async {
        held.reached().await;
        let mut off =
            std::pin::pin!(
                bed.hub
                    .set_enabled(&bed.scope, &acme, false, Confirm::in_use())
            );
        assert!(
            futures::poll!(off.as_mut()).is_pending(),
            "the switch waits for the move"
        );
        held.release();
        off.await
    };
    let (edited, disabled) = tokio::join!(edit, disable);
    edited.unwrap();
    disabled.unwrap();
    assert!(
        !enabled_of(&bed).await,
        "the operator's disable is the last word"
    );
}

#[tokio::test]
async fn ops_a_move_that_cannot_switch_the_provider_on_still_announces_the_change() {
    let (bed, _spy) = acme_bed().await;
    let _fault = bed
        .ports
        .config
        .hold(Hold::before(Call::Save).skip(1).fail());
    bed.ports.events.drain();
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap();
    let events = bed.ports.events.events();
    // Once each: the change is announced by the finished edit, not also by the
    // step that failed.
    let count =
        |wanted: fn(&crate::ports::HubEvent) -> bool| events.iter().filter(|e| wanted(e)).count();
    assert_eq!(
        count(|e| matches!(e, crate::ports::HubEvent::ProviderEdited { .. })),
        1,
        "{events:?}"
    );
    assert_eq!(
        count(|e| matches!(e, crate::ports::HubEvent::KeyChanged { .. })),
        1,
        "{events:?}"
    );
}

#[tokio::test]
async fn ops_an_add_whose_provider_another_writer_moved_before_its_key_landed_stores_no_key() {
    // Round 3: the id still matched, so the add wrote a key entered for the
    // endpoint it was added at over the provider another writer had just moved.
    let bed = Bed::new();
    let draft = ProviderDraft::new("custom")
        .with_label("Acme")
        .with_base_url(OLD)
        .with_key(Secret::new(K_OLD))
        .with_model(model("m"));
    let acme = slug("acme");
    let mut held = bed.ports.config.hold(Hold::after(Call::Save));
    let add = bed.hub.add(&bed.scope, draft);
    let mover = async {
        held.reached().await;
        bed.hub
            .edit(&bed.scope, &acme, ProviderPatch::new().base_url(NEW))
            .await
            .unwrap();
        held.release();
    };
    let (added, ()) = tokio::join!(add, mover);
    assert!(matches!(added, Err(HubError::Conflict)), "{added:?}");
    assert_eq!(state_of(&bed).await, (NEW.to_string(), None));
    assert!(
        bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, crate::ports::HubEvent::ProviderAdded { .. })),
        "the record was created, so it is announced"
    );
}

#[tokio::test]
async fn ops_a_connect_whose_provider_was_edited_before_its_check_is_not_undone() {
    // Round 3: connect treated the Conflict as its own failure and undid the add,
    // deleting the edit another writer had just made.
    let bed = Bed::new();
    let draft = ProviderDraft::new("custom")
        .with_label("Acme")
        .with_base_url(OLD)
        .with_key(Secret::new(K_OLD))
        .with_model(model("m"));
    let acme = slug("acme");
    // The last thing the add does, after its lock is released.
    let mut held = bed.ports.health.hold(Hold::before(Call::Forget));
    let connect = bed
        .hub
        .connect(&bed.scope, draft, ConnectOptions::default());
    // The parked `forget` holds the health tracker's lock for this provider, so the
    // edit runs as far as its own `forget` (its record, key and flag are all
    // committed by then) and is finished after the connect is released.
    let mut edit = std::pin::pin!(bed.hub.edit(&bed.scope, &acme, move_patch()));
    let mover = async {
        held.reached().await;
        assert!(futures::poll!(edit.as_mut()).is_pending());
        held.release();
    };
    let (connected, ()) = tokio::join!(connect, mover);
    edit.await.unwrap();
    assert!(
        matches!(connected, Err(HubError::Conflict)),
        "{connected:?}"
    );
    assert_eq!(state_of(&bed).await, (NEW.to_string(), Some(K_NEW.into())));
    assert_eq!(bed.ports.http.request_count(), 0, "nothing was sent");
}

#[tokio::test]
async fn ops_a_probe_of_a_provider_removed_while_it_waited_reports_not_found() {
    let (bed, _spy) = acme_bed().await;
    let acme = slug("acme");
    let mut held = bed.ports.config.hold(Hold::after(Call::Load));
    let test = bed
        .hub
        .test(&bed.scope, &acme, crate::taxonomy::TestDepth::Catalog, None);
    let removal = async {
        held.reached().await;
        bed.hub
            .remove(&bed.scope, &acme, Confirm::in_use())
            .await
            .unwrap();
        held.release();
    };
    let (tested, ()) = tokio::join!(test, removal);
    assert!(matches!(tested, Err(HubError::NotFound(_))), "{tested:?}");
}

#[tokio::test]
async fn ops_a_removal_that_waited_while_the_slug_was_re_added_does_not_delete_the_new_provider() {
    let (bed, _spy) = acme_bed().await;
    let acme = slug("acme");
    // The removal parks holding its lock; a second removal + add of the slug must
    // wait; the first removal then finds ITS provider gone. Stage the opposite: a
    // removal that read the record, then loses the lock to a remove+add.
    let mut held = bed.ports.config.hold(Hold::after(Call::Load));
    let stale_removal = bed.hub.remove(&bed.scope, &acme, Confirm::in_use());
    let replacement = async {
        held.reached().await;
        bed.hub
            .remove(&bed.scope, &acme, Confirm::in_use())
            .await
            .unwrap();
        bed.hub
            .add(
                &bed.scope,
                ProviderDraft::new("custom")
                    .with_label("Acme")
                    .with_base_url(OLD)
                    .with_key(Secret::new(K_NEW))
                    .with_model(model("m")),
            )
            .await
            .unwrap();
        held.release();
    };
    let (removed, ()) = tokio::join!(stale_removal, replacement);
    assert!(matches!(removed, Err(HubError::Conflict)), "{removed:?}");
    assert_eq!(state_of(&bed).await, (OLD.to_string(), Some(K_NEW.into())));
}

#[tokio::test]
async fn ops_an_undone_origin_move_puts_the_label_and_model_back_too() {
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _fault = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    bed.hub
        .edit(
            &bed.scope,
            &slug("acme"),
            move_patch().label("Renamed").model(model("other")),
        )
        .await
        .unwrap_err();
    let status = bed.hub.status(&bed.scope).await.unwrap();
    let record = &status
        .providers
        .iter()
        .find(|p| p.view.record.slug == slug("acme"))
        .unwrap()
        .view
        .record;
    assert_eq!(record.label, "Acme");
    assert_eq!(record.model.as_ref().map(ModelId::as_str), Some("m"));
    assert_eq!(record.base_url, OLD);
}

#[tokio::test]
async fn ops_an_edit_that_waited_while_the_slug_was_re_added_does_not_touch_the_new_provider() {
    // Round 4: the edit validated its patch against one provider and, after the
    // lock, must not apply it to another that took the slug.
    let (bed, _spy) = acme_bed().await;
    let acme = slug("acme");
    let mut held = bed.ports.config.hold(Hold::after(Call::Load));
    let stale_edit = bed.hub.edit(&bed.scope, &acme, move_patch());
    let replacement = async {
        held.reached().await;
        bed.hub
            .remove(&bed.scope, &acme, Confirm::in_use())
            .await
            .unwrap();
        bed.hub
            .add(
                &bed.scope,
                ProviderDraft::new("custom")
                    .with_label("Acme")
                    .with_base_url(OLD)
                    .with_key(Secret::new("sk-not-a-real-key-second"))
                    .with_model(model("m")),
            )
            .await
            .unwrap();
        held.release();
    };
    let (edited, ()) = tokio::join!(stale_edit, replacement);
    assert!(matches!(edited, Err(HubError::Conflict)), "{edited:?}");
    assert_eq!(
        state_of(&bed).await,
        (OLD.to_string(), Some("sk-not-a-real-key-second".into()))
    );
}

#[tokio::test]
async fn ops_a_key_set_and_a_disable_that_waited_while_the_slug_was_re_added_do_not_touch_the_new_provider()
 {
    let (bed, _spy) = acme_bed().await;
    let acme = slug("acme");
    let readd = |bed: &Bed| {
        let hub = bed.hub.clone();
        let scope = bed.scope.clone();
        async move {
            hub.remove(&scope, &slug("acme"), Confirm::in_use())
                .await
                .unwrap();
            hub.add(
                &scope,
                ProviderDraft::new("custom")
                    .with_label("Acme")
                    .with_base_url(OLD)
                    .with_key(Secret::new("sk-not-a-real-key-second"))
                    .with_model(model("m")),
            )
            .await
            .unwrap();
        }
    };
    let mut held = bed.ports.config.hold(Hold::after(Call::Load));
    let set = bed
        .hub
        .set_key(&bed.scope, &acme, Secret::new("sk-not-a-real-key-late"));
    let replacement = async {
        held.reached().await;
        readd(&bed).await;
        held.release();
    };
    let (set, ()) = tokio::join!(set, replacement);
    assert!(matches!(set, Err(HubError::Conflict)), "{set:?}");
    assert_eq!(
        bed.key_of("acme").await.as_deref(),
        Some("sk-not-a-real-key-second")
    );

    let mut held = bed.ports.config.hold(Hold::after(Call::Load));
    let off = bed
        .hub
        .set_enabled(&bed.scope, &acme, false, Confirm::in_use());
    let replacement = async {
        held.reached().await;
        readd(&bed).await;
        held.release();
    };
    let (off, ()) = tokio::join!(off, replacement);
    assert!(matches!(off, Err(HubError::Conflict)), "{off:?}");
    assert!(enabled_of(&bed).await, "the new provider is still on");
}

#[tokio::test]
async fn ops_an_undone_origin_move_leaves_a_label_another_writer_set_meanwhile() {
    let (bed, _spy) = acme_bed().await;
    let acme = slug("acme");
    let slot = acme.key_slot();
    // Park the new key's write; run a label edit (no lock) in the gap; make the
    // write fail; let the undo's slot cleanup through once the store is healed.
    let mut write = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot));
    let mut cleanup = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Delete).slot(&slot).skip(1));
    let moving = bed.hub.edit(&bed.scope, &acme, move_patch().label("Moved"));
    let meanwhile = async {
        write.reached().await;
        bed.hub
            .edit(
                &bed.scope,
                &acme,
                ProviderPatch::new().label("Someone else"),
            )
            .await
            .unwrap();
        bed.ports
            .credentials
            .inject(crate::ports::memory::CredentialFault::Write);
        write.release();
        cleanup.reached().await;
        bed.ports.credentials.heal();
        cleanup.release();
    };
    let (moved, ()) = tokio::join!(moving, meanwhile);
    moved.unwrap_err();
    let status = bed.hub.status(&bed.scope).await.unwrap();
    let record = &status
        .providers
        .iter()
        .find(|p| p.view.record.slug == acme)
        .unwrap()
        .view
        .record;
    assert_eq!(record.label, "Someone else", "not overwritten by the undo");
    assert_eq!(record.base_url, OLD);
    assert_eq!(bed.key_of("acme").await.as_deref(), Some(K_OLD));
}

#[tokio::test]
async fn ops_a_rejected_connect_never_undoes_an_edit_another_writer_made_meanwhile() {
    // Round 5: while a probe is in flight the provider lock is free, so another
    // writer can move the provider; the connect's undo then removed it and its
    // key. Park the connect at each of its config reads in turn, run the edit
    // there, and require that whatever the edit made survives.
    for n in 0..16usize {
        let bed = Bed::new();
        for base in [OLD, NEW] {
            bed.ports.http.route(
                crate::testkit::Match::prefix(base),
                crate::testkit::Scripted::json(
                    401,
                    &serde_json::json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
                ),
            );
        }
        let draft = ProviderDraft::new("custom")
            .with_label("Acme")
            .with_base_url(OLD)
            .with_key(Secret::new(K_OLD))
            .with_model(model("m"));
        let acme = slug("acme");
        let mut held = bed.ports.config.hold(Hold::before(Call::Load).skip(n));
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let (go_tx, go_rx) = tokio::sync::oneshot::channel::<()>();
        let connect = async {
            let result = bed
                .hub
                .connect(&bed.scope, draft, ConnectOptions::default())
                .await;
            let _ = done_tx.send(());
            result
        };
        // The edit starts once the connect is parked and then runs alongside it:
        // if the connect holds the provider's lock it queues behind it, and the
        // lock is handed over when the connect lets go.
        let edit = async {
            if go_rx.await.is_ok() {
                Some(bed.hub.edit(&bed.scope, &acme, move_patch()).await)
            } else {
                None
            }
        };
        let mover = async {
            tokio::select! {
                () = held.reached() => {
                    let _ = go_tx.send(());
                    tokio::task::yield_now().await;
                    held.release();
                }
                _ = done_rx => {
                    drop(go_tx);
                }
            }
        };
        let (connected, edited, ()) = tokio::join!(connect, edit, mover);
        drop(held);
        let _ = connected;
        let edited = edited.unwrap_or(Err(HubError::Conflict));
        if edited.is_ok() {
            // The edit went through: what it made is still there, with its key.
            assert_eq!(
                state_of(&bed).await,
                (NEW.to_string(), Some(K_NEW.into())),
                "park point {n}"
            );
        }
    }
}

#[tokio::test]
async fn ops_an_undo_whose_record_removal_failed_does_not_announce_a_removal() {
    // Round 6: `ProviderRemoved` was emitted whatever the removal did.
    let bed = Bed::new();
    bed.openai_rejects_key();
    // Saves: the add's record (#0), then the undo's removal (#1).
    let _fault = bed
        .ports
        .config
        .hold(Hold::before(Call::Save).skip(1).fail());
    bed.hub
        .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
        .await
        .unwrap_err();
    assert!(
        !bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, crate::ports::HubEvent::ProviderRemoved { .. })),
        "the record is still there"
    );
    assert!(
        bed.hub
            .status(&bed.scope)
            .await
            .unwrap()
            .providers
            .iter()
            .any(|p| p.view.record.slug == slug("openai"))
    );
}

#[tokio::test]
async fn ops_a_key_write_that_commits_and_then_reports_failure_is_put_back() {
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _fault = bed
        .ports
        .credentials
        .hold(Hold::after(Call::Set).slot(&slot).fail());
    bed.hub
        .set_key(&bed.scope, &slug("acme"), Secret::new(K_NEW))
        .await
        .unwrap_err();
    assert_eq!(bed.key_of("acme").await.as_deref(), Some(K_OLD));
    // And when the confirmation read after a good write fails, the key is stored
    // and announced.
    bed.ports.events.drain();
    // Loads: the first read (#0), the re-read under the lock (#1), the read after
    // the write (#2).
    let _blind = bed
        .ports
        .config
        .hold(Hold::before(Call::Load).skip(2).fail());
    bed.hub
        .set_key(&bed.scope, &slug("acme"), Secret::new(K_NEW))
        .await
        .unwrap_err();
    assert_eq!(bed.key_of("acme").await.as_deref(), Some(K_NEW));
    assert!(
        bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, crate::ports::HubEvent::KeyChanged { present: true, .. }))
    );
}

#[tokio::test]
async fn ops_a_key_that_cannot_be_put_back_is_announced_as_what_the_slot_holds() {
    // remove: the key is deleted, the record's removal fails, and putting the key
    // back fails too. The record stays with no key, and the event says so.
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _save = bed.ports.config.hold(Hold::before(Call::Save).fail());
    let _restore = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    bed.ports.events.drain();
    bed.hub
        .remove(&bed.scope, &slug("acme"), Confirm::in_use())
        .await
        .unwrap_err();
    assert_eq!(bed.key_of("acme").await, None);
    assert!(
        bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, crate::ports::HubEvent::KeyChanged { present: false, .. })),
        "{:?}",
        bed.ports.events.events()
    );
}

#[tokio::test]
async fn ops_a_move_refused_because_another_hub_moved_the_record_does_not_restore_the_key_beside_it()
 {
    // Round 6: the old key was put back beside a record another hub had moved to
    // a third origin. Only a record verifiably at the endpoint the key was
    // entered for gets it back.
    let (bed, _spy) = acme_bed().await;
    let acme = slug("acme");
    // Loads: the first read (#0), the re-read under the lock (#1), the
    // transaction's (#2).
    let mut held = bed.ports.config.hold(Hold::before(Call::Load).skip(2));
    let edit = bed.hub.edit(&bed.scope, &acme, move_patch());
    let other_hub = async {
        held.reached().await;
        let mut doc: serde_json::Value =
            serde_json::from_str(&bed.ports.config.raw(&bed.scope).unwrap()).unwrap();
        for row in doc["providers"].as_array_mut().unwrap() {
            if row["slug"] == "acme" {
                row["base_url"] = "https://llm.third.test/v1".into();
            }
        }
        bed.ports.config.put_raw(&bed.scope, doc.to_string());
        held.release();
    };
    let (edited, ()) = tokio::join!(edit, other_hub);
    assert!(matches!(edited, Err(HubError::Conflict)), "{edited:?}");
    let (base, key) = state_of(&bed).await;
    assert_eq!(base, "https://llm.third.test/v1");
    assert_eq!(
        key, None,
        "the old key is not put back beside a third origin"
    );
}

#[tokio::test]
async fn ops_a_removal_the_store_committed_and_then_reported_failed_does_not_bring_the_key_back() {
    // Round 7: the key was restored beside a record that was in fact gone, for the
    // next provider of that slug to find.
    let (bed, _spy) = acme_bed().await;
    let mut fault = bed.ports.config.hold(Hold::after(Call::Save).fail());
    // The removal did commit, so it is reported as done (and announced), not as a
    // failure the caller would retry.
    let removed = bed
        .hub
        .remove(&bed.scope, &slug("acme"), Confirm::in_use())
        .await;
    assert!(futures::poll!(std::pin::pin!(fault.reached())).is_ready());
    assert!(removed.is_ok(), "{removed:?}");
    assert!(
        bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, crate::ports::HubEvent::ProviderRemoved { .. }))
    );
    assert!(
        !bed.hub
            .status(&bed.scope)
            .await
            .unwrap()
            .providers
            .iter()
            .any(|p| p.view.record.slug == slug("acme")),
        "the removal did commit"
    );
    assert_eq!(bed.key_of("acme").await, None, "and its key stays gone");
}

#[tokio::test]
async fn ops_an_add_that_fails_after_its_record_committed_is_announced_as_an_add_and_its_undoing() {
    let bed = Bed::new();
    let slot = slug("acme").key_slot();
    let _fault = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("custom")
                .with_label("Acme")
                .with_base_url(OLD)
                .with_key(Secret::new(K_OLD))
                .with_model(model("m")),
        )
        .await
        .unwrap_err();
    let events = bed.ports.events.events();
    let position = |wanted: fn(&crate::ports::HubEvent) -> bool| events.iter().position(wanted);
    let added = position(|e| matches!(e, crate::ports::HubEvent::ProviderAdded { .. }));
    let removed = position(|e| matches!(e, crate::ports::HubEvent::ProviderRemoved { .. }));
    assert!(
        added.is_some() && removed.is_some() && added < removed,
        "{events:?}"
    );
}

#[tokio::test]
async fn ops_an_undo_whose_move_back_commits_and_then_reports_failure_still_gets_the_old_key_back()
{
    let (bed, _spy) = acme_bed().await;
    let slot = slug("acme").key_slot();
    let _write = bed
        .ports
        .credentials
        .hold(Hold::before(Call::Set).slot(&slot).fail());
    // Saves: the move (#0), then the undo's move back (#1): it lands, then errors.
    let mut back = bed
        .ports
        .config
        .hold(Hold::after(Call::Save).skip(1).fail());
    bed.hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap_err();
    assert!(futures::poll!(std::pin::pin!(back.reached())).is_ready());
    assert_eq!(state_of(&bed).await, (OLD.to_string(), Some(K_OLD.into())));
    assert!(enabled_of(&bed).await);
}

#[tokio::test]
async fn ops_an_undo_whose_record_removal_commits_and_then_reports_failure_is_announced() {
    let bed = Bed::new();
    bed.openai_rejects_key();
    // Saves: the add's record (#0), then the undo's removal (#1): it lands, then errors.
    let mut fault = bed
        .ports
        .config
        .hold(Hold::after(Call::Save).skip(1).fail());
    bed.hub
        .connect(&bed.scope, bed.openai_draft(), ConnectOptions::default())
        .await
        .unwrap_err();
    // The fault fired: the test is about that failure, not about a clean undo.
    assert!(futures::poll!(std::pin::pin!(fault.reached())).is_ready());
    assert!(
        bed.ports
            .events
            .events()
            .iter()
            .any(|e| matches!(e, crate::ports::HubEvent::ProviderRemoved { .. })),
        "the record is gone, so it is announced"
    );
}

#[tokio::test]
async fn ops_an_add_whose_provider_another_writer_only_re_pathed_still_gets_its_key() {
    // Same origin, different path: not a move (the key was entered for the origin).
    let bed = Bed::new();
    let draft = ProviderDraft::new("custom")
        .with_label("Acme")
        .with_base_url(OLD)
        .with_key(Secret::new(K_OLD))
        .with_model(model("m"));
    let acme = slug("acme");
    let mut held = bed.ports.config.hold(Hold::after(Call::Save));
    let add = bed.hub.add(&bed.scope, draft);
    let mover = async {
        held.reached().await;
        bed.hub
            .edit(
                &bed.scope,
                &acme,
                ProviderPatch::new().base_url("https://llm.acme.test/v2"),
            )
            .await
            .unwrap();
        held.release();
    };
    let (added, ()) = tokio::join!(add, mover);
    added.unwrap();
    assert_eq!(bed.key_of("acme").await.as_deref(), Some(K_OLD));
}

#[tokio::test]
async fn ops_a_switch_back_on_the_store_committed_and_then_reported_failed_is_not_a_warning() {
    let (bed, _spy) = acme_bed().await;
    // Saves: the move (#0), then switching back on (#1): it lands, then errors.
    let mut fault = bed
        .ports
        .config
        .hold(Hold::after(Call::Save).skip(1).fail());
    let mutation = bed
        .hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap();
    assert!(futures::poll!(std::pin::pin!(fault.reached())).is_ready());
    assert_eq!(mutation.status, crate::hub::MutationStatus::Saved);
    assert!(enabled_of(&bed).await);
}

#[tokio::test]
async fn ops_a_warning_on_a_moved_provider_survives_the_read_back_failing_too() {
    let (bed, _spy) = acme_bed().await;
    // The switch back on fails (Save #1), and so do the reads that follow it.
    let mut off = bed
        .ports
        .config
        .hold(Hold::before(Call::Save).skip(1).fail());
    // Loads: first read #0, lock re-read #1, the move's #2, the switch back on's
    // #3; then the switch-back check's read-back (#4) and finish_edit's (#5).
    let _blind1 = bed
        .ports
        .config
        .hold(Hold::before(Call::Load).skip(4).fail());
    let _blind2 = bed
        .ports
        .config
        .hold(Hold::before(Call::Load).skip(4).fail());
    let mutation = bed
        .hub
        .edit(&bed.scope, &slug("acme"), move_patch())
        .await
        .unwrap();
    assert!(futures::poll!(std::pin::pin!(off.reached())).is_ready());
    assert_eq!(
        mutation.status,
        crate::hub::MutationStatus::SavedWithWarning
    );
    assert!(mutation.note.contains("could not be switched back on"));
    assert!(
        mutation.record.is_none(),
        "the read-back failed, so no view"
    );
}

#[tokio::test]
async fn ops_a_removal_whose_record_another_writer_removed_meanwhile_reports_not_found_at_every_point()
 {
    // Whichever config read the removal is parked at, a record another writer
    // removes there is reported as not found: never as a removal this call made,
    // and never with its key put back.
    for n in 0..8usize {
        let (bed, _spy) = acme_bed().await;
        let acme = slug("acme");
        let mut held = bed.ports.config.hold(Hold::before(Call::Load).skip(n));
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let removal = async {
            let result = bed.hub.remove(&bed.scope, &acme, Confirm::in_use()).await;
            let _ = done_tx.send(());
            result
        };
        let other = async {
            tokio::select! {
                () = held.reached() => {
                    let mut doc: serde_json::Value =
                        serde_json::from_str(&bed.ports.config.raw(&bed.scope).unwrap()).unwrap();
                    doc["providers"]
                        .as_array_mut()
                        .unwrap()
                        .retain(|row| row["slug"] != "acme");
                    bed.ports.config.put_raw(&bed.scope, doc.to_string());
                    held.release();
                }
                _ = done_rx => {}
            }
        };
        let (removed, ()) = tokio::join!(removal, other);
        drop(held);
        let removed_by_us = removed.is_ok();
        assert!(
            !removed_by_us || bed.key_of("acme").await.is_none(),
            "park point {n}: {removed:?}"
        );
        if let Err(error) = &removed {
            assert!(
                matches!(error, HubError::NotFound(_)),
                "park point {n}: {error:?}"
            );
        }
    }
}
