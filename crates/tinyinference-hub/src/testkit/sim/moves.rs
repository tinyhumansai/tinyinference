//! Origin moves and kept models: the actions behind invariant 11 (a credential
//! is never sent to an origin other than the one it was entered for).

use std::sync::Arc;

use tinyinference_llm::model::{ChatModel, ModelRequest};

use super::action::StepResult;
use super::world::WORLD;
use super::{RaceStats, ScenarioRunner, is_infra};
use crate::hub::ProviderPatch;
use crate::ids::{ModelId, ScopeKey, Slug};
use crate::policy::same_origin;
use crate::ports::memory::{Call, Hold};
use crate::route::{ProviderRoute, TurnQuery};

/// How many places a race can park something at: the edit's own store calls
/// (see [`race_hold`]) and, past them, the kept model's credential read.
pub(crate) const RACE_POINTS: usize = 16;

/// The first race point that parks the **kept model** instead of the edit.
const MODEL_POINTS: usize = 14;

/// The `n`-th place an edit of one provider can be parked: every store call it
/// makes, before and after it takes effect. `true` is a configuration-store
/// call, `false` a credential-store call.
fn race_hold(n: u8, slot: &str) -> (bool, Hold) {
    let cfg = |hold| (true, hold);
    let cred = |hold: Hold| (false, hold.slot(slot));
    match usize::from(n) % RACE_POINTS {
        0 => cfg(Hold::before(Call::Save)),
        1 => cfg(Hold::after(Call::Save)),
        2 => cfg(Hold::before(Call::Load).skip(1)),
        3 => cfg(Hold::before(Call::Load).skip(2)),
        4 => cfg(Hold::after(Call::Load).skip(2)),
        5 => cfg(Hold::after(Call::Load).skip(3)),
        6 => cred(Hold::before(Call::Set)),
        7 => cred(Hold::after(Call::Set)),
        8 => cred(Hold::before(Call::Delete)),
        9 => cred(Hold::after(Call::Delete)),
        10 => cred(Hold::before(Call::Get)),
        11 => cred(Hold::after(Call::Get)),
        12 => cred(Hold::before(Call::Get).skip(1)),
        _ => cred(Hold::after(Call::Get).skip(1)),
    }
}

fn model_id() -> ModelId {
    ModelId::parse("m-a").expect("a constant model id is valid")
}

impl ScenarioRunner {
    /// The index of the provider with an editable endpoint.
    fn movable() -> Option<usize> {
        WORLD.iter().position(|w| w.alt_base.is_some())
    }

    /// The endpoint the scope's record for `prov` has now (the world's own when
    /// there is none): what a key entered now is entered for.
    pub(crate) async fn record_origin(&self, scope: usize, prov: usize) -> String {
        let world = &WORLD[prov % WORLD.len()];
        let scope = &self.scopes[scope % self.scopes.len()];
        match self.hub.read_config(scope).await {
            Ok(config) => config
                .provider(&Slug::parse(world.slug).expect("a constant slug is valid"))
                .map_or_else(|| world.base.to_string(), |r| r.base_url.clone()),
            Err(_) => world.base.to_string(),
        }
    }

    /// The other origin of the movable provider, given where its record is.
    async fn other_origin(&self, scope: usize) -> Option<(usize, String)> {
        let prov = Self::movable()?;
        let world = &WORLD[prov];
        let now = self.record_origin(scope, prov).await;
        let alt = world.alt_base?;
        Some((
            prov,
            if same_origin(&now, world.base) {
                alt.to_string()
            } else {
                world.base.to_string()
            },
        ))
    }

    pub(crate) async fn move_origin(&mut self, scope: usize, rotate: bool) -> StepResult {
        let Some((prov, target)) = self.other_origin(scope).await else {
            return StepResult::ok(String::from("no movable provider"));
        };
        let scope_key = self.scopes[scope % self.scopes.len()].clone();
        let mut patch = ProviderPatch::new().base_url(target.clone());
        if rotate {
            patch = patch.key(self.new_key(&target));
        }
        let slug = Slug::parse(WORLD[prov].slug).expect("a constant slug is valid");
        match self.hub.edit(&scope_key, &slug, patch).await {
            Ok(mutation) => {
                let mut result = StepResult::ok(format!("{mutation:?}"));
                result.changed = mutation.status != crate::hub::MutationStatus::Unchanged;
                result
            }
            Err(error) => StepResult::from_error(&error),
        }
    }

    pub(crate) async fn keep(&mut self, scope: usize) -> StepResult {
        let scope_key = self.scopes[scope % self.scopes.len()].clone();
        let turn = match self
            .hub
            .resolve_for_turn(&scope_key, &TurnQuery::new())
            .await
        {
            Ok(turn) => turn,
            Err(error) => return StepResult::from_error(&error),
        };
        match self.hub.chat_model(&scope_key, &turn).await {
            Ok(model) => {
                let index = scope % self.kept.len();
                self.kept[index] = Some(model);
                StepResult::ok(format!("kept {turn:?}"))
            }
            Err(error) => StepResult::from_error(&error),
        }
    }

    pub(crate) async fn use_kept(&mut self, scope: usize) -> StepResult {
        let index = scope % self.kept.len();
        let Some(model) = self.kept[index].clone() else {
            return StepResult::ok(String::from("nothing kept"));
        };
        Self::send(&model).await
    }

    async fn send(model: &Arc<dyn ChatModel<()>>) -> StepResult {
        match model.invoke(&(), ModelRequest::default()).await {
            Ok(_) => StepResult::ok(String::from("sent")),
            Err(error) => {
                // A refusal to send (a stale route, no key) is the point; the
                // text is scanned for secrets like any other.
                let mut result = StepResult::ok(format!("refused: {error}"));
                result.ok = false;
                result
            }
        }
    }

    /// Uses the movable provider the two ways a host does, mid-edit: through the
    /// model it kept and through a fresh resolve.
    async fn use_models(&self, scope: &ScopeKey, prov: usize, kept: &Arc<dyn ChatModel<()>>) {
        let _ = Self::send(kept).await;
        let query = TurnQuery::new().with_override(
            ProviderRoute::provider(Slug::parse(WORLD[prov].slug).expect("a constant slug"))
                .with_model(model_id()),
        );
        if let Ok(turn) = self.hub.resolve_for_turn(scope, &query).await
            && let Ok(fresh) = self.hub.chat_model(scope, &turn).await
        {
            let _ = Self::send(&fresh).await;
        }
    }

    /// The operations that send a credential of their own: a probe and a
    /// listing, which read the record and then the key. They wait for an endpoint
    /// move of the same provider (it holds the provider's lock throughout).
    async fn probe_and_list(&self, scope: &ScopeKey, prov: usize) {
        let slug = Slug::parse(WORLD[prov].slug).expect("a constant slug is valid");
        let _ = self
            .hub
            .test(scope, &slug, crate::taxonomy::TestDepth::Catalog, None)
            .await;
        let _ = self.hub.list_models(scope, &slug, true).await;
    }

    /// Everything a host can do with the provider once nothing is parked.
    async fn use_it(&self, scope: &ScopeKey, prov: usize, kept: &Arc<dyn ChatModel<()>>) {
        self.use_models(scope, prov, kept).await;
        self.probe_and_list(scope, prov).await;
    }

    /// The mirror race: the kept model is parked at its credential read (before
    /// it, or after it) while the whole edit runs, then released to send with
    /// whatever it read.
    async fn race_model(
        &self,
        scope: &ScopeKey,
        prov: usize,
        slug: &Slug,
        kept: &Arc<dyn ChatModel<()>>,
        patch: ProviderPatch,
        at: u8,
    ) -> StepResult {
        let hold = if usize::from(at) % RACE_POINTS == MODEL_POINTS {
            Hold::before(Call::Get)
        } else {
            Hold::after(Call::Get)
        }
        .slot(slot_of(slug));
        let mut held = self.ports.credentials.hold(hold);
        let (sent_tx, sent_rx) = tokio::sync::oneshot::channel::<()>();
        let send = async {
            let result = Self::send(kept).await;
            let _ = sent_tx.send(());
            result
        };
        let edit_meanwhile = async {
            // A model that refuses before reading its credential never parks.
            tokio::select! {
                () = held.reached() => {
                    RaceStats::bump(&self.races.parked);
                }
                _ = sent_rx => {
                    // The hold was never reached; the edit's own credential reads
                    // must not be parked by it.
                    held.release();
                }
            }
            let edited = self.hub.edit(scope, slug, patch).await;
            held.release();
            edited
        };
        let (_, edited) = tokio::join!(send, edit_meanwhile);
        drop(held);
        self.use_it(scope, prov, kept).await;
        match edited {
            Ok(mutation) => {
                let mut result = StepResult::ok(format!("{mutation:?}"));
                result.changed = mutation.status != crate::hub::MutationStatus::Unchanged;
                if result.changed {
                    RaceStats::bump(&self.races.moved);
                }
                result
            }
            Err(error) => {
                let mut result = StepResult::from_error(&error);
                result.infra = is_infra(&error);
                result
            }
        }
    }

    pub(crate) async fn race_move(&mut self, scope: usize, at: u8, rotate: bool) -> StepResult {
        let Some((prov, target)) = self.other_origin(scope).await else {
            return StepResult::ok(String::from("no movable provider"));
        };
        let scope_key = self.scopes[scope % self.scopes.len()].clone();
        let slug = Slug::parse(WORLD[prov].slug).expect("a constant slug is valid");
        let usable = self
            .hub
            .read_config(&scope_key)
            .await
            .is_ok_and(|c| c.provider(&slug).is_some_and(|r| r.enabled));
        if !usable {
            return StepResult::ok(String::from("nothing to race"));
        }
        let query = TurnQuery::new()
            .with_override(ProviderRoute::provider(slug.clone()).with_model(model_id()));
        let kept = match self.hub.resolve_for_turn(&scope_key, &query).await {
            Ok(turn) => match self.hub.chat_model(&scope_key, &turn).await {
                Ok(model) => model,
                Err(error) => return StepResult::from_error(&error),
            },
            Err(error) => return StepResult::from_error(&error),
        };
        // The host has used it once before the edit.
        let _ = Self::send(&kept).await;
        RaceStats::bump(&self.races.attempted);

        let mut patch = ProviderPatch::new().base_url(target.clone());
        if rotate {
            patch = patch.key(self.new_key(&target));
        }
        if usize::from(at) % RACE_POINTS >= MODEL_POINTS {
            return self
                .race_model(&scope_key, prov, &slug, &kept, patch, at)
                .await;
        }
        let (on_config, hold) = race_hold(at, &slug.key_slot());
        let mut held = if on_config {
            self.ports.config.hold(hold)
        } else {
            self.ports.credentials.hold(hold)
        };
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
        let edit = async {
            let result = self.hub.edit(&scope_key, &slug, patch).await;
            let _ = done_tx.send(());
            result
        };
        let others = async {
            tokio::select! {
                () = held.reached() => {
                    RaceStats::bump(&self.races.parked);
                    self.use_models(&scope_key, prov, &kept).await;
                    // A probe and a listing wait for the parked edit's lock:
                    // start them, let them queue, then let the edit go.
                    let mut probes = std::pin::pin!(self.probe_and_list(&scope_key, prov));
                    let finished = tokio::select! {
                        biased;
                        () = &mut probes => true,
                        () = std::future::ready(()) => false,
                    };
                    held.release();
                    if !finished {
                        probes.await;
                    }
                }
                _ = done_rx => {}
            }
        };
        let (edited, ()) = tokio::join!(edit, others);
        // A hold that was never reached must not park the calls made from here.
        drop(held);
        self.use_it(&scope_key, prov, &kept).await;
        match edited {
            Ok(mutation) => {
                let mut result = StepResult::ok(format!("{mutation:?}"));
                result.changed = mutation.status != crate::hub::MutationStatus::Unchanged;
                if result.changed {
                    RaceStats::bump(&self.races.moved);
                }
                result
            }
            Err(error) => {
                let mut result = StepResult::from_error(&error);
                result.infra = is_infra(&error);
                result
            }
        }
    }
}

fn slot_of(slug: &Slug) -> String {
    slug.key_slot()
}
