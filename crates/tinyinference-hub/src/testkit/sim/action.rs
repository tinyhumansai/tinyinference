//! The actions a scenario takes and how the runner performs them.

use std::sync::atomic::Ordering;
use std::time::Duration;

use super::moves::RACE_POINTS;
use super::world::{Mode, WORLD};
use super::{ScenarioRunner, is_infra};
use crate::catalog::Freshness;
use crate::config::{ModelChoice, ProviderDraft};
use crate::error::{HubError, ProviderFailure, ReasonCode, Retry};
use crate::health::Outcome;
use crate::hub::{Confirm, ConnectOptions, ProviderPatch};
use crate::ids::{AgentKey, ModelId, Slug, WorkloadKey};
use crate::route::{ProviderRoute, TurnQuery};
use crate::taxonomy::TestDepth;

/// One thing a scenario does. Indexes name a scope (`0` or `1`) and a provider
/// of [`WORLD`].
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Add a provider and check it.
    Connect {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// Whether the draft carries a key.
        keyed: bool,
        /// Keep the row even if the check fails.
        add_anyway: bool,
        /// Ask to make it the default.
        make_default: bool,
        /// Check with a completion instead of a catalog read.
        completion: bool,
    },
    /// Add a provider without checking it.
    Add {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// Whether the draft carries a key.
        keyed: bool,
    },
    /// Edit a provider.
    Edit {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// Give it a new key.
        rotate: bool,
        /// Give it a new model (an index into the model names).
        model: Option<u8>,
    },
    /// Remove a provider.
    Remove {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// Confirm an in-use removal.
        confirm: bool,
    },
    /// Enable or disable a provider.
    SetEnabled {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// The new state.
        on: bool,
        /// Confirm an in-use disable.
        confirm: bool,
    },
    /// Store a new key.
    SetKey {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
    },
    /// Delete the stored key.
    ClearKey {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// Confirm an in-use clear.
        confirm: bool,
    },
    /// Set the default.
    SetDefault {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// Model name index.
        model: u8,
    },
    /// Clear the default.
    ClearDefault {
        /// Scope index.
        scope: usize,
    },
    /// Pin (or with `None`, unpin) an agent.
    Pin {
        /// Scope index.
        scope: usize,
        /// Agent index.
        agent: u8,
        /// Provider index, or `None` to unpin.
        prov: Option<usize>,
    },
    /// Route (or with `None`, unroute) a workload.
    SetRoute {
        /// Scope index.
        scope: usize,
        /// Workload index.
        workload: u8,
        /// Provider index, or `None` to remove the route.
        prov: Option<usize>,
    },
    /// Resolve a turn.
    Resolve {
        /// Scope index.
        scope: usize,
        /// Agent index, if the turn has one.
        agent: Option<u8>,
        /// Workload index, if the turn has one.
        workload: Option<u8>,
    },
    /// List a provider's models.
    List {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// Bypass the cache.
        refresh: bool,
    },
    /// Test a provider.
    Test {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// A completion instead of a catalog read.
        completion: bool,
    },
    /// Report a real turn's outcome.
    RecordOutcome {
        /// Scope index.
        scope: usize,
        /// Provider index.
        prov: usize,
        /// Whether the turn worked.
        ok: bool,
        /// Which failure, when it did not (an index into a fixed list).
        reason: u8,
    },
    /// Re-test what is down on a rejected key.
    RetestDown {
        /// Scope index.
        scope: usize,
    },
    /// Move the fake clock.
    Advance {
        /// Seconds.
        secs: u32,
    },
    /// Change how a provider answers.
    Flip {
        /// Provider index.
        prov: usize,
        /// The new mode.
        mode: Mode,
    },
    /// Sign the platform token in or out.
    ToggleSignedOut,
    /// Move the provider with an editable endpoint to its other origin,
    /// optionally entering a key for the new origin in the same edit.
    MoveOrigin {
        /// Scope index.
        scope: usize,
        /// Enter a new key with the move.
        rotate: bool,
    },
    /// Hold on to a chat model for the scope's default provider, as a host does
    /// between turns.
    Keep {
        /// Scope index.
        scope: usize,
    },
    /// Send a request through the model kept for the scope.
    UseKept {
        /// Scope index.
        scope: usize,
    },
    /// Move the editable-endpoint provider to its other origin **while** a kept
    /// model and a fresh resolve use it, with the edit parked at one of the
    /// points where it touches a store (deterministic interleaving).
    RaceMove {
        /// Scope index.
        scope: usize,
        /// Which store call the edit is parked at (taken modulo the table).
        at: u8,
        /// Enter a new key with the move.
        rotate: bool,
    },
}

/// What a step did, for the invariants and the caller.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct StepResult {
    /// Whether the operation succeeded.
    pub ok: bool,
    /// The failure's reason code, when it did not.
    pub reason: Option<ReasonCode>,
    /// A rendering of the result (the error or the value), scanned for secrets.
    pub text: String,
    /// The provider index a successful resolve chose.
    pub resolved: Option<usize>,
    /// What a successful list read: how fresh and which models.
    pub listed: Option<(Freshness, Vec<String>)>,
    /// Whether the failure was one only an injected fault can cause.
    pub infra: bool,
    /// Whether a mutation changed anything (`false` for an `Unchanged` one and
    /// for a failure).
    pub changed: bool,
}

impl StepResult {
    pub(crate) fn ok(text: String) -> Self {
        Self {
            ok: true,
            reason: None,
            text,
            resolved: None,
            listed: None,
            infra: false,
            changed: false,
        }
    }

    fn mutated(mutation: &crate::hub::Mutation) -> Self {
        let mut result = Self::ok(format!("{mutation:?}"));
        result.changed = mutation.status != crate::hub::MutationStatus::Unchanged;
        result
    }

    pub(crate) fn from_error(error: &HubError) -> Self {
        Self::err(error)
    }

    fn err(error: &HubError) -> Self {
        Self {
            ok: false,
            reason: Some(error.reason()),
            text: format!("{error} | {error:?}"),
            resolved: None,
            listed: None,
            infra: is_infra(error),
            changed: false,
        }
    }
}

const MODELS: [&str; 3] = ["m-a", "m-b", "m-c"];
const REASONS: [ReasonCode; 5] = [
    ReasonCode::Timeout,
    ReasonCode::Auth,
    ReasonCode::RateLimited,
    ReasonCode::Endpoint,
    ReasonCode::Unknown,
];

fn model(index: u8) -> ModelId {
    ModelId::parse(MODELS[usize::from(index) % MODELS.len()]).expect("a constant model id is valid")
}

impl ScenarioRunner {
    pub(crate) fn random_action(&mut self) -> Action {
        let scope = self.rng.below(self.scopes.len());
        let prov = self.rng.below(WORLD.len());
        let flag = |rng: &mut super::Rng, p: f64| rng.chance(p);
        // The first hundred slots are the original mix; the twelve after them are
        // the origin-move and kept-model actions added in run 4. The modulus
        // changed with them, so a seed recorded under an earlier mix does not
        // replay the same trace (the seed file says which mix its seeds are for).
        match self.rng.below(112) {
            0..=17 => Action::Connect {
                scope,
                prov,
                keyed: flag(&mut self.rng, 0.9),
                add_anyway: flag(&mut self.rng, 0.15),
                make_default: flag(&mut self.rng, 0.1),
                completion: flag(&mut self.rng, 0.15),
            },
            18..=22 => Action::Add {
                scope,
                prov,
                keyed: flag(&mut self.rng, 0.8),
            },
            23..=29 => Action::Edit {
                scope,
                prov,
                rotate: flag(&mut self.rng, 0.5),
                model: flag(&mut self.rng, 0.5)
                    .then(|| u8::try_from(self.rng.below(3)).unwrap_or(0)),
            },
            30..=35 => Action::Remove {
                scope,
                prov,
                confirm: flag(&mut self.rng, 0.5),
            },
            36..=41 => Action::SetEnabled {
                scope,
                prov,
                on: flag(&mut self.rng, 0.6),
                confirm: flag(&mut self.rng, 0.5),
            },
            42..=46 => Action::SetKey { scope, prov },
            47..=50 => Action::ClearKey {
                scope,
                prov,
                confirm: flag(&mut self.rng, 0.5),
            },
            51..=56 => Action::SetDefault {
                scope,
                prov,
                model: u8::try_from(self.rng.below(3)).unwrap_or(0),
            },
            57..=58 => Action::ClearDefault { scope },
            59..=62 => Action::Pin {
                scope,
                agent: u8::try_from(self.rng.below(2)).unwrap_or(0),
                prov: flag(&mut self.rng, 0.7).then_some(prov),
            },
            63..=65 => Action::SetRoute {
                scope,
                workload: u8::try_from(self.rng.below(2)).unwrap_or(0),
                prov: flag(&mut self.rng, 0.7).then_some(prov),
            },
            66..=76 => Action::Resolve {
                scope,
                agent: flag(&mut self.rng, 0.5)
                    .then(|| u8::try_from(self.rng.below(2)).unwrap_or(0)),
                workload: flag(&mut self.rng, 0.5)
                    .then(|| u8::try_from(self.rng.below(2)).unwrap_or(0)),
            },
            77..=85 => Action::List {
                scope,
                prov,
                refresh: flag(&mut self.rng, 0.3),
            },
            86..=88 => Action::Test {
                scope,
                prov,
                completion: flag(&mut self.rng, 0.4),
            },
            89..=91 => Action::RecordOutcome {
                scope,
                prov,
                ok: flag(&mut self.rng, 0.5),
                reason: u8::try_from(self.rng.below(REASONS.len())).unwrap_or(0),
            },
            92 => Action::RetestDown { scope },
            93..=95 => Action::Advance {
                secs: [30, 61, 301, 3601][self.rng.below(4)],
            },
            96..=98 => Action::Flip {
                prov,
                mode: Mode::ALL[self.rng.below(Mode::ALL.len())],
            },
            99 => Action::ToggleSignedOut,
            100..=103 => Action::MoveOrigin {
                scope,
                rotate: flag(&mut self.rng, 0.7),
            },
            104..=106 => Action::Keep { scope },
            107..=109 => Action::UseKept { scope },
            _ => Action::RaceMove {
                scope,
                at: u8::try_from(self.rng.below(RACE_POINTS)).unwrap_or(0),
                rotate: flag(&mut self.rng, 0.8),
            },
        }
    }

    fn draft(&mut self, prov: usize, keyed: bool, model_index: u8) -> ProviderDraft {
        let world = &WORLD[prov];
        let mut draft = ProviderDraft::new(world.kind).with_model(model(model_index));
        if let Some(label) = world.label {
            draft = draft.with_label(label);
        }
        if let Some(base) = world.draft_base {
            draft = draft.with_base_url(base);
        }
        if keyed && world.keyed {
            let origin = world.draft_base.unwrap_or(world.base);
            draft = draft.with_key(self.new_key(origin));
        }
        draft
    }

    /// Performs one action against the hub.
    pub(crate) async fn apply(&mut self, action: &Action) -> StepResult {
        let hub = self.hub.clone();
        let scope_of = |runner: &Self, i: usize| runner.scopes[i % runner.scopes.len()].clone();
        let slug_of = |prov: usize| -> Slug {
            Slug::parse(WORLD[prov % WORLD.len()].slug).expect("a constant slug is valid")
        };
        match action {
            Action::Connect {
                scope,
                prov,
                keyed,
                add_anyway,
                make_default,
                completion,
            } => {
                let scope = scope_of(self, *scope);
                let draft = self.draft(*prov, *keyed, 0);
                let mut options = ConnectOptions::default()
                    .add_anyway(*add_anyway)
                    .make_default(*make_default);
                if *completion {
                    options = options.depth(TestDepth::Completion);
                }
                match hub.connect(&scope, draft, options).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::Add { scope, prov, keyed } => {
                let scope = scope_of(self, *scope);
                let draft = self.draft(*prov, *keyed, 1);
                match hub.add(&scope, draft).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::Edit {
                scope,
                prov,
                rotate,
                model: m,
            } => {
                let origin = self.record_origin(*scope, *prov).await;
                let scope = scope_of(self, *scope);
                let mut patch = ProviderPatch::new();
                if *rotate {
                    patch = patch.key(self.new_key(&origin));
                }
                if let Some(m) = m {
                    patch = patch.model(model(*m));
                }
                match hub.edit(&scope, &slug_of(*prov), patch).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::Remove {
                scope,
                prov,
                confirm,
            } => {
                let scope = scope_of(self, *scope);
                let confirm = if *confirm {
                    Confirm::in_use()
                } else {
                    Confirm::no()
                };
                match hub.remove(&scope, &slug_of(*prov), confirm).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::SetEnabled {
                scope,
                prov,
                on,
                confirm,
            } => {
                let scope = scope_of(self, *scope);
                let confirm = if *confirm {
                    Confirm::in_use()
                } else {
                    Confirm::no()
                };
                match hub.set_enabled(&scope, &slug_of(*prov), *on, confirm).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::SetKey { scope, prov } => {
                let origin = self.record_origin(*scope, *prov).await;
                let scope = scope_of(self, *scope);
                let key = self.new_key(&origin);
                match hub.set_key(&scope, &slug_of(*prov), key).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::ClearKey {
                scope,
                prov,
                confirm,
            } => {
                let scope = scope_of(self, *scope);
                let confirm = if *confirm {
                    Confirm::in_use()
                } else {
                    Confirm::no()
                };
                match hub.clear_key(&scope, &slug_of(*prov), confirm).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::SetDefault {
                scope,
                prov,
                model: m,
            } => {
                let scope = scope_of(self, *scope);
                let choice = ModelChoice::new(slug_of(*prov), model(*m));
                match hub.set_default(&scope, choice).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::ClearDefault { scope } => {
                let scope = scope_of(self, *scope);
                match hub.clear_default(&scope).await {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::Pin { scope, agent, prov } => {
                let scope = scope_of(self, *scope);
                let choice = prov.map(|p| ModelChoice::new(slug_of(p), model(0)));
                match hub
                    .pin_agent(&scope, &AgentKey::new(format!("agent:{agent}")), choice)
                    .await
                {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::SetRoute {
                scope,
                workload,
                prov,
            } => {
                let scope = scope_of(self, *scope);
                let route = prov.map(|p| ProviderRoute::provider(slug_of(p)).with_model(model(2)));
                match hub
                    .set_workload_route(
                        &scope,
                        &WorkloadKey::new(format!("tier:{workload}")),
                        route,
                    )
                    .await
                {
                    Ok(mutation) => StepResult::mutated(&mutation),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::Resolve {
                scope,
                agent,
                workload,
            } => {
                let scope = scope_of(self, *scope);
                let mut query = TurnQuery::new();
                if let Some(agent) = agent {
                    query = query.with_agent(AgentKey::new(format!("agent:{agent}")));
                }
                if let Some(workload) = workload {
                    query = query.with_workload(WorkloadKey::new(format!("tier:{workload}")));
                }
                match hub.resolve_for_turn(&scope, &query).await {
                    Ok(turn) => {
                        let mut result = StepResult::ok(format!("{turn:?}"));
                        result.resolved = WORLD.iter().position(|w| w.slug == turn.slug.as_str());
                        result
                    }
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::List {
                scope,
                prov,
                refresh,
            } => {
                let scope = scope_of(self, *scope);
                match hub.list_models(&scope, &slug_of(*prov), *refresh).await {
                    Ok(list) => {
                        let mut result = StepResult::ok(format!("{:?}", list.freshness));
                        result.listed = Some((
                            list.freshness.clone(),
                            list.models
                                .iter()
                                .map(|m| m.id.as_str().to_string())
                                .collect(),
                        ));
                        result
                    }
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::Test {
                scope,
                prov,
                completion,
            } => {
                let scope = scope_of(self, *scope);
                let depth = if *completion {
                    TestDepth::Completion
                } else {
                    TestDepth::Catalog
                };
                match hub.test(&scope, &slug_of(*prov), depth, None).await {
                    Ok(report) => StepResult::ok(format!("{report:?}")),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::RecordOutcome {
                scope,
                prov,
                ok,
                reason,
            } => {
                let scope = scope_of(self, *scope);
                let outcome = if *ok {
                    Outcome::Ok {
                        latency: Duration::from_millis(30),
                    }
                } else {
                    let reason = REASONS[usize::from(*reason) % REASONS.len()];
                    Outcome::Failed(ProviderFailure::new(reason, Retry::Later(None)))
                };
                match hub.record_outcome(&scope, &slug_of(*prov), outcome).await {
                    Ok(()) => StepResult::ok(String::from("recorded")),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::RetestDown { scope } => {
                let scope = scope_of(self, *scope);
                match hub.retest_down(&scope).await {
                    Ok(done) => StepResult::ok(format!("{done:?}")),
                    Err(error) => StepResult::err(&error),
                }
            }
            Action::Advance { secs } => {
                self.ports
                    .clock
                    .advance(Duration::from_secs(u64::from(*secs)));
                StepResult::ok(String::from("advanced"))
            }
            Action::Flip { prov, mode } => {
                self.modes[*prov % WORLD.len()] = *mode;
                self.reinstall();
                StepResult::ok(String::from("flipped"))
            }
            Action::ToggleSignedOut => {
                let now = self.signed_out.load(Ordering::SeqCst);
                self.signed_out.store(!now, Ordering::SeqCst);
                StepResult::ok(String::from("toggled"))
            }
            Action::MoveOrigin { scope, rotate } => self.move_origin(*scope, *rotate).await,
            Action::Keep { scope } => self.keep(*scope).await,
            Action::UseKept { scope } => self.use_kept(*scope).await,
            Action::RaceMove { scope, at, rotate } => self.race_move(*scope, *at, *rotate).await,
        }
    }
}
