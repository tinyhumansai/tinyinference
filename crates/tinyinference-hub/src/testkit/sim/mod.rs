//! The seeded scenario runner: random but reproducible sessions against a full
//! [`Hub`](crate::Hub) over in-memory ports, a scripted transport and a fake
//! clock, with the eleven invariants checked after **every** step.
//!
//! A run is `ScenarioRunner::new(seed)` then [`run_random`](ScenarioRunner::run_random).
//! The seed is always passed in code (never read from the environment), and a
//! failure prints `seed=<n> step=<k>` with the action trace, so it can be
//! replayed. Every seed that ever failed is kept in `tests/golden/sim_seeds.txt`
//! and replayed by `sim_random_regressions`.

mod action;
mod invariants;
mod model;
mod moves;
#[cfg(test)]
#[path = "test.rs"]
mod tests;
mod world;

pub use action::{Action, StepResult};
pub use invariants::InvariantViolation;
pub(crate) use world::install;
pub use world::{MANAGED_BASE, Mode, SimToken, WORLD, World};

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use crate::config::DefaultChoice;
use crate::credential::CredentialOrigin;
use crate::credential::TokenSourceAdapter;
use crate::error::HubError;
use crate::hub::{Hub, ManagedConfig};
use crate::ids::ScopeKey;
use crate::policy::EndpointPolicy;
use crate::ports::memory::CredentialFault;

use super::{FakeClock, MemoryPorts};

/// A deterministic generator (SplitMix64): no dependency, same sequence on
/// every platform.
#[derive(Clone, Debug)]
pub(crate) struct Rng(u64);

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub(crate) fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A number in `0..n` (`n` must be positive).
    pub(crate) fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % (n.max(1) as u64)).unwrap_or(0)
    }

    pub(crate) fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
}

/// How often the runner injects infrastructure faults. All zero by default.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaultPlan {
    /// The seed the plan is derived from (kept so a failure names it).
    pub seed: u64,
    /// The chance that a step runs with the configuration store losing its
    /// compare-and-swap (one to three times).
    pub p_conflict: f64,
    /// The chance that a step runs with the credential store down.
    pub p_credential_outage: f64,
}

impl FaultPlan {
    /// No faults.
    pub fn none(seed: u64) -> Self {
        Self {
            seed,
            p_conflict: 0.0,
            p_credential_outage: 0.0,
        }
    }

    /// Occasional faults of both kinds.
    pub fn flaky(seed: u64) -> Self {
        Self {
            seed,
            p_conflict: 0.08,
            p_credential_outage: 0.05,
        }
    }
}

/// A run that broke an invariant (or made the hub panic or misbehave).
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SimFailure {
    /// The seed that reproduces it.
    pub seed: u64,
    /// The step that broke it (from zero).
    pub step: usize,
    /// What broke.
    pub violation: InvariantViolation,
    /// Every action up to and including the failing step.
    pub trace: Vec<String>,
}

impl fmt::Display for SimFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "seed={} step={}: {}",
            self.seed, self.step, self.violation
        )?;
        for (n, line) in self.trace.iter().enumerate() {
            writeln!(f, "  {n:>4}  {line}")?;
        }
        Ok(())
    }
}

impl std::error::Error for SimFailure {}

/// The runner. See the module docs.
pub struct ScenarioRunner {
    pub(crate) seed: u64,
    pub(crate) rng: Rng,
    pub(crate) ports: MemoryPorts,
    pub(crate) hub: Hub,
    pub(crate) scopes: Vec<ScopeKey>,
    pub(crate) faults: FaultPlan,
    pub(crate) signed_out: Arc<AtomicBool>,
    pub(crate) modes: Vec<Mode>,
    /// Every credential the run created, with the model its listing names.
    pub(crate) keys: Vec<(String, usize)>,
    /// The one origin each credential was entered for (invariant 11): a request
    /// carrying the credential to any other origin is a leak.
    pub(crate) bound: HashMap<String, String>,
    /// A chat model the run holds on to per scope, as a host does between turns.
    pub(crate) kept: Vec<Option<Arc<dyn tinyinference_llm::model::ChatModel<()>>>>,
    pub(crate) next_key: usize,
    pub(crate) trace: Vec<String>,
    pub(crate) step: usize,
    pub(crate) requests_scanned: usize,
    pub(crate) requests_before: usize,
    pub(crate) refused_seen: usize,
    pub(crate) infra_faults_this_step: bool,
    /// What the race actions actually did, so a test can say they were not
    /// vacuous.
    pub(crate) races: RaceStats,
}

/// Counters for [`ScenarioRunner`]'s race actions.
#[derive(Debug, Default)]
pub(crate) struct RaceStats {
    /// Races started (a movable provider existed and was enabled).
    pub(crate) attempted: std::sync::atomic::AtomicUsize,
    /// Races in which the parked call was actually reached.
    pub(crate) parked: std::sync::atomic::AtomicUsize,
    /// Races whose edit succeeded and changed the record.
    pub(crate) moved: std::sync::atomic::AtomicUsize,
}

impl RaceStats {
    pub(crate) fn bump(counter: &std::sync::atomic::AtomicUsize) {
        counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    #[cfg(test)]
    pub(crate) fn get(counter: &std::sync::atomic::AtomicUsize) -> usize {
        counter.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl fmt::Debug for ScenarioRunner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ScenarioRunner")
            .field("seed", &self.seed)
            .field("step", &self.step)
            .finish_non_exhaustive()
    }
}

impl ScenarioRunner {
    /// A runner for `seed` under the desktop endpoint policy, with the managed
    /// provider configured (its token rotates every fake minute) and two
    /// scopes.
    ///
    /// # Panics
    ///
    /// Never: the hub is built from in-memory ports.
    pub fn new(seed: u64) -> Self {
        Self::with(seed, EndpointPolicy::desktop(), FaultPlan::none(seed))
    }

    /// A runner under a chosen policy and fault plan.
    ///
    /// # Panics
    ///
    /// Never: the hub is built from in-memory ports.
    pub fn with(seed: u64, policy: EndpointPolicy, faults: FaultPlan) -> Self {
        let ports = MemoryPorts::new();
        let signed_out = Arc::new(AtomicBool::new(false));
        let clock: FakeClock = ports.clock.clone();
        let token = Arc::new(SimToken::new(clock, signed_out.clone()));
        let factory = model::SimFactory {
            http: ports.http.clone(),
            policy: policy.clone(),
        };
        let hub = ports
            .builder()
            .model_factory(Arc::new(factory))
            .policy(policy)
            .managed(
                ManagedConfig::new(MANAGED_BASE).source(TokenSourceAdapter::new(
                    token,
                    CredentialOrigin::InstanceIdentity,
                )),
            )
            .build()
            .expect("the runner's hub builds from in-memory ports");
        let mut runner = Self {
            seed,
            rng: Rng::new(seed),
            ports,
            hub,
            scopes: vec![ScopeKey::new("company:a"), ScopeKey::new("company:b")],
            faults,
            signed_out,
            modes: vec![Mode::Healthy; WORLD.len()],
            keys: Vec::new(),
            bound: HashMap::new(),
            kept: vec![None; 2],
            next_key: 0,
            trace: Vec::new(),
            step: 0,
            requests_scanned: 0,
            requests_before: 0,
            refused_seen: 0,
            infra_faults_this_step: false,
            races: RaceStats::default(),
        };
        runner.reinstall();
        runner
    }

    /// The seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The hub under test, for a scenario that wants to look at it.
    pub fn hub(&self) -> &Hub {
        &self.hub
    }

    /// The ports, for a scenario that wants to inspect or script them.
    pub fn ports(&self) -> &MemoryPorts {
        &self.ports
    }

    /// The actions run so far, one line each.
    pub fn trace(&self) -> &[String] {
        &self.trace
    }

    /// Re-installs every provider's scripted rules from the current modes and
    /// keys.
    pub(crate) fn reinstall(&mut self) {
        self.ports.http.clear_rules();
        for (index, world) in WORLD.iter().enumerate() {
            install(&self.ports.http, world, self.modes[index], &self.keys);
        }
    }

    /// Mints a credential the run has never used, **entered for `origin`**, and
    /// installs its rules.
    pub(crate) fn new_key(&mut self, origin: &str) -> crate::secret::Secret {
        let id = self.next_key;
        self.next_key += 1;
        // The terminator keeps one key from being a prefix of another
        // (`...-1` and `...-10`), so a substring search is exact.
        let secret = format!("sk-sim-{}-{id}.end", self.seed);
        self.bound.insert(secret.clone(), origin.to_string());
        self.keys.push((secret.clone(), id));
        self.reinstall();
        crate::secret::Secret::new(secret)
    }

    fn inject_faults(&mut self) {
        self.infra_faults_this_step = false;
        if self.rng.chance(self.faults.p_conflict) {
            self.ports
                .config
                .conflict_next(1 + u32::try_from(self.rng.below(3)).unwrap_or(0));
            self.infra_faults_this_step = true;
        }
        if self.rng.chance(self.faults.p_credential_outage) {
            let fault = if self.rng.below(2) == 0 {
                CredentialFault::Read
            } else {
                CredentialFault::Write
            };
            self.ports.credentials.inject(fault);
            self.infra_faults_this_step = true;
        }
    }

    fn heal(&mut self) {
        self.ports.config.conflict_next(0);
        self.ports.credentials.heal();
    }

    /// Runs one action, then checks every invariant.
    ///
    /// # Errors
    ///
    /// [`SimFailure`] when an invariant broke.
    pub async fn step(&mut self, action: Action) -> Result<StepResult, SimFailure> {
        let step = self.step;
        self.step += 1;
        self.trace.push(format!("{action:?}"));
        let before = self.snapshot_defaults().await;
        self.requests_before = self.ports.http.request_count();
        let result = self.apply(&action).await;
        self.heal();
        let outcome = match self.after_step(&action, &result, &before).await {
            Ok(()) => self.check_invariants().await,
            Err(violation) => Err(violation),
        };
        match outcome {
            Ok(()) => Ok(result),
            Err(violation) => Err(SimFailure {
                seed: self.seed,
                step,
                violation,
                trace: self.trace.clone(),
            }),
        }
    }

    /// Runs `steps` random actions under the fault plan.
    ///
    /// # Errors
    ///
    /// The first [`SimFailure`], carrying the seed and the trace.
    pub async fn run_random(&mut self, steps: usize) -> Result<(), SimFailure> {
        for _ in 0..steps {
            self.inject_faults();
            let action = self.random_action();
            self.step(action).await?;
        }
        Ok(())
    }

    pub(crate) async fn snapshot_defaults(&self) -> Vec<DefaultChoice> {
        let mut out = Vec::new();
        for scope in &self.scopes {
            out.push(
                self.hub
                    .read_config(scope)
                    .await
                    .map_or(DefaultChoice::Unset, |config| config.default),
            );
        }
        out
    }
}

/// Whether an error is one only an injected infrastructure fault can cause.
pub(crate) fn is_infra(error: &HubError) -> bool {
    matches!(error, HubError::Conflict | HubError::StoreUnreadable { .. })
}
