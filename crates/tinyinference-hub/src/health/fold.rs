//! The latching rules: how signals become one [`ProviderHealth`].
//!
//! A snapshot keeps the **latest** result of each probe depth and of the real
//! turns. The status is a pure function of those, so it can be recomputed and
//! tested without any I/O:
//!
//! * a rejected credential (`auth`) or an exhausted account (`quota`) is `Down`
//!   at once, whatever else passes: nothing will work until the operator acts;
//! * failures alongside successes are `Degraded` (the partial-outage case: the
//!   catalog read fails while completions work);
//! * a run of [`FAILURES_TO_DOWN`] failed real turns is `Down`, however long ago
//!   a probe last passed, unless the failure is a rate limit or an unknown model
//!   (about the request, so `Degraded`);
//! * with nothing passing, an unreachable endpoint is `Down`; any other failure
//!   is `Degraded` until it repeats ([`FAILURES_TO_DOWN`] failed turns in a
//!   row, or several lanes failing) and then `Down`;
//! * a failing **chat** lane (a completion probe or a real turn) is cleared by a
//!   newer success in the other chat lane (see `supersede_chat_failures` for
//!   exactly which success clears which failure). The clearing is recorded on the
//!   failed signal, so it stays cleared. Read-only lanes (key check, catalog)
//!   are never cleared by anything but their own next result: a working
//!   completion says nothing about a broken listing, and that partial outage is
//!   exactly what `Degraded` is for;
//! * a probe that **proves the key** clears a rejected-key failure in the lanes
//!   shallower than itself (key-only, catalog, completion); only a completion,
//!   the deepest check, also clears the real-turn lane and a quota failure;
//! * the **turn lane is latest-wins**: only the most recent real turn is kept, so a
//!   turn that gets any answer (success, or a rate limit, which is returned only
//!   to an accepted credential) replaces an earlier turn failure, terminal or
//!   not. What a passive turn never does is clear a *probe* lane's terminal
//!   failure (above);
//! * a passing probe speaks only for failures recorded **before it started**
//!   (`started_ms`): a failure that landed while the probe was in flight is
//!   newer than anything the probe saw and is kept;
//! * a passing lane counts as evidence the provider is alive only if it is not
//!   more than 30 minutes older than the newest failure.

use crate::error::ReasonCode;
use crate::taxonomy::TestDepth;

use super::types::{FailureNote, HealthSnapshot, ProbeSignal, ProviderHealth, TurnSignal};

/// Consecutive failed turns after which a transient failure stops being
/// `Degraded` and becomes `Down`.
pub const FAILURES_TO_DOWN: u32 = 3;

/// How much a failure says the provider is unusable; higher wins.
fn severity(reason: ReasonCode) -> u8 {
    match reason {
        ReasonCode::Auth => 7,
        ReasonCode::Quota => 6,
        ReasonCode::Endpoint => 5,
        ReasonCode::Timeout => 4,
        ReasonCode::Model => 3,
        ReasonCode::RateLimited => 2,
        _ => 1,
    }
}

fn is_terminal(reason: ReasonCode) -> bool {
    matches!(reason, ReasonCode::Auth | ReasonCode::Quota)
}

/// Whether a repeating failure means the provider is *down*. A rate limit and an
/// unknown model are about this request, not about the provider: repeating them
/// says "slow down" or "pick another model", so they stay `Degraded` (a `Down`
/// provider is no longer routed to and would never produce the turn that clears
/// it).
fn repeats_mean_down(reason: ReasonCode) -> bool {
    !matches!(reason, ReasonCode::RateLimited | ReasonCode::Model)
}

/// A passing lane older than this, measured against the newest failure, is not
/// evidence the provider is alive *now*: an hours-old catalog pass must not keep
/// a dead endpoint at `Degraded`.
const PASS_FRESH_MS: u64 = 30 * 60 * 1000;

/// One lane's latest result, for comparison.
#[derive(Clone, Copy)]
struct Lane {
    ok: bool,
    superseded: bool,
    reason: Option<ReasonCode>,
    at_ms: u64,
    /// The probe depth, or `None` for a real turn (which is as deep as a
    /// completion).
    depth: Option<TestDepth>,
}

impl HealthSnapshot {
    fn lanes(&self) -> Vec<Lane> {
        let mut lanes: Vec<Lane> = self
            .probes
            .iter()
            .map(|(depth, signal)| Lane {
                ok: signal.ok,
                superseded: signal.superseded,
                reason: signal.reason,
                at_ms: signal.at_ms,
                depth: Some(*depth),
            })
            .collect();
        if let Some(turn) = &self.turn {
            lanes.push(Lane {
                ok: turn.ok,
                superseded: turn.superseded,
                reason: turn.reason,
                at_ms: turn.at_ms,
                depth: None,
            });
        }
        lanes
    }

    /// Marks failing chat lanes (the turn lane and the completion probe) as
    /// superseded when `matches` says their reason is one the new success
    /// proves over. Sticky, so overwriting the success lane later does not
    /// resurrect them.
    ///
    /// Which success clears which failure is the point of the rules:
    ///
    /// * a real turn that worked clears a failing completion probe, unless that
    ///   failure was terminal: a rejected credential or an exhausted account is
    ///   `Down` "whatever else passes", so a passive success never hides one;
    /// * a **deliberate** completion probe that passed clears a failing turn,
    ///   terminal or not: the operator re-testing after fixing the key or
    ///   topping up is fresh, current evidence, and without it a `Down`
    ///   provider that is no longer routed to would never produce a turn to
    ///   clear itself;
    /// * a key-only probe that passed clears a rejected credential (that is
    ///   exactly what it proves) and nothing else.
    fn supersede_chat_failures(
        &mut self,
        turn: bool,
        completion: bool,
        matches: impl Fn(ReasonCode) -> bool,
        evidence_from_ms: u64,
    ) {
        if turn
            && let Some(signal) = self.turn.as_mut()
            && !signal.ok
            && signal.at_ms <= evidence_from_ms
            && signal.reason.is_some_and(&matches)
        {
            signal.superseded = true;
            // The run of failed turns is over: without this the counter would
            // survive the recovery and one later blip would read as the next
            // failure in a row.
            self.consecutive_failures = 0;
        }
        if completion
            && let Some(signal) = self.probes.get_mut(&TestDepth::Completion)
            && !signal.ok
            && signal.at_ms <= evidence_from_ms
            && signal.reason.is_some_and(&matches)
        {
            signal.superseded = true;
        }
    }

    /// A probe that passed clears the failures of the **probe** lanes shallower
    /// than itself, for the reasons `matches` accepts (the real-turn lane is
    /// handled by `supersede_chat_failures`, which is the one place that rule
    /// lives): a deeper lane's
    /// explicit rejection is never hidden by a shallower pass (a scope-restricted
    /// key can list models and still be refused a completion), but a passing
    /// completion, the deepest check, speaks for every lane above it.
    ///
    /// Depth order is key-only, catalog, completion. The lane being overwritten
    /// right now is left to its own result. A key-only pass therefore clears
    /// nothing (no lane is shallower), and a catalog pass clears only a
    /// key-only rejection: to lift a chat lane's rejection the operator re-tests
    /// with a completion, the only check that shows the key can chat.
    ///
    /// Recency is by evidence time: a pass speaks only for failures recorded no
    /// later than the moment the probe **started** (`evidence_from_ms`). A
    /// failure recorded while the probe was in flight is newer than anything the
    /// probe saw, so it stays.
    fn clear_shallower_lanes(
        &mut self,
        passed: TestDepth,
        matches: impl Fn(ReasonCode) -> bool,
        evidence_from_ms: u64,
    ) {
        for (depth, signal) in &mut self.probes {
            if *depth < passed
                && !signal.ok
                && signal.at_ms <= evidence_from_ms
                && signal.reason.is_some_and(&matches)
            {
                signal.superseded = true;
            }
        }
    }

    /// Recomputes [`HealthSnapshot::health`] from the signals.
    ///
    /// Returns whether the status changed. Only called after a signal was
    /// recorded, so there is always at least one lane.
    fn refold(&mut self, now_ms: u64) -> bool {
        let lanes = self.lanes();
        let failing: Vec<&Lane> = lanes
            .iter()
            .filter(|lane| !lane.ok && !lane.superseded)
            .collect();
        let newest_failure = failing.iter().map(|lane| lane.at_ms).max().unwrap_or(0);
        let passing = lanes
            .iter()
            .filter(|lane| lane.ok && lane.at_ms.saturating_add(PASS_FRESH_MS) >= newest_failure)
            .count();
        // The reason of the failing *turns*, not the worst reason across lanes: a
        // provider that is only rate limiting is not down because some unrelated
        // probe timed out.
        let turn_reason = failing
            .iter()
            .find(|lane| lane.depth.is_none())
            .and_then(|lane| lane.reason);
        let next = match failing
            .iter()
            .filter_map(|lane| lane.reason)
            .max_by_key(|reason| severity(*reason))
        {
            None if failing.is_empty() => ProviderHealth::Ok,
            None => ProviderHealth::Degraded(ReasonCode::Unknown),
            Some(worst) if is_terminal(worst) => ProviderHealth::Down(worst),
            // Turns are what the operator cares about: a run of failed turns is
            // `Down` however long ago some probe last passed.
            Some(_)
                if self.consecutive_failures >= FAILURES_TO_DOWN
                    && turn_reason.is_some_and(repeats_mean_down) =>
            {
                ProviderHealth::Down(turn_reason.unwrap_or(ReasonCode::Unknown))
            }
            Some(worst) if passing > 0 => ProviderHealth::Degraded(worst),
            Some(worst) if worst == ReasonCode::Endpoint => ProviderHealth::Down(worst),
            // With nothing passing, several lanes failing (or a run of failed
            // turns) is an outage only if every failure that is counted is the
            // kind that repeating makes an outage: a rate-limited turn beside an
            // older timed-out probe is still just throttling.
            Some(worst)
                if repeats_mean_down(worst)
                    && (failing.len() >= 2 || self.consecutive_failures >= FAILURES_TO_DOWN)
                    && failing
                        .iter()
                        .filter_map(|lane| lane.reason)
                        .all(repeats_mean_down) =>
            {
                ProviderHealth::Down(worst)
            }
            Some(worst) => ProviderHealth::Degraded(worst),
        };
        let changed = next != self.health;
        if changed {
            self.health = next;
            self.changed_at_ms = now_ms;
        }
        changed
    }

    /// Records the result of a probe at `depth`. `proves_key` is the probe
    /// report's own verdict that a pass shows the presented credential works (a
    /// key-only or completion pass always does; a catalog pass only where the
    /// listing is not public). A pass that proves the key clears a rejected-key
    /// failure in the probe lanes shallower than itself; a completion also
    /// clears the real-turn lane and a quota failure. Returns whether the
    /// status changed.
    pub fn record_probe(
        &mut self,
        depth: TestDepth,
        failure: Option<(ReasonCode, Option<u16>)>,
        latency_ms: Option<u64>,
        proves_key: bool,
        now_ms: u64,
    ) -> bool {
        self.record_probe_started(depth, failure, latency_ms, proves_key, now_ms, now_ms)
    }

    /// [`record_probe`](Self::record_probe) for a probe that **started** at
    /// `started_ms` and finished (was recorded) at `now_ms`.
    ///
    /// A pass supersedes only failures recorded no later than `started_ms`: one
    /// that landed while the probe was in flight (a real turn failing during a
    /// slow completion ping) is newer than anything the probe saw and is kept.
    /// The start time is stored on the signal, so the snapshot always knows how
    /// old the evidence behind each lane is.
    pub fn record_probe_started(
        &mut self,
        depth: TestDepth,
        failure: Option<(ReasonCode, Option<u16>)>,
        latency_ms: Option<u64>,
        proves_key: bool,
        started_ms: u64,
        now_ms: u64,
    ) -> bool {
        let started_ms = started_ms.min(now_ms);
        if failure.is_none() {
            if depth == TestDepth::Completion {
                // A deliberate completion that passed ends a run of failed turns,
                // terminal or not, and shows the key and the account work to the
                // shallower lanes too.
                self.supersede_chat_failures(true, false, |_| true, started_ms);
                self.clear_shallower_lanes(depth, is_terminal, started_ms);
            } else if proves_key {
                // A key-proving pass at a shallower depth clears only rejected
                // keys, and only in lanes shallower than itself.
                self.clear_shallower_lanes(depth, |reason| reason == ReasonCode::Auth, started_ms);
            }
        }
        self.probes.insert(
            depth,
            ProbeSignal {
                ok: failure.is_none(),
                superseded: false,
                reason: failure.map(|(reason, _)| reason),
                at_ms: now_ms,
                started_ms: Some(started_ms),
                latency_ms,
            },
        );
        self.note(failure, now_ms);
        self.refold(now_ms)
    }

    /// Records the result of a real turn. Returns whether the status changed.
    pub fn record_turn(
        &mut self,
        failure: Option<(ReasonCode, Option<u16>)>,
        latency_ms: Option<u64>,
        now_ms: u64,
    ) -> bool {
        if failure.is_none() {
            self.supersede_chat_failures(false, true, |r| !is_terminal(r), now_ms);
        }
        self.turn = Some(TurnSignal {
            latency_ms,
            ok: failure.is_none(),
            superseded: false,
            reason: failure.map(|(reason, _)| reason),
            at_ms: now_ms,
        });
        self.consecutive_failures = if failure.is_none() {
            0
        } else {
            self.consecutive_failures.saturating_add(1)
        };
        self.note(failure, now_ms);
        self.refold(now_ms)
    }

    /// Marks the provider signed out. Clears every signal: what was learned
    /// while signed in says nothing about now, and the next probe or turn
    /// decides. Returns whether the status changed.
    pub fn record_signed_out(&mut self, now_ms: u64) -> bool {
        self.probes.clear();
        self.turn = None;
        self.consecutive_failures = 0;
        let changed = self.health != ProviderHealth::SignedOut;
        if changed {
            self.health = ProviderHealth::SignedOut;
            self.changed_at_ms = now_ms;
        }
        changed
    }

    fn note(&mut self, failure: Option<(ReasonCode, Option<u16>)>, now_ms: u64) {
        match failure {
            None => self.last_ok_ms = Some(now_ms),
            Some((reason, status)) => {
                self.last_failure = Some(FailureNote {
                    reason,
                    status,
                    at_ms: now_ms,
                });
            }
        }
    }
}
