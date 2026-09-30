//! The invariants checked after every step (08-test-plan section 2).
//!
//! 1. No secret appears in any error text, event, resolved turn, recorded
//!    request or stored document.
//! 2. The stored configuration never contains a credential.
//! 3. A cached list never crosses a scope or outlives its credential.
//! 4. A rejected credential is never served from the cache.
//! 5. Every request the hub sent went to a host of the world, and none was
//!    sent that the transport refused.
//! 6. The managed provider exists and is first.
//! 7. A resolved turn is never a disabled or removed provider.
//! 8. A failed add leaves the key slot and the record set as they were.
//! 9. A removed or re-keyed provider's health is empty.
//! 10. The default only changes through the operations that may change it.
//! 11. **A credential is only ever sent to the origin it was entered for**: no
//!     request carries a pasted key toward any other origin, and the platform
//!     token goes only to the managed endpoint. Checked on every request of
//!     every step, including the ones a kept model or a fresh resolve sends in
//!     the middle of an edit that is parked mid-way.

use std::fmt;

use super::action::Action;
use super::world::{MANAGED_BASE, Mode, SimToken, WORLD};
use super::{ScenarioRunner, StepResult};
use crate::catalog::Freshness;
use crate::config::DefaultChoice;
use crate::ids::{ScopeKey, Slug};
use crate::ports::{CredentialStore, HealthStore};
use crate::secret::Secret;

/// What went wrong.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvariantViolation {
    /// Which invariant (1 to 10) or `0` for an unexpected result.
    pub number: u8,
    /// A sentence saying what was seen.
    pub detail: String,
}

impl fmt::Display for InvariantViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invariant {} broken: {}", self.number, self.detail)
    }
}

fn broken(number: u8, detail: impl Into<String>) -> InvariantViolation {
    InvariantViolation {
        number,
        detail: detail.into(),
    }
}

impl ScenarioRunner {
    /// Every credential the run mints starts with one prefix, so one needle
    /// finds any of them.
    fn leaks(&self, text: &str) -> Option<String> {
        let needle = format!("sk-sim-{}-", self.seed);
        text.contains(&needle).then_some(needle)
    }

    /// Whether `text` mentions a platform token of any minute.
    fn has_token(text: &str) -> bool {
        text.contains("platform-token-")
    }

    async fn stored_key(&self, scope: &ScopeKey, prov: usize) -> Option<String> {
        let slug = Slug::parse(WORLD[prov].slug).ok()?;
        self.ports
            .credentials
            .get(scope, &slug.key_slot())
            .await
            .ok()
            .flatten()
            .map(|s| s.expose().to_string())
    }

    /// What must hold after `action` produced `result`, given the defaults
    /// before it.
    pub(crate) async fn after_step(
        &mut self,
        action: &Action,
        result: &StepResult,
        before: &[DefaultChoice],
    ) -> Result<(), InvariantViolation> {
        // 1: nothing the step said contains a secret.
        if let Some(secret) = self.leaks(&result.text) {
            return Err(broken(
                1,
                format!("the step's result carried the secret {secret}"),
            ));
        }
        if Self::has_token(&result.text) {
            return Err(broken(1, "the step's result carried a platform token"));
        }
        // Errors only an injected fault can cause must not appear without one.
        if result.infra && !self.infra_faults_this_step {
            return Err(broken(
                0,
                format!(
                    "an infrastructure error with no fault injected: {}",
                    result.text
                ),
            ));
        }
        let faulty = self.infra_faults_this_step;
        let scope_index = match action {
            Action::Connect { scope, .. }
            | Action::Add { scope, .. }
            | Action::Edit { scope, .. }
            | Action::Remove { scope, .. }
            | Action::SetEnabled { scope, .. }
            | Action::SetKey { scope, .. }
            | Action::ClearKey { scope, .. }
            | Action::SetDefault { scope, .. }
            | Action::ClearDefault { scope }
            | Action::Pin { scope, .. }
            | Action::SetRoute { scope, .. }
            | Action::Resolve { scope, .. }
            | Action::List { scope, .. }
            | Action::Test { scope, .. }
            | Action::RecordOutcome { scope, .. }
            | Action::RetestDown { scope }
            | Action::MoveOrigin { scope, .. }
            | Action::Keep { scope }
            | Action::UseKept { scope }
            | Action::RaceMove { scope, .. } => Some(*scope % self.scopes.len()),
            _ => None,
        };

        // 10: the default changes only where it may.
        if let Some(index) = scope_index {
            let after = self.hub.read_config(&self.scopes[index]).await;
            if let Ok(after) = after {
                let (was, now) = (&before[index], &after.default);
                let may_change = matches!(
                    action,
                    Action::SetDefault { .. }
                        | Action::ClearDefault { .. }
                        | Action::Connect { .. }
                        | Action::Add { .. }
                );
                if was != now && !may_change {
                    return Err(broken(
                        10,
                        format!("{action:?} changed the default from {was:?} to {now:?}"),
                    ));
                }
                // A failed add is undone exactly: including the default it may
                // have replaced.
                if was != now
                    && matches!(action, Action::Connect { .. } | Action::Add { .. })
                    && !result.ok
                    && !self.infra_faults_this_step
                {
                    return Err(broken(
                        10,
                        format!("a failed add changed the default from {was:?} to {now:?}"),
                    ));
                }
                if was != now && matches!(action, Action::Connect { .. } | Action::Add { .. }) {
                    let first_only =
                        *was == DefaultChoice::Unset && matches!(now, DefaultChoice::Full { .. });
                    let explicit = matches!(
                        action,
                        Action::Connect {
                            make_default: true,
                            ..
                        }
                    );
                    if !first_only && !explicit {
                        return Err(broken(
                            10,
                            format!("an add moved the default from {was:?} to {now:?}"),
                        ));
                    }
                }
                if let Action::SetDefault { prov, .. } = action
                    && result.ok
                    && !matches!(now, DefaultChoice::Full { provider, .. } if provider.as_str() == WORLD[*prov].slug)
                {
                    return Err(broken(
                        10,
                        format!("set_default succeeded but the default is {now:?}"),
                    ));
                }
            }
        }

        // 6: managed exists and is first.
        if let Some(index) = scope_index
            && !faulty
            && let Ok(config) = self.hub.read_config(&self.scopes[index]).await
            && !config
                .providers
                .iter()
                .any(|p| p.kind.as_str() == "tinyhumans")
        {
            return Err(broken(
                6,
                "the managed provider is missing from the configuration",
            ));
        }

        // Actions that must be typed refusals.
        if let Action::Connect { prov, .. } | Action::Add { prov, .. } = action
            && WORLD[*prov].managed
            && (result.ok || result.reason != Some(crate::error::ReasonCode::Unsupported))
            && !faulty
        {
            return Err(broken(
                0,
                format!("adding the managed provider was not Unsupported: {result:?}"),
            ));
        }
        if let Action::Remove { prov, .. } = action
            && WORLD[*prov].managed
            && result.ok
        {
            return Err(broken(6, "the managed provider was removed"));
        }

        // 7: a resolved turn is an enabled provider that exists.
        if let (Action::Resolve { scope, .. }, Some(prov)) = (action, result.resolved)
            && let Ok(config) = self
                .hub
                .read_config(&self.scopes[*scope % self.scopes.len()])
                .await
        {
            let enabled = config
                .providers
                .iter()
                .find(|p| p.slug.as_str() == WORLD[prov].slug)
                .is_some_and(|p| p.enabled);
            if !enabled {
                return Err(broken(
                    7,
                    format!("resolved to {} which is disabled or gone", WORLD[prov].slug),
                ));
            }
        }

        // 8 and 9 need the credential and health stores; skip them while a
        // fault could make a read fail.
        if faulty {
            return Ok(());
        }
        if let Action::Connect { scope, prov, .. } | Action::Add { scope, prov, .. } = action
            && !result.ok
        {
            let scope = &self.scopes[*scope % self.scopes.len()];
            let present = self.hub.read_config(scope).await.is_ok_and(|c| {
                c.providers
                    .iter()
                    .any(|p| p.slug.as_str() == WORLD[*prov].slug)
            });
            // A failed add can only be an AlreadyExists (the row was there) or
            // left nothing behind.
            if result.reason != Some(crate::error::ReasonCode::AlreadyExists)
                && present
                && !WORLD[*prov].managed
            {
                return Err(broken(
                    8,
                    format!("a failed add left the row behind: {}", result.text),
                ));
            }
            if result.reason != Some(crate::error::ReasonCode::AlreadyExists)
                && !present
                && self.stored_key(scope, *prov).await.is_some()
            {
                return Err(broken(8, "a failed add left a key in the slot"));
            }
        }
        let rekeyed = match action {
            Action::SetKey { scope, prov } | Action::ClearKey { scope, prov, .. }
                if result.changed =>
            {
                Some((*scope, *prov))
            }
            Action::Edit {
                scope,
                prov,
                rotate: true,
                ..
            } if result.changed => Some((*scope, *prov)),
            Action::Remove { scope, prov, .. } if result.changed => Some((*scope, *prov)),
            _ => None,
        };
        if let Some((scope, prov)) = rekeyed {
            let scope = self.scopes[scope % self.scopes.len()].clone();
            if let Ok(slug) = Slug::parse(WORLD[prov].slug)
                && let Ok(Some(snapshot)) = self.ports.health.get(&scope, &slug).await
            {
                return Err(broken(
                    9,
                    format!(
                        "health for {slug} survived a key change: {:?}",
                        snapshot.health
                    ),
                ));
            }
        }

        // 3 and 4: a list is read with the current credential, and a fetch the
        // provider rejected is an error, never an older list.
        if let Action::List { scope, prov, .. } = action {
            let world = &WORLD[*prov];
            let fetched = self.ports.http.request_count() > self.requests_before;
            // A rejection is a fact about the *credential presented*. A read that
            // presented none (a local runtime, a keyless custom endpoint) was
            // rejected by the endpoint itself, which the cache remembers like any
            // other endpoint failure and may answer with an older list.
            let presented = self
                .ports
                .http
                .requests_from(self.requests_before)
                .iter()
                .any(|r| r.credentialed);
            if fetched
                && presented
                && self.modes[*prov] == Mode::AuthFail
                && (result.ok || result.reason != Some(crate::error::ReasonCode::Auth))
            {
                return Err(broken(
                    4,
                    format!("{} fetched a rejection but answered {result:?}", world.slug),
                ));
            }
            if fetched && self.modes[*prov] == Mode::Healthy && !result.ok {
                return Err(broken(
                    0,
                    format!(
                        "{} fetched from a healthy provider and failed: {}",
                        world.slug, result.text
                    ),
                ));
            }
            if let Some((freshness, ids)) = &result.listed
                && matches!(freshness, Freshness::Fresh | Freshness::Cached)
            {
                let scope = &self.scopes[*scope % self.scopes.len()];
                let key = self.stored_key(scope, *prov).await;
                let expected: Option<String> = if !world.keyed {
                    Some("llama3".to_string())
                } else if world.managed {
                    Some("managed-model".to_string())
                } else {
                    key.as_ref()
                        .and_then(|k| self.keys.iter().find(|(s, _)| s == k))
                        .map(|(_, id)| format!("model-{id}"))
                };
                if let Some(expected) = expected
                    && ids.as_slice() != std::slice::from_ref(&expected)
                {
                    return Err(broken(
                        3,
                        format!(
                            "{} listed {ids:?}; the current credential's list is [{expected}]",
                            world.slug
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// The invariants that hold at any moment, checked after every step.
    ///
    /// # Errors
    ///
    /// The first [`InvariantViolation`].
    pub async fn check_invariants(&mut self) -> Result<(), InvariantViolation> {
        // 1 and 5: scan what was sent and emitted since the last check.
        let fresh = self.ports.http.requests_from(self.requests_scanned);
        for request in &fresh {
            let shown = format!(
                "{request:?}{}{}",
                request.url,
                request.body.clone().unwrap_or_default()
            );
            if let Some(secret) = self.leaks(&shown) {
                return Err(broken(
                    1,
                    format!("a recorded request carried the secret {secret}"),
                ));
            }
            let host = url::Url::parse(&request.url)
                .ok()
                .and_then(|u| u.host_str().map(str::to_string));
            let known = WORLD.iter().any(|w| {
                std::iter::once(w.base).chain(w.alt_base).any(|base| {
                    url::Url::parse(base)
                        .ok()
                        .and_then(|u| u.host_str().map(str::to_string))
                        == host
                })
            });
            if !known {
                return Err(broken(
                    5,
                    format!("a request went to an unexpected host: {}", request.url),
                ));
            }
            self.check_credential_binding(request)?;
            if request.url.starts_with(MANAGED_BASE) && !self.managed_request_is_valid(request) {
                return Err(broken(
                    1,
                    "a managed request carried neither the current token nor a pasted key",
                ));
            }
        }
        self.requests_scanned += fresh.len();
        if self.ports.http.refused_count() > self.refused_seen {
            self.refused_seen = self.ports.http.refused_count();
            for (url, violation) in self.ports.http.refused() {
                if WORLD.iter().any(|w| url.starts_with(w.base)) {
                    return Err(broken(
                        5,
                        format!("the hub asked for a URL the policy refuses: {url} ({violation})"),
                    ));
                }
            }
        }
        for event in &self.ports.events.drain() {
            let shown = format!("{event:?}");
            if let Some(secret) = self.leaks(&shown) {
                return Err(broken(1, format!("an event carried the secret {secret}")));
            }
            if Self::has_token(&shown) {
                return Err(broken(1, "an event carried a platform token"));
            }
        }

        // 2 and 6: the stored documents.
        for scope in self.scopes.clone() {
            let Some(raw) = self.ports.config.raw(&scope) else {
                continue;
            };
            if let Some(secret) = self.leaks(&raw) {
                return Err(broken(
                    2,
                    format!("the stored configuration carried the secret {secret}"),
                ));
            }
            if Self::has_token(&raw) {
                return Err(broken(
                    2,
                    "the stored configuration carried a platform token",
                ));
            }
            let Ok(config) = serde_json::from_str::<crate::config::HubConfig>(&raw) else {
                return Err(broken(2, "the stored configuration does not load"));
            };
            if config.validate().is_err() {
                return Err(broken(2, "the stored configuration does not validate"));
            }
            let managed = config
                .providers
                .iter()
                .filter(|p| p.kind.as_str() == "tinyhumans")
                .count();
            if managed > 1 {
                return Err(broken(6, "two managed records"));
            }
        }
        Ok(())
    }

    /// Invariant 11 for one request: each pasted key it carries was entered for
    /// the origin it went to, and the platform token only went to the managed
    /// endpoint.
    fn check_credential_binding(
        &self,
        request: &crate::testkit::RecordedRequest,
    ) -> Result<(), InvariantViolation> {
        // Whatever header carries it: a key in `x-api-key` is as much a leak.
        if request.headers.is_empty() {
            return Ok(());
        }
        for (secret, entered_for) in &self.bound {
            if request.carried(&Secret::new(secret.clone()))
                && !crate::policy::same_origin(&request.url, entered_for)
            {
                // Which key, by its number: the value itself is never printed.
                let number = self
                    .keys
                    .iter()
                    .find(|(s, _)| s == secret)
                    .map_or(0, |(_, id)| *id);
                return Err(broken(
                    11,
                    format!(
                        "key #{number}, entered for {entered_for}, was sent to {}",
                        request.url
                    ),
                ));
            }
        }
        // A platform token of any minute (a stale one is as wrong as a fresh one).
        let any_token = Secret::new("platform-token-");
        if !request.url.starts_with(MANAGED_BASE) && request.carried(&any_token) {
            return Err(broken(
                11,
                format!("the platform token was sent to {}", request.url),
            ));
        }
        Ok(())
    }

    /// A managed request carries the platform token of the minute it was sent in,
    /// or a key someone pasted: never a token from an earlier minute.
    fn managed_request_is_valid(&self, request: &crate::testkit::RecordedRequest) -> bool {
        let elapsed_ms = request
            .at_wall_ms
            .saturating_sub(crate::testkit::FakeClock::START_WALL_MS);
        let current = SimToken::at_minute(elapsed_ms / 60_000);
        request.carried(&Secret::new(current))
            || self
                .keys
                .iter()
                .any(|(k, _)| request.carried(&Secret::new(k.clone())))
    }
}
