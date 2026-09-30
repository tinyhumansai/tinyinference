//! [`HubModel`]: the `ChatModel` the hub hands out.

use std::fmt;
use std::sync::Mutex;

use async_trait::async_trait;
use serde_json::Value;
use tinyinference_llm::error::{Error, Result};
use tinyinference_llm::model::{
    ChatModel, ModelProfile, ModelRequest, ModelResponse, ModelStream, ProviderError,
};

use crate::credential::CredentialOrigin;
use crate::error::{HubError, ProviderFailure};
use crate::health::Outcome;
use crate::hub::Hub;
use crate::ids::ScopeKey;
use crate::route::ResolvedTurn;
use crate::secret::{Secret, SecretId};

use super::factory::ModelSpec;

/// The legacy billing key OpenCompany and OpenHuman read (D4).
const LEGACY_USAGE_META: &str = "openhuman_usage_meta";
/// The neutral spelling.
const USAGE_META: &str = "usage_meta";

/// The credential one call used: which source answered and which credential it
/// was, so a rejection is attributed to that credential and no other.
#[derive(Clone, Copy)]
struct Used<'a> {
    origin: &'a CredentialOrigin,
    id: SecretId,
}

/// The model for one call and the credential that answered for it.
type Current = (
    std::sync::Arc<dyn ChatModel<()>>,
    Option<(CredentialOrigin, SecretId)>,
);

type Built = (Option<Secret>, std::sync::Arc<dyn ChatModel<()>>);

pub(super) struct HubModel {
    hub: Hub,
    scope: ScopeKey,
    turn: ResolvedTurn,
    inner: Mutex<Option<Built>>,
}

impl HubModel {
    pub(super) fn new(hub: Hub, scope: ScopeKey, turn: ResolvedTurn) -> Self {
        Self {
            hub,
            scope,
            turn,
            inner: Mutex::new(None),
        }
    }

    /// Whether the kind needs a key to be used at all.
    fn key_required(&self) -> bool {
        self.turn.auth.needs_credential()
            && self
                .hub
                .inner
                .registry
                .get(&self.turn.kind)
                .is_some_and(|driver| driver.descriptor().needs_key)
    }

    fn provider_error(&self, code: &str, message: &str, retryable: bool) -> Error {
        Error::Provider(Box::new(ProviderError {
            provider: self.turn.kind.to_string(),
            model: self.turn.model.as_ref().map(ToString::to_string),
            status: None,
            code: Some(code.to_string()),
            message: message.to_string(),
            retryable,
            ..ProviderError::default()
        }))
    }

    /// Fails closed unless the live record is still enabled and at the endpoint
    /// this model was resolved for. The managed provider's endpoint is the host's,
    /// not the record's, and is not checked.
    async fn check_route_is_current(&self) -> Result<()> {
        if self.turn.group == crate::taxonomy::ProviderGroup::Managed {
            return Ok(());
        }
        let stale = match self.hub.read_config(&self.scope).await {
            Ok(config) => config
                .provider(&self.turn.slug)
                .is_none_or(|record| !record.enabled || record.base_url != self.turn.base_url),
            Err(_) => {
                return Err(self.provider_error(
                    "store_unreadable",
                    "the settings store could not be read",
                    true,
                ));
            }
        };
        if stale {
            return Err(self.provider_error(
                "stale_route",
                "the provider changed or was removed since this model was resolved",
                false,
            ));
        }
        Ok(())
    }

    /// The model for this call: the credential chain is resolved **now**, and
    /// the underlying client is rebuilt only when the credential changed.
    async fn current(&self) -> Result<Current> {
        // A model the host kept must not send whatever key the chain holds now
        // to an endpoint the record no longer has (G3): the live record must
        // still be enabled and at the endpoint this model was resolved for.
        self.check_route_is_current().await?;
        let epoch = self.hub.inner.health.epoch(&self.scope, &self.turn.slug);
        let resolved = self
            .hub
            .chain_for(&self.turn.kind)
            .resolve(&self.scope, &self.turn.slug)
            .await;
        // The chain only ever fails as an unreadable source: not "no key", and not
        // a reason to fall through to a call without one.
        let key = match resolved {
            Ok(found) => found,
            Err(_) => {
                return Err(self.provider_error(
                    "store_unreadable",
                    "the credential store could not be read",
                    true,
                ));
            }
        };
        // The record again, **after** the key was read: an endpoint move commits
        // its record (disabled, at the new endpoint) before it writes the new key,
        // so a key read here that was entered for another endpoint implies the
        // record has already moved, and this second look sees it. (Record, key,
        // record: the pair is only used if the record did not move across the key
        // read.) Only a key the hub stores per provider is entered for an
        // endpoint; a host's own source (environment, keychain, a rotating token)
        // is not, and skips the extra read.
        if matches!(&key, Some((_, CredentialOrigin::ProviderKey))) {
            self.check_route_is_current().await?;
        }
        let (key, origin) = match key {
            Some((secret, origin)) => {
                let id = secret.id();
                (Some(secret), Some((origin, id)))
            }
            None => (None, None),
        };
        let managed = self.turn.group == crate::taxonomy::ProviderGroup::Managed;
        if key.is_none() && !managed && self.key_required() {
            // Same rule as `resolve_for_turn`: a keyed kind with no key fails
            // closed. A model the host kept from before the key was cleared
            // must not quietly send an empty credential.
            return Err(self.provider_error("no_key", "no key is configured", false));
        }
        if key.is_none() && managed {
            let _ = self
                .hub
                .inner
                .health
                .mark_signed_out_at(&self.scope, &self.turn.slug, Some(epoch))
                .await;
            return Err(self.provider_error("signed_out", "signed out", false));
        }
        {
            let held = self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some((built_for, model)) = held.as_ref()
                && built_for.as_ref().map(Secret::expose) == key.as_ref().map(Secret::expose)
            {
                return Ok((model.clone(), origin));
            }
        }
        let extra = self.hub.request_headers(&self.turn);
        let responses_api = self.hub.serves_responses_api(&self.turn);
        let spec = ModelSpec {
            turn: &self.turn,
            key: key.as_ref().map(Secret::expose),
            extra_headers: &extra,
            responses_api,
        };
        let built = self
            .hub
            .inner
            .models
            .build(&spec)
            .map_err(|e| Error::Unsupported(e.to_string()))?;
        *self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some((key, built.clone()));
        Ok((built, origin))
    }

    async fn observe_ok(&self, started: std::time::Instant, epoch: u64) {
        let latency = self
            .hub
            .inner
            .clock
            .now()
            .saturating_duration_since(started);
        let _ = self
            .hub
            .record_outcome_lenient(&self.scope, &self.turn, Outcome::Ok { latency }, epoch)
            .await;
    }

    async fn observe_err(&self, error: &Error, epoch: u64, used: Option<Used<'_>>) {
        let Some(failure) = failure_of(error) else {
            return;
        };
        let current = self.hub.inner.health.epoch(&self.scope, &self.turn.slug) == epoch;
        if failure.is_rejection() && current {
            // The source that supplied the credential this request used, not
            // whatever answers now: a stale rejection must not refresh a token
            // that has since been rotated.
            if let Some(used) = used {
                self.hub
                    .chain_for(&self.turn.kind)
                    .invalidate_origin_rejected(&self.scope, used.origin, used.id);
            }
            // The cached client holds the rejected key: drop it.
            *self
                .inner
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
        }
        let _ = self
            .hub
            .record_outcome_lenient(&self.scope, &self.turn, Outcome::Failed(failure), epoch)
            .await;
    }
}

/// The provider failure inside an llm error, by reference (llm's `Error` is not
/// `Clone`, but what it carries is).
fn failure_of(error: &Error) -> Option<ProviderFailure> {
    let owned = match error {
        Error::Provider(provider) => Error::Provider(Box::new((**provider).clone())),
        Error::Model(text) => Error::Model(text.clone()),
        Error::Catalog(text) => Error::Catalog(text.clone()),
        Error::Unsupported(_) | Error::Validation(_) | Error::Serialization(_) => return None,
    };
    match HubError::from(owned) {
        HubError::Provider(failure) => Some(failure),
        _ => None,
    }
}

/// Mirrors the billing key both ways so the neutral and the legacy spellings
/// are both present.
fn mirror_usage_meta(response: &mut ModelResponse) {
    let Some(Value::Object(raw)) = response.raw.as_mut() else {
        return;
    };
    match (
        raw.get(USAGE_META).cloned(),
        raw.get(LEGACY_USAGE_META).cloned(),
    ) {
        (None, Some(legacy)) => {
            raw.insert(USAGE_META.to_string(), legacy);
        }
        (Some(neutral), None) => {
            raw.insert(LEGACY_USAGE_META.to_string(), neutral);
        }
        _ => {}
    }
}

#[async_trait]
impl ChatModel<()> for HubModel {
    fn profile(&self) -> Option<&ModelProfile> {
        None
    }

    fn cache_identity(&self) -> Option<String> {
        // Names the route, never a credential.
        Some(format!(
            "hub:{}:{}:{}",
            self.turn.slug,
            self.turn.kind,
            self.turn.model.as_ref().map_or("", |m| m.as_str())
        ))
    }

    async fn invoke(&self, state: &(), request: ModelRequest) -> Result<ModelResponse> {
        // Read before the credential is resolved: what this turn learns is about
        // that credential, and is dropped if the provider's key changes meanwhile.
        let epoch = self.hub.inner.health.epoch(&self.scope, &self.turn.slug);
        let (model, origin) = self.current().await?;
        let started = self.hub.inner.clock.now();
        match model.invoke(state, request).await {
            Ok(mut response) => {
                mirror_usage_meta(&mut response);
                self.observe_ok(started, epoch).await;
                Ok(response)
            }
            Err(error) => {
                self.observe_err(&error, epoch, used(&origin)).await;
                Err(error)
            }
        }
    }

    async fn stream(&self, state: &(), request: ModelRequest) -> Result<ModelStream> {
        let epoch = self.hub.inner.health.epoch(&self.scope, &self.turn.slug);
        let (model, origin) = self.current().await?;
        let started = self.hub.inner.clock.now();
        match model.stream(state, request).await {
            Ok(stream) => {
                // Streaming health (time to first token, mid-stream failures) is
                // the router's; a stream that starts is recorded as a turn that
                // worked.
                self.observe_ok(started, epoch).await;
                Ok(stream)
            }
            Err(error) => {
                self.observe_err(&error, epoch, used(&origin)).await;
                Err(error)
            }
        }
    }
}

fn used(origin: &Option<(CredentialOrigin, SecretId)>) -> Option<Used<'_>> {
    origin.as_ref().map(|(origin, id)| Used { origin, id: *id })
}

#[cfg(test)]
impl HubModel {
    /// Lets a test hand the failure path a stale (or current) epoch directly.
    pub(super) async fn observe_err_for_test(
        &self,
        error: &Error,
        epoch: u64,
        origin: Option<(&CredentialOrigin, SecretId)>,
    ) {
        let used = origin.map(|(origin, id)| Used { origin, id });
        self.observe_err(error, epoch, used).await;
    }
}

impl fmt::Debug for HubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HubModel")
            .field("turn", &self.turn)
            .finish_non_exhaustive()
    }
}
