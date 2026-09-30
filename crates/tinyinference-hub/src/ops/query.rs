//! Reading and checking providers: probes, model lists, health and status.

use crate::catalog::{CatalogKey, Freshness, ModelList, merge_metadata};
use crate::catalogue::catalog_shape_for;
use crate::config::ProviderDraft;
use crate::descriptor::ProviderRecord;
use crate::error::{
    HubError, InputField, InvalidInput, NotFound, Operation, PolicyViolation, ReasonCode,
};
use crate::health::{Outcome, ProviderHealth};
use crate::hub::{Credential, Hub, HubStatus, ProviderStatus, Retested};
use crate::ids::{ModelId, ScopeKey, Slug};
use crate::kinds::Target;
use crate::policy::check_endpoint_with_credential;
use crate::ports::HubEvent;
use crate::probe::{ProbeReport, run_probe};
use crate::taxonomy::{ProviderGroup, TestDepth};

impl Hub {
    /// Runs a probe against a saved record with the credential already
    /// resolved, records the result as the record's health, and tells the
    /// credential source when the credential was rejected.
    pub(crate) async fn probe_record(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
        depth: TestDepth,
        model: Option<&ModelId>,
        credential: &Credential,
    ) -> Result<ProbeReport, HubError> {
        let driver = self.driver(&record.kind)?;
        let descriptor = driver.descriptor();
        let auth = Self::auth_of(record, descriptor);
        let headers = self.headers_for(&auth);
        let mut target = Target::new(
            &record.slug,
            &record.kind,
            descriptor.group,
            &record.base_url,
            &auth,
        );
        if let Some(key) = credential.key.as_ref() {
            target = target.with_credential(key);
        }
        if let Some(model) = model.or(record.model.as_ref()) {
            target = target.with_model(model);
        }
        let cx = self.cx(&headers);
        // The epoch was read before the credential was resolved (see
        // `Credential`): a key change while the probe is in flight makes its
        // result about a credential that is gone, and it must not be recorded.
        let epoch = credential.epoch;
        match run_probe(&cx, &*driver, &target, depth).await {
            Ok(report) => {
                // Recording is best effort: the check ran, and its report is what
                // the caller asked for. A health store that is down says so on the
                // next read of health.
                if let Err(error) = self
                    .inner
                    .health
                    .record_probe_at(scope, &record.slug, &report, Some(epoch))
                    .await
                {
                    tracing::warn!(slug = %record.slug, reason = %error.reason(), "could not record a probe");
                }
                if let Some(failure) = &report.failure {
                    self.note_rejection(scope, &record.kind, credential, failure);
                }
                Ok(report)
            }
            Err(HubError::SignedOut { provider }) => {
                if let Err(error) = self
                    .inner
                    .health
                    .mark_signed_out_at(scope, &record.slug, Some(epoch))
                    .await
                {
                    tracing::warn!(slug = %record.slug, reason = %error.reason(), "could not record signed out");
                }
                Err(HubError::SignedOut { provider })
            }
            Err(other) => Err(other),
        }
    }

    /// Checks a provider that is not saved. Nothing is stored, cached or
    /// recorded as health.
    ///
    /// A provider *failing* the check is a fact in the report, not an error:
    /// the operator is typing a key and wants to hear why it does not work. The
    /// draft's own key is the only credential used; a saved provider's key is
    /// never lent to a draft.
    ///
    /// # Errors
    ///
    /// The draft's validation errors, [`HubError::Unsupported`] for a depth the
    /// kind does not offer, and [`HubError::Invalid`] for a missing key or model
    /// the depth needs.
    pub async fn probe_draft(
        &self,
        _scope: &ScopeKey,
        draft: &ProviderDraft,
        depth: TestDepth,
    ) -> Result<ProbeReport, HubError> {
        let plan = self.plan_add(Operation::ProbeDraft, draft)?;
        let driver = self.driver(&plan.kind)?;
        let descriptor = driver.descriptor();
        if depth == TestDepth::Completion && plan.model.is_none() {
            return Err(HubError::Invalid(InvalidInput::Empty(InputField::ModelId)));
        }
        let auth = descriptor.auth.clone();
        let headers = self.headers_for(&auth);
        let mut target = Target::new(&plan.slug, &plan.kind, plan.group, &plan.base_url, &auth);
        if let Some(key) = plan.key.as_ref() {
            target = target.with_credential(key);
        }
        if let Some(model) = plan.model.as_ref() {
            target = target.with_model(model);
        }
        let cx = self.cx(&headers);
        run_probe(&cx, &*driver, &target, depth).await
    }

    /// Checks a saved provider at `depth` and records the result as its health.
    ///
    /// `model` overrides the record's model for a `Completion` check.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`], [`HubError::Unsupported`] for a depth the kind
    /// does not offer, [`HubError::SignedOut`] for a managed provider with no
    /// credential, [`HubError::Invalid`] for a missing key or model,
    /// [`HubError::Conflict`] when the provider was replaced by another under the
    /// same slug, or kept moving, while this resolved its credential (a single
    /// endpoint move is followed, never mixed: the credential is only ever paired
    /// with the endpoint it was entered for), and [`HubError::StoreUnreadable`].
    pub async fn test(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        depth: TestDepth,
        model: Option<&ModelId>,
    ) -> Result<ProbeReport, HubError> {
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        let (record, credential) = self.checked_credential(scope, &record).await?;
        self.probe_record(scope, &record, depth, model, &credential)
            .await
    }

    /// The models a provider lists.
    ///
    /// Cached per endpoint (an hour for a listing, a minute for a failure; a
    /// rejected key or a `403` is never remembered) and single-flight.
    /// `refresh` skips the cache (a Refresh button). When the provider fails and
    /// an older list exists, the older list is returned marked
    /// [`Freshness::Stale`]. A managed provider with no credential is
    /// [`HubError::SignedOut`], never an empty list.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`], [`HubError::SignedOut`], [`HubError::Unsupported`]
    /// (a CLI has no listing), [`HubError::Policy`] for an endpoint the policy
    /// refuses, [`HubError::Provider`] with the classified failure,
    /// [`HubError::Conflict`] when the provider was replaced by another under the
    /// same slug, or kept moving, while this resolved its credential (a single
    /// endpoint move is followed), and [`HubError::StoreUnreadable`].
    pub async fn list_models(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        refresh: bool,
    ) -> Result<ModelList, HubError> {
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        let driver = self.driver(&record.kind)?;
        let descriptor = driver.descriptor();
        if descriptor.group == ProviderGroup::Cli {
            return Err(HubError::Unsupported {
                op: Operation::ListModels,
                kind: record.kind.clone(),
            });
        }
        let (record, credential) = self.checked_credential(scope, &record).await?;
        let auth = Self::auth_of(&record, descriptor);
        if credential.key.is_none() && auth.needs_credential() {
            if descriptor.group == ProviderGroup::Managed {
                if let Err(error) = self
                    .inner
                    .health
                    .mark_signed_out_at(scope, slug, Some(credential.epoch))
                    .await
                {
                    tracing::warn!(%slug, reason = %error.reason(), "could not record signed out");
                }
                return Err(HubError::SignedOut {
                    provider: slug.clone(),
                });
            }
            if descriptor.needs_key {
                return Err(HubError::Invalid(InvalidInput::Empty(InputField::Key)));
            }
        }
        let credentialed = credential.key.is_some() && auth.needs_credential();
        check_endpoint_with_credential(&record.base_url, &self.inner.policy, credentialed)
            .map_err(|refusal| HubError::Policy(PolicyViolation::from(refusal)))?;

        let shape = if descriptor.group == ProviderGroup::Managed {
            descriptor.catalog
        } else {
            catalog_shape_for(record.kind.as_str(), &record.base_url)
        };
        let headers = self.headers_for(&auth);
        let mut target = Target::new(
            &record.slug,
            &record.kind,
            descriptor.group,
            &record.base_url,
            &auth,
        );
        if let Some(key) = credential.key.as_ref() {
            target = target.with_credential(key);
        }
        let cx = self.cx(&headers);
        let key = CatalogKey::new(scope, slug, credentialed, &record.base_url, shape);
        let read = self
            .inner
            .cache
            .read_as(
                key,
                refresh,
                credential.cache_identity(credentialed),
                || async { driver.list_models(&cx, &target).await },
            )
            .await;
        // A key change while the list was being read makes what was cached the
        // old credential's entitlement list: drop it rather than let it serve the
        // new key for an hour.
        if self.inner.health.epoch(scope, slug) != credential.epoch {
            self.inner.cache.evict_scope(scope);
        }
        let mut list = match read {
            Ok(list) => list,
            Err(error) => {
                if let HubError::Provider(failure) = &error {
                    self.note_rejection(scope, &record.kind, &credential, failure);
                }
                return Err(error);
            }
        };
        match &list.freshness {
            Freshness::Fresh => self.inner.events.emit(HubEvent::CatalogFetched {
                scope: scope.clone(),
                slug: slug.clone(),
                models: list.models.len(),
            }),
            Freshness::Stale { failure } => self.inner.events.emit(HubEvent::CatalogServedStale {
                scope: scope.clone(),
                slug: slug.clone(),
                reason: failure.reason,
            }),
            Freshness::Cached => {}
        }
        if self.inner.metadata.is_some() || !self.inner.overrides.is_empty() {
            merge_metadata(
                std::sync::Arc::make_mut(&mut list.models),
                &record.kind,
                self.inner.metadata.as_deref(),
                &self.inner.overrides,
            );
        }
        Ok(list)
    }

    async fn status_of(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
        config: &crate::config::HubConfig,
    ) -> Result<ProviderStatus, HubError> {
        let view = self.view(scope, record, config).await;
        let (health, snapshot) = self.effective_health(scope, record, &view.key).await?;
        Ok(ProviderStatus {
            view,
            health,
            snapshot,
        })
    }

    /// One provider's health, key state and the signals behind them.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`] and [`HubError::StoreUnreadable`].
    pub async fn health(&self, scope: &ScopeKey, slug: &Slug) -> Result<ProviderStatus, HubError> {
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?;
        self.status_of(scope, record, &config).await
    }

    /// Every provider at a glance: the managed provider first (D5), then the
    /// operator's rows in the order they were added.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the configuration or health store
    /// cannot be read.
    pub async fn status(&self, scope: &ScopeKey) -> Result<HubStatus, HubError> {
        let config = self.read_config(scope).await?;
        let mut providers = Vec::with_capacity(config.providers.len());
        for record in &config.providers {
            providers.push(self.status_of(scope, record, &config).await?);
        }
        // Managed first, everything else in order (a stable sort).
        providers.sort_by_key(|p| p.view.group != ProviderGroup::Managed);
        let primary = match &config.default {
            crate::config::DefaultChoice::Full { provider, .. }
            | crate::config::DefaultChoice::ProviderOnly { provider }
                if providers
                    .iter()
                    .any(|p| &p.view.record.slug == provider && p.view.record.enabled) =>
            {
                Some(provider.clone())
            }
            _ => providers
                .iter()
                .find(|p| p.view.record.enabled)
                .map(|p| p.view.record.slug.clone()),
        };
        Ok(HubStatus {
            providers,
            default: config.default.clone(),
            primary,
        })
    }

    /// Feeds a real turn's outcome into the provider's health.
    ///
    /// The client built by [`Hub::chat_model`] does this itself; call it only
    /// for turns that bypass the client. A rejected credential is passed to the
    /// source that supplied it.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`] and [`HubError::StoreUnreadable`].
    pub async fn record_outcome(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
        outcome: Outcome,
    ) -> Result<(), HubError> {
        let config = self.read_config(scope).await?;
        let record = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?;
        self.inner
            .health
            .record_outcome(scope, slug, &outcome)
            .await?;
        if let Outcome::Failed(failure) = &outcome
            && failure.is_rejection()
            && let Ok(credential) = self.credential(scope, record).await
        {
            self.note_rejection_unattributed(scope, &record.kind, &credential, failure);
        }
        Ok(())
    }

    /// Re-tests every enabled provider that is `Down` on a rejected key or an
    /// exhausted account and whose last failure is older than
    /// [`HubPolicy::retest_after`](crate::hub::HubPolicy).
    ///
    /// Such a provider is no longer routed to, so no real turn will ever clear
    /// it; only a deliberate check can. The hub spawns nothing, so a host calls
    /// this on its own schedule. The check is a completion when the record has a
    /// model (the only depth that clears a rejected chat key), otherwise a
    /// catalog read (which lifts only what a catalog read failed). Every re-test
    /// stamps its time whatever it finds, so the next one waits again.
    ///
    /// # Errors
    ///
    /// [`HubError::StoreUnreadable`] when the configuration or health store
    /// cannot be read. A provider whose credential store cannot be read is
    /// skipped.
    pub async fn retest_down(&self, scope: &ScopeKey) -> Result<Vec<Retested>, HubError> {
        let config = self.read_config(scope).await?;
        let now = self.inner.clock.wall_ms();
        let wait =
            u64::try_from(self.inner.hub_policy.retest_after.as_millis()).unwrap_or(u64::MAX);
        let mut done = Vec::new();
        for record in config.providers.iter().filter(|r| r.enabled) {
            let snapshot = self.inner.health.snapshot(scope, &record.slug).await?;
            let ProviderHealth::Down(reason) = snapshot.health else {
                continue;
            };
            if !matches!(reason, ReasonCode::Auth | ReasonCode::Quota) {
                continue;
            }
            let last_retest = self
                .inner
                .retests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(&(scope.clone(), record.slug.clone()))
                .copied()
                .unwrap_or(0);
            let since = snapshot
                .last_failure
                .map_or(snapshot.changed_at_ms, |f| f.at_ms)
                .max(last_retest);
            if now < since.saturating_add(wait) {
                continue;
            }
            let driver = self.driver(&record.kind)?;
            let descriptor = driver.descriptor();
            let depth =
                if record.model.is_some() && descriptor.supports_depth(TestDepth::Completion) {
                    TestDepth::Completion
                } else if descriptor.supports_depth(TestDepth::Catalog) {
                    TestDepth::Catalog
                } else {
                    continue;
                };
            let Ok((record, credential)) = self.checked_credential(scope, record).await else {
                continue;
            };
            let record = &record;
            if credential.key.is_none() && Self::auth_of(record, descriptor).needs_credential() {
                continue;
            }
            // Stamped before the check, whatever it finds: a check that passes
            // but cannot lift the failure (a catalog read for a provider that
            // failed a chat completion) must not be repeated on every tick.
            self.inner
                .retests
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert((scope.clone(), record.slug.clone()), now);
            let Ok(report) = self
                .probe_record(scope, record, depth, None, &credential)
                .await
            else {
                continue;
            };
            let health = self.inner.health.health(scope, &record.slug).await?;
            self.inner.events.emit(HubEvent::Retested {
                scope: scope.clone(),
                slug: record.slug.clone(),
                ok: report.ok(),
            });
            done.push(Retested {
                slug: record.slug.clone(),
                failure: report.failure,
                health,
            });
        }
        Ok(done)
    }
}

impl Hub {
    /// Headers every request of a turn carries: the kind's own and the product
    /// header when (and only when) the endpoint is first-party (guard G26).
    pub(crate) fn request_headers(
        &self,
        turn: &crate::route::ResolvedTurn,
    ) -> Vec<(String, String)> {
        let mut headers: Vec<(String, String)> = self
            .inner
            .registry
            .get(&turn.kind)
            .map(|d| {
                d.descriptor()
                    .extra_headers
                    .iter()
                    .map(|(n, v)| ((*n).to_string(), (*v).to_string()))
                    .collect()
            })
            .unwrap_or_default();
        if let Some((name, value)) = &self.inner.product {
            let mut product = vec![(name.clone(), value.clone())];
            self.inner
                .headers
                .strip_product_header_unless_first_party(&mut product, &turn.base_url);
            headers.extend(product);
        }
        headers
    }

    /// Whether the turn's endpoint serves the OpenAI Responses API: the OpenAI
    /// row on OpenAI's own host, the only one that does (limitation 1).
    pub(crate) fn serves_responses_api(&self, turn: &crate::route::ResolvedTurn) -> bool {
        self.inner.registry.get(&turn.kind).is_some_and(|d| {
            d.descriptor()
                .has_quirk(crate::descriptor::Quirk::ResponsesApi)
        }) && crate::endpoint::endpoint_host(&turn.base_url).as_deref() == Some("api.openai.com")
    }

    /// Records a turn's outcome without failing the turn if the health store is
    /// down: observing a turn must never be what breaks it.
    pub(crate) async fn record_outcome_lenient(
        &self,
        scope: &ScopeKey,
        turn: &crate::route::ResolvedTurn,
        outcome: Outcome,
        epoch: u64,
    ) -> Result<(), HubError> {
        self.inner
            .health
            .record_outcome_at(scope, &turn.slug, &outcome, Some(epoch))
            .await?;
        Ok(())
    }
}
