//! Resolving a turn to one provider and model.
//!
//! Precedence, first match wins and a match that cannot be served **fails
//! closed** (it never falls through to a lower rule):
//!
//! 1. the caller's per-turn override,
//! 2. the agent's pin,
//! 3. the workload's route (an opaque host key, D7),
//! 4. the scope's default: a full default resolves, a provider-only default
//!    fails closed on the turn path, and no default at all is `Unresolved`.
//!
//! The "first enabled provider" fallback exists only in
//! [`Hub::status`](crate::Hub::status) for display, never here. The model is
//! passed through verbatim (guard G25 keeps a host's reserved word from being
//! sent to a provider as a model), and the resolved endpoint is checked against
//! the endpoint policy again.

use crate::config::DefaultChoice;
use crate::descriptor::ProviderRecord;
use crate::error::{HubError, InvalidInput, Operation, Unresolved};
use crate::hub::{Credential, Hub};
use crate::ids::{KindId, ModelId, ScopeKey, Slug};
use crate::policy::check_endpoint_with_credential;
use crate::taxonomy::{AuthStyle, CliKind, LocalRuntime, Protocol, ProviderGroup};

use super::types::{ProviderRoute, ResolvedTurn, ResolvedVia, RouteTarget, Temperature, TurnQuery};

impl Hub {
    /// The credential a provider needs to serve a turn, or the typed reason it
    /// cannot: signed out (managed), or no key where one is required.
    pub(crate) async fn usable_credential(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
    ) -> Result<Credential, HubError> {
        let driver = self.driver(&record.kind)?;
        let descriptor = driver.descriptor();
        let credential = self.credential(scope, record).await?;
        if credential.key.is_none() {
            if descriptor.group == ProviderGroup::Managed {
                return Err(HubError::SignedOut {
                    provider: record.slug.clone(),
                });
            }
            if descriptor.needs_key && Self::auth_of(record, descriptor).needs_credential() {
                return Err(HubError::Unresolved(Unresolved::NoKey(record.slug.clone())));
            }
        }
        Ok(credential)
    }

    /// Resolves what a turn will call.
    ///
    /// The result carries no credential: [`Hub::chat_model`] resolves the
    /// credential chain on every request.
    ///
    /// # Errors
    ///
    /// [`HubError::Unresolved`] naming why nothing resolves (no provider, a
    /// provider-only default, a missing, disabled or keyless provider, no
    /// model); [`HubError::SignedOut`] for the managed provider with no
    /// credential; [`HubError::Policy`] when the endpoint is refused;
    /// [`HubError::Unsupported`] for an ephemeral route; and the stores'
    /// errors.
    pub async fn resolve_for_turn(
        &self,
        scope: &ScopeKey,
        query: &TurnQuery,
    ) -> Result<ResolvedTurn, HubError> {
        let config = self.read_config(scope).await?;
        let (route, via) = if let Some(route) = &query.override_route {
            (route.clone(), ResolvedVia::Override)
        } else if let Some(pin) = query.agent.as_ref().and_then(|a| config.agent_pins.get(a)) {
            (
                ProviderRoute::provider(pin.provider.clone()).with_model(pin.model.clone()),
                ResolvedVia::Pin,
            )
        } else if let Some(route) = query
            .workload
            .as_ref()
            .and_then(|w| config.workload_routes.get(w))
        {
            (route.clone(), ResolvedVia::Workload)
        } else {
            (ProviderRoute::default_route(), ResolvedVia::Default)
        };

        let mut model = route.model.clone();
        let mut via = via;
        let target = match &route.target {
            RouteTarget::Default => {
                via = ResolvedVia::Default;
                match &config.default {
                    DefaultChoice::Full {
                        provider,
                        model: default_model,
                    } => {
                        model = model.or_else(|| Some(default_model.clone()));
                        RouteTarget::Provider(provider.clone())
                    }
                    DefaultChoice::ProviderOnly { provider } => {
                        return Err(HubError::Unresolved(Unresolved::ProviderOnlyDefault(
                            provider.clone(),
                        )));
                    }
                    DefaultChoice::Unset => {
                        return Err(HubError::Unresolved(Unresolved::NoProvider));
                    }
                }
            }
            other => other.clone(),
        };
        if let Some(model) = &model {
            self.check_model(model)?;
        }
        match target {
            RouteTarget::Provider(slug) => {
                let record = config
                    .provider(&slug)
                    .ok_or_else(|| HubError::Unresolved(Unresolved::Missing(slug.clone())))?;
                self.resolve_record(scope, record, model, route.temperature, via)
                    .await
            }
            RouteTarget::Managed => {
                let record = config
                    .providers
                    .iter()
                    .find(|p| self.group_of(p) == ProviderGroup::Managed)
                    .ok_or(HubError::Unresolved(Unresolved::NoTarget(
                        "managed provider",
                    )))?;
                self.resolve_record(scope, record, model, route.temperature, via)
                    .await
            }
            RouteTarget::Local(runtime) => {
                let record = self.pick_local(&config, runtime)?;
                self.resolve_record(scope, record, model, route.temperature, via)
                    .await
            }
            RouteTarget::Cli(cli) => Self::resolve_cli(cli, model, via),
            RouteTarget::Ephemeral => Err(HubError::Unsupported {
                op: Operation::ResolveForTurn,
                kind: KindId::new("ephemeral-route"),
            }),
            RouteTarget::Default => Err(HubError::Unresolved(Unresolved::NoProvider)),
        }
    }

    /// The first enabled local record (of `runtime`, when one is named).
    fn pick_local<'a>(
        &self,
        config: &'a crate::config::HubConfig,
        runtime: Option<LocalRuntime>,
    ) -> Result<&'a ProviderRecord, HubError> {
        let mut candidates = config.providers.iter().filter(|p| {
            self.group_of(p) == ProviderGroup::Local
                && runtime.is_none_or(|rt| {
                    self.inner
                        .registry
                        .get(&p.kind)
                        .and_then(|d| d.descriptor().local_runtime)
                        == Some(rt)
                })
        });
        let first = candidates.clone().next();
        let enabled = candidates.find(|p| p.enabled);
        match (enabled, first) {
            (Some(record), _) => Ok(record),
            (None, Some(disabled)) => Err(HubError::Unresolved(Unresolved::Disabled(
                disabled.slug.clone(),
            ))),
            (None, None) => Err(HubError::Unresolved(Unresolved::NoTarget("local runtime"))),
        }
    }

    fn resolve_cli(
        cli: CliKind,
        model: Option<ModelId>,
        via: ResolvedVia,
    ) -> Result<ResolvedTurn, HubError> {
        let slug = Slug::parse(cli.option_slug()).map_err(HubError::Invalid)?;
        Ok(ResolvedTurn {
            kind: KindId::new(cli.option_slug()),
            slug,
            group: ProviderGroup::Cli,
            base_url: String::new(),
            model,
            protocol: Protocol::CliStream,
            auth: AuthStyle::None,
            via,
            origin: None,
            temperature: None,
            cli: Some(cli),
        })
    }

    async fn resolve_record(
        &self,
        scope: &ScopeKey,
        record: &ProviderRecord,
        model: Option<ModelId>,
        temperature: Option<Temperature>,
        via: ResolvedVia,
    ) -> Result<ResolvedTurn, HubError> {
        if !record.enabled {
            return Err(HubError::Unresolved(Unresolved::Disabled(
                record.slug.clone(),
            )));
        }
        let driver = self.driver(&record.kind)?;
        let descriptor = driver.descriptor();
        let credential = self.usable_credential(scope, record).await?;
        let model = model
            .or_else(|| record.model.clone())
            .ok_or_else(|| HubError::Unresolved(Unresolved::NoModel(record.slug.clone())))?;
        let auth = Self::auth_of(record, descriptor);
        let credentialed = credential.key.is_some() && auth.needs_credential();
        check_endpoint_with_credential(&record.base_url, &self.inner.policy, credentialed)
            .map_err(|refusal| HubError::Policy(crate::error::PolicyViolation::from(refusal)))?;
        Ok(ResolvedTurn {
            slug: record.slug.clone(),
            kind: record.kind.clone(),
            group: descriptor.group,
            base_url: record.base_url.clone(),
            model: Some(model),
            protocol: descriptor.protocol,
            auth,
            via,
            origin: credential.origin,
            temperature,
            cli: descriptor.cli,
        })
    }
}

/// Rejects a route whose model or target could never be stored or resolved:
/// used by importers and hosts before they save a route.
///
/// # Errors
///
/// [`InvalidInput`] naming what is wrong.
pub fn check_route(route: &ProviderRoute) -> Result<(), InvalidInput> {
    if matches!(route.target, RouteTarget::Default | RouteTarget::Ephemeral) {
        return Err(InvalidInput::Malformed {
            field: crate::error::InputField::Config,
            reason: "a workload route must name a provider, the managed provider, a local runtime or a CLI login",
        });
    }
    Ok(())
}
