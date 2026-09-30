//! The default choice, per-agent pins and per-workload routes.
//!
//! Guards: G8 (a default needs a model; the provider exists and is enabled; the
//! row's model is rewritten in the same save, so there is nothing to restore),
//! G22 (a pin refuses a removed, disabled or keyless provider), G25 (a model on
//! the wire is never a host's reserved word).

use crate::config::{DefaultChoice, ModelChoice};
use crate::descriptor::RecordOrigin;
use crate::error::{HubError, InputField, InvalidInput, NotFound, Unresolved};
use crate::hub::{Hub, Mutation, MutationStatus};
use crate::ids::{AgentKey, ScopeKey, Slug, WorkloadKey};
use crate::ports::HubEvent;
use crate::route::{ProviderRoute, RouteTarget};

impl Hub {
    fn no_provider(slug: &Slug) -> HubError {
        HubError::NotFound(NotFound::Provider(slug.clone()))
    }

    /// Finishes a mutation that names one provider.
    async fn provider_mutation(
        &self,
        scope: &ScopeKey,
        slug: Option<&Slug>,
        changed: bool,
        note: String,
    ) -> Result<Mutation, HubError> {
        let config = self.read_config(scope).await?;
        let view = match slug.and_then(|s| config.provider(s)) {
            Some(record) => Some(self.view(scope, record, &config).await),
            None => None,
        };
        Ok(Mutation {
            status: if changed {
                MutationStatus::Saved
            } else {
                MutationStatus::Unchanged
            },
            note: if changed {
                note
            } else {
                "Nothing changed.".to_string()
            },
            probe: None,
            used_by: None,
            record: view,
        })
    }

    /// Makes `choice` the default: this provider and this model.
    ///
    /// The provider's own model is set to the same value in the same save.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`] for an unknown provider,
    /// [`HubError::Unresolved`] naming a disabled one, [`HubError::Invalid`] for
    /// a model that is a host reserved word, and the stores' errors.
    pub async fn set_default(
        &self,
        scope: &ScopeKey,
        choice: ModelChoice,
    ) -> Result<Mutation, HubError> {
        let model = self.check_model(&choice.model)?;
        let slug = choice.provider.clone();
        let committed = self
            .transact(scope, |config| {
                let record = config
                    .provider_mut(&slug)
                    .ok_or_else(|| Self::no_provider(&slug))?;
                if !record.enabled {
                    return Err(HubError::Unresolved(Unresolved::Disabled(slug.clone())));
                }
                if record.origin != RecordOrigin::EntryZero {
                    record.model = Some(model.clone());
                }
                config.default = DefaultChoice::Full {
                    provider: slug.clone(),
                    model: model.clone(),
                };
                Ok(())
            })
            .await?;
        if committed.changed {
            self.inner.events.emit(HubEvent::DefaultChanged {
                scope: scope.clone(),
            });
        }
        self.provider_mutation(
            scope,
            Some(&slug),
            committed.changed,
            format!("{slug} is now the default."),
        )
        .await
    }

    /// Clears the default. The only way the default is ever unset besides
    /// [`Hub::set_default`] replacing it.
    ///
    /// # Errors
    ///
    /// The stores' errors.
    pub async fn clear_default(&self, scope: &ScopeKey) -> Result<Mutation, HubError> {
        let committed = self
            .transact(scope, |config| {
                config.default = DefaultChoice::Unset;
                Ok(())
            })
            .await?;
        if committed.changed {
            self.inner.events.emit(HubEvent::DefaultChanged {
                scope: scope.clone(),
            });
        }
        self.provider_mutation(
            scope,
            None,
            committed.changed,
            "The default was cleared.".to_string(),
        )
        .await
    }

    /// Pins an agent to a provider and model, or (with `None`) removes its pin.
    ///
    /// A pin refuses a provider that does not exist, is disabled, or has no key
    /// where one is needed (guard G22), so a pin cannot be created that would
    /// fail on its first turn.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`], [`HubError::Unresolved`] naming the disabled or
    /// keyless provider, [`HubError::Invalid`], and the stores' errors.
    pub async fn pin_agent(
        &self,
        scope: &ScopeKey,
        agent: &AgentKey,
        choice: Option<ModelChoice>,
    ) -> Result<Mutation, HubError> {
        let Some(choice) = choice else {
            let committed = self
                .transact(scope, |config| {
                    config.agent_pins.remove(agent);
                    Ok(())
                })
                .await?;
            if committed.changed {
                self.inner.events.emit(HubEvent::DefaultChanged {
                    scope: scope.clone(),
                });
            }
            return self
                .provider_mutation(
                    scope,
                    None,
                    committed.changed,
                    "The pin was removed.".to_string(),
                )
                .await;
        };
        let model = self.check_model(&choice.model)?;
        let slug = choice.provider.clone();
        let config = self.read_config(scope).await?;
        let record = config
            .provider(&slug)
            .ok_or_else(|| Self::no_provider(&slug))?
            .clone();
        // Guard G22: a pin refuses a provider that is disabled or has no key
        // where one is needed. (Re-checked for existence and enabled inside the
        // change; the key lives outside the configuration's version.)
        if !record.enabled {
            return Err(HubError::Unresolved(Unresolved::Disabled(slug)));
        }
        self.usable_credential(scope, &record).await?;
        let pinned = ModelChoice::new(slug.clone(), model);
        let committed = self
            .transact(scope, |config| {
                let record = config
                    .provider(&slug)
                    .ok_or_else(|| Self::no_provider(&slug))?;
                if !record.enabled {
                    return Err(HubError::Unresolved(Unresolved::Disabled(slug.clone())));
                }
                config.agent_pins.insert(agent.clone(), pinned.clone());
                Ok(())
            })
            .await?;
        if committed.changed {
            self.inner.events.emit(HubEvent::DefaultChanged {
                scope: scope.clone(),
            });
        }
        self.provider_mutation(
            scope,
            Some(&slug),
            committed.changed,
            format!("{agent} is pinned to {slug}."),
        )
        .await
    }

    /// Sets a workload's route, or (with `None`) removes it. The workload key is
    /// the host's; the hub never interprets it.
    ///
    /// A route to a provider is not pre-checked for being enabled or keyed: it
    /// is the host's to keep, and it fails closed on the turn path.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`] for a route naming an unknown provider,
    /// [`HubError::Invalid`] for a default or ephemeral route (neither is
    /// stored) or a reserved model word, and the stores' errors.
    pub async fn set_workload_route(
        &self,
        scope: &ScopeKey,
        workload: &WorkloadKey,
        route: Option<ProviderRoute>,
    ) -> Result<Mutation, HubError> {
        if let Some(route) = &route {
            if matches!(route.target, RouteTarget::Default | RouteTarget::Ephemeral) {
                return Err(HubError::Invalid(InvalidInput::Malformed {
                    field: InputField::Config,
                    reason: "a workload route must name a provider, the managed provider, a local runtime or a CLI login",
                }));
            }
            if let Some(model) = &route.model {
                self.check_model(model)?;
            }
        }
        let slug = route.as_ref().and_then(|r| r.provider_slug()).cloned();
        let committed = self
            .transact(scope, |config| {
                match &route {
                    Some(route) => {
                        if let Some(slug) = route.provider_slug()
                            && !config.contains(slug)
                        {
                            return Err(Self::no_provider(slug));
                        }
                        config
                            .workload_routes
                            .insert(workload.clone(), route.clone());
                    }
                    None => {
                        config.workload_routes.remove(workload);
                    }
                }
                Ok(())
            })
            .await?;
        if committed.changed {
            self.inner.events.emit(HubEvent::DefaultChanged {
                scope: scope.clone(),
            });
        }
        self.provider_mutation(
            scope,
            slug.as_ref(),
            committed.changed,
            format!("The route for {workload} was saved."),
        )
        .await
    }
}
