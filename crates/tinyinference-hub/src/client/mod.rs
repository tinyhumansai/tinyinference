//! The client: a `ChatModel` for a resolved turn.
//!
//! [`Hub::chat_model`] returns a hub-owned model that **resolves the
//! credential chain on every call**, so a rotated key, a refreshed platform
//! token or a switched credential source is used on the next request without
//! rebuilding anything. It builds (and reuses, while the credential is
//! unchanged) the underlying `tinyinference-llm` model through a
//! [`ModelFactory`], reports each outcome to the provider's health, and on a
//! rejected credential tells the source that supplied it.
//!
//! A model the host keeps reflects **credential** changes on its next call
//! (a rotated key is used, a cleared key fails closed as `no_key`, a rejected
//! token is refreshed). It also re-reads the provider's record on each call and
//! fails closed (`stale_route`) when the provider was removed, disabled, or
//! moved to another endpoint since the model was resolved, so a key can never
//! follow an edit to an origin it was not entered for. The cost is one settings
//! read per call.
//!
//! The underlying transport is `tinyinference-llm`'s own, so the hub's
//! per-redirect policy and address pinning cover probes and catalogs but not
//! turn traffic (open question Q2); the resolved endpoint **is** checked
//! against the endpoint policy at resolve time.
//!
//! # Usage metadata (D4)
//!
//! `ModelResponse::raw` already carries a provider payload, and OpenCompany
//! and OpenHuman both read a billing key out of it (`openhuman_usage_meta`).
//! The hub mirrors it: a response that carries one spelling gets the other, so
//! `usage_meta` and the legacy key are both present for one release without any
//! change to a frozen llm type. The hub never *computes* usage metadata; the
//! backend model that knows the charge does.

mod factory;
mod model;

pub use factory::{LlmModelFactory, ModelFactory, ModelSpec};

use std::sync::Arc;

use tinyinference_llm::model::ChatModel;

use crate::error::{HubError, Operation, Unresolved};
use crate::hub::Hub;
use crate::ids::ScopeKey;
use crate::route::ResolvedTurn;
use crate::taxonomy::Protocol;

impl Hub {
    /// A chat model for a resolved turn.
    ///
    /// Nothing is sent here; the model resolves its credential and builds the
    /// underlying client on first use.
    ///
    /// # Errors
    ///
    /// [`HubError::Unsupported`] for a CLI route (the host runs the binary) and
    /// [`HubError::Unresolved`] when the turn has no model.
    pub async fn chat_model(
        &self,
        scope: &ScopeKey,
        turn: &ResolvedTurn,
    ) -> Result<Arc<dyn ChatModel<()>>, HubError> {
        if turn.protocol == Protocol::CliStream || turn.cli.is_some() {
            return Err(HubError::Unsupported {
                op: Operation::ChatModel,
                kind: turn.kind.clone(),
            });
        }
        if turn.model.is_none() {
            return Err(HubError::Unresolved(Unresolved::NoModel(turn.slug.clone())));
        }
        Ok(Arc::new(model::HubModel::new(
            self.clone(),
            scope.clone(),
            turn.clone(),
        )))
    }
}

#[cfg(test)]
#[path = "test.rs"]
mod tests;
