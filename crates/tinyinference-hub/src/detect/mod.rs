//! First-run detection: what is already on this machine.
//!
//! Two sources, both **read-only and never persisted**: local runtimes, found
//! by fingerprinting a known port (a runtime is reported only if it answers the
//! request that only that runtime answers, so the host's own server on `8080` is
//! not mistaken for llama.cpp), and provider keys in the environment (reported
//! as a draft **without** the key: the value is read per request through the
//! credential chain, never copied). Detection is off under a policy that does
//! not allow loopback (a hosted tenant), and sends nothing then.

mod env;
mod local;

pub use env::{ENV_KEYS, detect_env, env_var_for_kind, env_vars_for_kind};
pub use local::{Fingerprint, LocalRuntimeStatus, detect_local, fingerprints};

use crate::config::ProviderDraft;
use crate::error::HubError;
use crate::hub::Hub;
use crate::ports::DetectOptions;

impl Hub {
    /// Finds providers without being told about them.
    ///
    /// Returns drafts, in a stable order (local runtimes by port, then
    /// environment keys in table order), for the operator to confirm with
    /// [`Hub::connect`]. Nothing is stored, and a draft never carries a key. A
    /// policy that does not allow loopback (a hosted tenant) makes this return
    /// nothing without sending anything. A host that supplied its own
    /// [`Detector`](crate::ports::Detector) gets that instead.
    ///
    /// # Errors
    ///
    /// Never today; the `Result` leaves room for a detector that can fail.
    pub async fn detect(&self, options: &DetectOptions) -> Result<Vec<ProviderDraft>, HubError> {
        if !self.inner.policy.allow_loopback {
            return Ok(Vec::new());
        }
        if let Some(detector) = &self.inner.detector {
            return Ok(detector.detect(options).await);
        }
        let mut drafts = detect_local(&*self.inner.http, &self.inner.policy, options).await;
        if let Some(env) = &self.inner.env {
            drafts.extend(detect_env(&**env));
        }
        Ok(drafts)
    }
}

#[cfg(test)]
#[path = "test.rs"]
mod tests;

use crate::descriptor::ProviderRecord;
use crate::error::{NotFound, Operation};
use crate::ids::{ScopeKey, Slug};
use crate::taxonomy::ProviderGroup;

impl Hub {
    /// What a saved local runtime is doing: does it answer, is it who it says,
    /// and how many models does it list.
    ///
    /// # Errors
    ///
    /// [`HubError::NotFound`] and [`HubError::Unsupported`] for a provider that
    /// is not a local runtime.
    pub async fn local_status(
        &self,
        scope: &ScopeKey,
        slug: &Slug,
    ) -> Result<LocalRuntimeStatus, HubError> {
        let config = self.read_config(scope).await?;
        let record: ProviderRecord = config
            .provider(slug)
            .ok_or_else(|| HubError::NotFound(NotFound::Provider(slug.clone())))?
            .clone();
        let driver = self.driver(&record.kind)?;
        let descriptor = driver.descriptor();
        if descriptor.group != ProviderGroup::Local {
            return Err(HubError::Unsupported {
                op: Operation::LocalStatus,
                kind: record.kind.clone(),
            });
        }
        let root = record
            .base_url
            .trim_end_matches('/')
            .trim_end_matches("/v1");
        let mut fingerprinted = None;
        let mut version = None;
        if let Some(runtime) = descriptor.local_runtime
            && let Some(print) = fingerprints().iter().find(|p| p.runtime == runtime)
            && let Some(found) =
                local::fingerprint_at(&*self.inner.http, &self.inner.policy, root, print).await
        {
            fingerprinted = Some(runtime);
            version = found;
        }
        // A list served from before because the runtime is down says nothing
        // about it now, so only a fresh read counts.
        let models = self
            .list_models(scope, slug, true)
            .await
            .ok()
            .filter(|list| !list.is_stale())
            .map(|list| list.models.len());
        Ok(LocalRuntimeStatus {
            reachable: fingerprinted.is_some() || models.is_some(),
            fingerprinted,
            version,
            models,
        })
    }
}
