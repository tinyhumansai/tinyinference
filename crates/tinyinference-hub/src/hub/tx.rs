//! Loading and changing the configuration: compare-and-swap with re-checked
//! guards.

use crate::catalogue;
use crate::config::HubConfig;
use crate::descriptor::{ProviderRecord, RecordOrigin};
use crate::error::{HubError, PortName};
use crate::ids::{ScopeKey, Slug};
use crate::ports::{PortError, Version};
use crate::taxonomy::ProviderGroup;

use super::{Hub, MAX_CAS_ATTEMPTS};

/// The result of a committed change.
pub(crate) struct Committed<T> {
    /// What the change closure returned on the attempt that stuck.
    pub(crate) value: T,
    /// Whether the document differed from what was loaded (nothing is saved
    /// when it did not).
    pub(crate) changed: bool,
}

impl Hub {
    /// Makes sure the managed provider's record exists and points at the host's
    /// endpoint. The endpoint is the host's to change, so it is rewritten from
    /// the builder on every load rather than trusted from the stored copy.
    pub(crate) fn ensure_managed(&self, config: &mut HubConfig) {
        let (Some(endpoint), Some(descriptor)) = (
            self.inner.managed_endpoint.as_ref(),
            catalogue::descriptors_in(ProviderGroup::Managed).next(),
        ) else {
            return;
        };
        let Ok(slug) = Slug::parse(descriptor.slug()) else {
            return;
        };
        if let Some(record) = config.provider_mut(&slug) {
            record.base_url.clone_from(endpoint);
            return;
        }
        let mut record = ProviderRecord::new(
            "managed",
            slug,
            descriptor.label,
            descriptor.kind.clone(),
            endpoint.clone(),
        );
        record.synthetic = true;
        record.origin = RecordOrigin::Indexed;
        config.providers.insert(0, record);
    }

    /// Loads the scope's configuration and its version (`None` when nothing has
    /// been saved), with the managed record ensured.
    pub(crate) async fn load(
        &self,
        scope: &ScopeKey,
    ) -> Result<(HubConfig, Option<Version>), HubError> {
        let stored = self
            .inner
            .config
            .load(scope)
            .await
            .map_err(|e| e.into_hub(PortName::Config))?;
        let (mut config, version) = match stored {
            Some((config, version)) => (config, Some(version)),
            None => (HubConfig::new(), None),
        };
        self.ensure_managed(&mut config);
        Ok((config, version))
    }

    /// The configuration as of now, for a read-only operation.
    pub(crate) async fn read_config(&self, scope: &ScopeKey) -> Result<HubConfig, HubError> {
        Ok(self.load(scope).await?.0)
    }

    /// Changes the configuration: load, run `body` (which validates the guards
    /// against exactly what was loaded), save with the loaded version as the
    /// expectation. A lost compare-and-swap loads again and re-runs `body`, up
    /// to [`MAX_CAS_ATTEMPTS`] times.
    ///
    /// # Errors
    ///
    /// Whatever `body` returned, a store failure, or [`HubError::Conflict`] when
    /// every attempt lost.
    pub(crate) async fn transact<T>(
        &self,
        scope: &ScopeKey,
        mut body: impl FnMut(&mut HubConfig) -> Result<T, HubError>,
    ) -> Result<Committed<T>, HubError> {
        for _ in 0..MAX_CAS_ATTEMPTS {
            let (mut config, expect) = self.load(scope).await?;
            let before = config.clone();
            let value = body(&mut config)?;
            config.validate().map_err(HubError::Invalid)?;
            if config == before && expect.is_some() {
                return Ok(Committed {
                    value,
                    changed: false,
                });
            }
            match self.inner.config.save(scope, &config, expect).await {
                Ok(_) => {
                    return Ok(Committed {
                        value,
                        changed: config != before,
                    });
                }
                Err(PortError::Conflict) => {}
                Err(other) => return Err(other.into_hub(PortName::Config)),
            }
        }
        Err(HubError::Conflict)
    }
}
