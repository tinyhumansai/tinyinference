//! [`CliDriver`]: CLI-login kinds have no HTTP surface.

use async_trait::async_trait;

use crate::catalog::Fetched;
use crate::descriptor::ProviderDescriptor;
use crate::error::{HubError, Operation};

use super::{DriverContext, KindDriver, Target};

/// A CLI login (Claude Code, Codex). Readiness is decided by launching the
/// binary through the `ProcessSpawner` port (feature `cli`, a later milestone),
/// never by an HTTP request, so every HTTP-shaped operation is a typed
/// [`HubError::Unsupported`] here.
///
/// The Claude subscription is used only by running the `claude` binary, never
/// by replaying a token (D12).
#[derive(Clone, Debug)]
pub struct CliDriver {
    descriptor: ProviderDescriptor,
}

impl CliDriver {
    /// A driver for one CLI catalogue row.
    pub fn for_descriptor(descriptor: ProviderDescriptor) -> Self {
        Self { descriptor }
    }
}

#[async_trait]
impl KindDriver for CliDriver {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    async fn list_models(
        &self,
        _cx: &DriverContext<'_>,
        _target: &Target<'_>,
    ) -> Result<Fetched, HubError> {
        Err(HubError::Unsupported {
            op: Operation::ListModels,
            kind: self.descriptor.kind.clone(),
        })
    }
}
