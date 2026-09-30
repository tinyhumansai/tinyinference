//! The [`KindDriver`] trait.

use std::fmt::Debug;

use async_trait::async_trait;

use crate::catalog::Fetched;
use crate::descriptor::ProviderDescriptor;
use crate::error::{HubError, Operation, ProviderFailure, classify_for};
use crate::ids::ModelId;
use crate::taxonomy::{Protocol, TestDepth};

use super::ping::ping_by_protocol;
use super::{DriverContext, Target};

/// How one provider kind is reached.
///
/// A kind that cannot do an operation returns
/// [`HubError::Unsupported`](crate::HubError::Unsupported), never a silent
/// no-op and never a plain string (D6).
#[async_trait]
pub trait KindDriver: Send + Sync + Debug {
    /// The catalogue row this driver serves.
    fn descriptor(&self) -> &ProviderDescriptor;

    /// Classifies a failed answer. The default is the shared classifier with
    /// this kind's rules; override only for a vendor with its own error
    /// dialect.
    fn classify(&self, status: u16, headers: &[(&str, &str)], body: &str) -> ProviderFailure {
        classify_for(Some(self.descriptor().kind.as_str()), status, headers, body)
    }

    /// A cheap key validation call ([`TestDepth::KeyOnly`]).
    ///
    /// # Errors
    ///
    /// [`HubError::Unsupported`] by default; a provider failure when the call
    /// ran and was refused.
    async fn key_check(&self, cx: &DriverContext<'_>, target: &Target<'_>) -> Result<(), HubError> {
        let _ = (cx, target);
        Err(HubError::Unsupported {
            op: Operation::Test(TestDepth::KeyOnly),
            kind: self.descriptor().kind.clone(),
        })
    }

    /// Reads the model listing.
    ///
    /// # Errors
    ///
    /// A classified provider failure, a policy refusal, or
    /// [`HubError::Unsupported`] for a kind with no listing.
    async fn list_models(
        &self,
        cx: &DriverContext<'_>,
        target: &Target<'_>,
    ) -> Result<Fetched, HubError>;

    /// Sends a one-token completion to prove the provider serves `model`
    /// ([`TestDepth::Completion`]). The default speaks the descriptor's
    /// protocol.
    ///
    /// # Errors
    ///
    /// A classified provider failure, or [`HubError::Unsupported`] for a
    /// protocol the hub does not ping.
    async fn completion_ping(
        &self,
        cx: &DriverContext<'_>,
        target: &Target<'_>,
        model: &ModelId,
    ) -> Result<(), HubError> {
        match self.descriptor().protocol {
            Protocol::OpenAiChat | Protocol::OpenAiResponses | Protocol::AnthropicMessages => {
                ping_by_protocol(
                    cx,
                    self.descriptor(),
                    &|status, headers, body| self.classify(status, headers, body),
                    target,
                    model,
                )
                .await
            }
            _ => Err(HubError::Unsupported {
                op: Operation::Test(TestDepth::Completion),
                kind: self.descriptor().kind.clone(),
            }),
        }
    }
}
