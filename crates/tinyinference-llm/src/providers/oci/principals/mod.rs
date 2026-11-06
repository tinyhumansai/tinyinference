//! Instance principal and OKE workload identity auth.
//!
//! Both flows exchange node/pod identity for a short-lived OCI session token
//! bound to a freshly generated RSA session keypair, then sign requests with
//! `ST$<token>` (oci-go-sdk `common/auth`). They need OpenSSL for RSA keygen
//! and X.509 parsing, so they sit behind the `oci-principals` cargo feature;
//! without it [`instance_source`] and [`oke_source`] return
//! [`Error::Unsupported`](crate::Error::Unsupported) naming the feature.

#[cfg(feature = "oci-principals")]
mod instance;
#[cfg(feature = "oci-principals")]
mod keys;
#[cfg(all(test, feature = "oci-principals"))]
mod mock_http;
#[cfg(feature = "oci-principals")]
mod oke;
#[cfg(feature = "oci-principals")]
mod token;

#[cfg(feature = "oci-principals")]
pub use instance::{InstancePrincipalSettings, InstancePrincipalSource};
#[cfg(feature = "oci-principals")]
pub use oke::{OkeWorkloadIdentitySettings, OkeWorkloadIdentitySource};

use std::sync::Arc;

use super::auth::OciTokenSource;
use crate::Result;

/// Instance-principal token source using IMDS defaults (honours
/// `OCI_METADATA_BASE_URL`).
#[cfg(feature = "oci-principals")]
pub(super) fn instance_source() -> Result<Arc<dyn OciTokenSource>> {
    Ok(Arc::new(InstancePrincipalSource::new(
        InstancePrincipalSettings::from_env(),
    )))
}

/// OKE workload-identity token source from the pod environment.
#[cfg(feature = "oci-principals")]
pub(super) fn oke_source() -> Result<Arc<dyn OciTokenSource>> {
    Ok(Arc::new(OkeWorkloadIdentitySource::new(
        OkeWorkloadIdentitySettings::from_env()?,
    )))
}

/// Region implied by OKE workload identity (`OCI_RESOURCE_PRINCIPAL_REGION`).
#[cfg(feature = "oci-principals")]
pub(super) fn oke_region() -> Option<String> {
    oke::region_from_env()
}

/// Region implied by instance principal (read from IMDS).
#[cfg(feature = "oci-principals")]
pub(super) fn instance_region() -> Option<String> {
    instance::fetch_region(&InstancePrincipalSettings::from_env()).ok()
}

#[cfg(not(feature = "oci-principals"))]
fn disabled(flow: &str) -> crate::Error {
    crate::Error::Unsupported(format!(
        "OCI {flow} auth requires the `oci-principals` cargo feature of tinyinference-llm (it needs OpenSSL for RSA keygen and X.509 parsing)"
    ))
}

#[cfg(not(feature = "oci-principals"))]
pub(super) fn instance_source() -> Result<Arc<dyn OciTokenSource>> {
    Err(disabled("instance principal"))
}

#[cfg(not(feature = "oci-principals"))]
pub(super) fn oke_source() -> Result<Arc<dyn OciTokenSource>> {
    Err(disabled("OKE workload identity"))
}

#[cfg(not(feature = "oci-principals"))]
pub(super) fn oke_region() -> Option<String> {
    None
}

#[cfg(not(feature = "oci-principals"))]
pub(super) fn instance_region() -> Option<String> {
    None
}

#[cfg(all(test, not(feature = "oci-principals")))]
#[path = "disabled_tests.rs"]
mod disabled_tests;
