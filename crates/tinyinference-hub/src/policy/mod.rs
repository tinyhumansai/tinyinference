//! The endpoint policy: which URLs and addresses the hub may talk to, and
//! which headers may travel with a request.
//!
//! Generalising "test this credential against this URL" to a scope creates an
//! authenticated *send a request to an arbitrary address* primitive, which is
//! SSRF-shaped. [`EndpointPolicy`] is the answer, ported from OpenCompany's
//! `probe.rs` and tightened where noted in `endpoint`. The pure decisions
//! live here; the `Http` port (a later milestone) is contractually bound to
//! apply them on every redirect hop and to connect to the address it checked.

mod endpoint;
mod headers;

pub use endpoint::{
    EndpointPolicy, EndpointRefusal, check_address, check_endpoint, check_endpoint_with_credential,
    check_redirect, resolve_redirect, same_origin,
};
pub use headers::HeaderPolicy;

#[cfg(test)]
#[path = "test.rs"]
mod tests;
