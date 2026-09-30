//! Endpoint text handling: host extraction, local-endpoint normalisation, and
//! credential refusal/redaction/scrubbing.
//!
//! Pure string functions with no I/O, ported from OpenCompany
//! (`catalogue.rs:590,765,806,900`, `probe.rs:1064`).

mod normalize;
mod redact;

pub use normalize::{endpoint_host, normalize_local_endpoint};
pub(crate) use redact::url_query_has_credential;
pub use redact::{
    REDACTED_USERINFO, endpoint_has_credentials, endpoint_query_has_credential, redact_endpoint,
    scrub_endpoint_credential,
};

/// What an operator is told when an endpoint they typed carries a credential.
///
/// One sentence for every place an endpoint is typed, so two surfaces cannot
/// drift into different advice.
pub const ENDPOINT_CREDENTIAL_REFUSAL: &str = "That endpoint carries a username or password \
    in the URL. Remove them and put the credential in the API key field. An endpoint is stored \
    as written and is readable by everyone who can see this configuration.";

#[cfg(test)]
#[path = "test.rs"]
mod tests;
