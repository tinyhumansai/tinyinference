//! [`HeaderPolicy`]: which headers may travel with a request.
//!
//! Two rules, both from OpenCompany: the product-identity header goes only to
//! first-party hosts (guard G26), and a credential header never follows a
//! redirect to another origin (guard G17). `reqwest` strips `Authorization`
//! across hosts but keeps custom headers such as `x-api-key`, so the hub
//! strips the whole credential set itself.

use crate::endpoint::endpoint_host;
use crate::taxonomy::AuthStyle;

use super::endpoint::same_origin;

/// Header-handling rules as data.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HeaderPolicy {
    /// The name of the product-identity header.
    pub product_header: String,
    /// Hosts (and their subdomains) that may receive the product header.
    pub first_party_hosts: Vec<String>,
    /// Lowercase names of headers that carry a credential.
    pub credential_headers: Vec<String>,
}

impl HeaderPolicy {
    /// The built-in policy: the product header goes only to `tinyhumans.ai`
    /// and `openhuman.ai`, and the credential set is `authorization`,
    /// `proxy-authorization`, `x-api-key`, `api-key`, and `cookie`.
    pub fn builtin() -> Self {
        Self {
            product_header: "x-sdk-name".to_string(),
            first_party_hosts: vec!["tinyhumans.ai".to_string(), "openhuman.ai".to_string()],
            credential_headers: [
                "authorization",
                "proxy-authorization",
                "x-api-key",
                "api-key",
                "cookie",
            ]
            .iter()
            .map(|h| (*h).to_string())
            .collect(),
        }
    }

    /// This policy plus the credential header(s) an [`AuthStyle`] sends, so a
    /// [`AuthStyle::Custom`] header (Azure's `api-key`, a gateway's
    /// `x-acme-key`) is stripped on a cross-origin redirect like the built-in
    /// ones. `reqwest` keeps custom headers across origins, so the hub must
    /// know the name.
    #[must_use]
    pub fn with_auth(mut self, auth: &AuthStyle) -> Self {
        for name in auth.credential_headers() {
            if !self.credential_headers.contains(&name) {
                self.credential_headers.push(name);
            }
        }
        self
    }

    /// Whether the product header may be sent to `url`.
    pub fn allows_product_header_to(&self, url: &str) -> bool {
        let Some(host) = endpoint_host(url) else {
            return false;
        };
        // An absolute name (`api.tinyhumans.ai.`) is the same host, and a
        // configured entry is compared case-insensitively.
        let host = host.trim_end_matches('.');
        self.first_party_hosts.iter().any(|first| {
            let first = first.trim_end_matches('.').to_ascii_lowercase();
            host == first || host.ends_with(&format!(".{first}"))
        })
    }

    /// Whether `name` carries a credential (case-insensitive).
    pub fn is_credential_header(&self, name: &str) -> bool {
        let lower = name.trim().to_ascii_lowercase();
        self.credential_headers.contains(&lower)
    }

    /// Removes credential headers (and the product header) from `headers` when
    /// following a redirect from `from_url` to `to_url` crosses an origin.
    /// Same-origin redirects keep everything.
    pub fn strip_for_redirect(
        &self,
        headers: &mut Vec<(String, String)>,
        from_url: &str,
        to_url: &str,
    ) {
        if same_origin(from_url, to_url) {
            return;
        }
        headers.retain(|(name, _)| {
            !self.is_credential_header(name) && !name.eq_ignore_ascii_case(&self.product_header)
        });
    }

    /// Removes the product header from `headers` unless `url` is first-party.
    pub fn strip_product_header_unless_first_party(
        &self,
        headers: &mut Vec<(String, String)>,
        url: &str,
    ) {
        if !self.allows_product_header_to(url) {
            headers.retain(|(name, _)| !name.eq_ignore_ascii_case(&self.product_header));
        }
    }
}

impl Default for HeaderPolicy {
    fn default() -> Self {
        Self::builtin()
    }
}
