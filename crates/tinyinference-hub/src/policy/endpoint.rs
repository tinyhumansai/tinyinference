//! [`EndpointPolicy`] and the pure endpoint checks.
//!
//! Ported from OpenCompany `probe.rs:447-552`, with three deliberate
//! tightenings (each has a test):
//!
//! * the URL is parsed with the WHATWG parser (`url` crate), so alternative
//!   IPv4 spellings (`http://2130706433/`, `http://0x7f.1/`) are checked as the
//!   address a client will actually connect to, instead of slipping through as
//!   "a hostname";
//! * `localhost` and `*.localhost` are loopback **by name**, so a policy that
//!   refuses loopback also refuses them;
//! * NAT64 (`64:ff9b::/96`) and the deprecated IPv4-compatible IPv6 form are
//!   unwrapped to the IPv4 address they embed, as IPv4-mapped already was.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::time::Duration;

use url::{Host, Url};

use crate::error::PolicyViolation;

/// Why an endpoint may not be used.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EndpointRefusal {
    /// Not a URL a host can be read out of.
    Unparseable,
    /// Something other than `http` or `https`.
    Scheme,
    /// Loopback, and this deployment does not offer local runtimes.
    Loopback,
    /// A link-local, site-local or cloud-metadata address. Never allowed.
    LinkLocal,
    /// A private, carrier-grade-NAT, reserved or otherwise non-routable
    /// address.
    PrivateNetwork,
    /// `http` to somewhere other than this host, with a credential to present.
    Cleartext,
    /// A public host, and this deployment is local-only.
    NonLocal,
    /// The URL carries `user:password@`. A credential belongs in the key
    /// field, never in an endpoint (guard G16).
    CredentialInUrl,
}

impl fmt::Display for EndpointRefusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unparseable => write!(f, "that is not an endpoint address"),
            Self::Scheme => write!(f, "an endpoint must be http or https"),
            Self::Loopback => write!(f, "this host does not offer local model runtimes"),
            Self::LinkLocal => write!(f, "a model endpoint is never on a link-local address"),
            Self::PrivateNetwork => write!(
                f,
                "a model endpoint is never on this host's private network"
            ),
            Self::Cleartext => write!(
                f,
                "a key cannot be sent to an http endpoint off this host; use https"
            ),
            Self::NonLocal => write!(f, "this deployment only reaches local model runtimes"),
            Self::CredentialInUrl => write!(
                f,
                "an endpoint cannot carry a username or password; put the credential in the key field"
            ),
        }
    }
}

impl std::error::Error for EndpointRefusal {}

/// What a deployment lets the hub reach, and the caps it applies to what comes
/// back.
///
/// Start from a preset ([`EndpointPolicy::hosted`], [`EndpointPolicy::desktop`],
/// [`EndpointPolicy::local_only`]) and adjust with the `with_*` methods. The
/// default is [`EndpointPolicy::hosted`], the most restrictive.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EndpointPolicy {
    /// Loopback is acceptable. An explicit allowance made because local
    /// runtimes exist, not a hole left open.
    pub allow_loopback: bool,
    /// RFC 1918, CGNAT and unique-local addresses are acceptable (a model on a
    /// LAN box). Link-local and metadata addresses are refused regardless.
    pub allow_private: bool,
    /// Public hosts (any name, any routable address) are acceptable. False for
    /// an offline, local-only deployment.
    pub allow_public: bool,
    /// A credential over cleartext `http` may only go to this host's loopback.
    pub credentialed_http_loopback_only: bool,
    /// Redirects followed per request; every hop is re-checked.
    pub max_redirects: usize,
    /// Total time for a request, including connect, TLS and body.
    pub timeout: Duration,
    /// Bytes of a **failure** body read before the rest is discarded.
    pub fail_body_cap: usize,
    /// Bytes of a successful model-catalog body read before it is refused as
    /// too large.
    pub catalog_cap: usize,
    /// Bytes of one page of a paged catalog.
    pub page_cap: usize,
}

impl EndpointPolicy {
    fn base() -> Self {
        Self {
            allow_loopback: false,
            allow_private: false,
            allow_public: true,
            credentialed_http_loopback_only: true,
            max_redirects: 3,
            timeout: Duration::from_secs(10),
            fail_body_cap: 64 * 1024,
            catalog_cap: 16 * 1024 * 1024,
            page_cap: 4 * 1024 * 1024,
        }
    }

    /// A hosted, multi-tenant deployment: no loopback, no private ranges.
    pub fn hosted() -> Self {
        Self::base()
    }

    /// A desktop or single-user deployment: loopback allowed so Ollama and LM
    /// Studio are reachable.
    pub fn desktop() -> Self {
        Self {
            allow_loopback: true,
            ..Self::base()
        }
    }

    /// An offline deployment: only this machine's runtimes are reachable, and
    /// cloud rows are expected to report a disabled health.
    pub fn local_only() -> Self {
        Self {
            allow_loopback: true,
            allow_public: false,
            ..Self::base()
        }
    }

    /// Sets whether loopback is allowed.
    #[must_use]
    pub fn with_loopback(mut self, allow: bool) -> Self {
        self.allow_loopback = allow;
        self
    }

    /// Sets whether private ranges are allowed.
    #[must_use]
    pub fn with_private(mut self, allow: bool) -> Self {
        self.allow_private = allow;
        self
    }

    /// Sets the redirect limit.
    #[must_use]
    pub fn with_max_redirects(mut self, max: usize) -> Self {
        self.max_redirects = max;
        self
    }

    /// Sets the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for EndpointPolicy {
    fn default() -> Self {
        Self::hosted()
    }
}

fn parse(url: &str) -> Result<Url, EndpointRefusal> {
    let parsed = Url::parse(url.trim()).map_err(|_| EndpointRefusal::Unparseable)?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(EndpointRefusal::Scheme);
    }
    Ok(parsed)
}

fn is_loopback_name(domain: &str) -> bool {
    let name = domain.trim_end_matches('.').to_ascii_lowercase();
    name == "localhost" || name.ends_with(".localhost")
}

/// Whether `url` may be used under `policy`.
///
/// * the scheme must be `http` or `https`;
/// * link-local and cloud metadata addresses (`169.254.0.0/16`, `fe80::/10`,
///   `fec0::/10`) are refused outright, because that range is where a
///   container's credentials live;
/// * private, CGNAT, unspecified, multicast and reserved ranges are refused
///   unless the policy allows private networks (and unspecified, multicast and
///   broadcast are refused regardless);
/// * loopback, including `localhost` by name, only where the policy allows it;
/// * a public name or address only where the policy allows public hosts;
/// * userinfo (`user:password@`) and credential-named query parameters
///   (`?key=`, `?api_key=`) are refused after the host checks, so the
///   address answer for a metadata host still wins, but no URL that carries a
///   credential is ever accepted (guard G16). OpenCompany refused it on its
///   write path; the hub refuses it wherever an endpoint is checked.
///
/// A hostname is **not resolved** here: resolving would be a DNS lookup in a
/// pure function, and a check performed before a resolve is defeated by the
/// resolve changing underneath it anyway. The `Http` port's contract is to
/// check every resolved address with [`check_address`] and connect to that
/// pinned address. Apply this to every redirect target too: a permitted host
/// that redirects to the metadata address is the whole trick.
///
/// # Errors
///
/// An [`EndpointRefusal`] naming why.
pub fn check_endpoint(url: &str, policy: &EndpointPolicy) -> Result<(), EndpointRefusal> {
    let parsed = parse(url)?;
    // A special scheme (`http`, `https`) always has a host once it parses; the
    // `ok_or` keeps the impossible case an error rather than a panic.
    match parsed.host().ok_or(EndpointRefusal::Unparseable)? {
        Host::Ipv4(v4) => check_address(IpAddr::V4(v4), policy)?,
        Host::Ipv6(v6) => check_address(IpAddr::V6(v6), policy)?,
        Host::Domain(name) => {
            if is_loopback_name(name) {
                loopback(policy)?;
            } else if !policy.allow_public {
                return Err(EndpointRefusal::NonLocal);
            }
        }
    }
    if !parsed.username().is_empty()
        || parsed.password().is_some()
        || crate::endpoint::url_query_has_credential(&parsed)
    {
        return Err(EndpointRefusal::CredentialInUrl);
    }
    Ok(())
}

/// [`check_endpoint`], plus the rule that only applies when there is a key.
///
/// **A bearer over plain `http` is the key, in the clear, to everything on the
/// path.** `http` stays allowed because Ollama documents
/// `http://localhost:11434` and there is no certificate to have there, so the
/// scheme cannot simply be narrowed to `https`. What separates the two cases is
/// the destination: loopback never leaves this host, and anything else with a
/// credential attached does. Loopback counts by the exact name `localhost` as well as by literal.
///
/// # Errors
///
/// Everything [`check_endpoint`] returns, plus [`EndpointRefusal::Cleartext`].
pub fn check_endpoint_with_credential(
    url: &str,
    policy: &EndpointPolicy,
    has_credential: bool,
) -> Result<(), EndpointRefusal> {
    check_endpoint(url, policy)?;
    if !has_credential || !policy.credentialed_http_loopback_only {
        return Ok(());
    }
    let parsed = parse(url)?;
    if parsed.scheme() != "http" {
        return Ok(());
    }
    let on_this_host = match parsed.host().ok_or(EndpointRefusal::Unparseable)? {
        // Only the exact name `localhost`: a `*.localhost` subdomain is loopback
        // by RFC 6761 but only some resolvers pin it there, so a key must not
        // be sent to it in the clear.
        Host::Domain(name) => name.trim_end_matches('.').eq_ignore_ascii_case("localhost"),
        Host::Ipv4(v4) => v4.is_loopback(),
        // Only an IPv4-mapped loopback stays on this machine; NAT64, 6to4 and
        // IPv4-compatible forms leave it through a gateway or relay.
        Host::Ipv6(v6) => {
            v6.is_loopback() || v6.to_ipv4_mapped().is_some_and(|v4| v4.is_loopback())
        }
    };
    if on_this_host {
        Ok(())
    } else {
        Err(EndpointRefusal::Cleartext)
    }
}

/// The scheme, host and effective port of a URL, lowercased, or `None`.
fn origin(url: &str) -> Option<(String, String, u16)> {
    let parsed = Url::parse(url.trim()).ok()?;
    let host = parsed.host_str()?.to_ascii_lowercase();
    Some((
        parsed.scheme().to_ascii_lowercase(),
        host,
        parsed.port_or_known_default()?,
    ))
}

/// Whether two URLs name the same origin: scheme, host and port.
///
/// **A credentialed request must not follow a redirect off its origin.**
/// `reqwest` strips `Authorization` when the host changes but keeps a custom
/// header, and a non-bearer style sends the key as `x-api-key`, so a provider
/// that can answer `302` could hand a key to any host it names. An unparseable
/// URL on either side is not a match: refusing to follow costs a catalogue
/// read; following costs the key.
pub fn same_origin(a: &str, b: &str) -> bool {
    match (origin(a), origin(b)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

/// The IPv4 address an IPv6 address embeds, if it is one of the forms that is
/// "the same machine wearing a longer name": IPv4-mapped (`::ffff:a.b.c.d`),
/// the well-known NAT64 prefix (`64:ff9b::a.b.c.d`), 6to4
/// (`2002:aabb:ccdd::/48`, which embeds the address in bits 16..48), IPv4-translated
/// SIIT (`::ffff:0:a.b.c.d`), or the
/// deprecated IPv4-compatible form (`::a.b.c.d`, excluding `::` and `::1`).
fn embedded_v4(ip: Ipv6Addr) -> Option<Ipv4Addr> {
    if let Some(mapped) = ip.to_ipv4_mapped() {
        return Some(mapped);
    }
    let s = ip.segments();
    if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        return Some(Ipv4Addr::new(
            (s[6] >> 8) as u8,
            s[6] as u8,
            (s[7] >> 8) as u8,
            s[7] as u8,
        ));
    }
    // IPv4-translated (SIIT) `::ffff:0:a.b.c.d`.
    if s[..4] == [0, 0, 0, 0] && s[4] == 0xffff && s[5] == 0 {
        return Some(Ipv4Addr::new(
            (s[6] >> 8) as u8,
            s[6] as u8,
            (s[7] >> 8) as u8,
            s[7] as u8,
        ));
    }
    if s[0] == 0x2002 {
        return Some(Ipv4Addr::new(
            (s[1] >> 8) as u8,
            s[1] as u8,
            (s[2] >> 8) as u8,
            s[2] as u8,
        ));
    }
    if s[..6] == [0, 0, 0, 0, 0, 0] && ip != Ipv6Addr::UNSPECIFIED && ip != Ipv6Addr::LOCALHOST {
        return Some(Ipv4Addr::new(
            (s[6] >> 8) as u8,
            s[6] as u8,
            (s[7] >> 8) as u8,
            s[7] as u8,
        ));
    }
    None
}

/// The address half of [`check_endpoint`], exposed so a redirect target or a
/// resolved DNS answer can be checked before connecting to it.
///
/// # Errors
///
/// An [`EndpointRefusal`] naming why the address is refused.
pub fn check_address(ip: IpAddr, policy: &EndpointPolicy) -> Result<(), EndpointRefusal> {
    match ip {
        IpAddr::V4(v4) => check_v4(v4, policy),
        IpAddr::V6(v6) => {
            // The same machine wearing a longer name gets the same answer.
            // Checking only the v6 shape is how `::ffff:169.254.169.254`
            // reaches a metadata service.
            if let Some(v4) = embedded_v4(v6) {
                return check_v4(v4, policy);
            }
            if v6.is_loopback() {
                return loopback(policy);
            }
            let s = v6.segments();
            // fe80::/10 link-local, and fec0::/10 site-local.
            if (s[0] & 0xffc0) == 0xfe80 || (s[0] & 0xffc0) == 0xfec0 {
                return Err(EndpointRefusal::LinkLocal);
            }
            // Unspecified, multicast, Teredo (2001::/32, which obfuscates a
            // client IPv4), the local-use NAT64 prefix (64:ff9b:1::/48) and the
            // documentation prefix (2001:db8::/32) are never a real endpoint.
            let teredo = s[0] == 0x2001 && s[1] == 0;
            let local_nat64 = s[0] == 0x64 && s[1] == 0xff9b && s[2] == 1;
            let documentation = s[0] == 0x2001 && s[1] == 0x0db8;
            if v6 == Ipv6Addr::UNSPECIFIED
                || v6.is_multicast()
                || teredo
                || local_nat64
                || documentation
            {
                return Err(EndpointRefusal::PrivateNetwork);
            }
            // fc00::/7 unique-local.
            if (s[0] & 0xfe00) == 0xfc00 {
                return private(policy);
            }
            public(policy)
        }
    }
}

fn check_v4(ip: Ipv4Addr, policy: &EndpointPolicy) -> Result<(), EndpointRefusal> {
    if ip.is_loopback() {
        return loopback(policy);
    }
    // 169.254.0.0/16: link-local, and the address every cloud puts its
    // instance credentials behind.
    if ip.is_link_local() {
        return Err(EndpointRefusal::LinkLocal);
    }
    let [a, b, ..] = ip.octets();
    // Never routable, whatever the policy says. 0.0.0.0/8 reaches localhost on
    // several stacks; 240.0.0.0/4 is reserved.
    if ip.is_unspecified() || ip.is_broadcast() || ip.is_multicast() || a == 0 || a >= 240 {
        return Err(EndpointRefusal::PrivateNetwork);
    }
    // Documentation and benchmarking ranges are never a real endpoint:
    // 192.0.0.0/24, 192.0.2.0/24, 198.18.0.0/15, 198.51.100.0/24, 203.0.113.0/24.
    let [_, _, c, _] = ip.octets();
    let reserved = (a == 192 && b == 0 && (c == 0 || c == 2))
        || (a == 198 && (b == 18 || b == 19))
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113);
    if reserved {
        return Err(EndpointRefusal::PrivateNetwork);
    }
    // 100.64.0.0/10, carrier-grade NAT: where a container network often lives.
    let cgnat = a == 100 && (64..128).contains(&b);
    if ip.is_private() || cgnat {
        return private(policy);
    }
    public(policy)
}

fn loopback(policy: &EndpointPolicy) -> Result<(), EndpointRefusal> {
    if policy.allow_loopback {
        Ok(())
    } else {
        Err(EndpointRefusal::Loopback)
    }
}

fn private(policy: &EndpointPolicy) -> Result<(), EndpointRefusal> {
    if policy.allow_private {
        Ok(())
    } else {
        Err(EndpointRefusal::PrivateNetwork)
    }
}

fn public(policy: &EndpointPolicy) -> Result<(), EndpointRefusal> {
    if policy.allow_public {
        Ok(())
    } else {
        Err(EndpointRefusal::NonLocal)
    }
}

/// Resolves a `Location` header against the URL that answered it, or `None`
/// when either is not a URL.
pub fn resolve_redirect(from_url: &str, location: &str) -> Option<String> {
    let base = Url::parse(from_url.trim()).ok()?;
    base.join(location.trim()).ok().map(String::from)
}

/// Decides whether a request may follow a redirect.
///
/// `hop` is the 1-based number of the redirect about to be followed. The
/// target is re-checked against the policy exactly like the first URL, the
/// hop count is bounded by [`EndpointPolicy::max_redirects`], and a
/// credentialed request never leaves its origin.
///
/// # Errors
///
/// A [`PolicyViolation`]: [`PolicyViolation::TooManyRedirects`],
/// [`PolicyViolation::CrossOriginRedirect`], or
/// [`PolicyViolation::Endpoint`] for the target.
pub fn check_redirect(
    policy: &EndpointPolicy,
    from_url: &str,
    to_url: &str,
    credentialed: bool,
    hop: usize,
) -> Result<(), PolicyViolation> {
    if hop > policy.max_redirects {
        return Err(PolicyViolation::TooManyRedirects {
            max: policy.max_redirects,
        });
    }
    check_endpoint_with_credential(to_url, policy, credentialed)?;
    if credentialed && !same_origin(from_url, to_url) {
        return Err(PolicyViolation::CrossOriginRedirect);
    }
    Ok(())
}
