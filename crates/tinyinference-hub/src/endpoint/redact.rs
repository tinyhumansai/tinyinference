//! Redaction of endpoint credentials, ported from OpenCompany
//! (`company/inference/catalogue.rs` and `probe.rs`).
//!
//! An endpoint is stored as written and read back on every page load, echoed
//! into failure text, and written to plaintext stores, so a password in one is a
//! password in all three. Two independent mechanisms keep it out: **refusal**
//! ([`endpoint_has_credentials`]) keeps userinfo out of anything written from
//! now on, and **redaction** ([`redact_endpoint`]) keeps it out of anything
//! *said*, including about values stored before the refusal existed.

/// What [`redact_endpoint`] leaves where the userinfo was.
///
/// It replaces the userinfo only; the delimiting `@` survives, so the result
/// still reads as a URL and the operator can see *that* something was embedded
/// — which is the sentence they need in order to go and move it into the key
/// field.
pub const REDACTED_USERINFO: &str = "***";

/// Every byte range of an endpoint that **might** be userinfo — the redaction's
/// reading, deliberately wider than the refusal's.
///
/// A candidate authority starts at the beginning of the value, after every
/// `://`, and after every `http:` or `https:` in any case; any run of `/` or `\`
/// after that start is skipped, the candidate ends at the next `/` or `\`, and
/// only an `@` inside it counts. Every candidate is taken, including ones inside
/// a path (`https://gw/proxy/http://bob:pw@inner`), and overlapping ranges are
/// merged. That over-reads on purpose: this decides what is **said** about an
/// endpoint, and masking a path segment that only looks like a credential costs
/// a log line some readability, where missing a real one costs the credential.
/// Whether an endpoint is **refused** is the narrower
/// `endpoint_credential_range`, so a well-formed endpoint is never rejected
/// for path text (Codex and CodeRabbit review on #2281).
fn endpoint_userinfo_ranges(endpoint: &str) -> Vec<std::ops::Range<usize>> {
    let head = &endpoint[..endpoint.find(['?', '#']).unwrap_or(endpoint.len())];
    // ASCII lowercasing keeps every byte offset, so indices into it are
    // indices into `head`.
    let lower = head.to_ascii_lowercase();
    let mut starts = vec![0];
    starts.extend(head.match_indices("://").map(|(i, _)| i + ":".len()));
    for scheme in ["https:", "http:"] {
        starts.extend(lower.match_indices(scheme).map(|(i, _)| i + scheme.len()));
    }
    let mut ranges: Vec<std::ops::Range<usize>> = starts
        .into_iter()
        .filter_map(|start| {
            let rest = &head[start..];
            let slashes = rest
                .bytes()
                .take_while(|b| matches!(b, b'/' | b'\\'))
                .count();
            let authority = &rest[slashes..];
            let len = authority.find(['/', '\\']).unwrap_or(authority.len());
            // `rfind`, not `find`: a password may itself contain an `@`, and
            // the last one in the authority is the delimiter per RFC 3986.
            let at = authority[..len].rfind('@')?;
            let from = start + slashes;
            Some(from..from + at)
        })
        .collect();
    ranges.sort_by_key(|range| (range.start, range.end));
    let mut merged: Vec<std::ops::Range<usize>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

/// The userinfo an HTTP client would actually read from `endpoint`, if any.
///
/// Read the way WHATWG URL parsing reads a special scheme: a leading `http:` or
/// `https:` (any case) or another `scheme://`, then any run of `/` or `\`, then
/// the authority up to the next `/` or `\`. So `http:/alice:pw@host`,
/// `http:///alice:pw@host`, `http:\\alice:pw@host` and `HTTP:alice:pw@host` all
/// carry a credential. The read continues past an authority **only** for a doubled
/// scheme — an authority that is itself a bare scheme **and** is followed by `//`
/// (`http://HTTP://alice:pw@host`). One slash after it is a host with an empty
/// port: `http://http:/v1@beta` is host `http` and path `/v1@beta` to every
/// parser, and has no userinfo. Never into ordinary path text either:
/// `https://gateway.example/proxy/http:user@example.com/v1` has no userinfo, and
/// refusing it would reject an endpoint every client accepts (Codex review on
/// #2281). Every range this returns is also one `endpoint_userinfo_ranges`
/// returns, so whatever is refused is also redacted.
fn endpoint_credential_range(endpoint: &str) -> Option<std::ops::Range<usize>> {
    let head = &endpoint[..endpoint.find(['?', '#']).unwrap_or(endpoint.len())];
    let mut pos = 0;
    // Bounded: each hop consumes a scheme, so a real value needs two or three.
    for _ in 0..8 {
        let scheme_len = scheme_prefix_len(&head[pos..]);
        if scheme_len == 0 && pos > 0 {
            return None;
        }
        let after = pos + scheme_len;
        let from = after
            + head[after..]
                .bytes()
                .take_while(|b| matches!(b, b'/' | b'\\'))
                .count();
        let tail = &head[from..];
        let authority = &tail[..tail.find(['/', '\\']).unwrap_or(tail.len())];
        // `rfind`, not `find`: a password may itself contain an `@`, and the
        // last one in the authority is the delimiter per RFC 3986.
        if let Some(at) = authority.rfind('@') {
            return Some(from..from + at);
        }
        let bare_scheme = authority.strip_suffix(':').is_some_and(is_scheme_name);
        // A doubled `scheme://scheme://` only. `http://http:/v1@beta` is a host
        // named `http` with an empty port, not a second scheme (Codex review on
        // #2281), so a single slash after the bare scheme ends the read.
        let doubled = tail[authority.len()..]
            .bytes()
            .take_while(|b| matches!(b, b'/' | b'\\'))
            .count()
            >= 2;
        if from == pos || !bare_scheme || !doubled {
            return None;
        }
        pos = from;
    }
    None
}

/// The length of the scheme `rest` opens with: `http:`/`https:` in any case, or
/// any other valid `scheme://`. Zero when it opens with none.
fn scheme_prefix_len(rest: &str) -> usize {
    let lower = rest.to_ascii_lowercase();
    if lower.starts_with("https:") {
        return "https:".len();
    }
    if lower.starts_with("http:") {
        return "http:".len();
    }
    match rest.find("://") {
        Some(i) if is_scheme_name(&rest[..i]) => i + "://".len(),
        _ => 0,
    }
}

/// Whether `name` is a URI scheme name (RFC 3986 §3.1).
fn is_scheme_name(name: &str) -> bool {
    name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

/// Whether an endpoint URL carries a credential in its authority
/// (`http://user:password@host/v1`).
///
/// The check every place that **accepts** an endpoint makes, so that a
/// credential in a URL never reaches storage. It is the same class of rule as
/// the rule that an API key is never a config value: a secret belongs in the
/// credential slot, which is write-only to a console, and nowhere else. An
/// endpoint is read back by every console reader on every page load, echoed
/// into operator-facing failure text, and written to a plaintext store — so a
/// password in one is a password in all three.
pub fn endpoint_has_credentials(endpoint: &str) -> bool {
    endpoint_credential_range(&as_url_parser_reads(endpoint)).is_some()
}

/// The endpoint as a URL parser sees it: trimmed, with every ASCII tab, line
/// feed and carriage return removed.
///
/// WHATWG URL parsing strips those three wherever they appear, so
/// `http:\t//alice:pw@host` reaches a client as `http://alice:pw@host`. A scan
/// that stopped at the tab saw no `@` in the authority, which let the credential
/// through both the refusal and the redaction (Codex review on #2281).
fn as_url_parser_reads(endpoint: &str) -> String {
    endpoint
        .trim()
        .chars()
        .filter(|c| !matches!(c, '\t' | '\n' | '\r'))
        .collect()
}

/// The same endpoint with any embedded credential replaced by
/// [`REDACTED_USERINFO`].
///
/// The second, independent mechanism behind the same invariant as
/// [`endpoint_has_credentials`]. Rejection keeps userinfo out of anything
/// written from now on; this keeps it out of anything **said**, including about
/// values stored before the rejection existed and values that arrived from a
/// `company.toml` or an `OPENCOMPANY_INFERENCE_URL` this host does not own.
///
/// Every endpoint that reaches a response body, an operator-facing message or a
/// log goes through here. The endpoint used to actually *make* a request does
/// not — redacting there would break the request, which is the difference
/// between the two call sites and the reason this is a separate function rather
/// than something done at the point of storage.
pub fn redact_endpoint(endpoint: &str) -> String {
    // Said as a client would read it. Tabs and line breaks are dropped before
    // the scan rather than masked around, because a client drops them too and
    // an offset into the raw value would not line up with the credential.
    let mut out = as_url_parser_reads(endpoint);
    let ranges = endpoint_userinfo_ranges(&out);
    // Last range first, so each replacement leaves the earlier offsets valid.
    for range in ranges.into_iter().rev() {
        out.replace_range(range, REDACTED_USERINFO);
    }
    redact_query_credentials(&out)
}

/// Replaces the value of every credential-named query parameter
/// (`?key=...&api_key=...`) with [`REDACTED_USERINFO`], leaving the rest of the
/// URL, the other parameters and the fragment as written.
fn redact_query_credentials(url: &str) -> String {
    let Some(q) = url.find('?') else {
        return url.to_string();
    };
    let host = super::normalize::endpoint_host(url);
    let (head, rest) = url.split_at(q + 1);
    let (query, fragment) = match rest.find('#') {
        Some(h) => rest.split_at(h),
        None => (rest, ""),
    };
    let redacted: Vec<String> = query
        .split('&')
        .map(|pair| match pair.split_once('=') {
            Some((name, _)) if raw_query_name_is_credential(name, host.as_deref()) => {
                format!("{name}={REDACTED_USERINFO}")
            }
            _ => pair.to_string(),
        })
        .collect();
    format!("{head}{}{fragment}", redacted.join("&"))
}

/// Whether an already-decoded query-parameter name carries a credential. A
/// bare `code` counts only on an Azure Functions host (`*.azurewebsites.net`),
/// where `?code=` is the function key; elsewhere it is a routing parameter as
/// often as a secret.
pub(crate) fn query_name_is_credential(name: &str, host: Option<&str>) -> bool {
    crate::secret::is_credential_name(name)
        || (name.trim().eq_ignore_ascii_case("code")
            && host.is_some_and(|h| h == "azurewebsites.net" || h.ends_with(".azurewebsites.net")))
}

/// [`query_name_is_credential`] for a name as written in the URL text, which is
/// percent-decoded first (`%6Bey` is `key`).
fn raw_query_name_is_credential(name: &str, host: Option<&str>) -> bool {
    let decoded = url::form_urlencoded::parse(format!("{name}=").as_bytes())
        .next()
        .map(|(decoded, _)| decoded.into_owned())
        .unwrap_or_default();
    query_name_is_credential(&decoded, host)
}

/// `text` with every credential removed that a request to `endpoint` carried
/// **because of the endpoint's own userinfo**.
///
/// `reqwest` lifts `user:password@` out of a URL and sends it as `Authorization:
/// Basic base64(user:password)`, percent-decoded first. So three forms can come
/// back in a response body: the password as written in the URL, the password
/// decoded, and the Basic token. Each is replaced with
/// [`REDACTED_USERINFO`], both padded and unpadded.
///
/// **A username with no password is the credential.** `http://sk-secret@host`
/// is how a token gets pasted into a URL, and `reqwest` sends it as
/// `Basic base64(sk-secret:)` — so it is scrubbed as written and decoded, like a
/// password (Codex review on #2281). Beside a password the username is an
/// account name, and is left so an error naming the account still reads.
pub fn scrub_endpoint_credential(endpoint: &str, text: &str) -> String {
    let Ok(parsed) = url::Url::parse(endpoint.trim()) else {
        return text.to_string();
    };
    let username = percent_decode(parsed.username());
    let password = parsed.password();
    if username.is_empty() && password.is_none() {
        return text.to_string();
    }
    let decoded = password.map(percent_decode).unwrap_or_default();
    let token = base64_standard(format!("{username}:{decoded}").as_bytes());
    let mut secrets = vec![token.trim_end_matches('=').to_string(), token];
    match password {
        Some(raw) => {
            secrets.push(raw.to_string());
            secrets.push(decoded);
        }
        None => {
            secrets.push(parsed.username().to_string());
            secrets.push(username);
        }
    }
    secrets.retain(|secret| !secret.is_empty());
    // Longest first, so a shorter secret never splits a longer one it sits in.
    secrets.sort_by_key(|secret| std::cmp::Reverse(secret.len()));
    secrets.dedup();
    let mut out = text.to_string();
    for secret in &secrets {
        out = out.replace(secret.as_str(), REDACTED_USERINFO);
    }
    out
}

/// Percent-decodes `s` the way `reqwest` decodes URL userinfo; an invalid
/// escape is kept as written.
pub(super) fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = |b: u8| (b as char).to_digit(16);
            if let (Some(high), Some(low)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((high * 16 + low) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Standard padded base64, for matching a Basic token without a dependency this
/// crate only takes behind a feature.
pub(super) fn base64_standard(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = (u32::from(chunk[0]) << 16) | (u32::from(b1) << 8) | u32::from(b2);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Whether an endpoint carries a credential in its **query string**
/// (`?key=...`, `?api_key=...`, `?access_token=...`), judged by the parameter
/// name. Gemini-style `?key=` URLs are a common paste; like userinfo, the value
/// would be stored and echoed to every reader of the configuration.
pub fn endpoint_query_has_credential(endpoint: &str) -> bool {
    url::Url::parse(as_url_parser_reads(endpoint).as_str())
        .is_ok_and(|url| url_query_has_credential(&url))
}

/// Whether a parsed URL carries a credential in its query string.
pub(crate) fn url_query_has_credential(url: &url::Url) -> bool {
    let host = url
        .host_str()
        .map(|h| h.trim_end_matches('.').to_ascii_lowercase());
    url.query_pairs()
        .any(|(name, _)| query_name_is_credential(&name, host.as_deref()))
}
