//! Endpoint text-handling tests: local-endpoint normalisation, credential
//! refusal and redaction, and scrubbing. The refusal/redaction corpus is
//! ported from OpenCompany's `catalogue_tests_endpoints.rs` (each case pins a
//! Codex/CodeRabbit review finding on #2281), extended with scrubbing tests
//! from `probe_tests_catalog.rs` and property tests.

use proptest::prelude::*;

use super::redact::{base64_standard, percent_decode};
use super::*;

#[test]
fn a_bare_origin_gains_the_v1_an_openai_surface_lives_at() {
    // `http://localhost:11434` is what Ollama's own documentation prints,
    // and it is not where the OpenAI-compatible surface is.
    assert_eq!(
        normalize_local_endpoint("http://localhost:11434").as_deref(),
        Some("http://localhost:11434/v1")
    );
    assert_eq!(
        normalize_local_endpoint("  http://localhost:11434/  ").as_deref(),
        Some("http://localhost:11434/v1")
    );
}

#[test]
fn a_path_the_operator_supplied_is_left_exactly_as_typed() {
    // Appending is not guessing. Someone who typed a path meant it.
    assert_eq!(
        normalize_local_endpoint("https://acme.example/api/gateway").as_deref(),
        Some("https://acme.example/api/gateway")
    );
    assert_eq!(
        normalize_local_endpoint("http://127.0.0.1:1234/v1/").as_deref(),
        Some("http://127.0.0.1:1234/v1")
    );
}

#[test]
fn an_endpoint_carrying_a_credential_is_not_an_endpoint() {
    // The security half of the same refusal. `normalize_local_endpoint` is
    // the funnel every stored endpoint passes through, so refusing here is
    // what makes "no credential is ever stored in a `base_url`" a property
    // of the store rather than of whichever handler remembered to check.
    for bad in [
        "http://alice:hunter2@127.0.0.1:8597/v1",
        "https://alice@api.acme.example/v1",
        "http://alice:hunter2@127.0.0.1:8597",
        // A password may itself contain an `@`; the authority still has one.
        "http://alice:hun@ter2@127.0.0.1:8597/v1",
    ] {
        assert!(endpoint_has_credentials(bad), "`{bad}` carries userinfo");
        assert!(
            normalize_local_endpoint(bad).is_none(),
            "`{bad}` must not normalise into something storable"
        );
    }
}

#[test]
fn a_second_scheme_does_not_hide_the_credential_behind_it() {
    // Codex review on #2281: an uppercase scheme was once prefixed with a
    // second one by setup normalisation, and the first authority (`HTTP:`)
    // has no `@` — so reading only that one missed the credential entirely.
    for bad in [
        "http://HTTP://alice:hunter2@127.0.0.1:8597/v1",
        "https://http://alice@api.acme.example/v1",
    ] {
        assert!(endpoint_has_credentials(bad), "`{bad}` carries userinfo");
        assert!(
            normalize_local_endpoint(bad).is_none(),
            "`{bad}` must not be storable"
        );
        let said = redact_endpoint(bad);
        assert!(
            !said.contains("alice") && !said.contains("hunter2"),
            "`{bad}` redacted to `{said}`"
        );
    }
    // An uppercase scheme on its own is an ordinary endpoint.
    assert!(endpoint_has_credentials(
        "HTTP://alice:hunter2@127.0.0.1:8597/v1"
    ));
    assert!(!endpoint_has_credentials("HTTPS://api.acme.example/v1"));
}

#[test]
fn a_credential_is_found_in_every_authority_an_http_client_could_read() {
    // Codex and CodeRabbit review on #2281. Each shape once hid its
    // credential from the refusal, the redaction, or both.
    for (bad, said) in [
        // One slash: WHATWG URL parsing still reads the authority after it.
        (
            "http:/alice:hunter2@127.0.0.1:8597/v1",
            "http:/***@127.0.0.1:8597/v1",
        ),
        // Three slashes: the extra one is skipped, not an empty authority.
        (
            "http:///alice:hunter2@127.0.0.1:8597/v1",
            "http:///***@127.0.0.1:8597/v1",
        ),
        // Backslashes, which a special scheme reads as slashes.
        (
            "http:\\\\alice:hunter2@127.0.0.1:8597/v1",
            "http:\\\\***@127.0.0.1:8597/v1",
        ),
        // No slash at all.
        (
            "HTTP:alice:hunter2@127.0.0.1:8597/v1",
            "***@127.0.0.1:8597/v1",
        ),
        // Two authorities, two credentials: both go.
        (
            "http://alice:one@outer/http://bob:two@inner/v1",
            "http://***@outer/http://***@inner/v1",
        ),
    ] {
        assert!(endpoint_has_credentials(bad), "`{bad}` carries userinfo");
        assert!(
            normalize_local_endpoint(bad).is_none(),
            "`{bad}` must not be storable"
        );
        assert_eq!(redact_endpoint(bad), said, "`{bad}`");
    }
    // A path is still a path. A gateway that proxies to another URL, with
    // an `@` later in that path, carries no credential and stays storable.
    let gateway = "https://gateway.example/proxy/http://upstream/@me";
    assert!(!endpoint_has_credentials(gateway));
    assert_eq!(normalize_local_endpoint(gateway).as_deref(), Some(gateway));
    for good in [
        gateway,
        "http://127.0.0.1:8597/v1",
        "http://[::1]:11434/v1",
        "https://api.acme.example:8443/v1/@me",
        "localhost:1234/v1",
    ] {
        assert!(!endpoint_has_credentials(good), "`{good}` has no userinfo");
        assert_eq!(redact_endpoint(good), good);
    }
}

#[test]
fn a_scheme_in_a_well_formed_path_is_not_an_authority() {
    // Codex review on #2281: WHATWG parsing gives this endpoint no userinfo.
    // `http:user@example.com` is path text, so the endpoint is accepted.
    let gateway = "https://gateway.example/proxy/http:user@example.com/v1";
    assert!(!endpoint_has_credentials(gateway));
    assert_eq!(normalize_local_endpoint(gateway).as_deref(), Some(gateway));
    // What is *said* about it still masks the segment that looks like one:
    // the redaction reads wider than the refusal, by design.
    assert_eq!(
        redact_endpoint(gateway),
        "https://gateway.example/proxy/http:***@example.com/v1"
    );
    // A port-less host that happens to end in `:` does not start a hop.
    assert!(!endpoint_has_credentials("http://localhost:/v1/@me"));
    // Nor does a host *named* like a scheme with one slash after it: every
    // parser reads `http://http:/v1@beta` as host `http`, empty port, path
    // `/v1@beta`. It is storable; the wider redaction still masks the
    // lookalike when it is said.
    let empty_port = "http://http:/v1@beta";
    assert!(!endpoint_has_credentials(empty_port));
    assert_eq!(
        normalize_local_endpoint(empty_port).as_deref(),
        Some(empty_port)
    );
    assert_eq!(redact_endpoint(empty_port), "http://http:/***@beta");
    // And a doubled scheme still does — the one case a hop exists for.
    assert!(endpoint_has_credentials(
        "https://http://alice@api.acme.example/v1"
    ));
}

#[test]
fn tabs_and_line_breaks_do_not_hide_a_credential() {
    // Codex review on #2281: a URL parser removes ASCII tab, LF and CR
    // wherever they appear, so each of these reaches a client carrying
    // `alice:hunter2`.
    for bad in [
        "http:\t//alice:hunter2@127.0.0.1:8597/v1",
        "http://ali\nce:hunter2@127.0.0.1:8597/v1",
        "http://http:\t//alice:hunter2@127.0.0.1:8597/v1",
        "http://alice:hunter2\r@127.0.0.1:8597/v1",
    ] {
        assert!(endpoint_has_credentials(bad), "{bad:?} carries userinfo");
        assert!(
            normalize_local_endpoint(bad).is_none(),
            "{bad:?} must not be storable"
        );
        let said = redact_endpoint(bad);
        assert!(
            !said.contains("hunter2") && !said.contains("alice"),
            "{bad:?} redacted to {said:?}"
        );
    }
    assert_eq!(
        redact_endpoint("http:\t//alice:hunter2@127.0.0.1:8597/v1"),
        "http://***@127.0.0.1:8597/v1"
    );
}

#[test]
fn an_at_sign_in_the_path_is_not_a_credential() {
    // The `@` has to be inside the authority. A path may legitimately carry
    // one, and refusing those would reject perfectly good endpoints.
    for good in [
        "https://api.acme.example/v1/@me",
        "https://api.acme.example/v1?to=a@b",
        "https://api.acme.example/v1#a@b",
    ] {
        assert!(!endpoint_has_credentials(good), "`{good}` has no userinfo");
        assert_eq!(redact_endpoint(good), good);
    }
}

#[test]
fn redacting_an_endpoint_removes_the_credential_and_nothing_else() {
    // Observed in the incident: reqwest masks userinfo in its own error
    // Display (`for url (http://127.0.0.1:8597/v1/models)`), and then the
    // handler's own `format!` put it back from the endpoint we hold.
    assert_eq!(
        redact_endpoint("http://alice:hunter2@127.0.0.1:8597/v1"),
        "http://***@127.0.0.1:8597/v1"
    );
    assert_eq!(
        redact_endpoint("https://alice@api.acme.example/v1"),
        "https://***@api.acme.example/v1"
    );
    // The last `@` in the authority is the delimiter, so a password
    // containing one is removed whole rather than half-left behind.
    assert_eq!(
        redact_endpoint("http://alice:hun@ter2@127.0.0.1:8597/v1"),
        "http://***@127.0.0.1:8597/v1"
    );
    // Scheme-less, as `normalize_setup_base_url` accepts.
    assert_eq!(
        redact_endpoint("alice:hunter2@localhost:1234/v1"),
        "***@localhost:1234/v1"
    );
    // Nothing to redact: byte-for-byte the same endpoint, trimmed.
    assert_eq!(
        redact_endpoint("  https://api.openai.com/v1  "),
        "https://api.openai.com/v1"
    );
}

#[test]
fn only_http_and_https_are_endpoints() {
    // Rejected here rather than at the probe, because this is the one
    // category whose endpoint the operator types — and the connect flow's
    // ordering says reject before any write.
    for bad in [
        "file:///etc/passwd",
        "ftp://acme.example/v1",
        "localhost:11434",
        "",
        "   ",
        "http://",
    ] {
        assert!(
            normalize_local_endpoint(bad).is_none(),
            "`{bad}` is not an endpoint"
        );
    }
}

#[test]
fn a_basic_token_is_encoded_the_way_reqwest_sends_it() {
    assert_eq!(base64_standard(b"alice:hunter2"), "YWxpY2U6aHVudGVyMg==");
    assert_eq!(base64_standard(b"a"), "YQ==");
    assert_eq!(base64_standard(b"ab"), "YWI=");
    assert_eq!(base64_standard(b""), "");
    assert_eq!(percent_decode("p%40ss%zz"), "p@ss%zz");
    assert_eq!(percent_decode("a%2"), "a%2");
}

#[test]
fn every_form_of_the_endpoint_credential_is_scrubbed_from_text() {
    let endpoint = "http://alice:p%40ss@127.0.0.1:9/v1/models";
    let token = base64_standard(b"alice:p@ss");
    let echoed = format!(
        "rejected Basic {token} / {} for alice:p@ss (raw p%40ss)",
        token.trim_end_matches('=')
    );
    let scrubbed = scrub_endpoint_credential(endpoint, &echoed);
    for secret in [
        "p@ss",
        "p%40ss",
        token.as_str(),
        token.trim_end_matches('='),
    ] {
        assert!(
            !scrubbed.contains(secret),
            "{secret:?} survived: {scrubbed}"
        );
    }
    assert!(
        scrubbed.contains("alice"),
        "the account name still reads: {scrubbed}"
    );

    // Username only: that username is the token, in every form it can echo.
    let token_only = "http://sk-not%2Ba-real-key@127.0.0.1:9/v1/models";
    let basic = base64_standard(b"sk-not+a-real-key:");
    let echoed = format!(
        "bad key sk-not+a-real-key (sent sk-not%2Ba-real-key) in Basic {basic} / {}",
        basic.trim_end_matches('=')
    );
    let scrubbed = scrub_endpoint_credential(token_only, &echoed);
    for secret in [
        "sk-not+a-real-key",
        "sk-not%2Ba-real-key",
        basic.as_str(),
        basic.trim_end_matches('='),
    ] {
        assert!(
            !scrubbed.contains(secret),
            "{secret:?} survived: {scrubbed}"
        );
    }
    // No userinfo: the text is untouched. So is text for an unparseable URL.
    assert_eq!(
        scrub_endpoint_credential("http://127.0.0.1:9/v1", "Basic abc"),
        "Basic abc"
    );
    assert_eq!(
        scrub_endpoint_credential("not a url", "Basic abc"),
        "Basic abc"
    );
}

#[test]
fn endpoint_host_reads_the_authority_only() {
    assert_eq!(
        endpoint_host("https://API.OpenAI.com/v1").as_deref(),
        Some("api.openai.com")
    );
    assert_eq!(
        endpoint_host("http://user:pw@host.test:8080/x").as_deref(),
        Some("host.test")
    );
    assert_eq!(
        endpoint_host("http://[::1]:11434/v1").as_deref(),
        Some("::1")
    );
    assert_eq!(endpoint_host("[::1]").as_deref(), Some("::1"));
    assert_eq!(
        endpoint_host("localhost:1234/v1").as_deref(),
        Some("localhost")
    );
    assert_eq!(endpoint_host("host.test").as_deref(), Some("host.test"));
    assert_eq!(
        endpoint_host("https://host.test?x=1").as_deref(),
        Some("host.test")
    );
    assert_eq!(endpoint_host(""), None);
    assert_eq!(endpoint_host("http://"), None);
    // WHATWG reads `http:///path` as host `path`, which is what a client
    // connects to.
    assert_eq!(endpoint_host("http:///path").as_deref(), Some("path"));
}

#[test]
fn the_credential_refusal_sentence_is_neutral_and_secret_free() {
    assert!(ENDPOINT_CREDENTIAL_REFUSAL.contains("API key field"));
    assert!(!ENDPOINT_CREDENTIAL_REFUSAL.contains("company"));
    assert_eq!(REDACTED_USERINFO, "***");
}

#[test]
fn a_bare_scheme_word_is_recognised_only_in_a_doubled_authority() {
    // Exercises the scheme-hop walk with an unknown scheme and with the
    // depth bound: a pathological chain of doubled schemes terminates.
    assert!(endpoint_has_credentials("ftp://ftp://alice@host/v1"));
    let deep = "http://".repeat(20) + "alice@host";
    let _ = endpoint_has_credentials(&deep);
    let _ = redact_endpoint(&deep);
    assert!(!endpoint_has_credentials("://alice@host"));
    assert!(!endpoint_has_credentials(""));
}

proptest! {
    #[test]
    fn redaction_never_leaks_the_userinfo_it_was_given(
        user in "[a-z][a-z0-9]{3,12}",
        pass in "[A-Za-z0-9]{6,16}",
        host in "[a-z]{3,10}\\.test",
        path in "(/[a-z0-9]{1,8}){0,3}",
    ) {
        let endpoint = format!("https://{user}:{pass}@{host}{path}");
        let said = redact_endpoint(&endpoint);
        prop_assert!(!said.contains(&pass), "{said}");
        prop_assert!(!said.contains(&format!("{user}:")), "{said}");
        prop_assert!(said.contains(&host));
        prop_assert!(endpoint_has_credentials(&endpoint));
        prop_assert!(normalize_local_endpoint(&endpoint).is_none());
    }

    #[test]
    fn a_credential_free_endpoint_is_redacted_to_itself(
        host in "[a-z]{3,10}\\.test",
        path in "(/[a-z0-9]{1,8}){0,3}",
    ) {
        let endpoint = format!("https://{host}{path}");
        prop_assert_eq!(redact_endpoint(&endpoint), endpoint.clone());
        prop_assert!(!endpoint_has_credentials(&endpoint));
    }

    #[test]
    fn normalize_local_endpoint_is_idempotent(
        host in "[a-z]{3,10}",
        port in 1024u16..60000,
        path in "(/[a-z0-9]{1,8}){0,3}",
        slash in proptest::bool::ANY,
    ) {
        let raw = format!("http://{host}:{port}{path}{}", if slash { "/" } else { "" });
        if let Some(once) = normalize_local_endpoint(&raw) {
            prop_assert_eq!(normalize_local_endpoint(&once), Some(once.clone()));
            prop_assert!(!once.ends_with('/'));
        }
    }

    #[test]
    fn scrubbing_never_panics_and_is_a_noop_without_userinfo(
        endpoint in "[ -~]{0,60}",
        text in "[ -~]{0,80}",
    ) {
        let _ = scrub_endpoint_credential(&endpoint, &text);
        let _ = redact_endpoint(&endpoint);
        let _ = endpoint_has_credentials(&endpoint);
        prop_assert_eq!(scrub_endpoint_credential("https://ok.test/v1", &text), text);
    }
}

#[test]
fn endpoint_host_agrees_with_the_host_a_client_connects_to() {
    // Regression (review finding): a special scheme reads `\` as `/`, so this
    // URL connects to evil.test. A splitter that only knew `/ ? #` reported
    // `tinyhumans.ai` and let first-party headers go to the attacker.
    assert_eq!(
        endpoint_host("https://evil.test\\@tinyhumans.ai/").as_deref(),
        Some("evil.test")
    );
    assert_eq!(
        endpoint_host("https://tinyhumans.ai\\.evil.test/x").as_deref(),
        Some("tinyhumans.ai")
    );
    assert_eq!(
        endpoint_host("http://user:pw@Host.TEST:8080/a?b#c").as_deref(),
        Some("host.test")
    );
    assert_eq!(
        endpoint_host("http://[2001:db8::1]:80/").as_deref(),
        Some("2001:db8::1")
    );
    assert_eq!(
        endpoint_host("HTTPS://EXAMPLE.com").as_deref(),
        Some("example.com")
    );
    // IDN hosts are compared in their ASCII form, as a client resolves them.
    assert_eq!(
        endpoint_host("https://b\u{fc}cher.example/").as_deref(),
        Some("xn--bcher-kva.example")
    );
    // The tolerant fallback (no scheme, other schemes) also treats `\` as a
    // delimiter.
    assert_eq!(
        endpoint_host("host.test\\@other.test/x").as_deref(),
        Some("host.test")
    );
    assert_eq!(
        endpoint_host("ftp://host.test/x").as_deref(),
        Some("host.test")
    );
}

#[test]
fn a_query_or_fragment_is_not_a_path_when_appending_v1() {
    // Regression (review round 2): `?x=1/v1` corrupted the query, and a `/` in
    // the query suppressed the append.
    for (raw, expected) in [
        (
            "http://localhost:11434?x=1",
            "http://localhost:11434/v1?x=1",
        ),
        ("http://host?next=/a", "http://host/v1?next=/a"),
        ("http://host#frag", "http://host/v1#frag"),
        ("http://host:1/api?x=1", "http://host:1/api?x=1"),
        ("https://Host.test", "https://Host.test/v1"),
    ] {
        let once = normalize_local_endpoint(raw).unwrap();
        assert_eq!(once, expected, "{raw}");
        assert_eq!(
            normalize_local_endpoint(&once),
            Some(once.clone()),
            "idempotent: {raw}"
        );
    }
}

#[test]
fn a_bare_slash_before_a_query_or_fragment_still_gets_v1() {
    // Regression (review round 3): `host/?x=1` kept the root path.
    for (raw, expected) in [
        ("http://host/?x=1", "http://host/v1?x=1"),
        ("http://host:11434/#frag", "http://host:11434/v1#frag"),
        ("http://host//?x=1", "http://host/v1?x=1"),
        ("http://host/api/?x=1", "http://host/api?x=1"),
    ] {
        let once = normalize_local_endpoint(raw).unwrap();
        assert_eq!(once, expected, "{raw}");
        assert_eq!(
            normalize_local_endpoint(&once),
            Some(once.clone()),
            "idempotent {raw}"
        );
    }
}

#[test]
fn an_endpoint_with_a_credential_query_is_not_storable() {
    assert!(endpoint_query_has_credential("https://h.test/v1?key=abc"));
    assert!(endpoint_query_has_credential(
        "https://h.test/v1?x=1&apiKey=abc"
    ));
    assert!(!endpoint_query_has_credential(
        "https://h.test/v1?version=2"
    ));
    assert!(!endpoint_query_has_credential("not a url ?key=abc"));
    assert_eq!(
        normalize_local_endpoint("http://h.test:1234/v1?api_key=abc"),
        None
    );
}

// ---- round 4 review regressions ----------------------------------------------------------

#[test]
fn redacting_an_endpoint_masks_credential_query_values_too() {
    assert_eq!(
        redact_endpoint("https://generativelanguage.googleapis.com/v1beta/openai?key=AIzaSECRET"),
        "https://generativelanguage.googleapis.com/v1beta/openai?key=***"
    );
    assert_eq!(
        redact_endpoint("https://h.test/v1?a=1&api_key=abc&Signature=x%2By&limit=5#frag"),
        "https://h.test/v1?a=1&api_key=***&Signature=***&limit=5#frag"
    );
    assert_eq!(
        redact_endpoint("https://u:pw@h.test/v1?subscription-key=abc"),
        "https://***@h.test/v1?subscription-key=***"
    );
    // Ordinary queries and query-less URLs are untouched.
    assert_eq!(
        redact_endpoint("https://h.test/v1?version=2&flag"),
        "https://h.test/v1?version=2&flag"
    );
    assert_eq!(redact_endpoint("https://h.test/v1"), "https://h.test/v1");
}

#[test]
fn azure_and_cookie_style_query_names_count_as_credentials() {
    for url in [
        "https://h.test/v1?subscription-key=a",
        "https://h.test/v1?Ocp-Apim-Subscription-Key=a",
        "https://h.test/v1?x-functions-key=a",
        "https://h.test/v1?cookie=a",
        "https://h.test/v1?app_key=a",
        "https://h.test/v1?X-Amz-Signature=a",
        "https://h.test/v1?pwd=a",
    ] {
        assert!(endpoint_query_has_credential(url), "{url}");
    }
    for url in [
        // A bare `code` is a routing parameter as often as a secret.
        "https://h.test/v1?code=eu",
        "https://h.test/v1?public_key=a",
        "https://h.test/v1?page_token=a",
        "https://h.test/v1?secretary=a",
        "https://h.test/v1?monkey=a",
        "https://h.test/v1?max_tokens=5",
    ] {
        assert!(!endpoint_query_has_credential(url), "{url}");
    }
}

#[test]
fn an_endpoint_with_no_host_or_a_backslash_is_not_normalisable() {
    // Regression (review round 4): these returned Some and failed to parse later.
    for bad in [
        "http://:8080",
        "http://?x=1",
        "http:///",
        "http://host\\@evil.test",
        "https://[::1",
    ] {
        assert_eq!(normalize_local_endpoint(bad), None, "{bad}");
    }
    // The query value is kept exactly as typed.
    assert_eq!(
        normalize_local_endpoint("http://host/?a=b/").as_deref(),
        Some("http://host/v1?a=b/")
    );
}

#[test]
fn an_absolute_host_name_is_the_same_host() {
    assert_eq!(
        endpoint_host("https://api.groq.com./openai/v1").as_deref(),
        Some("api.groq.com")
    );
    assert_eq!(
        endpoint_host("api.groq.com./x").as_deref(),
        Some("api.groq.com")
    );
}

// ---- round 5 review regressions ----------------------------------------------------------

#[test]
fn a_percent_encoded_credential_name_is_masked_as_well_as_detected() {
    // Regression (review round 5): detection decoded names, masking did not.
    assert_eq!(
        redact_endpoint("https://h.test/v1?%6Bey=SECRET1&a=b"),
        "https://h.test/v1?%6Bey=***&a=b"
    );
    assert_eq!(
        redact_endpoint("https://h.test/v1?api%5Fkey=SECRET2"),
        "https://h.test/v1?api%5Fkey=***"
    );
    assert!(endpoint_query_has_credential(
        "https://h.test/v1?%6Bey=SECRET1"
    ));
    assert!(endpoint_query_has_credential(
        "https://h.test/v1?api%5Fkey=S"
    ));
}

#[test]
fn run_together_credential_names_in_a_query_are_caught() {
    for name in ["openaiapikey", "dbpassword", "clientsecret", "APITOKEN"] {
        let url = format!("https://h.test/v1?{name}=S5");
        assert!(endpoint_query_has_credential(&url), "{name}");
        assert_eq!(
            redact_endpoint(&url),
            format!("https://h.test/v1?{name}=***"),
            "{name}"
        );
    }
}

// ---- round 6 review regressions ----------------------------------------------------------

#[test]
fn a_code_query_parameter_is_a_credential_only_on_an_azure_functions_host() {
    // Regression (review round 6): dropping `code` globally let Azure Functions
    // keys through; keeping it globally refused routing parameters.
    for url in [
        "https://myapp.azurewebsites.net/api/chat?code=FUNCKEY",
        "https://azurewebsites.net/api/chat?code=FUNCKEY",
    ] {
        assert!(endpoint_query_has_credential(url), "{url}");
        assert!(redact_endpoint(url).ends_with("code=***"), "{url}");
        assert_eq!(normalize_local_endpoint(url), None, "{url}");
    }
    for url in [
        "http://gw:8080/v1?code=eu",
        "https://notazurewebsites.net/api?code=eu",
    ] {
        assert!(!endpoint_query_has_credential(url), "{url}");
        assert_eq!(redact_endpoint(url), url);
    }
}

#[test]
fn a_leading_slash_after_the_scheme_is_not_a_host() {
    for bad in ["http:///v1", "http://\\v1", "https:////x"] {
        assert_eq!(normalize_local_endpoint(bad), None, "{bad}");
    }
}

#[test]
fn a_double_encoded_query_name_is_judged_by_what_the_server_receives() {
    // Regression (review round 6): detection decoded twice, so `%256Bey` (the
    // literal name `%6Bey`) was refused although it carries no credential.
    assert!(!endpoint_query_has_credential(
        "https://h.test/v1?%256Bey=1"
    ));
    assert_eq!(
        redact_endpoint("https://h.test/v1?%256Bey=1"),
        "https://h.test/v1?%256Bey=1"
    );
    assert!(endpoint_query_has_credential("https://h.test/v1?%6Bey=1"));
}

#[test]
fn an_absolute_azure_host_is_still_an_azure_host_for_the_code_parameter() {
    // Regression (review round 7): `host_str()` kept the trailing dot, so the
    // refusal missed what redaction masked.
    let url = "https://f.azurewebsites.net./api/x?code=SECRET";
    assert!(endpoint_query_has_credential(url));
    assert_eq!(normalize_local_endpoint(url), None);
    assert!(redact_endpoint(url).ends_with("code=***"));
}
