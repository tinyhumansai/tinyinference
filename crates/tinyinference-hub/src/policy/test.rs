//! Endpoint-policy tests. The SSRF corpus is ported from OpenCompany's
//! `probe_tests_ssrf.rs`; the additions pin the three tightenings (WHATWG
//! parsing, loopback names, embedded-IPv4 forms) and the redirect helpers.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::time::Duration;

use proptest::prelude::*;

use super::*;
use crate::error::PolicyViolation;

fn server_side() -> EndpointPolicy {
    EndpointPolicy::hosted()
}

fn local_offered() -> EndpointPolicy {
    EndpointPolicy::desktop()
}

// ---- ported corpus -------------------------------------------------------

#[test]
fn an_ordinary_endpoint_is_allowed() {
    assert_eq!(
        check_endpoint("https://api.openai.com/v1", &server_side()),
        Ok(())
    );
    assert_eq!(check_endpoint("https://8.8.8.8/v1", &server_side()), Ok(()));
}

#[test]
fn only_http_and_https_are_probeable() {
    let p = server_side();
    assert_eq!(
        check_endpoint("file:///etc/passwd", &p),
        Err(EndpointRefusal::Scheme)
    );
    assert_eq!(
        check_endpoint("gopher://acme.test/v1", &p),
        Err(EndpointRefusal::Scheme)
    );
    assert_eq!(
        check_endpoint("api.openai.com/v1", &p),
        Err(EndpointRefusal::Unparseable)
    );
    assert_eq!(check_endpoint("", &p), Err(EndpointRefusal::Unparseable));
    assert_eq!(
        check_endpoint("http://", &p),
        Err(EndpointRefusal::Unparseable)
    );
}

#[test]
fn the_cloud_metadata_address_is_refused_wherever_it_is_offered() {
    for policy in [local_offered(), server_side(), EndpointPolicy::local_only()] {
        assert_eq!(
            check_endpoint("http://169.254.169.254/latest/meta-data/", &policy),
            Err(EndpointRefusal::LinkLocal)
        );
    }
    // Even a policy that allows private networks never allows link-local.
    let lan = EndpointPolicy::desktop().with_private(true);
    assert_eq!(
        check_endpoint("http://169.254.169.254/", &lan),
        Err(EndpointRefusal::LinkLocal)
    );
}

#[test]
fn link_local_is_refused_in_both_address_families() {
    assert_eq!(
        check_endpoint("http://169.254.1.1/v1", &local_offered()),
        Err(EndpointRefusal::LinkLocal)
    );
    assert_eq!(
        check_endpoint("http://[fe80::1]/v1", &local_offered()),
        Err(EndpointRefusal::LinkLocal)
    );
    assert_eq!(
        check_endpoint("http://[fec0::1]/v1", &local_offered()),
        Err(EndpointRefusal::LinkLocal)
    );
}

#[test]
fn an_ipv4_mapped_ipv6_address_gets_the_ipv4_answer() {
    assert_eq!(
        check_endpoint("http://[::ffff:169.254.169.254]/v1", &local_offered()),
        Err(EndpointRefusal::LinkLocal)
    );
    assert_eq!(
        check_endpoint("http://[::ffff:127.0.0.1]:11434/v1", &server_side()),
        Err(EndpointRefusal::Loopback)
    );
    assert_eq!(
        check_endpoint("http://[::ffff:10.0.0.1]/v1", &local_offered()),
        Err(EndpointRefusal::PrivateNetwork)
    );
}

#[test]
fn loopback_is_an_explicit_allowance_not_a_hole() {
    assert_eq!(
        check_endpoint("http://127.0.0.1:11434/v1", &local_offered()),
        Ok(())
    );
    assert_eq!(
        check_endpoint("http://[::1]:11434/v1", &local_offered()),
        Ok(())
    );
    assert_eq!(
        check_endpoint("http://127.0.0.1:11434/v1", &server_side()),
        Err(EndpointRefusal::Loopback)
    );
    assert_eq!(
        check_endpoint("http://[::1]:11434/v1", &server_side()),
        Err(EndpointRefusal::Loopback)
    );
}

#[test]
fn private_and_carrier_grade_ranges_are_refused() {
    for addr in [
        "http://10.0.0.5/v1",
        "http://192.168.1.10/v1",
        "http://172.16.0.1/v1",
        "http://100.64.0.1/v1",
        "http://100.127.255.254/v1",
        "http://0.0.0.0/v1",
        "http://255.255.255.255/v1",
        "http://224.0.0.1/v1",
    ] {
        assert_eq!(
            check_endpoint(addr, &local_offered()),
            Err(EndpointRefusal::PrivateNetwork),
            "{addr}"
        );
    }
    for addr in [
        "http://[fc00::1]/v1",
        "http://[fd12:3456::1]/v1",
        "http://[::]/v1",
        "http://[ff02::1]/v1",
    ] {
        assert_eq!(
            check_endpoint(addr, &local_offered()),
            Err(EndpointRefusal::PrivateNetwork),
            "{addr}"
        );
    }
    // Just outside CGNAT is public.
    assert_eq!(
        check_endpoint("http://100.63.255.255/v1", &local_offered()),
        Ok(())
    );
    assert_eq!(
        check_endpoint("http://100.128.0.1/v1", &local_offered()),
        Ok(())
    );
}

#[test]
fn a_key_is_never_sent_to_an_http_endpoint_off_this_host() {
    let p = server_side();
    assert_eq!(
        check_endpoint_with_credential("http://gateway.acme.test/v1", &p, true),
        Err(EndpointRefusal::Cleartext)
    );
    assert_eq!(
        check_endpoint_with_credential("http://gateway.acme.test/v1", &p, false),
        Ok(())
    );
    assert_eq!(
        check_endpoint_with_credential("https://gateway.acme.test/v1", &p, true),
        Ok(())
    );
    for local in [
        "http://localhost:11434/v1",
        "http://localhost.:11434/v1",
        "http://127.0.0.1:11434/v1",
        "http://[::1]:11434/v1",
        "http://[::ffff:127.0.0.1]:11434/v1",
    ] {
        assert_eq!(
            check_endpoint_with_credential(local, &local_offered(), true),
            Ok(()),
            "{local} is this host"
        );
    }
    assert_eq!(
        check_endpoint_with_credential("https://169.254.169.254/v1", &p, true),
        Err(EndpointRefusal::LinkLocal)
    );
}

#[test]
fn a_credentialed_request_does_not_follow_a_redirect_off_its_origin() {
    let origin = "https://api.acme.test/v1/models";
    assert!(same_origin(origin, "https://api.acme.test/v2/models"));
    assert!(same_origin(
        origin,
        "https://API.ACME.TEST/v1/models?page=2"
    ));
    assert!(!same_origin(origin, "https://elsewhere.test/v1/models"));
    assert!(!same_origin(origin, "http://api.acme.test/v1/models"));
    assert!(!same_origin(origin, "https://api.acme.test:8443/v1/models"));
    assert!(!same_origin(origin, "api.acme.test/v1/models"));
    assert!(!same_origin("api.acme.test/v1", origin));
}

#[test]
fn a_redirect_target_gets_the_same_answer_as_the_first_hop() {
    assert_eq!(
        check_endpoint("https://acme.test/v1", &server_side()),
        Ok(())
    );
    assert_eq!(
        check_address("169.254.169.254".parse().unwrap(), &server_side()),
        Err(EndpointRefusal::LinkLocal)
    );
}

#[test]
fn userinfo_and_ports_do_not_hide_the_host() {
    assert_eq!(
        check_endpoint("http://user:pw@169.254.169.254:80/v1", &server_side()),
        Err(EndpointRefusal::LinkLocal)
    );
    // The authority after the last @ is the real host (so this is not the
    // metadata address), but the URL still carries a credential and is refused
    // as one (guard G16; review finding).
    assert_eq!(
        check_endpoint("http://169.254.169.254@example.test/v1", &server_side()),
        Err(EndpointRefusal::CredentialInUrl)
    );
}

#[test]
fn a_hostname_is_allowed_because_resolving_it_here_would_prove_nothing() {
    assert_eq!(
        check_endpoint("https://localhost.acme.test/v1", &server_side()),
        Ok(())
    );
}

// ---- tightenings beyond OpenCompany ----------------------------------------

#[test]
fn alternative_ipv4_spellings_are_checked_as_the_address_a_client_connects_to() {
    let p = server_side();
    // `2130706433` is 127.0.0.1 in decimal, `0x7f.1` mixes hex and decimal, and
    // `0177.0.0.1` is octal: all read as "a hostname" by a string parser.
    for (spelling, expected) in [
        ("http://2130706433/v1", EndpointRefusal::Loopback),
        ("http://0x7f.1/v1", EndpointRefusal::Loopback),
        ("http://0177.0.0.1/v1", EndpointRefusal::Loopback),
        ("http://127.1/v1", EndpointRefusal::Loopback),
        ("http://2852039166/v1", EndpointRefusal::LinkLocal),
        ("http://0xa9fea9fe/v1", EndpointRefusal::LinkLocal),
        ("http://167772161/v1", EndpointRefusal::PrivateNetwork),
    ] {
        assert_eq!(check_endpoint(spelling, &p), Err(expected), "{spelling}");
    }
}

#[test]
fn localhost_is_loopback_by_name() {
    let p = server_side();
    for url in [
        "http://localhost:8080/v1",
        "http://LOCALHOST/v1",
        "http://localhost./v1",
        "http://ollama.localhost/v1",
    ] {
        assert_eq!(
            check_endpoint(url, &p),
            Err(EndpointRefusal::Loopback),
            "{url}"
        );
        assert_eq!(check_endpoint(url, &local_offered()), Ok(()), "{url}");
    }
}

#[test]
fn nat64_and_ipv4_compatible_forms_embed_an_ipv4_address() {
    let p = local_offered();
    assert_eq!(
        check_endpoint("http://[64:ff9b::a9fe:a9fe]/v1", &p),
        Err(EndpointRefusal::LinkLocal),
        "NAT64 of 169.254.169.254"
    );
    assert_eq!(
        check_endpoint("http://[64:ff9b::a00:1]/v1", &p),
        Err(EndpointRefusal::PrivateNetwork),
        "NAT64 of 10.0.0.1"
    );
    assert_eq!(
        check_endpoint("http://[::a9fe:a9fe]/v1", &p),
        Err(EndpointRefusal::LinkLocal),
        "IPv4-compatible 169.254.169.254"
    );
    assert_eq!(
        check_endpoint("http://[64:ff9b::808:808]/v1", &p),
        Ok(()),
        "NAT64 of 8.8.8.8"
    );
    // `::1` is the v6 loopback, not the IPv4-compatible 0.0.0.1.
    assert_eq!(check_endpoint("http://[::1]/v1", &p), Ok(()));
    assert_eq!(
        check_endpoint("http://[::1]/v1", &server_side()),
        Err(EndpointRefusal::Loopback)
    );
}

#[test]
fn reserved_and_this_network_ranges_are_never_routable() {
    for addr in [
        "http://0.1.2.3/v1",
        "http://240.0.0.1/v1",
        "http://250.1.1.1/v1",
    ] {
        assert_eq!(
            check_endpoint(addr, &local_offered().with_private(true)),
            Err(EndpointRefusal::PrivateNetwork),
            "{addr}"
        );
    }
}

#[test]
fn allow_private_admits_lan_addresses_only() {
    let lan = EndpointPolicy::desktop().with_private(true);
    for addr in [
        "http://10.0.0.5/v1",
        "http://192.168.1.10/v1",
        "http://172.16.0.1/v1",
        "http://100.64.0.1/v1",
        "http://[fc00::1]/v1",
    ] {
        assert_eq!(check_endpoint(addr, &lan), Ok(()), "{addr}");
    }
    assert_eq!(
        check_endpoint("http://0.0.0.0/v1", &lan),
        Err(EndpointRefusal::PrivateNetwork)
    );
}

#[test]
fn a_local_only_policy_reaches_this_machine_and_the_lan_but_no_public_host() {
    let p = EndpointPolicy::local_only();
    assert_eq!(check_endpoint("http://127.0.0.1:11434/v1", &p), Ok(()));
    assert_eq!(check_endpoint("http://localhost:11434/v1", &p), Ok(()));
    assert_eq!(
        check_endpoint("https://api.openai.com/v1", &p),
        Err(EndpointRefusal::NonLocal)
    );
    assert_eq!(
        check_endpoint("https://8.8.8.8/v1", &p),
        Err(EndpointRefusal::NonLocal)
    );
    assert_eq!(
        check_endpoint("https://[2001:4860:4860::8888]/v1", &p),
        Err(EndpointRefusal::NonLocal)
    );
    assert_eq!(
        check_endpoint("http://10.0.0.5/v1", &p),
        Err(EndpointRefusal::PrivateNetwork)
    );
    assert_eq!(
        check_endpoint("http://10.0.0.5/v1", &p.with_private(true)),
        Ok(())
    );
}

#[test]
fn presets_have_the_documented_caps_and_flags() {
    let hosted = EndpointPolicy::hosted();
    assert!(!hosted.allow_loopback && !hosted.allow_private && hosted.allow_public);
    assert!(hosted.credentialed_http_loopback_only);
    assert_eq!(hosted.max_redirects, 3);
    assert_eq!(hosted.timeout, Duration::from_secs(10));
    assert_eq!(hosted.fail_body_cap, 64 * 1024);
    assert_eq!(hosted.catalog_cap, 16 * 1024 * 1024);
    assert_eq!(hosted.page_cap, 4 * 1024 * 1024);
    assert_eq!(EndpointPolicy::default(), hosted);
    let desktop = EndpointPolicy::desktop();
    assert!(desktop.allow_loopback && desktop.allow_public);
    let offline = EndpointPolicy::local_only();
    assert!(offline.allow_loopback && !offline.allow_public);
    let tuned = EndpointPolicy::hosted()
        .with_loopback(true)
        .with_max_redirects(1)
        .with_timeout(Duration::from_secs(2));
    assert!(tuned.allow_loopback);
    assert_eq!(tuned.max_redirects, 1);
    assert_eq!(tuned.timeout, Duration::from_secs(2));
}

#[test]
fn credentialed_cleartext_is_allowed_when_the_policy_relaxes_it() {
    let mut p = server_side();
    p.credentialed_http_loopback_only = false;
    assert_eq!(
        check_endpoint_with_credential("http://gateway.acme.test/v1", &p, true),
        Ok(())
    );
}

#[test]
fn every_refusal_has_a_sentence_that_names_the_next_step() {
    for refusal in [
        EndpointRefusal::Unparseable,
        EndpointRefusal::Scheme,
        EndpointRefusal::Loopback,
        EndpointRefusal::LinkLocal,
        EndpointRefusal::PrivateNetwork,
        EndpointRefusal::Cleartext,
        EndpointRefusal::NonLocal,
    ] {
        let said = refusal.to_string();
        assert!(said.len() > 15, "{refusal:?}: {said}");
        let boxed: Box<dyn std::error::Error> = Box::new(refusal);
        assert_eq!(boxed.to_string(), said);
    }
    assert!(EndpointRefusal::Cleartext.to_string().contains("https"));
}

// ---- redirects -----------------------------------------------------------

#[test]
fn a_location_header_is_resolved_against_the_answering_url() {
    assert_eq!(
        resolve_redirect("https://api.acme.test/v1/models", "/v2/models").as_deref(),
        Some("https://api.acme.test/v2/models")
    );
    assert_eq!(
        resolve_redirect("https://api.acme.test/v1/", "models").as_deref(),
        Some("https://api.acme.test/v1/models")
    );
    assert_eq!(
        resolve_redirect("https://a.test/", "http://169.254.169.254/x").as_deref(),
        Some("http://169.254.169.254/x")
    );
    assert_eq!(resolve_redirect("not a url", "/x"), None);
    assert_eq!(resolve_redirect("https://a.test/", "http://"), None);
}

#[test]
fn three_redirects_are_followed_and_the_fourth_is_refused() {
    let p = server_side();
    for hop in 1..=3 {
        assert_eq!(
            check_redirect(&p, "https://a.test/1", "https://a.test/2", true, hop),
            Ok(())
        );
    }
    assert_eq!(
        check_redirect(&p, "https://a.test/1", "https://a.test/2", true, 4),
        Err(PolicyViolation::TooManyRedirects { max: 3 })
    );
}

#[test]
fn a_redirect_to_the_metadata_address_is_refused_on_every_hop() {
    let p = server_side();
    assert_eq!(
        check_redirect(
            &p,
            "https://a.test/1",
            "http://169.254.169.254/latest",
            false,
            1
        ),
        Err(PolicyViolation::Endpoint(EndpointRefusal::LinkLocal))
    );
}

#[test]
fn a_credentialed_request_never_leaves_its_origin() {
    let p = server_side();
    assert_eq!(
        check_redirect(&p, "https://a.test/1", "https://b.test/1", true, 1),
        Err(PolicyViolation::CrossOriginRedirect)
    );
    // Without a credential a cross-origin redirect is fine.
    assert_eq!(
        check_redirect(&p, "https://a.test/1", "https://b.test/1", false, 1),
        Ok(())
    );
    // A credentialed redirect to cleartext off-host is refused by the cleartext
    // rule first.
    assert_eq!(
        check_redirect(&p, "https://a.test/1", "http://a.test/1", true, 1),
        Err(PolicyViolation::Endpoint(EndpointRefusal::Cleartext))
    );
}

// ---- headers ---------------------------------------------------------------

fn header_set() -> Vec<(String, String)> {
    [
        ("Authorization", "Bearer test-token"),
        ("X-Api-Key", "sk-not-a-real-key"),
        ("Content-Type", "application/json"),
        ("x-sdk-name", "opencompany"),
        ("Cookie", "s=1"),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
    .collect()
}

#[test]
fn the_product_header_goes_only_to_first_party_hosts() {
    let policy = HeaderPolicy::builtin();
    assert!(
        policy.allows_product_header_to("https://api.tinyhumans.ai/agent-integrations/openrouter")
    );
    assert!(policy.allows_product_header_to("https://tinyhumans.ai/x"));
    assert!(policy.allows_product_header_to("https://api.openhuman.ai/v1"));
    for third_party in [
        "https://openrouter.ai/api/v1",
        "http://localhost:11434/v1",
        "https://tinyhumans.ai.evil.test/v1",
        "https://nottinyhumans.ai/v1",
        "not a url",
    ] {
        assert!(
            !policy.allows_product_header_to(third_party),
            "{third_party}"
        );
    }
}

#[test]
fn the_product_header_is_stripped_unless_first_party() {
    let policy = HeaderPolicy::default();
    let mut third = header_set();
    policy.strip_product_header_unless_first_party(&mut third, "https://openrouter.ai/api/v1");
    assert!(
        !third
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("x-sdk-name"))
    );
    assert_eq!(third.len(), 4);
    let mut first = header_set();
    policy.strip_product_header_unless_first_party(&mut first, "https://api.tinyhumans.ai/v1");
    assert_eq!(first.len(), 5);
}

#[test]
fn a_cross_origin_redirect_strips_every_credential_header() {
    let policy = HeaderPolicy::builtin();
    let mut headers = header_set();
    policy.strip_for_redirect(&mut headers, "https://a.test/1", "https://b.test/1");
    assert_eq!(
        headers,
        vec![("Content-Type".to_string(), "application/json".to_string())]
    );
    // A same-origin hop keeps everything.
    let mut kept = header_set();
    policy.strip_for_redirect(&mut kept, "https://a.test/1", "https://a.test/2");
    assert_eq!(kept.len(), 5);
}

#[test]
fn credential_header_names_are_case_insensitive() {
    let policy = HeaderPolicy::builtin();
    for name in [
        "authorization",
        "AUTHORIZATION",
        "X-API-Key",
        " api-key ",
        "Proxy-Authorization",
        "COOKIE",
    ] {
        assert!(policy.is_credential_header(name), "{name}");
    }
    assert!(!policy.is_credential_header("Content-Type"));
    assert!(!policy.is_credential_header("x-sdk-name"));
}

// ---- properties ------------------------------------------------------------

proptest! {
    #[test]
    fn check_endpoint_never_panics(url in "[ -~]{0,80}") {
        let _ = check_endpoint(&url, &server_side());
        let _ = check_endpoint_with_credential(&url, &local_offered(), true);
        let _ = same_origin(&url, "https://a.test/");
        let _ = resolve_redirect("https://a.test/", &url);
    }

    #[test]
    fn rfc1918_link_local_and_cgnat_are_always_refused_under_hosted_and_desktop(
        b in 0u8..=255, c in 0u8..=255, d in 0u8..=255, which in 0usize..5,
    ) {
        let ip = match which {
            0 => Ipv4Addr::new(10, b, c, d),
            1 => Ipv4Addr::new(172, 16 + (b % 16), c, d),
            2 => Ipv4Addr::new(192, 168, c, d),
            3 => Ipv4Addr::new(169, 254, c, d),
            _ => Ipv4Addr::new(100, 64 + (b % 64), c, d),
        };
        for policy in [server_side(), local_offered()] {
            prop_assert!(check_address(IpAddr::V4(ip), &policy).is_err(), "{ip}");
            let mapped = IpAddr::V6(ip.to_ipv6_mapped());
            prop_assert!(check_address(mapped, &policy).is_err(), "mapped {ip}");
        }
    }

    #[test]
    fn a_public_v4_address_is_allowed_when_public_hosts_are(
        a in 1u8..=223, b in 0u8..=255, c in 0u8..=255, d in 1u8..=254,
    ) {
        let ip = Ipv4Addr::new(a, b, c, d);
        let special = ip.is_private()
            || ip.is_loopback()
            || ip.is_link_local()
            || (a == 100 && (64..128).contains(&b))
            || ip.is_multicast()
            || a >= 240
            || (a == 192 && b == 0 && (c == 0 || c == 2))
            || (a == 198 && (b == 18 || b == 19 || (b == 51 && c == 100)))
            || (a == 203 && b == 0 && c == 113);
        prop_assume!(!special);
        prop_assert_eq!(check_address(IpAddr::V4(ip), &server_side()), Ok(()));
        prop_assert_eq!(
            check_address(IpAddr::V4(ip), &EndpointPolicy::local_only()),
            Err(EndpointRefusal::NonLocal)
        );
    }

    #[test]
    fn loopback_answer_depends_only_on_the_policy(port in 1u16..65535) {
        let url = format!("http://127.0.0.1:{port}/v1");
        prop_assert_eq!(check_endpoint(&url, &server_side()), Err(EndpointRefusal::Loopback));
        prop_assert_eq!(check_endpoint(&url, &local_offered()), Ok(()));
        let v6 = IpAddr::V6(Ipv6Addr::LOCALHOST);
        prop_assert_eq!(check_address(v6, &server_side()), Err(EndpointRefusal::Loopback));
    }
}

proptest! {
    #[test]
    fn desktop_allows_everything_hosted_allows_and_local_only_allows_nothing_more(
        a in 0u8..=255, b in 0u8..=255, c in 0u8..=255, d in 0u8..=255,
        host in prop::sample::select(vec!["localhost", "api.acme.test", "ollama.localhost"]),
        use_ip in proptest::bool::ANY,
    ) {
        let url = if use_ip {
            format!("http://{a}.{b}.{c}.{d}/v1")
        } else {
            format!("https://{host}/v1")
        };
        let hosted = check_endpoint(&url, &EndpointPolicy::hosted());
        let desktop = check_endpoint(&url, &EndpointPolicy::desktop());
        let offline = check_endpoint(&url, &EndpointPolicy::local_only());
        if hosted.is_ok() {
            prop_assert!(desktop.is_ok(), "{url}");
        }
        if offline.is_ok() {
            prop_assert!(desktop.is_ok(), "{url}");
        }
        // Link-local is refused everywhere, however the policy is tuned.
        if use_ip && a == 169 && b == 254 {
            for policy in [EndpointPolicy::hosted(), EndpointPolicy::desktop().with_private(true)] {
                prop_assert_eq!(check_endpoint(&url, &policy), Err(EndpointRefusal::LinkLocal));
            }
        }
    }
}

#[test]
fn userinfo_is_refused_everywhere_an_endpoint_is_checked() {
    for url in [
        "https://alice:hunter2@api.acme.test/v1",
        "https://sk-not-a-real-key@api.acme.test/v1",
        "http://u:pw@lan-host.test/v1",
        "https://:pw@api.acme.test/v1",
    ] {
        for policy in [
            server_side(),
            local_offered(),
            EndpointPolicy::desktop().with_private(true),
        ] {
            assert_eq!(
                check_endpoint(url, &policy),
                Err(EndpointRefusal::CredentialInUrl),
                "{url}"
            );
            assert_eq!(
                check_endpoint_with_credential(url, &policy, true),
                Err(EndpointRefusal::CredentialInUrl),
                "{url}"
            );
        }
    }
    // The address answer still wins for a metadata host with userinfo.
    assert_eq!(
        check_endpoint("http://u:pw@169.254.169.254/x", &server_side()),
        Err(EndpointRefusal::LinkLocal)
    );
    assert!(
        EndpointRefusal::CredentialInUrl
            .to_string()
            .contains("key field")
    );
}

#[test]
fn a_redirect_target_with_userinfo_is_a_credential_violation() {
    let p = server_side();
    assert_eq!(
        check_redirect(&p, "https://a.test/1", "https://u:pw@a.test/2", false, 1),
        Err(PolicyViolation::CredentialInEndpoint)
    );
}

#[test]
fn a_custom_credential_header_is_stripped_when_the_policy_knows_its_name() {
    use crate::taxonomy::AuthStyle;
    // Regression (review finding): the fixed credential list ignored
    // `AuthStyle::Custom`, so `x-acme-key` followed a redirect off-origin.
    let mut headers = vec![
        ("X-Acme-Key".to_string(), "sk-not-a-real-key".to_string()),
        ("Content-Type".to_string(), "application/json".to_string()),
    ];
    let base = HeaderPolicy::builtin();
    let mut unaware = headers.clone();
    base.strip_for_redirect(&mut unaware, "https://a.test/1", "https://b.test/1");
    assert_eq!(
        unaware.len(),
        2,
        "the built-in list does not know x-acme-key"
    );
    let aware = base.with_auth(&AuthStyle::Custom("X-Acme-Key".into()));
    assert!(aware.is_credential_header("x-acme-key"));
    aware.strip_for_redirect(&mut headers, "https://a.test/1", "https://b.test/1");
    assert_eq!(
        headers,
        vec![("Content-Type".to_string(), "application/json".to_string())]
    );
    // Adding the same style twice does not duplicate the name.
    let twice = aware
        .clone()
        .with_auth(&AuthStyle::Custom("x-acme-key".into()));
    assert_eq!(twice, aware);
    // The built-in styles add nothing new.
    assert_eq!(
        HeaderPolicy::builtin().with_auth(&AuthStyle::Bearer),
        HeaderPolicy::builtin()
    );
    assert_eq!(
        HeaderPolicy::builtin().with_auth(&AuthStyle::None),
        HeaderPolicy::builtin()
    );
}

#[test]
fn a_first_party_check_reads_the_host_a_client_connects_to() {
    // Regression (review finding): `endpoint_host` used to disagree with the
    // WHATWG parser on `\`, so this URL (which connects to evil.test) counted
    // as tinyhumans.ai and received the product header.
    let policy = HeaderPolicy::builtin();
    assert!(!policy.allows_product_header_to("https://evil.test\\@tinyhumans.ai/"));
    assert!(policy.allows_product_header_to("https://tinyhumans.ai\\.evil.test/"));
    assert!(!policy.allows_product_header_to("https://evil.test:443\\@api.tinyhumans.ai/v1"));
}

#[test]
fn reserved_documentation_and_transition_ranges_are_never_endpoints() {
    // Regression (review round 2): these fell through to "public", and only
    // three IPv6-embedded-IPv4 forms were unwrapped.
    let p = local_offered().with_private(true);
    for url in [
        "http://192.0.0.1/v1",
        "http://192.0.2.10/v1",
        "http://198.18.0.1/v1",
        "http://198.19.255.254/v1",
        "http://198.51.100.7/v1",
        "http://203.0.113.9/v1",
        "http://[2001:0:4136:e378:8000:63bf:3fff:fdd2]/v1",
        "http://[2001:db8::1]/v1",
        "http://[64:ff9b:1::1]/v1",
    ] {
        assert_eq!(
            check_endpoint(url, &p),
            Err(EndpointRefusal::PrivateNetwork),
            "{url}"
        );
    }
    // 6to4 embeds the IPv4 address in bits 16..48: metadata and loopback stay refused.
    assert_eq!(
        check_endpoint("http://[2002:a9fe:a9fe::1]/v1", &p),
        Err(EndpointRefusal::LinkLocal)
    );
    assert_eq!(
        check_endpoint("http://[2002:7f00:1::1]/v1", &server_side()),
        Err(EndpointRefusal::Loopback)
    );
    assert_eq!(
        check_endpoint("http://[2002:0808:0808::1]/v1", &server_side()),
        Ok(()),
        "6to4 of 8.8.8.8"
    );
    // The neighbours of the reserved blocks are ordinary public addresses.
    for url in [
        "http://192.0.1.1/v1",
        "http://198.17.0.1/v1",
        "http://198.20.0.1/v1",
        "http://198.51.101.1/v1",
        "http://203.0.114.1/v1",
    ] {
        assert_eq!(check_endpoint(url, &server_side()), Ok(()), "{url}");
    }
}

#[test]
fn an_absolute_host_name_and_a_mixed_case_entry_still_match_first_party() {
    // Regression (review round 2).
    let mut policy = HeaderPolicy::builtin();
    assert!(policy.allows_product_header_to("https://api.tinyhumans.ai./v1"));
    policy.first_party_hosts.push("Example.Org.".to_string());
    assert!(policy.allows_product_header_to("https://api.example.org/v1"));
    assert!(!policy.allows_product_header_to("https://example.org.evil.test/v1"));
}

#[test]
fn only_an_ipv4_mapped_loopback_counts_as_this_host_for_cleartext() {
    // Regression (review round 3): NAT64, 6to4 and IPv4-compatible forms of
    // 127.0.0.1 leave the machine, so a key must not go to them over http.
    let p = local_offered();
    for url in [
        "http://[64:ff9b::7f00:1]/v1",
        "http://[2002:7f00:1::]/v1",
        "http://[::7f00:1]/v1",
    ] {
        assert!(
            check_endpoint_with_credential(url, &p, true).is_err(),
            "{url}"
        );
    }
    assert_eq!(
        check_endpoint_with_credential("http://[::ffff:127.0.0.1]/v1", &p, true),
        Ok(())
    );
    assert_eq!(
        check_endpoint_with_credential("http://[::1]/v1", &p, true),
        Ok(())
    );
}

#[test]
fn a_credential_named_query_parameter_is_refused_like_userinfo() {
    for url in [
        "https://generativelanguage.test/v1?key=AIzaFAKE",
        "https://h.test/v1?api_key=x",
        "https://h.test/v1?access_token=x&a=1",
        "https://h.test/v1?a=1&Signature=x",
    ] {
        assert_eq!(
            check_endpoint(url, &server_side()),
            Err(EndpointRefusal::CredentialInUrl),
            "{url}"
        );
    }
    assert_eq!(
        check_endpoint("https://h.test/v1?version=2&limit=5", &server_side()),
        Ok(())
    );
}

#[test]
fn ipv4_translated_siit_addresses_get_the_embedded_ipv4_answer() {
    // Regression (review round 4): `::ffff:0:a.b.c.d` fell through to public.
    let p = server_side();
    assert_eq!(
        check_endpoint("http://[::ffff:0:169.254.169.254]/", &p),
        Err(EndpointRefusal::LinkLocal)
    );
    assert_eq!(
        check_endpoint("http://[::ffff:0:10.0.0.1]/", &p),
        Err(EndpointRefusal::PrivateNetwork)
    );
    assert_eq!(
        check_endpoint("http://[::ffff:0:127.0.0.1]/", &p),
        Err(EndpointRefusal::Loopback)
    );
    assert_eq!(
        check_endpoint("http://[::ffff:0:127.0.0.1]/", &local_offered()),
        Ok(())
    );
    assert_eq!(check_endpoint("http://[::ffff:0:8.8.8.8]/", &p), Ok(()));
    // ...but it does not count as this host for a cleartext credential.
    assert!(
        check_endpoint_with_credential("http://[::ffff:0:127.0.0.1]/", &local_offered(), true)
            .is_err()
    );
}

#[test]
fn a_localhost_subdomain_never_receives_a_cleartext_key() {
    // Regression (review round 4): `*.localhost` is loopback only on resolvers
    // that pin it; the exemption is for the exact name `localhost`.
    let p = local_offered();
    for url in [
        "http://evil.localhost/v1",
        "http://foo.bar.localhost:11434/v1",
    ] {
        assert_eq!(
            check_endpoint(url, &p),
            Ok(()),
            "still loopback for the allowance: {url}"
        );
        assert_eq!(
            check_endpoint_with_credential(url, &p, true),
            Err(EndpointRefusal::Cleartext),
            "{url}"
        );
        assert_eq!(
            check_endpoint(url, &server_side()),
            Err(EndpointRefusal::Loopback),
            "{url}"
        );
        // https is fine: the certificate names the host.
        let https = url.replace("http://", "https://");
        assert_eq!(
            check_endpoint_with_credential(&https, &p, true),
            Ok(()),
            "{https}"
        );
    }
    assert_eq!(
        check_endpoint_with_credential("http://localhost/v1", &p, true),
        Ok(())
    );
    assert_eq!(
        check_endpoint_with_credential("http://LOCALHOST./v1", &p, true),
        Ok(())
    );
}

#[test]
fn a_double_encoded_query_name_is_not_a_credential_and_azure_code_is() {
    assert_eq!(
        check_endpoint("https://h.test/v1?%256Bey=1", &server_side()),
        Ok(())
    );
    assert_eq!(
        check_endpoint("https://h.test/v1?%6Bey=1", &server_side()),
        Err(EndpointRefusal::CredentialInUrl)
    );
    assert_eq!(
        check_endpoint("https://app.azurewebsites.net/api?code=k", &server_side()),
        Err(EndpointRefusal::CredentialInUrl)
    );
    assert_eq!(
        check_endpoint("https://gw.test/api?code=eu", &server_side()),
        Ok(())
    );
}

#[test]
fn headers_the_product_header_is_never_allowed_to_an_unreadable_url() {
    let policy = HeaderPolicy::builtin();
    for url in ["", "not a url", "http://", "://x"] {
        assert!(!policy.allows_product_header_to(url), "{url:?}");
    }
}
