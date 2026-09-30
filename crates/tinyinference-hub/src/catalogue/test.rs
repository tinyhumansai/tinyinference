//! Catalogue tests: counts, uniqueness, lookups, and the rules ported from
//! OpenCompany's `catalogue_tests_basics.rs` / `catalogue_tests_endpoints.rs`.

use std::collections::HashSet;

use super::*;
use crate::descriptor::Quirk;
use crate::policy::{EndpointPolicy, check_endpoint};
use crate::taxonomy::{CliKind, LocalRuntime, Protocol, TestDepth, Transport};

fn slugs_in(group: ProviderGroup) -> Vec<&'static str> {
    descriptors_in(group).map(|d| d.slug()).collect()
}

#[test]
fn the_catalogue_ships_the_counts_the_plan_names() {
    assert_eq!(descriptors_in(ProviderGroup::Managed).count(), 1, "managed");
    assert_eq!(descriptors_in(ProviderGroup::Cloud).count(), 26, "cloud");
    assert_eq!(descriptors_in(ProviderGroup::Local).count(), 5, "local");
    assert_eq!(descriptors_in(ProviderGroup::Cli).count(), 2, "cli");
    assert_eq!(descriptors_in(ProviderGroup::Custom).count(), 0);
    assert_eq!(descriptors_in(ProviderGroup::OAuthBacked).count(), 0);
    assert_eq!(descriptors().len(), 34);
    // Managed sorts first (D5).
    assert_eq!(descriptors()[0].slug(), "tinyhumans");
}

#[test]
fn kind_ids_and_aliases_are_unique_across_the_whole_catalogue() {
    let mut seen = HashSet::new();
    for d in descriptors() {
        assert!(
            seen.insert(d.slug().to_string()),
            "duplicate kind {}",
            d.slug()
        );
        for alias in d.aliases {
            assert!(
                seen.insert((*alias).to_ascii_lowercase()),
                "alias {alias} collides"
            );
        }
    }
}

#[test]
fn every_descriptor_is_well_formed() {
    for d in descriptors() {
        assert!(!d.label.is_empty(), "{}", d.slug());
        assert_eq!(
            d.slug(),
            d.slug().to_ascii_lowercase(),
            "kind ids are lowercase"
        );
        assert!(
            d.slug()
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
            "{}",
            d.slug()
        );
        match d.group {
            ProviderGroup::Cloud => {
                assert!(
                    d.default_endpoint.unwrap().starts_with("https://"),
                    "{}",
                    d.slug()
                );
                assert!(
                    !d.endpoint_editable,
                    "cloud endpoints are presets (G2): {}",
                    d.slug()
                );
                assert!(d.needs_key, "{}", d.slug());
                assert_eq!(d.transport, Transport::Http);
            }
            ProviderGroup::Local => {
                assert!(d.endpoint_editable, "{}", d.slug());
                assert!(!d.needs_key, "no local runtime demands a key: {}", d.slug());
                assert!(d.local_runtime.is_some());
            }
            ProviderGroup::Cli => {
                assert_eq!(d.transport, Transport::Subprocess);
                assert_eq!(d.protocol, Protocol::CliStream);
                assert_eq!(d.default_endpoint, None);
                assert!(d.test_depths.is_empty());
                assert!(d.cli.is_some());
            }
            ProviderGroup::Managed => {
                assert!(!d.endpoint_editable);
                assert_eq!(
                    d.default_endpoint, None,
                    "the host supplies the managed endpoint (Q3)"
                );
            }
            _ => panic!("unexpected group for {}", d.slug()),
        }
        if d.group != ProviderGroup::Cli {
            assert!(d.supports_depth(TestDepth::Completion), "{}", d.slug());
            assert!(d.supports_depth(TestDepth::Catalog), "{}", d.slug());
        }
    }
}

#[test]
fn every_default_endpoint_passes_its_own_policy() {
    for d in descriptors() {
        let Some(endpoint) = d.default_endpoint else {
            continue;
        };
        let policy = match d.group {
            ProviderGroup::Cloud => EndpointPolicy::hosted(),
            _ => EndpointPolicy::desktop(),
        };
        assert_eq!(
            check_endpoint(endpoint, &policy),
            Ok(()),
            "{}: {endpoint}",
            d.slug()
        );
        assert!(endpoint_host(endpoint).is_some());
    }
}

#[test]
fn lookups_resolve_kinds_and_aliases_case_insensitively() {
    assert_eq!(descriptor("openai").unwrap().slug(), "openai");
    assert_eq!(descriptor("  OpenAI ").unwrap().slug(), "openai");
    assert_eq!(descriptor("openhuman").unwrap().slug(), "tinyhumans");
    assert_eq!(descriptor("managed").unwrap().slug(), "tinyhumans");
    assert_eq!(descriptor("cloud").unwrap().slug(), "tinyhumans");
    assert_eq!(descriptor("lm-studio").unwrap().slug(), "lmstudio");
    assert_eq!(descriptor("LM_STUDIO").unwrap().slug(), "lmstudio");
    assert_eq!(descriptor("vllm").unwrap().slug(), "local-openai");
    assert_eq!(descriptor("llama.cpp").unwrap().slug(), "local-openai");
    assert_eq!(descriptor("mlx_lm").unwrap().slug(), "mlx");
    assert_eq!(descriptor("codex").unwrap().slug(), "codex");
    assert_eq!(
        resolve_kind("Custom-OpenAI"),
        Some(KindId::new("local-openai"))
    );
    assert_eq!(resolve_kind("nope"), None);
    assert!(descriptor("").is_none() && descriptor("   ").is_none());
    // `custom` is a group of operator records, not a catalogue row.
    assert!(descriptor("custom").is_none());
    assert_eq!(group_of("custom"), ProviderGroup::Custom);
    assert_eq!(group_of("my-gateway"), ProviderGroup::Custom);
}

#[test]
fn a_bare_openai_is_the_hosted_row_never_a_local_runtime() {
    let d = descriptor("openai").unwrap();
    assert_eq!(d.group, ProviderGroup::Cloud);
    assert_eq!(d.local_runtime, None);
    assert_eq!(group_of("openai"), ProviderGroup::Cloud);
}

#[test]
fn every_local_runtime_maps_to_a_local_descriptor() {
    for runtime in LocalRuntime::ALL {
        let d = descriptor_for_runtime(runtime).unwrap();
        assert_eq!(d.group, ProviderGroup::Local, "{runtime}");
    }
    assert_eq!(
        descriptor_for_runtime(LocalRuntime::Vllm)
            .unwrap()
            .local_runtime,
        Some(LocalRuntime::OpenAiCompatible)
    );
    assert_eq!(
        descriptor_for_runtime(LocalRuntime::Ollama)
            .unwrap()
            .local_runtime,
        Some(LocalRuntime::Ollama)
    );
}

// ---- auth (ported) --------------------------------------------------------------------

#[test]
fn anthropic_is_the_only_non_bearer_cloud_entry() {
    assert_eq!(auth_style_for("anthropic"), AuthStyle::Anthropic);
    for d in descriptors_in(ProviderGroup::Cloud).filter(|d| d.slug() != "anthropic") {
        assert_eq!(auth_style_for(d.slug()), AuthStyle::Bearer, "{}", d.slug());
    }
    assert_eq!(
        descriptor("anthropic").unwrap().protocol,
        Protocol::AnthropicMessages
    );
    assert_eq!(ANTHROPIC_VERSION, "2023-06-01");
}

#[test]
fn a_keyless_local_runtime_sends_no_auth_header_and_omlx_does() {
    assert_eq!(auth_style_for("ollama"), AuthStyle::None);
    assert_eq!(auth_style_for("lmstudio"), AuthStyle::None);
    assert_eq!(auth_style_for("mlx"), AuthStyle::None);
    assert_eq!(auth_style_for("omlx"), AuthStyle::Bearer);
    assert!(
        !descriptor("omlx").unwrap().needs_key,
        "accepting a key is not demanding one"
    );
}

#[test]
fn an_unknown_kind_is_a_custom_openai_compatible_endpoint() {
    assert_eq!(auth_style_for("my-gateway"), AuthStyle::Bearer);
    assert_eq!(auth_style_for("custom"), AuthStyle::Bearer);
    assert_eq!(auth_style_for("claude-code"), AuthStyle::None);
}

#[test]
fn minimax_keeps_the_openai_surface_that_is_the_fix() {
    let minimax = descriptor("minimax").unwrap();
    assert_eq!(minimax.default_endpoint, Some("https://api.minimax.io/v1"));
    assert_eq!(minimax.auth, AuthStyle::Bearer);
}

// ---- catalog shape (ported) ------------------------------------------------------------

#[test]
fn catalog_shape_is_paged_only_for_the_managed_kind_or_the_proxy_path() {
    let proxy = "https://api.tinyhumans.ai/agent-integrations/openrouter";
    assert_eq!(
        catalog_shape_for("tinyhumans", proxy),
        CatalogShape::PagedEnvelope
    );
    assert_eq!(
        catalog_shape_for(" tinyhumans ", ""),
        CatalogShape::PagedEnvelope
    );
    assert_eq!(
        catalog_shape_for("openhuman", ""),
        CatalogShape::PagedEnvelope
    );
    assert_eq!(
        catalog_shape_for("openrouter", &format!("{proxy}/")),
        CatalogShape::PagedEnvelope
    );
    assert_eq!(MANAGED_PROXY_PATH, "/agent-integrations/openrouter");
    assert_eq!(
        catalog_shape_for("openrouter", "https://api.tinyhumans.ai/openai/v1"),
        CatalogShape::OpenAi
    );
    assert_eq!(
        catalog_shape_for("custom", "http://127.0.0.1:8099/v1"),
        CatalogShape::OpenAi
    );
    for d in descriptors_in(ProviderGroup::Cloud) {
        assert_eq!(
            catalog_shape_for(d.slug(), d.default_endpoint.unwrap()),
            CatalogShape::OpenAi,
            "{}: only the managed kind pages",
            d.slug()
        );
    }
    assert_eq!(catalog_shape_for("claude-code", ""), CatalogShape::None);
}

// ---- endpoint helpers (ported) -----------------------------------------------------------

#[test]
fn only_openai_serves_the_responses_api_fallback() {
    assert!(!endpoint_is_chat_completions_only(
        "https://api.openai.com/v1"
    ));
    assert!(endpoint_is_chat_completions_only(
        "https://integrate.api.nvidia.com/v1"
    ));
    assert!(endpoint_is_chat_completions_only(
        "https://api.groq.com/openai/v1"
    ));
    assert!(!endpoint_is_chat_completions_only(
        "https://proxy.acme.dev/v1"
    ));
    assert!(!endpoint_is_chat_completions_only("not a url"));
    assert!(!endpoint_is_chat_completions_only(""));
}

#[test]
fn azure_is_detected_by_host_including_subdomains_and_sovereign_clouds() {
    for ok in [
        "https://my-resource.openai.azure.com/openai/v1",
        "https://r.services.ai.azure.com/openai/v1",
        "https://r.cognitiveservices.azure.com/openai/v1",
        "https://r.openai.azure.us/openai/v1",
        "https://r.openai.azure.cn/openai/v1",
        "https://openai.azure.com/x",
    ] {
        assert!(is_azure_endpoint(ok), "{ok}");
    }
    assert!(!is_azure_endpoint("https://api.openai.com/v1"));
    assert!(!is_azure_endpoint(""));
    assert_eq!(AZURE_ENDPOINT_HOSTS.len(), 5);
}

#[test]
fn the_foundry_serverless_hosts_and_lookalike_suffixes_are_not_azure() {
    assert!(!is_azure_endpoint("https://r.inference.ai.azure.com/v1"));
    assert!(!is_azure_endpoint("https://myopenai.azure.comx/v1"));
    assert!(!is_azure_endpoint("https://openai.azure.com.evil.test/v1"));
    assert!(!is_azure_endpoint("https://notopenai.azure.com/v1"));
}

#[test]
fn openrouter_reads_ask_for_the_whole_catalogue_and_only_on_its_own_host() {
    for endpoint in [
        "https://openrouter.ai/api/v1",
        "https://openrouter.ai/api/v1/",
        "https://eu.openrouter.ai/api/v1",
    ] {
        let query = catalog_query(endpoint);
        assert!(
            query.contains("output_modalities=all") && query.contains("limit=1000"),
            "{endpoint}"
        );
        assert!(is_openrouter_endpoint(endpoint));
    }
    for endpoint in [
        "https://api.anthropic.com/v1",
        "http://localhost:11434/v1",
        "https://openrouter.ai.example.com/v1",
        "https://api.tinyhumans.ai/openai/v1",
    ] {
        assert_eq!(catalog_query(endpoint), "", "{endpoint}");
        assert!(!is_openrouter_endpoint(endpoint), "{endpoint}");
    }
    let scoped = scoped_catalog_path("https://openrouter.ai/api/v1", true).unwrap();
    assert!(
        scoped.starts_with("/models/user")
            && scoped.contains("output_modalities=all")
            && scoped.contains("limit=1000")
    );
    assert!(scoped_catalog_path("https://openrouter.ai/api/v1", false).is_none());
    assert!(scoped_catalog_path("https://api.openai.com/v1", true).is_none());
}

// ---- reserved slugs ----------------------------------------------------------------------

#[test]
fn reserved_slugs_cover_kinds_aliases_cli_names_and_internal_words() {
    for reserved in [
        "openrouter",
        "ollama",
        "claude-code",
        "codex",
        "openai",
        "tinyhumans",
        "managed",
        "openhuman",
        "cloud",
        "pid",
        "byok-inference",
        "ephemeral-route",
        "vllm",
        "lm-studio",
        "llama.cpp",
        "OPENAI",
        "  groq ",
    ] {
        assert!(is_reserved_slug(reserved), "{reserved}");
    }
    // Route words and the custom group are not reserved.
    for free in ["local", "default", "custom", "acme-gateway", "", "  "] {
        assert!(!is_reserved_slug(free), "{free:?}");
    }
}

#[test]
fn the_reserved_list_is_sorted_unique_and_agrees_with_the_predicate() {
    let list = reserved_slugs();
    let mut sorted = list.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(list, sorted);
    for slug in &list {
        assert!(is_reserved_slug(slug), "{slug}");
    }
    for d in descriptors() {
        assert!(list.iter().any(|s| s == d.slug()), "{}", d.slug());
    }
    assert!(list.contains(&"openai".to_string()));
    assert_eq!(INTERNAL_SLUGS.len(), 7);
}

// ---- quirks and depths ----------------------------------------------------------------------

#[test]
fn quirks_and_depths_carry_the_documented_limitations() {
    assert!(descriptor("openai").unwrap().has_quirk(Quirk::ResponsesApi));
    assert!(
        descriptor("huggingface")
            .unwrap()
            .has_quirk(Quirk::CatalogUnauthenticated)
    );
    assert!(
        descriptor("venice")
            .unwrap()
            .has_quirk(Quirk::CatalogUnauthenticated)
    );
    assert!(
        descriptor("fireworks")
            .unwrap()
            .has_quirk(Quirk::CatalogAccountScoped)
    );
    assert!(
        descriptor("cerebras")
            .unwrap()
            .has_quirk(Quirk::CatalogKeylessTwin)
    );
    assert!(
        descriptor("modelscope")
            .unwrap()
            .has_quirk(Quirk::MeteredFreeTier)
    );
    assert!(
        descriptor("tinyhumans")
            .unwrap()
            .has_quirk(Quirk::CredentialChain)
    );
    assert!(
        descriptor("ollama")
            .unwrap()
            .has_quirk(Quirk::KeyCausesLocalAuthErrors)
    );
    assert!(!descriptor("groq").unwrap().has_quirk(Quirk::ResponsesApi));
    // Only OpenRouter has a KeyOnly depth (`GET /key`).
    let key_only: Vec<_> = descriptors()
        .iter()
        .filter(|d| d.supports_depth(TestDepth::KeyOnly))
        .map(|d| d.slug())
        .collect();
    assert_eq!(key_only, vec!["openrouter"]);
    let headers = descriptor("openrouter").unwrap().extra_headers;
    assert_eq!(headers.len(), 2);
    assert!(
        headers.iter().any(|(k, _)| *k == "HTTP-Referer")
            && headers.iter().any(|(k, _)| *k == "X-Title")
    );
    assert!(
        descriptors()
            .iter()
            .filter(|d| d.slug() != "openrouter")
            .all(|d| d.extra_headers.is_empty())
    );
}

#[test]
fn the_slug_sets_match_the_plan() {
    assert_eq!(
        slugs_in(ProviderGroup::Local),
        vec!["ollama", "lmstudio", "omlx", "mlx", "local-openai"]
    );
    assert_eq!(slugs_in(ProviderGroup::Cli), vec!["claude-code", "codex"]);
    assert_eq!(descriptor("codex").unwrap().cli, Some(CliKind::Codex));
    assert!(descriptor("claude-code").unwrap().test_depths.is_empty());
}

#[test]
fn a_descriptor_answers_to_its_aliases_and_kind() {
    let d = descriptor("lmstudio").unwrap();
    assert!(d.answers_to("LMSTUDIO") && d.answers_to("lm-studio") && !d.answers_to("ollama"));
}

#[test]
fn every_local_runtime_spelling_resolves_the_same_way_in_the_taxonomy_and_the_catalogue() {
    // Regression (review round 2): `openai_compatible` and `llama-cpp` were
    // runtimes to `LocalRuntime` but custom endpoints to the catalogue.
    let spellings = [
        "ollama",
        "lmstudio",
        "lm-studio",
        "lm_studio",
        "llamacpp",
        "llama.cpp",
        "llama_cpp",
        "llama-cpp",
        "vllm",
        "mlx",
        "mlx-server",
        "mlx_lm",
        "omlx",
        "omlx-server",
        "local-openai",
        "local_openai",
        "custom-openai",
        "custom_openai",
        "openai_compatible",
        "open_ai_compatible",
    ];
    for spelling in spellings {
        let runtime = LocalRuntime::parse_loose(spelling).unwrap_or_else(|| panic!("{spelling}"));
        let by_runtime = descriptor_for_runtime(runtime).unwrap();
        let by_name =
            descriptor(spelling).unwrap_or_else(|| panic!("{spelling} is not in the catalogue"));
        assert_eq!(by_name.slug(), by_runtime.slug(), "{spelling}");
        assert_eq!(group_of(spelling), ProviderGroup::Local, "{spelling}");
        assert!(is_reserved_slug(spelling), "{spelling}");
        // The serde spellings agree too.
        let parsed: LocalRuntime = serde_json::from_value(serde_json::json!(spelling))
            .unwrap_or_else(|_| panic!("serde {spelling}"));
        assert_eq!(parsed, runtime, "{spelling}");
    }
    // And every alias a local descriptor carries is a runtime spelling.
    for d in descriptors_in(ProviderGroup::Local) {
        for alias in d.aliases {
            assert!(LocalRuntime::parse_loose(alias).is_some(), "{alias}");
        }
    }
}

#[test]
fn an_absolute_fqdn_spelling_matches_the_same_provider() {
    // Regression (review round 4): a trailing dot made the host a different one.
    assert!(endpoint_is_chat_completions_only(
        "https://api.groq.com./openai/v1"
    ));
    assert!(is_azure_endpoint("https://x.openai.azure.com./openai/v1"));
    assert!(is_openrouter_endpoint("https://openrouter.ai./api/v1"));
    assert!(!endpoint_is_chat_completions_only(
        "https://api.openai.com./v1"
    ));
}
