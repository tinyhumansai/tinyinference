//! Taxonomy tests: serde spellings, legacy aliases, and the conversions that
//! reconcile the four local-runtime enums.

use serde_json::json;
use tinyinference_llm::ProviderKind;
use tinyinference_llm::providers::openai::LocalRuntimeKind;

use super::*;

fn round_trip<T>(value: &T) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    serde_json::from_value(serde_json::to_value(value).unwrap()).unwrap()
}

#[test]
fn every_group_round_trips_and_matches_its_wire_string() {
    for (group, wire) in [
        (ProviderGroup::Managed, "managed"),
        (ProviderGroup::Cloud, "cloud"),
        (ProviderGroup::Local, "local"),
        (ProviderGroup::Cli, "cli"),
        (ProviderGroup::Custom, "custom"),
        (ProviderGroup::OAuthBacked, "oauth_backed"),
    ] {
        assert_eq!(serde_json::to_value(group).unwrap(), json!(wire));
        assert_eq!(group.as_str(), wire);
        assert_eq!(group.to_string(), wire);
        assert_eq!(round_trip(&group), group);
    }
    assert_eq!(
        serde_json::from_value::<ProviderGroup>(json!("o_auth_backed")).unwrap(),
        ProviderGroup::OAuthBacked
    );
}

#[test]
fn transports_protocols_shapes_and_depths_round_trip() {
    for value in [Transport::Http, Transport::Subprocess] {
        assert_eq!(round_trip(&value), value);
    }
    assert_eq!(
        serde_json::to_value(Transport::Subprocess).unwrap(),
        json!("subprocess")
    );
    for value in [
        Protocol::OpenAiChat,
        Protocol::OpenAiResponses,
        Protocol::AnthropicMessages,
        Protocol::CliStream,
    ] {
        assert_eq!(round_trip(&value), value);
    }
    assert_eq!(
        serde_json::to_value(Protocol::OpenAiChat).unwrap(),
        json!("open_ai_chat")
    );
    for value in [
        CatalogShape::OpenAi,
        CatalogShape::PagedEnvelope,
        CatalogShape::OllamaTags,
        CatalogShape::LmStudioV0,
        CatalogShape::None,
    ] {
        assert_eq!(round_trip(&value), value);
    }
    assert_eq!(
        serde_json::to_value(CatalogShape::LmStudioV0).unwrap(),
        json!("lm_studio_v0")
    );
}

#[test]
fn test_depths_are_ordered_by_cost_and_named_stably() {
    assert!(TestDepth::KeyOnly < TestDepth::Catalog);
    assert!(TestDepth::Catalog < TestDepth::Completion);
    for (depth, wire) in [
        (TestDepth::KeyOnly, "key_only"),
        (TestDepth::Catalog, "catalog"),
        (TestDepth::Completion, "completion"),
    ] {
        assert_eq!(depth.as_str(), wire);
        assert_eq!(depth.to_string(), wire);
        assert_eq!(serde_json::to_value(depth).unwrap(), json!(wire));
        assert_eq!(round_trip(&depth), depth);
    }
}

#[test]
fn auth_style_wire_spellings_and_legacy_aliases() {
    for (style, wire) in [
        (AuthStyle::Bearer, "bearer"),
        (AuthStyle::XApiKey, "x_api_key"),
        (AuthStyle::Anthropic, "anthropic"),
        (AuthStyle::SessionJwt, "session_jwt"),
        (AuthStyle::None, "none"),
    ] {
        assert_eq!(serde_json::to_value(&style).unwrap(), json!(wire));
        assert_eq!(style.as_str(), wire);
        assert_eq!(round_trip(&style), style);
    }
    // OpenHuman's `OpenhumanJwt` was spelled two ways on disk; both load.
    for legacy in ["openhuman_jwt", "openhumanjwt"] {
        assert_eq!(
            serde_json::from_value::<AuthStyle>(json!(legacy)).unwrap(),
            AuthStyle::SessionJwt,
            "{legacy}"
        );
    }
    let custom = AuthStyle::Custom("api-key".to_string());
    assert_eq!(
        serde_json::to_value(&custom).unwrap(),
        json!({"custom": "api-key"})
    );
    assert_eq!(round_trip(&custom), custom);
    assert_eq!(custom.as_str(), "custom");
}

#[test]
fn only_none_needs_no_credential() {
    assert!(!AuthStyle::None.needs_credential());
    for style in [
        AuthStyle::Bearer,
        AuthStyle::XApiKey,
        AuthStyle::Anthropic,
        AuthStyle::SessionJwt,
        AuthStyle::Custom("x".into()),
    ] {
        assert!(style.needs_credential(), "{style:?}");
    }
}

#[test]
fn each_auth_style_names_the_headers_that_carry_its_credential() {
    assert_eq!(
        AuthStyle::Bearer.credential_headers(),
        vec!["authorization"]
    );
    assert_eq!(
        AuthStyle::SessionJwt.credential_headers(),
        vec!["authorization"]
    );
    assert_eq!(AuthStyle::XApiKey.credential_headers(), vec!["x-api-key"]);
    assert_eq!(AuthStyle::Anthropic.credential_headers(), vec!["x-api-key"]);
    assert!(AuthStyle::None.credential_headers().is_empty());
    assert_eq!(
        AuthStyle::Custom(" API-Key ".into()).credential_headers(),
        vec!["api-key"]
    );
    assert!(
        AuthStyle::Custom("  ".into())
            .credential_headers()
            .is_empty()
    );
}

#[test]
fn cli_kinds_map_option_slugs_to_stored_slugs() {
    assert_eq!(CliKind::ClaudeCode.option_slug(), "claude-code");
    assert_eq!(CliKind::ClaudeCode.stored_slug(), "claude-code");
    assert_eq!(CliKind::Codex.option_slug(), "codex");
    // Codex has no row of its own; its login is stored under `openai`.
    assert_eq!(CliKind::Codex.stored_slug(), "openai");
    assert_eq!(CliKind::from_option_slug(" codex "), Some(CliKind::Codex));
    assert_eq!(
        CliKind::from_option_slug("claude-code"),
        Some(CliKind::ClaudeCode)
    );
    assert_eq!(CliKind::from_option_slug("openai"), None);
    assert_eq!(
        serde_json::to_value(CliKind::ClaudeCode).unwrap(),
        json!("claude_code")
    );
    assert_eq!(round_trip(&CliKind::Codex), CliKind::Codex);
}

// ---- LocalRuntime -----------------------------------------------------------

#[test]
fn local_runtime_serialises_canonically_and_round_trips() {
    for runtime in LocalRuntime::ALL {
        let value = serde_json::to_value(runtime).unwrap();
        assert_eq!(value, json!(runtime.as_str()), "{runtime:?}");
        assert_eq!(runtime.to_string(), runtime.as_str());
        assert_eq!(round_trip(&runtime), runtime);
        assert_eq!(LocalRuntime::parse_loose(runtime.as_str()), Some(runtime));
    }
    assert_eq!(
        serde_json::to_value(LocalRuntime::OpenAiCompatible).unwrap(),
        json!("openai_compatible")
    );
}

#[test]
fn every_legacy_slug_still_loads() {
    for (legacy, expected) in [
        ("lmstudio", LocalRuntime::LmStudio),
        ("lm_studio", LocalRuntime::LmStudio),
        ("lm-studio", LocalRuntime::LmStudio),
        ("local-openai", LocalRuntime::OpenAiCompatible),
        ("local_openai", LocalRuntime::OpenAiCompatible),
        ("custom-openai", LocalRuntime::OpenAiCompatible),
        ("custom_openai", LocalRuntime::OpenAiCompatible),
        ("open_ai_compatible", LocalRuntime::OpenAiCompatible),
        ("llama_cpp", LocalRuntime::LlamaCpp),
        ("llamacpp", LocalRuntime::LlamaCpp),
        ("llama.cpp", LocalRuntime::LlamaCpp),
        ("vllm", LocalRuntime::Vllm),
        ("mlx", LocalRuntime::Mlx),
        ("mlx-server", LocalRuntime::Mlx),
        ("mlx_lm", LocalRuntime::Mlx),
        ("omlx", LocalRuntime::Omlx),
        ("omlx-server", LocalRuntime::Omlx),
        ("ollama", LocalRuntime::Ollama),
    ] {
        assert_eq!(
            serde_json::from_value::<LocalRuntime>(json!(legacy)).unwrap(),
            expected,
            "serde {legacy}"
        );
        assert_eq!(
            LocalRuntime::parse_loose(legacy),
            Some(expected),
            "loose {legacy}"
        );
        assert_eq!(
            LocalRuntime::parse_loose(&legacy.to_ascii_uppercase()),
            Some(expected),
            "loose upper {legacy}"
        );
    }
}

#[test]
fn a_bare_openai_is_never_a_local_runtime() {
    // OpenHuman's loose parser reads bare `openai` as LocalOpenai while its
    // `openai:` prefix form does not; the hub refuses the trap.
    for name in ["openai", "OpenAI", " openai ", "anthropic", "", "custom"] {
        assert_eq!(LocalRuntime::parse_loose(name), None, "{name:?}");
    }
    assert!(serde_json::from_value::<LocalRuntime>(json!("openai")).is_err());
}

#[test]
fn catalogue_slugs_collapse_the_generic_runtimes_onto_local_openai() {
    assert_eq!(LocalRuntime::Ollama.catalogue_slug(), "ollama");
    assert_eq!(LocalRuntime::LmStudio.catalogue_slug(), "lmstudio");
    assert_eq!(LocalRuntime::Mlx.catalogue_slug(), "mlx");
    assert_eq!(LocalRuntime::Omlx.catalogue_slug(), "omlx");
    for generic in [
        LocalRuntime::LlamaCpp,
        LocalRuntime::Vllm,
        LocalRuntime::OpenAiCompatible,
    ] {
        assert_eq!(generic.catalogue_slug(), "local-openai");
    }
}

#[test]
fn llm_provider_kinds_map_to_runtimes_only_for_local_ones() {
    assert_eq!(
        LocalRuntime::from_provider_kind(&ProviderKind::Ollama),
        Some(LocalRuntime::Ollama)
    );
    assert_eq!(
        LocalRuntime::from_provider_kind(&ProviderKind::LmStudio),
        Some(LocalRuntime::LmStudio)
    );
    assert_eq!(
        LocalRuntime::from_provider_kind(&ProviderKind::LlamaCpp),
        Some(LocalRuntime::LlamaCpp)
    );
    assert_eq!(
        LocalRuntime::from_provider_kind(&ProviderKind::Vllm),
        Some(LocalRuntime::Vllm)
    );
    for hosted in [
        ProviderKind::OpenAi,
        ProviderKind::Anthropic,
        ProviderKind::DeepSeek,
        ProviderKind::Groq,
        ProviderKind::Xai,
        ProviderKind::OpenRouter,
        ProviderKind::Fireworks,
        ProviderKind::TinyHumans,
        ProviderKind::Together,
        ProviderKind::Mistral,
        ProviderKind::Compatible,
    ] {
        assert_eq!(
            LocalRuntime::from_provider_kind(&hosted),
            None,
            "{hosted:?}"
        );
    }
}

#[test]
fn the_llm_local_runtime_enum_converts_both_ways_where_it_can() {
    for kind in [
        LocalRuntimeKind::Ollama,
        LocalRuntimeKind::LmStudio,
        LocalRuntimeKind::LlamaCpp,
        LocalRuntimeKind::Vllm,
    ] {
        let runtime = LocalRuntime::from(kind);
        assert_eq!(LocalRuntimeKind::try_from(runtime), Ok(kind));
    }
    for wider in [
        LocalRuntime::Mlx,
        LocalRuntime::Omlx,
        LocalRuntime::OpenAiCompatible,
    ] {
        assert_eq!(
            LocalRuntimeKind::try_from(wider),
            Err(UnsupportedRuntime(wider))
        );
    }
    let said = UnsupportedRuntime(LocalRuntime::Mlx).to_string();
    assert!(said.contains("mlx"), "{said}");
    let boxed: Box<dyn std::error::Error> = Box::new(UnsupportedRuntime(LocalRuntime::Omlx));
    assert!(boxed.to_string().contains("omlx"));
}

#[test]
fn legacy_as_str_outputs_of_the_old_enums_are_untouched() {
    // The hub adds an enum; it must not change what the existing ones print.
    assert_eq!(LocalRuntimeKind::LmStudio.as_str(), "lm_studio");
    assert_eq!(ProviderKind::LmStudio.as_str(), "lmstudio");
    assert_eq!(ProviderKind::LlamaCpp.as_str(), "llama_cpp");
}

#[cfg(feature = "local-bridge")]
mod bridge {
    use tinyinference_local::profile::LocalProviderKind;
    use tinyinference_local::provider::LocalAiProvider;

    use super::*;

    #[test]
    fn local_provider_kind_round_trips_except_where_it_collapses() {
        for (kind, runtime) in [
            (LocalProviderKind::Ollama, LocalRuntime::Ollama),
            (LocalProviderKind::LmStudio, LocalRuntime::LmStudio),
            (LocalProviderKind::Mlx, LocalRuntime::Mlx),
            (LocalProviderKind::Omlx, LocalRuntime::Omlx),
            (
                LocalProviderKind::LocalOpenai,
                LocalRuntime::OpenAiCompatible,
            ),
        ] {
            assert_eq!(LocalRuntime::from(kind), runtime);
            assert_eq!(LocalProviderKind::from(runtime), kind);
        }
        // llama.cpp and vLLM collapse to LocalOpenai, as the local crate's own
        // loose parser does.
        assert_eq!(
            LocalProviderKind::from(LocalRuntime::LlamaCpp),
            LocalProviderKind::LocalOpenai
        );
        assert_eq!(
            LocalProviderKind::from(LocalRuntime::Vllm),
            LocalProviderKind::LocalOpenai
        );
    }

    #[test]
    fn local_ai_provider_converts_for_the_two_managed_runtimes_only() {
        assert_eq!(
            LocalRuntime::from(LocalAiProvider::Ollama),
            LocalRuntime::Ollama
        );
        assert_eq!(
            LocalRuntime::from(LocalAiProvider::LmStudio),
            LocalRuntime::LmStudio
        );
        assert_eq!(
            LocalAiProvider::try_from(LocalRuntime::Ollama),
            Ok(LocalAiProvider::Ollama)
        );
        assert_eq!(
            LocalAiProvider::try_from(LocalRuntime::LmStudio),
            Ok(LocalAiProvider::LmStudio)
        );
        assert_eq!(
            LocalAiProvider::try_from(LocalRuntime::Vllm),
            Err(UnsupportedRuntime(LocalRuntime::Vllm))
        );
    }

    #[test]
    fn every_legacy_spelling_of_the_local_crate_parses_to_the_same_runtime() {
        for legacy in [
            "ollama",
            "lmstudio",
            "lm-studio",
            "lm_studio",
            "mlx",
            "mlx-server",
            "mlx_lm",
            "omlx",
            "omlx-server",
            "local-openai",
            "local_openai",
            "custom-openai",
            "custom_openai",
            "llamacpp",
            "llama.cpp",
            "vllm",
        ] {
            let via_local = LocalProviderKind::from_str_loose(legacy).unwrap();
            let via_hub = LocalRuntime::parse_loose(legacy).unwrap();
            assert_eq!(LocalProviderKind::from(via_hub), via_local, "{legacy}");
        }
    }
}
