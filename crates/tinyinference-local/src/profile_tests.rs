use super::*;

fn model_dir(config: &str) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), config).unwrap();
    dir
}

#[test]
fn mlx_window_comes_from_the_model_directory_config() {
    let flat = model_dir(r#"{"max_position_embeddings": 32768}"#);
    let nested = model_dir(r#"{"text_config": {"max_position_embeddings": 262144}}"#);
    let flat_id = flat.path().to_str().unwrap();
    let nested_id = nested.path().to_str().unwrap();
    assert_eq!(
        context_window_with_local_fallback(flat_id, None, Some(LocalProviderKind::Mlx)),
        Some(32_768)
    );
    assert_eq!(
        context_window_with_local_fallback(nested_id, None, Some(LocalProviderKind::Omlx)),
        Some(262_144)
    );
    // An explicit window still wins, and other runtimes ignore the directory.
    assert_eq!(
        context_window_with_local_fallback(flat_id, Some(8_000), Some(LocalProviderKind::Mlx)),
        Some(8_000)
    );
    assert_eq!(
        context_window_with_local_fallback(flat_id, None, Some(LocalProviderKind::Ollama)),
        Some(8_192)
    );
}

#[test]
fn mlx_window_falls_back_when_the_model_directory_is_unusable() {
    let missing_value = model_dir(r#"{"hidden_size": 4096}"#);
    let broken = model_dir("not json");
    for dir in [&missing_value, &broken] {
        assert_eq!(
            context_window_with_local_fallback(
                dir.path().to_str().unwrap(),
                None,
                Some(LocalProviderKind::Mlx)
            ),
            Some(4_096)
        );
    }
    assert_eq!(model_dir_context_window("relative/path"), None);
    assert_eq!(model_dir_context_window("/no/such/dir"), None);
}

#[test]
fn local_context_fallback_uses_profiles_and_conservative_floor() {
    assert_eq!(
        context_window_with_local_fallback("unknown", None, Some(LocalProviderKind::Ollama)),
        Some(8_192)
    );
    assert_eq!(
        context_window_with_local_fallback("unknown", None, Some(LocalProviderKind::Mlx)),
        Some(4_096)
    );
    assert_eq!(
        context_window_with_local_fallback("unknown", None, None),
        None
    );
    assert_eq!(
        context_window_with_local_fallback("unknown", None, Some(LocalProviderKind::LocalOpenai)),
        Some(4_096)
    );
    assert_eq!(
        context_window_with_local_fallback("known", Some(128_000), Some(LocalProviderKind::Ollama)),
        Some(128_000)
    );
}

#[test]
fn kind_from_str_loose_accepts_aliases() {
    assert_eq!(
        LocalProviderKind::from_str_loose("ollama"),
        Some(LocalProviderKind::Ollama)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("LM-Studio"),
        Some(LocalProviderKind::LmStudio)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("lm_studio"),
        Some(LocalProviderKind::LmStudio)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("mlx"),
        Some(LocalProviderKind::Mlx)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("mlx-server"),
        Some(LocalProviderKind::Mlx)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("llamacpp"),
        Some(LocalProviderKind::LocalOpenai)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("vllm"),
        Some(LocalProviderKind::LocalOpenai)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("custom-openai"),
        Some(LocalProviderKind::LocalOpenai)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("custom_openai"),
        Some(LocalProviderKind::LocalOpenai)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("openai"),
        Some(LocalProviderKind::LocalOpenai)
    );
    assert_eq!(LocalProviderKind::from_str_loose("unknown"), None);
}

#[test]
fn kind_from_provider_string_parses_prefixes() {
    for (provider, expected) in [
        ("lmstudio", LocalProviderKind::LmStudio),
        ("mlx", LocalProviderKind::Mlx),
        ("omlx", LocalProviderKind::Omlx),
        ("local-openai", LocalProviderKind::LocalOpenai),
        ("custom-openai", LocalProviderKind::LocalOpenai),
        ("custom_openai", LocalProviderKind::LocalOpenai),
    ] {
        assert_eq!(kind_from_provider_string(provider), Some(expected));
        assert!(is_local_provider_string(provider));
    }
    assert_eq!(
        kind_from_provider_string("ollama:qwen3:14b"),
        Some(LocalProviderKind::Ollama)
    );
    assert_eq!(
        kind_from_provider_string("lmstudio:mistral"),
        Some(LocalProviderKind::LmStudio)
    );
    assert_eq!(
        kind_from_provider_string("mlx:llama-3.1-8b"),
        Some(LocalProviderKind::Mlx)
    );
    assert_eq!(
        kind_from_provider_string("local-openai:qwen2"),
        Some(LocalProviderKind::LocalOpenai)
    );
    assert_eq!(kind_from_provider_string("openai:gpt-4o"), None);
    assert_eq!(kind_from_provider_string("openhuman"), None);
}

#[test]
fn is_local_identifies_local_strings() {
    assert!(is_local_provider_string("ollama:phi3"));
    assert!(is_local_provider_string("mlx:model"));
    assert!(!is_local_provider_string("openai:gpt-4"));
    assert!(!is_local_provider_string("openhuman"));
}

#[test]
fn profiles_have_correct_defaults() {
    let ollama = profile_for_kind(LocalProviderKind::Ollama);
    assert_eq!(ollama.tool_support, ToolSupport::PromptGuided);
    assert_eq!(ollama.default_context_window, Some(8_192));
    assert!(!ollama.supports_responses_api);

    let mlx = profile_for_kind(LocalProviderKind::Mlx);
    assert_eq!(mlx.default_context_window, Some(4_096));
}

#[test]
fn ollama_profile_is_conservative_on_tools() {
    let profile = profile_for_kind(LocalProviderKind::Ollama);
    assert_eq!(profile.tool_support, ToolSupport::PromptGuided);
}

#[test]
fn omlx_kind_and_profile() {
    assert_eq!(
        LocalProviderKind::from_str_loose("omlx"),
        Some(LocalProviderKind::Omlx)
    );
    assert_eq!(
        LocalProviderKind::from_str_loose("omlx-server"),
        Some(LocalProviderKind::Omlx)
    );
    assert_eq!(LocalProviderKind::Omlx.as_str(), "omlx");
    assert_eq!(LocalProviderKind::Omlx.display_name(), "OMLX");
    assert_eq!(
        profile_for_kind(LocalProviderKind::Omlx).default_base_url,
        "http://127.0.0.1:8000/v1"
    );
    assert_eq!(
        kind_from_provider_string("omlx:my-model"),
        Some(LocalProviderKind::Omlx)
    );
    assert!(is_local_provider_string("omlx:my-model"));
}
