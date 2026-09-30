//! The five local runtimes.
//!
//! OpenCompany ships three of them (ollama, lmstudio, omlx) with no default
//! endpoint for the last two; OpenHuman adds `mlx` and `local-openai` and
//! defaults. The hub carries all five. `needs_key` is a floor, not a ceiling: a
//! runtime that takes no key can still be given one.

use crate::descriptor::{ProviderDescriptor, Quirk};
use crate::ids::KindId;
use crate::taxonomy::{
    AuthStyle, CatalogShape, LocalRuntime, Protocol, ProviderGroup, TestDepth, Transport,
};

const DEPTHS: &[TestDepth] = &[TestDepth::Catalog, TestDepth::Completion];

struct Local {
    slug: &'static str,
    label: &'static str,
    default_endpoint: Option<&'static str>,
    auth: AuthStyle,
    runtime: LocalRuntime,
    aliases: &'static [&'static str],
    quirks: &'static [Quirk],
}

pub(super) fn descriptors() -> Vec<ProviderDescriptor> {
    let rows = [
        Local {
            slug: "ollama",
            label: "Ollama",
            // A bare origin; `normalize_local_endpoint` appends `/v1`.
            default_endpoint: Some("http://localhost:11434"),
            // "No authentication is required ... locally", and `OLLAMA_API_KEY`
            // is a Cloud-only variable whose presence locally is a documented
            // cause of spurious 401s.
            auth: AuthStyle::None,
            runtime: LocalRuntime::Ollama,
            aliases: &[],
            quirks: &[Quirk::KeyCausesLocalAuthErrors],
        },
        Local {
            slug: "lmstudio",
            label: "LM Studio",
            default_endpoint: Some("http://localhost:1234/v1"),
            auth: AuthStyle::None,
            runtime: LocalRuntime::LmStudio,
            aliases: &["lm-studio", "lm_studio"],
            quirks: &[Quirk::NoModelLoadedError],
        },
        Local {
            slug: "omlx",
            label: "OMLX",
            // Three different projects answer to this name and listen on three
            // different ports, so a default would be a guess.
            default_endpoint: None,
            // Bearer when there is a key to send (the `jundot/omlx --api-key`
            // operator), nothing when there is not.
            auth: AuthStyle::Bearer,
            runtime: LocalRuntime::Omlx,
            aliases: &["omlx-server"],
            quirks: &[Quirk::AmbiguousRuntime],
        },
        Local {
            slug: "mlx",
            label: "MLX",
            default_endpoint: Some("http://127.0.0.1:8080/v1"),
            auth: AuthStyle::None,
            runtime: LocalRuntime::Mlx,
            aliases: &["mlx-server", "mlx_lm"],
            quirks: &[Quirk::PortCollidesWithHost],
        },
        Local {
            slug: "local-openai",
            label: "Local OpenAI-compatible",
            default_endpoint: Some("http://127.0.0.1:8080/v1"),
            auth: AuthStyle::Bearer,
            runtime: LocalRuntime::OpenAiCompatible,
            // Never a bare `openai`: that is the hosted row.
            aliases: &[
                "custom-openai",
                "custom_openai",
                "local_openai",
                "openai_compatible",
                "open_ai_compatible",
                "llamacpp",
                "llama.cpp",
                "llama_cpp",
                "llama-cpp",
                "vllm",
            ],
            quirks: &[Quirk::PortCollidesWithHost],
        },
    ];
    rows.into_iter()
        .map(|row| ProviderDescriptor {
            kind: KindId::new(row.slug),
            label: row.label,
            group: ProviderGroup::Local,
            transport: Transport::Http,
            protocol: Protocol::OpenAiChat,
            auth: row.auth,
            catalog: CatalogShape::OpenAi,
            default_endpoint: row.default_endpoint,
            endpoint_editable: true,
            needs_key: false,
            key_placeholder: None,
            local_runtime: Some(row.runtime),
            cli: None,
            aliases: row.aliases,
            test_depths: DEPTHS,
            free_text_models: false,
            extra_headers: &[],
            quirks: row.quirks,
        })
        .collect()
}
