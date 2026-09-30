//! The 26 hosted providers both hosts share.
//!
//! ## The endpoints are presets, not a pattern
//!
//! Look at the paths: `/openai/v1`, `/inference/v1`, `/v1beta/openai`,
//! `/v1/openai`, `/v3/openai`, `/api/paas/v4`, `/api/gateway`, and DeepSeek's
//! bare host with no version segment. Deriving an endpoint as
//! `https://{host}/v1` is wrong for roughly a third of this list, which is why
//! each row carries its own URL and the cloud group never asks the operator to
//! type one. That is also a limitation: a preset is one value, so a vendor
//! running two products or regions behind different paths can be served for at
//! most one of them (typed as [`Quirk::RegionalEndpointSplit`]).

use crate::descriptor::{ProviderDescriptor, Quirk};
use crate::ids::KindId;
use crate::taxonomy::{AuthStyle, CatalogShape, Protocol, ProviderGroup, TestDepth, Transport};

/// How a row's credential is presented (a `Copy` subset of [`AuthStyle`] so the
/// table can be a `const`).
#[derive(Clone, Copy)]
enum RowAuth {
    Bearer,
    Anthropic,
}

struct CloudRow {
    slug: &'static str,
    label: &'static str,
    endpoint: &'static str,
    auth: RowAuth,
    key_placeholder: Option<&'static str>,
    quirks: &'static [Quirk],
}

fn row(
    slug: &'static str,
    label: &'static str,
    endpoint: &'static str,
    key_placeholder: Option<&'static str>,
    quirks: &'static [Quirk],
) -> CloudRow {
    CloudRow {
        slug,
        label,
        endpoint,
        auth: RowAuth::Bearer,
        key_placeholder,
        quirks,
    }
}

const NO_QUIRKS: &[Quirk] = &[];
const MOVING_ALIASES: &[Quirk] = &[Quirk::MovingAliasModels];
const REGIONAL: &[Quirk] = &[Quirk::RegionalEndpointSplit];
const UNVERIFIED: &[Quirk] = &[Quirk::EndpointUnverified];
const UNAUTH_CATALOG: &[Quirk] = &[Quirk::CatalogUnauthenticated];

/// Ported verbatim from OpenCompany `catalogue.rs:145`, in its order. Built at
/// runtime rather than as a `const` so the table is exercised by every test
/// that reads the catalogue (and counted by the coverage gate).
fn cloud_rows() -> Vec<CloudRow> {
    vec![
        row(
            "openai",
            "OpenAI",
            "https://api.openai.com/v1",
            Some("sk-..."),
            &[Quirk::ResponsesApi, Quirk::MaxCompletionTokens],
        ),
        CloudRow {
            slug: "anthropic",
            label: "Anthropic",
            endpoint: "https://api.anthropic.com/v1",
            auth: RowAuth::Anthropic,
            key_placeholder: Some("sk-ant-..."),
            quirks: &[Quirk::NativeMessagesFirstParty],
        },
        row(
            "openrouter",
            "OpenRouter",
            "https://openrouter.ai/api/v1",
            Some("sk-or-..."),
            &[
                Quirk::AttributionHeaders,
                Quirk::ScopedCatalogPath,
                Quirk::KeyCheckEndpoint,
            ],
        ),
        row(
            "orcarouter",
            "OrcaRouter",
            "https://api.orcarouter.ai/v1",
            Some("sk-orca-..."),
            UNVERIFIED,
        ),
        row(
            "gmi",
            "GMI",
            "https://api.gmi-serving.com/v1",
            Some("eyJ...."),
            NO_QUIRKS,
        ),
        row(
            "fireworks",
            "Fireworks",
            "https://api.fireworks.ai/inference/v1",
            Some("fw-..."),
            &[Quirk::CatalogAccountScoped],
        ),
        row(
            "moonshot",
            "Kimi (Moonshot)",
            "https://api.moonshot.ai/v1",
            Some("sk-..."),
            NO_QUIRKS,
        ),
        row(
            "groq",
            "Groq",
            "https://api.groq.com/openai/v1",
            Some("gsk_..."),
            NO_QUIRKS,
        ),
        row(
            "mistral",
            "Mistral",
            "https://api.mistral.ai/v1",
            None,
            &[Quirk::RegionalEndpointSplit, Quirk::MovingAliasModels],
        ),
        row(
            "deepseek",
            "DeepSeek",
            "https://api.deepseek.com",
            Some("sk-..."),
            MOVING_ALIASES,
        ),
        row(
            "together",
            "Together AI",
            "https://api.together.ai/v1",
            None,
            NO_QUIRKS,
        ),
        row(
            "google",
            "Google Gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            None,
            NO_QUIRKS,
        ),
        row(
            "cerebras",
            "Cerebras",
            "https://api.cerebras.ai/v1",
            None,
            &[Quirk::CatalogKeylessTwin],
        ),
        row("xai", "xAI", "https://api.x.ai/v1", None, MOVING_ALIASES),
        row(
            "huggingface",
            "Hugging Face",
            "https://router.huggingface.co/v1",
            Some("hf_..."),
            UNAUTH_CATALOG,
        ),
        row(
            "nvidia",
            "NVIDIA",
            "https://integrate.api.nvidia.com/v1",
            None,
            NO_QUIRKS,
        ),
        row(
            "zai",
            "Z.AI",
            "https://api.z.ai/api/paas/v4",
            None,
            REGIONAL,
        ),
        row(
            "minimax",
            "MiniMax",
            "https://api.minimax.io/v1",
            None,
            NO_QUIRKS,
        ),
        row(
            "stepfun",
            "StepFun",
            "https://api.stepfun.ai/v1",
            None,
            REGIONAL,
        ),
        row(
            "kilocode",
            "Kilo Code",
            "https://api.kilo.ai/api/gateway",
            None,
            UNVERIFIED,
        ),
        row(
            "deepinfra",
            "DeepInfra",
            "https://api.deepinfra.com/v1/openai",
            None,
            NO_QUIRKS,
        ),
        row(
            "novita",
            "Novita",
            "https://api.novita.ai/v3/openai",
            None,
            NO_QUIRKS,
        ),
        row(
            "venice",
            "Venice",
            "https://api.venice.ai/api/v1",
            None,
            UNAUTH_CATALOG,
        ),
        row(
            "vercel-ai-gateway",
            "Vercel AI Gateway",
            "https://ai-gateway.vercel.sh/v1",
            None,
            NO_QUIRKS,
        ),
        row(
            "sumopod",
            "SumoPod",
            "https://ai.sumopod.com/v1",
            Some("sk-..."),
            UNVERIFIED,
        ),
        row(
            "modelscope",
            "ModelScope",
            "https://api-inference.modelscope.cn/v1",
            Some("ms-..."),
            &[Quirk::MeteredFreeTier],
        ),
    ]
}

const STANDARD_DEPTHS: &[TestDepth] = &[TestDepth::Catalog, TestDepth::Completion];
const OPENROUTER_DEPTHS: &[TestDepth] = &[
    TestDepth::KeyOnly,
    TestDepth::Catalog,
    TestDepth::Completion,
];

/// OpenRouter's attribution headers. The hub's default values are neutral
/// (TinyHumans); a host that wants its own name in OpenRouter's rankings
/// overrides them when it builds the request.
const OPENROUTER_HEADERS: &[(&str, &str)] = &[
    ("HTTP-Referer", "https://tinyhumans.ai"),
    ("X-Title", "TinyHumans"),
];

pub(super) fn descriptors() -> Vec<ProviderDescriptor> {
    cloud_rows()
        .into_iter()
        .map(|row| {
            let openrouter = row.slug == "openrouter";
            let anthropic = matches!(row.auth, RowAuth::Anthropic);
            ProviderDescriptor {
                kind: KindId::new(row.slug),
                label: row.label,
                group: ProviderGroup::Cloud,
                transport: Transport::Http,
                protocol: if anthropic {
                    Protocol::AnthropicMessages
                } else {
                    Protocol::OpenAiChat
                },
                auth: match row.auth {
                    RowAuth::Bearer => AuthStyle::Bearer,
                    RowAuth::Anthropic => AuthStyle::Anthropic,
                },
                catalog: CatalogShape::OpenAi,
                default_endpoint: Some(row.endpoint),
                endpoint_editable: false,
                needs_key: true,
                key_placeholder: row.key_placeholder,
                local_runtime: None,
                cli: None,
                aliases: &[],
                test_depths: if openrouter {
                    OPENROUTER_DEPTHS
                } else {
                    STANDARD_DEPTHS
                },
                free_text_models: false,
                extra_headers: if openrouter { OPENROUTER_HEADERS } else { &[] },
                quirks: row.quirks,
            }
        })
        .collect()
}
