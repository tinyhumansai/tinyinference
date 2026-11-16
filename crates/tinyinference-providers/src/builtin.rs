//! Known cloud endpoints and their protocol capabilities.
//!
//! Host-specific managed providers are deliberately absent. Hosts register
//! their own descriptors alongside this provider-neutral catalog.

/// Authentication protocol for a builtin cloud provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinAuthStyle {
    /// OpenAI-compatible Authorization bearer token.
    Bearer,
    /// Anthropic x-api-key and version headers.
    Anthropic,
    /// OCI Generative AI: a GenAI API key, OCI request signing, or a
    /// principal, served by the `tinyinference-llm` OCI provider rather than
    /// a plain OpenAI-compatible client.
    Oci,
}

/// A known cloud-provider preset, independent of any saved host configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinCloudProvider {
    /// Stable provider identifier.
    pub slug: &'static str,
    /// Display name of the provider.
    pub label: &'static str,
    /// Canonical API base URL.
    pub endpoint: &'static str,
    /// Provider authentication protocol.
    pub auth_style: BuiltinAuthStyle,
}

/// Known third-party cloud-provider presets. Managed host routes are extensions.
pub const BUILTIN_CLOUD_PROVIDERS: &[BuiltinCloudProvider] = &[
    BuiltinCloudProvider {
        slug: "openai",
        label: "OpenAI",
        endpoint: "https://api.openai.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "anthropic",
        label: "Anthropic",
        endpoint: "https://api.anthropic.com/v1",
        auth_style: BuiltinAuthStyle::Anthropic,
    },
    BuiltinCloudProvider {
        slug: "openrouter",
        label: "OpenRouter",
        endpoint: "https://openrouter.ai/api/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "orcarouter",
        label: "OrcaRouter",
        endpoint: "https://api.orcarouter.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "gmi",
        label: "GMI",
        endpoint: "https://api.gmi-serving.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "fireworks",
        label: "Fireworks",
        endpoint: "https://api.fireworks.ai/inference/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "moonshot",
        label: "Kimi (Moonshot)",
        endpoint: "https://api.moonshot.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "groq",
        label: "Groq",
        endpoint: "https://api.groq.com/openai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "mistral",
        label: "Mistral",
        endpoint: "https://api.mistral.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "deepseek",
        label: "DeepSeek",
        endpoint: "https://api.deepseek.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "together",
        label: "Together AI",
        endpoint: "https://api.together.xyz/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "google",
        label: "Google Gemini",
        endpoint: "https://generativelanguage.googleapis.com/v1beta/openai",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "cerebras",
        label: "Cerebras",
        endpoint: "https://api.cerebras.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "xai",
        label: "xAI",
        endpoint: "https://api.x.ai/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "huggingface",
        label: "Hugging Face",
        endpoint: "https://router.huggingface.co/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "nvidia",
        label: "NVIDIA",
        endpoint: "https://integrate.api.nvidia.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "zai",
        label: "Z.AI",
        endpoint: "https://api.z.ai/api/paas/v4",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "minimax",
        label: "MiniMax",
        // MiniMax exposes a full OpenAI-compatible surface at `/v1`
        // (`/v1/chat/completions`, `/v1/models`). The previous `/anthropic`
        // base + Anthropic auth pointed at MiniMax's Messages-protocol API,
        // which OpenHuman does not speak — it only builds OpenAI-style
        // `/chat/completions` and `/models` — so both chat and model-listing
        // 404'd (`/anthropic/chat/completions`, `/anthropic/models`). The
        // 404 on model-listing was Sentry TAURI-RUST-8X3. Use the `/v1`
        // OpenAI surface with Bearer auth so both paths resolve.
        endpoint: "https://api.minimax.io/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "stepfun",
        label: "StepFun",
        endpoint: "https://api.stepfun.ai/step_plan/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "kilocode",
        label: "Kilo Code",
        endpoint: "https://api.kilo.ai/api/gateway",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "deepinfra",
        label: "DeepInfra",
        endpoint: "https://api.deepinfra.com/v1/openai",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "novita",
        label: "Novita",
        endpoint: "https://api.novita.ai/v3/openai",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "venice",
        label: "Venice",
        endpoint: "https://api.venice.ai/api/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "vercel-ai-gateway",
        label: "Vercel AI Gateway",
        endpoint: "https://ai-gateway.vercel.sh/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "sumopod",
        label: "SumoPod",
        endpoint: "https://ai.sumopod.com/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
    BuiltinCloudProvider {
        slug: "oci",
        label: "OCI Generative AI",
        // Display default only: hosts route OCI entries by region or an
        // explicit endpoint through the OCI provider, not by this URL.
        endpoint: "https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
        auth_style: BuiltinAuthStyle::Oci,
    },
    BuiltinCloudProvider {
        slug: "modelscope",
        label: "ModelScope",
        endpoint: "https://api-inference.modelscope.cn/v1",
        auth_style: BuiltinAuthStyle::Bearer,
    },
];

/// A generic cloud preset catalog extended with host-owned presets.
///
/// Hosts can add their managed endpoint without making the provider-neutral
/// catalog depend on host-specific names or policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltinPresets {
    providers: Vec<BuiltinCloudProvider>,
}

impl BuiltinPresets {
    /// Creates a catalog from the generic presets and host-supplied additions.
    ///
    /// The generic presets are searched first when a host supplies a duplicate
    /// slug, so host additions cannot silently change a generic provider's
    /// canonical endpoint or protocol.
    #[must_use]
    pub fn new(extra_presets: impl IntoIterator<Item = BuiltinCloudProvider>) -> Self {
        let providers = BUILTIN_CLOUD_PROVIDERS
            .iter()
            .copied()
            .chain(extra_presets)
            .collect();
        Self { providers }
    }

    /// Finds a preset by its exact provider slug.
    #[must_use]
    pub fn builtin_cloud_provider(&self, slug: &str) -> Option<&BuiltinCloudProvider> {
        self.providers.iter().find(|provider| provider.slug == slug)
    }

    /// Returns whether `slug` matches any preset in this catalog.
    #[must_use]
    pub fn is_builtin_cloud_slug(&self, slug: &str) -> bool {
        self.builtin_cloud_provider(slug).is_some()
    }

    /// Returns whether a preset is known to support the OpenAI Responses API.
    ///
    /// Only the generic OpenAI, OpenRouter, and xAI presets are known to expose
    /// it. Host-supplied and unknown providers retain caller-selected behavior.
    #[must_use]
    pub fn builtin_cloud_supports_responses_api(&self, slug: &str) -> bool {
        self.builtin_cloud_provider(slug).is_some()
            && matches!(slug, "openai" | "openrouter" | "xai")
    }

    /// Extracts the normalized authority host from an endpoint URL.
    #[must_use]
    pub fn endpoint_host(&self, endpoint: &str) -> Option<String> {
        endpoint_host(endpoint)
    }

    /// Returns whether `host` belongs to a preset in this catalog.
    #[must_use]
    pub fn host_is_builtin_cloud_provider(&self, host: &str) -> bool {
        let host = host.trim().to_ascii_lowercase();
        !host.is_empty()
            && self.providers.iter().any(|provider| {
                self.endpoint_host(provider.endpoint).as_deref() == Some(host.as_str())
            })
    }

    /// Returns whether an endpoint host in this catalog is chat-completions-only.
    #[must_use]
    pub fn endpoint_host_is_chat_completions_only(&self, endpoint: &str) -> bool {
        let Some(host) = self.endpoint_host(endpoint) else {
            return false;
        };
        let mut found = false;
        for provider in &self.providers {
            if self.endpoint_host(provider.endpoint).as_deref() == Some(host.as_str()) {
                if self.builtin_cloud_supports_responses_api(provider.slug) {
                    return false;
                }
                found = true;
            }
        }
        found
    }
}

impl Default for BuiltinPresets {
    fn default() -> Self {
        Self::new([])
    }
}

/// Finds a generic builtin preset by its exact provider identifier.
#[must_use]
pub fn builtin_cloud_provider(type_str: &str) -> Option<&'static BuiltinCloudProvider> {
    BUILTIN_CLOUD_PROVIDERS
        .iter()
        .find(|provider| provider.slug == type_str)
}

/// Whether `slug` matches a generic built-in cloud provider preset.
#[must_use]
pub fn is_builtin_cloud_slug(slug: &str) -> bool {
    builtin_cloud_provider(slug).is_some()
}

/// Whether a generic builtin preset supports the OpenAI Responses API.
#[must_use]
pub fn builtin_cloud_supports_responses_api(slug: &str) -> bool {
    matches!(slug, "openai" | "openrouter" | "xai")
}

/// Extract the lowercased authority host from an endpoint URL, dropping the
/// scheme, any userinfo, the port, and the path. Returns `None` when no host
/// can be parsed. Tolerates a missing scheme and bracketed IPv6 literals.
/// Bare IPv6 authorities are unsupported; use `[::1]` or `[::1]:8080`.
pub fn endpoint_host(endpoint: &str) -> Option<String> {
    let s = endpoint.trim();
    // Drop the scheme (`https://…`); tolerate a bare `host/path` form.
    let after_scheme = s.split_once("://").map(|(_, rest)| rest).unwrap_or(s);
    // The authority ends at the first path / query / fragment delimiter.
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    // Strip any `user:pass@` userinfo prefix.
    let host_port = authority
        .rsplit_once('@')
        .map(|(_, host)| host)
        .unwrap_or(authority);
    // Strip the port, handling bracketed IPv6 literals (`[::1]:8080`).
    let host = if let Some(rest) = host_port.strip_prefix('[') {
        rest.split_once(']').map(|(h, _)| h).unwrap_or(rest)
    } else {
        match host_port.split_once(':') {
            // More than one colon without brackets is not a valid host:port.
            Some((_, rest)) if rest.contains(':') => return None,
            Some((h, _)) => h,
            None => host_port,
        }
    };
    let host = host.trim().to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

/// Whether an authority host belongs to a known inference provider.
///
/// Comparisons are case-insensitive and derived from the builtin catalog.
pub fn host_is_builtin_cloud_provider(host: &str) -> bool {
    let host = host.trim().to_ascii_lowercase();
    !host.is_empty()
        && BUILTIN_CLOUD_PROVIDERS
            .iter()
            .any(|provider| endpoint_host(provider.endpoint).as_deref() == Some(host.as_str()))
}

/// Whether an endpoint **host** is a known cloud host that does NOT serve the
/// OpenAI Responses API (`/v1/responses`) — i.e. it is chat-completions-only,
/// regardless of which user slug points at it.
///
/// Derived entirely from [`BUILTIN_CLOUD_PROVIDERS`]: a host is chat-only when
/// some built-in preset uses it AND no preset at that host advertises the
/// Responses API (only OpenAI's `api.openai.com` does). This closes the
/// custom-slug gap behind the builtin-slug gate
/// ([`builtin_cloud_supports_responses_api`]): a user slug pointed at, e.g.,
/// `integrate.api.nvidia.com` must never attempt `/responses` (TAURI-RUST-5A1),
/// while a genuinely unknown proxy host keeps the permissive fallback so a real
/// OpenAI proxy still gets `/responses`.
pub fn endpoint_host_is_chat_completions_only(endpoint: &str) -> bool {
    let Some(host) = endpoint_host(endpoint) else {
        return false;
    };
    let mut matched_chat_only = false;
    for provider in BUILTIN_CLOUD_PROVIDERS {
        if endpoint_host(provider.endpoint).as_deref() == Some(host.as_str()) {
            if builtin_cloud_supports_responses_api(provider.slug) {
                return false;
            }
            matched_chat_only = true;
        }
    }
    matched_chat_only
}

#[cfg(test)]
#[path = "builtin_tests.rs"]
mod tests;
