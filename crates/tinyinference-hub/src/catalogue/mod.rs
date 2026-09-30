//! The provider catalogue: data, and lookup by kind.
//!
//! This is the one source for *which providers exist, where they live, and how
//! they authenticate*. It replaces two hosts' separate tables (OpenCompany's
//! `CLOUD_PROVIDERS`/`LOCAL_RUNTIMES`/`CLI_LOGINS` and OpenHuman's
//! `cloud_providers.rs`, plus a third copy in its frontend). Values are ported
//! verbatim from OpenCompany because its endpoints, auth styles and quirks are
//! the result of several rounds of real bugs; the three places where the two
//! hosts disagreed (deepseek, together, stepfun) take OpenCompany's value
//! (open question Q1, default applied).
//!
//! The rows are `const` tables of plain data; [`descriptors`] builds the typed
//! [`ProviderDescriptor`]s from them once.

mod azure;
mod cli;
mod cloud;
mod local;
mod managed;
mod reserved;

use std::sync::OnceLock;

use crate::descriptor::ProviderDescriptor;
use crate::endpoint::endpoint_host;
use crate::ids::KindId;
use crate::taxonomy::{AuthStyle, CatalogShape, LocalRuntime, ProviderGroup};

pub use azure::{AZURE_ENDPOINT_HOSTS, is_azure_endpoint};
pub use managed::MANAGED_PROXY_PATH;
pub use reserved::{INTERNAL_SLUGS, is_reserved_slug, reserved_slugs};

/// The `anthropic-version` header value that rides with [`AuthStyle::Anthropic`].
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// The slug of the one row that serves the OpenAI **Responses API**
/// (`/v1/responses`) and may fall back to it on a chat-completions 404. Every
/// other preset is chat-completions-only; enabling the fallback for those
/// guarantees a second 404 against a path that does not exist.
const RESPONSES_API_SLUG: &str = "openai";

/// OpenRouter's own host. Named rather than compared whole, so a trailing slash
/// or a `/api/v1/` spelling still matches.
const OPENROUTER_ENDPOINT_HOST: &str = "openrouter.ai";

/// Every built-in descriptor: managed first, then cloud, local and CLI kinds.
pub fn descriptors() -> &'static [ProviderDescriptor] {
    static ALL: OnceLock<Vec<ProviderDescriptor>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut all = vec![managed::descriptor()];
        all.extend(cloud::descriptors());
        all.extend(local::descriptors());
        all.extend(cli::descriptors());
        all
    })
}

/// The descriptor a name resolves to, by kind id or alias (case-insensitive).
///
/// `None` for anything the catalogue does not ship, including `custom`, which
/// is a group of operator-defined records rather than a catalogue row. A bare
/// `openai` resolves to the hosted OpenAI row, never to a local runtime.
pub fn descriptor(name: &str) -> Option<&'static ProviderDescriptor> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    descriptors().iter().find(|d| d.answers_to(name))
}

/// The canonical kind id a name resolves to.
pub fn resolve_kind(name: &str) -> Option<KindId> {
    descriptor(name).map(|d| d.kind.clone())
}

/// The descriptors of one group, in catalogue order.
pub fn descriptors_in(group: ProviderGroup) -> impl Iterator<Item = &'static ProviderDescriptor> {
    descriptors().iter().filter(move |d| d.group == group)
}

/// The descriptor that represents a [`LocalRuntime`].
pub fn descriptor_for_runtime(runtime: LocalRuntime) -> Option<&'static ProviderDescriptor> {
    descriptor(runtime.catalogue_slug())
}

/// Which group a kind belongs to; anything the catalogue does not ship is a
/// [`ProviderGroup::Custom`] endpoint.
pub fn group_of(kind: &str) -> ProviderGroup {
    descriptor(kind).map_or(ProviderGroup::Custom, |d| d.group)
}

/// The catalog shape a provider's `GET {base}/models` answers in.
///
/// The managed kind, or any endpoint whose path ends in
/// [`MANAGED_PROXY_PATH`], answers in the paged envelope; everything else
/// follows its descriptor and defaults to the OpenAI shape.
pub fn catalog_shape_for(kind: &str, base_url: &str) -> CatalogShape {
    let managed = descriptor(kind).is_some_and(|d| d.group == ProviderGroup::Managed);
    if managed
        || base_url
            .trim()
            .trim_end_matches('/')
            .ends_with(MANAGED_PROXY_PATH)
    {
        return CatalogShape::PagedEnvelope;
    }
    descriptor(kind).map_or(CatalogShape::OpenAi, |d| d.catalog)
}

/// How a kind expects its credential, read off its descriptor; an unknown kind
/// is a custom OpenAI-compatible endpoint and takes a bearer.
pub fn auth_style_for(kind: &str) -> AuthStyle {
    descriptor(kind).map_or(AuthStyle::Bearer, |d| d.auth.clone())
}

/// Whether an endpoint is OpenRouter's own (host `openrouter.ai` or a
/// subdomain), as opposed to any gateway that proxies it.
pub fn is_openrouter_endpoint(endpoint: &str) -> bool {
    endpoint_host(endpoint).is_some_and(|host| {
        host == OPENROUTER_ENDPOINT_HOST || host.ends_with(&format!(".{OPENROUTER_ENDPOINT_HOST}"))
    })
}

/// OpenRouter's authenticated, account-scoped listing path, when the endpoint
/// is OpenRouter's and a credential is in hand. The caller falls back to the
/// public `/models` on a 404.
pub fn scoped_catalog_path(endpoint: &str, authenticated: bool) -> Option<&'static str> {
    (authenticated && is_openrouter_endpoint(endpoint))
        .then_some("/models/user?limit=1000&output_modalities=all")
}

/// The query string appended to a `/models` request: OpenRouter's default
/// listing hides most models unless asked for all output modalities.
pub fn catalog_query(endpoint: &str) -> &'static str {
    if is_openrouter_endpoint(endpoint) {
        "?limit=1000&output_modalities=all"
    } else {
        ""
    }
}

/// Whether an endpoint is one of the catalogue's chat-completions-only hosts,
/// so a chat-completions 404 must not fall back to `/v1/responses`.
///
/// Compared on host rather than whole URL, so a hand-typed variant of a preset
/// still matches; a host shared with the Responses-API row is not
/// chat-completions-only. The managed host is supplied by the embedding
/// application and is not in this table.
pub fn endpoint_is_chat_completions_only(endpoint: &str) -> bool {
    let Some(host) = endpoint_host(endpoint) else {
        return false;
    };
    let mut matched = false;
    let presets = descriptors_in(ProviderGroup::Cloud).filter_map(|descriptor| {
        descriptor
            .default_endpoint
            .map(|preset| (descriptor, preset))
    });
    for (descriptor, preset) in presets {
        if endpoint_host(preset).as_deref() == Some(host.as_str()) {
            if descriptor.slug() == RESPONSES_API_SLUG {
                return false;
            }
            matched = true;
        }
    }
    matched
}

#[cfg(test)]
#[path = "test.rs"]
mod tests;
