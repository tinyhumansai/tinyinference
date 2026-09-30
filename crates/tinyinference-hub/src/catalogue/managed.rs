//! The managed (TinyHumans) kind.
//!
//! An ordinary descriptor (D1): the group is the only thing that marks it
//! special, and no host-facing API branches on "is this managed". The hub does
//! **not** hardcode either backend's URL (open question Q3): OpenCompany reaches
//! `{api_url}/agent-integrations/openrouter` with a paged catalog, OpenHuman
//! reaches `{api_url}/openai/v1` with a session token, so the endpoint, catalog
//! shape and query are supplied by the host when it configures the kind.

use crate::descriptor::{ProviderDescriptor, Quirk};
use crate::ids::KindId;
use crate::taxonomy::{AuthStyle, CatalogShape, Protocol, ProviderGroup, TestDepth, Transport};

/// The path OpenCompany's TinyHumans OpenRouter proxy is served under. An
/// endpoint ending in it answers its catalog in the paged envelope.
pub const MANAGED_PROXY_PATH: &str = "/agent-integrations/openrouter";

pub(super) fn descriptor() -> ProviderDescriptor {
    ProviderDescriptor {
        kind: KindId::new("tinyhumans"),
        label: "TinyHumans",
        group: ProviderGroup::Managed,
        transport: Transport::Http,
        protocol: Protocol::OpenAiChat,
        // The chain may yield a pasted key, an account key, an instance
        // identity or a session token; the descriptor's own style is the
        // common bearer form.
        auth: AuthStyle::Bearer,
        catalog: CatalogShape::PagedEnvelope,
        default_endpoint: None,
        endpoint_editable: false,
        // The credential comes from a chain and can be absent, which is the
        // typed signed-out state rather than a missing-key error.
        needs_key: false,
        key_placeholder: Some("th-..."),
        local_runtime: None,
        cli: None,
        aliases: &["openhuman", "managed", "cloud"],
        test_depths: &[TestDepth::Catalog, TestDepth::Completion],
        free_text_models: false,
        extra_headers: &[],
        quirks: &[Quirk::ProductHeader, Quirk::CredentialChain],
    }
}
