//! [`ProviderDescriptor`], the static template for a provider kind.

use serde::Serialize;

use crate::ids::KindId;
use crate::taxonomy::{
    AuthStyle, CatalogShape, CliKind, LocalRuntime, Protocol, ProviderGroup, TestDepth, Transport,
};

/// A typed provider limitation, so a UI or a test can act on it instead of
/// parsing prose. Numbers in comments refer to OpenCompany's
/// `provider-contracts.md` limitation list.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Quirk {
    /// `/models` answers without a key, so a `Catalog` test cannot fail a bad
    /// key; only `Completion` can (Hugging Face, Venice; limitation 4).
    CatalogUnauthenticated,
    /// The listing is scoped to the account, so a fresh key sees no public
    /// models and the connect probe cannot pass (Fireworks; limitation 2).
    CatalogAccountScoped,
    /// A keyless twin of the listing exists (Cerebras `/public/v1/models`); do
    /// not "fix" the authed path to it.
    CatalogKeylessTwin,
    /// The only row with the Responses API (OpenAI; limitation 1).
    ResponsesApi,
    /// Attribution headers are expected (OpenRouter).
    AttributionHeaders,
    /// An authenticated listing path exists (`/models/user`) with a public
    /// fallback (OpenRouter).
    ScopedCatalogPath,
    /// `GET /key` is a natural `KeyOnly` test (OpenRouter).
    KeyCheckEndpoint,
    /// Native Messages with prompt caching, first-party endpoint only
    /// (Anthropic).
    NativeMessagesFirstParty,
    /// Moving alias ids (`-latest`, `latest`, `deepseek-flash`; limitation 5).
    MovingAliasModels,
    /// A regional split the preset cannot express (Mistral EU/US, StepFun
    /// `.com`, Z.AI Coding Plan; limitation 7).
    RegionalEndpointSplit,
    /// A metered free tier that probes spend (ModelScope).
    MeteredFreeTier,
    /// The endpoint is secondary-sourced and unverified first-party.
    EndpointUnverified,
    /// Sending a key to this local runtime causes spurious 401s (Ollama).
    KeyCausesLocalAuthErrors,
    /// The name covers several projects on different ports (OMLX;
    /// limitation 9).
    AmbiguousRuntime,
    /// The default port collides with the host's own bind (MLX, llama.cpp).
    PortCollidesWithHost,
    /// A 400 "no models loaded" means "load a model", not a bad request (LM
    /// Studio).
    NoModelLoadedError,
    /// The product-identity header is sent only to first-party hosts
    /// (managed; `providers.md:264-279`).
    ProductHeader,
    /// The credential comes from a host-built chain and can be absent, which
    /// is the typed signed-out state (managed; D3).
    CredentialChain,
}

/// The static template for a provider kind: one catalogue row.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ProviderDescriptor {
    /// The kind id (the catalogue key and the default slug).
    pub kind: KindId,
    /// Display label. Never used for routing.
    pub label: &'static str,
    /// The group.
    pub group: ProviderGroup,
    /// How the hub reaches the provider.
    pub transport: Transport,
    /// The chat wire protocol.
    pub protocol: Protocol,
    /// How the credential is presented.
    pub auth: AuthStyle,
    /// The model listing shape.
    pub catalog: CatalogShape,
    /// The preset (cloud) or suggested (local) endpoint.
    pub default_endpoint: Option<&'static str>,
    /// Whether an operator may type a different endpoint. False for cloud
    /// presets: a typed URL is ignored on edit (guard G2).
    pub endpoint_editable: bool,
    /// Whether a key is required. A floor, not a ceiling: a keyless local
    /// runtime can still be given one.
    pub needs_key: bool,
    /// What a key tends to look like, for an input placeholder. `None` where
    /// the vendor has no recognisable prefix; inventing one would teach a
    /// shape that is not real.
    pub key_placeholder: Option<&'static str>,
    /// The local runtime this row represents.
    pub local_runtime: Option<LocalRuntime>,
    /// The CLI kind this row represents.
    pub cli: Option<CliKind>,
    /// Other spellings that resolve to this kind.
    pub aliases: &'static [&'static str],
    /// The test depths the kind supports; anything else is `Unsupported`.
    pub test_depths: &'static [TestDepth],
    /// Model ids are typed by hand because there is no listing (Azure
    /// deployments).
    pub free_text_models: bool,
    /// Extra request headers (OpenRouter attribution).
    pub extra_headers: &'static [(&'static str, &'static str)],
    /// Typed limitations.
    pub quirks: &'static [Quirk],
}

impl ProviderDescriptor {
    /// The kind id as a string slice.
    pub fn slug(&self) -> &str {
        self.kind.as_str()
    }

    /// Whether the kind supports a test depth.
    pub fn supports_depth(&self, depth: TestDepth) -> bool {
        self.test_depths.contains(&depth)
    }

    /// Whether the descriptor carries a quirk.
    pub fn has_quirk(&self, quirk: Quirk) -> bool {
        self.quirks.contains(&quirk)
    }

    /// Whether `name` (already trimmed) is this kind's id or one of its
    /// aliases, compared case-insensitively.
    pub fn answers_to(&self, name: &str) -> bool {
        self.kind.as_str().eq_ignore_ascii_case(name)
            || self.aliases.iter().any(|a| a.eq_ignore_ascii_case(name))
    }
}
