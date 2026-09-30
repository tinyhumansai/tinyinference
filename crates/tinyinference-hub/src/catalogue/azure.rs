//! Azure OpenAI endpoint recognition.

use crate::endpoint::endpoint_host;

/// The hosts that serve Azure's OpenAI-compatible surface, with their
/// sovereign-cloud counterparts.
///
/// The first three are what Microsoft documents (`https://<resource>.<host>/
/// openai/v1/`); the sovereign clouds are separate DNS parents, so the `.com`
/// entries do not cover them. Serverless `inference.ai.azure.com` is
/// deliberately excluded (it has a model listing).
pub const AZURE_ENDPOINT_HOSTS: &[&str] = &[
    "openai.azure.com",
    "services.ai.azure.com",
    "cognitiveservices.azure.com",
    "openai.azure.us",
    "openai.azure.cn",
];

/// Whether an endpoint is an Azure OpenAI resource, whose models are
/// operator-named *deployments* that are never in `/models`, so the model id
/// is free text.
pub fn is_azure_endpoint(endpoint: &str) -> bool {
    let Some(host) = endpoint_host(endpoint) else {
        return false;
    };
    AZURE_ENDPOINT_HOSTS
        .iter()
        .any(|known| host == *known || host.ends_with(&format!(".{known}")))
}
