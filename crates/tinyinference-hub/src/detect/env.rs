//! Environment detection: which provider keys are set.

use crate::config::ProviderDraft;
use crate::ports::EnvSource;

/// The well-known environment variables and the catalogue kind each one is the
/// key for, in the order detection reports them. The first variable listed for
/// a kind is the primary one; the credential chain reads every variable listed
/// for the kind (first non-blank wins) when the builder enables environment
/// credentials.
pub const ENV_KEYS: &[(&str, &str)] = &[
    ("OPENAI_API_KEY", "openai"),
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("OPENROUTER_API_KEY", "openrouter"),
    ("GROQ_API_KEY", "groq"),
    ("MISTRAL_API_KEY", "mistral"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("TOGETHER_API_KEY", "together"),
    ("XAI_API_KEY", "xai"),
    ("GEMINI_API_KEY", "google"),
    ("GOOGLE_API_KEY", "google"),
    ("CEREBRAS_API_KEY", "cerebras"),
    ("FIREWORKS_API_KEY", "fireworks"),
    ("HF_TOKEN", "huggingface"),
    ("HUGGINGFACE_API_KEY", "huggingface"),
    ("NVIDIA_API_KEY", "nvidia"),
    ("DEEPINFRA_API_KEY", "deepinfra"),
    ("MOONSHOT_API_KEY", "moonshot"),
    ("NOVITA_API_KEY", "novita"),
    ("VENICE_API_KEY", "venice"),
];

/// The environment variable a kind's key is read from, when the catalogue has
/// one for it.
pub fn env_var_for_kind(kind: &str) -> Option<&'static str> {
    ENV_KEYS
        .iter()
        .find(|(_, k)| *k == kind)
        .map(|(var, _)| *var)
}

/// Every environment variable that is a kind's key, in table order. The
/// credential chain reads all of them, so what detection reports is what the
/// chain will find.
pub fn env_vars_for_kind(kind: &str) -> impl Iterator<Item = &'static str> + '_ {
    ENV_KEYS
        .iter()
        .filter(move |(_, k)| *k == kind)
        .map(|(var, _)| *var)
}

/// Drafts for every provider whose environment variable is set and non-blank.
/// A kind with two variables set is reported once. The draft carries **no key**.
pub fn detect_env(env: &dyn EnvSource) -> Vec<ProviderDraft> {
    let mut seen: Vec<&str> = Vec::new();
    let mut drafts = Vec::new();
    for (var, kind) in ENV_KEYS {
        if seen.contains(kind) {
            continue;
        }
        if env.var(var).is_some_and(|value| !value.trim().is_empty()) {
            seen.push(kind);
            drafts.push(ProviderDraft::new(kind));
        }
    }
    drafts
}
