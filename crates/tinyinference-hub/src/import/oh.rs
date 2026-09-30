//! The OpenHuman reader: `config.toml` cloud providers, primary cloud, role
//! routes and local AI to a [`HubConfig`](crate::config::HubConfig).
//!
//! OpenHuman's shapes are lossy in twelve documented ways (06-migration-mapping
//! section 3). The reader takes the reading OpenHuman's backend actually ran
//! (what the UI showed is not evidence) and reports every judgement call.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::catalog::ModelOverride;
use crate::catalogue;
use crate::config::DefaultChoice;
use crate::descriptor::{ProviderRecord, RecordOrigin, Tri};
use crate::error::{HubError, InputField, InvalidInput};
use crate::ids::{KindId, ModelId, Slug, WorkloadKey};
use crate::route::RouteTarget;
use crate::route::legacy_oh::{self, OhContext};
use crate::taxonomy::{AuthStyle, LocalRuntime, ProviderGroup};

use super::{Imported, ImportedCredential, LossKind};
use crate::secret::Secret;

/// A key found in a stored shape. It redacts itself in `Debug` (an input struct
/// is easy to log by accident) and deserialises from the plain string the
/// source stored.
#[derive(Clone)]
pub struct StoredKey(Secret);

impl StoredKey {
    /// Wraps a key.
    pub fn new(key: impl Into<String>) -> Self {
        Self(Secret::new(key))
    }

    /// The key, trimmed, or `None` when it is blank.
    fn usable(&self) -> Option<Secret> {
        let key = self.0.expose().trim();
        (!key.is_empty()).then(|| Secret::new(key))
    }
}

impl From<String> for StoredKey {
    fn from(key: String) -> Self {
        Self::new(key)
    }
}

impl<'de> Deserialize<'de> for StoredKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        String::deserialize(deserializer).map(Self::new)
    }
}

impl std::fmt::Debug for StoredKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("StoredKey").field(&self.0).finish()
    }
}

/// One entry of `cloud_providers`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OhCloudRow {
    /// `p_<slug>_<rand>`, kept verbatim.
    pub id: String,
    /// The slug.
    pub slug: String,
    /// The display name.
    pub label: String,
    /// The endpoint.
    pub endpoint: String,
    /// `bearer`, `anthropic`, `none`, `openhuman_jwt` (or `openhumanjwt`).
    pub auth_style: String,
    /// A legacy per-row model, read once.
    pub default_model: Option<String>,
    /// A legacy row type, read once and dropped.
    #[serde(rename = "type")]
    pub kind_type: Option<String>,
}

/// `local_ai`: the one local runtime OpenHuman configures.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OhLocalAi {
    /// The runtime name, in any spelling OpenHuman accepts.
    pub provider: String,
    /// The one base URL OpenHuman shares between local runtimes.
    pub base_url: Option<String>,
    /// A key, when the runtime takes one.
    pub api_key: Option<StoredKey>,
    /// The model.
    pub model_id: Option<String>,
}

/// The persisted `byok-inference` synthetic route.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OhByok {
    /// `inference_url`.
    pub url: String,
    /// The key.
    pub api_key: Option<StoredKey>,
    /// The model.
    pub model: Option<String>,
}

/// One entry of `model_registry`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OhRegistryRow {
    /// The model id.
    pub id: String,
    /// The provider it was registered for.
    pub provider: Option<String>,
    /// Price per million input tokens.
    #[serde(alias = "cost_per_1m_input")]
    pub cost_per_1m_in: Option<f64>,
    /// Price per million output tokens.
    #[serde(alias = "cost_per_1m_output")]
    pub cost_per_1m_out: Option<f64>,
    /// Context window in tokens.
    pub context_window: Option<u64>,
    /// Whether the model takes images.
    pub vision: Option<bool>,
}

/// Everything the reader needs from one OpenHuman config.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OhSnapshot {
    /// `cloud_providers`.
    pub cloud_providers: Vec<OhCloudRow>,
    /// `primary_cloud`: a row **id**, while every route names a slug (case 10).
    pub primary_cloud: Option<String>,
    /// The top-level default model.
    pub default_model: Option<String>,
    /// Role routes: `chat`, `reasoning`, `agentic`, `coding`, `vision`,
    /// `memory`, `embeddings`, `heartbeat`, `learning`, `subconscious`, with or
    /// without a `_provider` suffix, plus the aliases `burst` and
    /// `summarization`.
    pub routes: BTreeMap<String, String>,
    /// `local_ai`.
    pub local_ai: Option<OhLocalAi>,
    /// The `byok-inference` route, when persisted.
    pub byok: Option<OhByok>,
    /// `model_registry`.
    pub model_registry: Vec<OhRegistryRow>,
    /// `temperature_unsupported_models`.
    pub temperature_unsupported_models: Vec<String>,
    /// Whether an OpenAI OAuth profile exists (the Codex login).
    pub openai_oauth: bool,
}

const OLD_OH_PRESETS: &[(&str, &str, &str)] = &[
    (
        "deepseek",
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com",
    ),
    (
        "together",
        "https://api.together.xyz/v1",
        "https://api.together.ai/v1",
    ),
    (
        "stepfun",
        "https://api.stepfun.ai/step_plan/v1",
        "https://api.stepfun.ai/v1",
    ),
];

fn parse_auth(raw: &str) -> Option<AuthStyle> {
    match raw
        .trim()
        .to_ascii_lowercase()
        .replace(['_', '-'], "")
        .as_str()
    {
        "" | "bearer" => Some(AuthStyle::Bearer),
        "anthropic" => Some(AuthStyle::Anthropic),
        "none" => Some(AuthStyle::None),
        "openhumanjwt" => Some(AuthStyle::SessionJwt),
        "xapikey" => Some(AuthStyle::XApiKey),
        _ => None,
    }
}

fn role_name(key: &str) -> (String, bool) {
    let stem = key.trim().strip_suffix("_provider").unwrap_or(key.trim());
    match stem {
        "burst" => ("agentic".to_string(), true),
        "summarization" => ("memory".to_string(), true),
        other => (other.to_string(), false),
    }
}

/// Reads an OpenHuman snapshot.
///
/// # Errors
///
/// [`HubError::Invalid`] for a slug or model id that is not valid.
pub fn import(snapshot: &OhSnapshot) -> Result<Imported, HubError> {
    let mut out = Imported::new();
    let mut row_slug_by_id: BTreeMap<String, Option<Slug>> = BTreeMap::new();
    let mut context = OhContext::new();

    for row in &snapshot.cloud_providers {
        let key = format!("cloud_providers/{}", row.slug);
        let slug_text = row.slug.trim().to_ascii_lowercase();
        if slug_text == "claude-code" || row.endpoint.trim().starts_with("cli://") {
            out.loss.push(
                &key,
                LossKind::Dropped,
                "the Claude Code row is a UI marker with a fake endpoint; a CLI login is a route target",
            );
            row_slug_by_id.insert(row.id.clone(), None);
            continue;
        }
        let descriptor = catalogue::descriptor(&slug_text);
        if descriptor.is_some_and(|d| d.group == ProviderGroup::Managed) {
            out.loss.push(
                &key,
                LossKind::Normalised,
                "the managed row is always listed by the hub and is not stored as a record",
            );
            row_slug_by_id.insert(row.id.clone(), Slug::parse("tinyhumans").ok());
            continue;
        }
        let slug = Slug::parse(&slug_text)?;
        if out.config.contains(&slug) {
            out.loss.push(
                &key,
                LossKind::Dropped,
                "a second row with the same slug; the first one wins",
            );
            continue;
        }
        let kind = descriptor.map_or_else(|| KindId::new("custom"), |d| d.kind.clone());
        let is_local = descriptor.is_some_and(|d| d.group == ProviderGroup::Local);
        if is_local {
            out.loss.push(
                &key,
                LossKind::Normalised,
                "a synthetic local row became an ordinary local record",
            );
        }
        let mut base_url = row.endpoint.trim().to_string();
        if base_url.is_empty() {
            base_url = descriptor
                .and_then(|d| d.default_endpoint)
                .unwrap_or("")
                .to_string();
        }
        if let Some((_, old, hub)) = OLD_OH_PRESETS
            .iter()
            .find(|(name, old, _)| *name == slug_text && base_url.trim_end_matches('/') == *old)
        {
            out.loss.push(
                &key,
                LossKind::Normalised,
                format!("the stored endpoint is OpenHuman's old preset; the hub's preset is {hub}. The stored value was kept"),
            );
            let _ = old;
        }
        let mut record = ProviderRecord::new(
            row.id.clone(),
            slug.clone(),
            if row.label.trim().is_empty() {
                row.slug.clone()
            } else {
                row.label.clone()
            },
            kind,
            base_url,
        );
        record.origin = RecordOrigin::Imported;
        match parse_auth(&row.auth_style) {
            Some(style) => {
                if descriptor.is_none() && style != AuthStyle::Bearer {
                    record.auth_override = Some(style);
                }
            }
            None => out.loss.push(
                &key,
                LossKind::Ambiguous,
                "the auth style is not one the hub knows; the kind's own style is used",
            ),
        }
        if let Some(model) = row
            .default_model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
        {
            let model = ModelId::parse(model)?;
            context = context.with_default_model(slug.clone(), model.clone());
            record.model = Some(model);
            out.loss.push(
                &key,
                LossKind::Normalised,
                "the legacy per-row default_model moved onto the record's model",
            );
        }
        if row.kind_type.is_some() {
            out.loss.push(
                &key,
                LossKind::Dropped,
                "the legacy row `type` has no home in the hub",
            );
        }
        record.validate()?;
        row_slug_by_id.insert(row.id.clone(), Some(slug));
        out.config.providers.push(record);
    }

    if let Some(byok) = &snapshot.byok
        && !byok.url.trim().is_empty()
    {
        let slug = Slug::parse("byok-inference")?;
        let mut record = ProviderRecord::new(
            "byok-inference",
            slug.clone(),
            "BYOK inference",
            KindId::new("custom"),
            byok.url.trim(),
        );
        record.synthetic = true;
        record.origin = RecordOrigin::Imported;
        if let Some(model) = byok
            .model
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty())
        {
            record.model = Some(ModelId::parse(model)?);
        }
        record.validate()?;
        if let Some(key) = byok.api_key.as_ref().and_then(StoredKey::usable) {
            out.credentials.push(ImportedCredential { slug, key });
        }
        out.loss.push(
            "byok-inference",
            LossKind::Synthesised,
            "the persisted BYOK route became a synthetic custom record",
        );
        out.config.providers.push(record);
    }

    let mut local_runtime = None;
    if let Some(local) = &snapshot.local_ai
        && !local.provider.trim().is_empty()
    {
        let runtime = LocalRuntime::parse_loose(&local.provider).ok_or(HubError::Invalid(
            InvalidInput::Malformed {
                field: InputField::Kind,
                reason: "local_ai.provider is not a local runtime the hub knows",
            },
        ))?;
        local_runtime = Some(runtime);
        let descriptor = catalogue::descriptor_for_runtime(runtime).ok_or(HubError::Invalid(
            InvalidInput::Malformed {
                field: InputField::Kind,
                reason: "the local runtime has no catalogue row",
            },
        ))?;
        let slug = Slug::parse(descriptor.slug())?;
        let base_url = local
            .base_url
            .as_deref()
            .map(str::trim)
            .filter(|u| !u.is_empty())
            .or(descriptor.default_endpoint)
            .unwrap_or("")
            .to_string();
        let model = local
            .model_id
            .as_deref()
            .map(str::trim)
            .filter(|m| !m.is_empty());
        if let Some(existing) = out.config.provider_mut(&slug) {
            existing.base_url = base_url;
            if let Some(model) = model {
                existing.model = Some(ModelId::parse(model)?);
            }
            existing.validate()?;
        } else {
            let mut record = ProviderRecord::new(
                format!("local_{}", descriptor.slug()),
                slug.clone(),
                descriptor.label,
                descriptor.kind.clone(),
                base_url,
            );
            record.origin = RecordOrigin::Imported;
            if let Some(model) = model {
                record.model = Some(ModelId::parse(model)?);
            }
            record.validate()?;
            out.config.providers.push(record);
        }
        if let Some(key) = local.api_key.as_ref().and_then(StoredKey::usable) {
            out.credentials.push(ImportedCredential { slug, key });
        }
        out.loss.push(
            "local_ai",
            LossKind::Normalised,
            "OpenHuman shares one base URL between local runtimes; only the configured runtime got a record, the others use their descriptor defaults when added",
        );
        context = context.with_local_ai_runtime(runtime);
    }

    match snapshot.primary_cloud.as_deref().map(str::trim).filter(|p| !p.is_empty()) {
        None => out.loss.push(
            "primary_cloud",
            LossKind::Ambiguous,
            "an empty primary_cloud meant the managed provider in OpenHuman; the hub needs a model to resolve it, so no default was set",
        ),
        Some(id) => match row_slug_by_id.get(id) {
            Some(Some(slug)) => {
                let model = out
                    .config
                    .provider(slug)
                    .and_then(|p| p.model.clone())
                    .or_else(|| {
                        snapshot
                            .default_model
                            .as_deref()
                            .map(str::trim)
                            .filter(|m| !m.is_empty())
                            .and_then(|m| ModelId::parse(m).ok())
                    });
                out.config.default = match model {
                    Some(model) => DefaultChoice::Full { provider: slug.clone(), model },
                    None => {
                        out.loss.push(
                            "primary_cloud",
                            LossKind::Ambiguous,
                            "the primary row has no model, so the default names a provider only and fails closed on a turn",
                        );
                        DefaultChoice::ProviderOnly { provider: slug.clone() }
                    }
                };
            }
            Some(None) | None => out.loss.push(
                "primary_cloud",
                LossKind::FailClosed,
                "primary_cloud names a row id that does not exist or is not a provider; no default was set",
            ),
        },
    }

    // Explicit role names first, aliases after: when both are present the
    // explicit one wins and the alias is reported dropped.
    let mut ordered: Vec<(&String, &String)> = snapshot.routes.iter().collect();
    ordered.sort_by_key(|(raw_role, _)| role_name(raw_role).1);
    for (raw_role, text) in ordered {
        let (role, aliased) = role_name(raw_role);
        let source = format!("routes/{role}");
        if aliased {
            if out
                .config
                .workload_routes
                .contains_key(&WorkloadKey::new(role.clone()))
            {
                out.loss.push(
                    &source,
                    LossKind::Dropped,
                    "a role alias was ignored because the role itself is set",
                );
                continue;
            }
            out.loss.push(
                &source,
                LossKind::Normalised,
                "a role alias was mapped to its role",
            );
        }
        let parsed = legacy_oh::parse(text, &context);
        for mut entry in parsed.loss {
            entry.source_key = source.clone();
            out.loss.entries.push(entry);
        }
        let Some(route) = parsed.route else {
            continue;
        };
        if route.target == RouteTarget::Default {
            out.loss.push(
                &source,
                LossKind::Normalised,
                "a route to the primary cloud is the absence of a route",
            );
            continue;
        }
        if route.target == RouteTarget::Ephemeral {
            out.loss.push(
                &source,
                LossKind::Dropped,
                "an ephemeral route is never persisted",
            );
            continue;
        }
        out.config
            .workload_routes
            .insert(WorkloadKey::new(role), route);
    }

    for row in &snapshot.model_registry {
        let Ok(model) = ModelId::parse(row.id.trim()) else {
            out.loss.push(
                "model_registry",
                LossKind::Dropped,
                "a registry row with an invalid model id was skipped",
            );
            continue;
        };
        let mut over = ModelOverride::new(model);
        over.input_per_1m = row.cost_per_1m_in;
        over.output_per_1m = row.cost_per_1m_out;
        over.context_window = row.context_window;
        over.vision = row.vision.map(|v| if v { Tri::Yes } else { Tri::No });
        if row.provider.is_some() {
            out.loss.push(
                "model_registry",
                LossKind::Normalised,
                "registry rows were scoped to a provider; overrides apply to the model id everywhere",
            );
        }
        out.overrides.push(over);
    }
    for id in &snapshot.temperature_unsupported_models {
        match ModelId::parse(id.trim()) {
            Ok(model) => {
                let mut over = ModelOverride::new(model);
                over.temperature = Some(Tri::No);
                out.overrides.push(over);
            }
            Err(_) => out.loss.push(
                "temperature_unsupported_models",
                LossKind::Dropped,
                "an entry that is not a valid model id was skipped",
            ),
        }
    }

    if snapshot.openai_oauth {
        out.loss.push(
            "auth-profiles",
            LossKind::Normalised,
            "the Codex login is an OpenAI OAuth profile; there is no codex record, and the origin is not carried over",
        );
    }
    let _ = local_runtime;
    out.config.validate()?;
    Ok(out)
}
