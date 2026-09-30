//! The OpenCompany reader: `inference/*` records to a [`HubConfig`].
//!
//! Input mirrors what OpenCompany stores per company (see 06-migration-mapping
//! section 1). Nothing is rewritten in OpenCompany: the reader is pure, and an
//! adapter that writes back must still emit OpenCompany's own shapes (the
//! uniform per-tier model map, the JSON default) so rollback binaries keep
//! working.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::catalogue;
use crate::config::{DefaultChoice, HubConfig};
use crate::descriptor::{ProviderRecord, RecordOrigin};
use crate::error::{HubError, InputField, InvalidInput, NotFound, ReasonCode};
use crate::health::ProviderHealth;
use crate::ids::{KindId, ModelId, Slug, WorkloadKey};
use crate::route::{RouteTarget, legacy_oc};
use crate::taxonomy::ProviderGroup;

use super::{Imported, LossKind, LossReport};

/// One entry of OpenCompany's `inference/providers`.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OcStoredProvider {
    /// `prv_<32hex>`, kept verbatim.
    pub id: String,
    /// The routing key.
    pub slug: String,
    /// The display name.
    pub label: String,
    /// The catalogue kind or `custom`.
    pub kind: String,
    /// The endpoint.
    pub base_url: String,
    /// The per-tier model map (tier to model id).
    pub models: BTreeMap<String, String>,
    /// Whether the row is enabled (absent means enabled).
    #[serde(default = "enabled_default")]
    pub enabled: bool,
}

fn enabled_default() -> bool {
    true
}

/// OpenCompany's `inference/config`, the legacy "entry zero".
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OcEntryZero {
    /// The provider name (`managed`, `openrouter`, `openai_compatible`,
    /// `ollama`, or a catalogue slug).
    pub provider: String,
    /// The endpoint, when the row carries one.
    pub base_url: Option<String>,
    /// The per-tier model map.
    pub models: BTreeMap<String, String>,
}

/// Everything the reader needs from one OpenCompany company.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct OcSnapshot {
    /// `inference/providers`.
    pub providers: Vec<OcStoredProvider>,
    /// `inference/default`: a bare slug (before the model was stored) or JSON
    /// `{"provider","model"}`.
    pub default: Option<String>,
    /// `inference/routes`: tier to route string.
    pub routes: BTreeMap<String, String>,
    /// `inference/config`.
    pub entry_zero: Option<OcEntryZero>,
    /// `inference/managed/enabled`: `"true"`/`"false"`, anything else means on.
    pub managed_enabled: Option<String>,
    /// `inference/health`, as the raw stored text.
    pub health: Option<String>,
}

/// The model a row carries: the id every tier agrees on, none, or a
/// disagreement.
enum ModelOnRow {
    None,
    One(String),
    Ambiguous,
}

fn model_on_row(models: &BTreeMap<String, String>) -> ModelOnRow {
    let mut ids = models.values().map(|m| m.trim()).filter(|m| !m.is_empty());
    let Some(first) = ids.next() else {
        return ModelOnRow::None;
    };
    if ids.all(|other| other == first) {
        ModelOnRow::One(first.to_string())
    } else {
        ModelOnRow::Ambiguous
    }
}

/// A slug that names the managed provider by a legacy alias (`cloud`) becomes
/// the managed provider's own slug, so a default, route or health entry that
/// pointed at a dropped alias row still resolves. Reported when it rewrites.
fn canon(loss: &mut LossReport, config: &HubConfig, at: &str, slug: Slug) -> Slug {
    // A real imported row that happens to be called `cloud` is that row.
    if config.contains(&slug) || catalogue::group_of(slug.as_str()) != ProviderGroup::Managed {
        return slug;
    }
    let Some(managed) = catalogue::descriptors_in(ProviderGroup::Managed)
        .next()
        .and_then(|d| Slug::parse(d.slug()).ok())
    else {
        return slug;
    };
    if managed != slug {
        loss.push(
            at,
            LossKind::Normalised,
            "a legacy alias of the managed provider was rewritten to its own slug",
        );
    }
    managed
}

/// Maps a stored kind to a catalogue kind, or refuses it (guard G27: an unknown
/// kind fails loudly instead of being read as something else).
fn normalise_kind(raw: &str) -> Result<(KindId, bool), HubError> {
    let name = raw.trim().to_ascii_lowercase();
    match name.as_str() {
        "openai_compatible" | "openai-compatible" | "custom" => {
            Ok((KindId::new("custom"), name != "custom"))
        }
        other => catalogue::resolve_kind(other)
            .map(|kind| {
                let changed = kind.as_str() != other;
                (kind, changed)
            })
            .ok_or_else(|| HubError::NotFound(NotFound::Kind(KindId::new(other)))),
    }
}

fn parse_default(raw: &str) -> Result<DefaultChoice, InvalidInput> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(DefaultChoice::Unset);
    }
    if raw.starts_with('{') {
        #[derive(Deserialize)]
        struct Stored {
            provider: String,
            #[serde(default)]
            model: Option<String>,
        }
        return match serde_json::from_str::<Stored>(raw) {
            Ok(stored) => {
                let provider = Slug::parse(&stored.provider)?;
                match stored
                    .model
                    .as_deref()
                    .map(str::trim)
                    .filter(|m| !m.is_empty())
                {
                    Some(model) => Ok(DefaultChoice::Full {
                        provider,
                        model: ModelId::parse(model)?,
                    }),
                    None => Ok(DefaultChoice::ProviderOnly { provider }),
                }
            }
            Err(_) => Err(InvalidInput::Malformed {
                field: InputField::Config,
                reason: "the stored default is neither a slug nor a provider/model object",
            }),
        };
    }
    Ok(DefaultChoice::ProviderOnly {
        provider: Slug::parse(raw)?,
    })
}

fn health_state(raw: &str) -> ProviderHealth {
    let reason = match raw.trim() {
        "ok" => return ProviderHealth::Ok,
        "auth" => ReasonCode::Auth,
        "quota" => ReasonCode::Quota,
        "model" => ReasonCode::Model,
        "rate_limited" => ReasonCode::RateLimited,
        "endpoint" => ReasonCode::Endpoint,
        "timeout" => ReasonCode::Timeout,
        _ => ReasonCode::Unknown,
    };
    if matches!(reason, ReasonCode::Auth | ReasonCode::Quota) {
        ProviderHealth::Down(reason)
    } else {
        ProviderHealth::Degraded(reason)
    }
}

/// Reads an OpenCompany snapshot.
///
/// # Errors
///
/// [`HubError::NotFound`] for a stored kind the catalogue does not know
/// (guard G27), and [`HubError::Invalid`] for a slug, model id or default that
/// is not valid or a credential-shaped legacy field.
pub fn import(snapshot: &OcSnapshot) -> Result<Imported, HubError> {
    let mut out = Imported::new();
    let mut managed_off = false;

    for stored in &snapshot.providers {
        let (kind, kind_changed) = normalise_kind(&stored.kind)?;
        if kind_changed {
            out.loss.push(
                format!("inference/providers/{}", stored.slug),
                LossKind::Normalised,
                "a legacy kind spelling was mapped to its catalogue kind",
            );
        }
        let slug = Slug::parse(&stored.slug)?;
        let descriptor = catalogue::descriptor(kind.as_str());
        if descriptor.is_some_and(|d| d.group == ProviderGroup::Cli) {
            out.loss.push(
                format!("inference/providers/{}", stored.slug),
                LossKind::Dropped,
                "a CLI login is a route target, not a provider record",
            );
            continue;
        }
        if descriptor.is_some_and(|d| d.group == ProviderGroup::Managed) {
            // The hub always lists the managed provider under its own slug, so a
            // stored row of the managed kind (a legacy alias) is not a second
            // record: it only says whether the operator had it switched off.
            out.loss.push(
                format!("inference/providers/{}", stored.slug),
                LossKind::Normalised,
                "a row of the managed kind is the managed provider, which the hub lists itself; only its enabled flag was kept",
            );
            managed_off |= !stored.enabled;
            continue;
        }
        let mut disabled_by_import = false;
        let preset = descriptor
            .filter(|d| !d.endpoint_editable)
            .and_then(|d| d.default_endpoint);
        let base_url = if stored.base_url.trim().is_empty() {
            descriptor
                .and_then(|d| d.default_endpoint)
                .unwrap_or("")
                .to_string()
        } else if let Some(preset) = preset
            && !crate::policy::same_origin(stored.base_url.trim(), preset)
        {
            // G2/G3: a cloud preset's endpoint is data, and a stored key must not
            // be repointed at another origin. The row is kept as stored but
            // disabled, so the operator decides.
            out.loss.push(
                format!("inference/providers/{}", stored.slug),
                LossKind::FailClosed,
                "the stored base_url is not the cloud preset's origin; the row was imported disabled with its stored url",
            );
            disabled_by_import = true;
            stored.base_url.trim().to_string()
        } else {
            stored.base_url.trim().to_string()
        };
        let mut record = ProviderRecord::new(
            stored.id.clone(),
            slug.clone(),
            if stored.label.trim().is_empty() {
                stored.slug.clone()
            } else {
                stored.label.clone()
            },
            kind,
            base_url,
        );
        record.enabled = stored.enabled && !disabled_by_import;
        record.origin = RecordOrigin::Imported;
        match model_on_row(&stored.models) {
            ModelOnRow::None => {}
            ModelOnRow::One(model) => record.model = Some(ModelId::parse(&model)?),
            ModelOnRow::Ambiguous => {
                out.loss.push(
                    format!("inference/providers/{}", stored.slug),
                    LossKind::Ambiguous,
                    "the row names different models per tier; none was chosen and the map is kept under legacy.tier_models",
                );
                record.legacy.insert(
                    "tier_models".to_string(),
                    serde_json::to_value(&stored.models).unwrap_or_default(),
                );
            }
        }
        record.validate()?;
        out.config.providers.push(record);
    }

    if let Some(zero) = &snapshot.entry_zero {
        let provider = zero.provider.trim().to_ascii_lowercase();
        if provider == "managed" || provider == "tinyhumans" || provider.is_empty() {
            out.loss.push(
                "inference/config",
                LossKind::Synthesised,
                "entry zero is the managed provider, which the hub always lists; no separate record was made",
            );
        } else {
            let (kind, changed) = normalise_kind(&provider)?;
            if changed {
                out.loss.push(
                    "inference/config",
                    LossKind::Normalised,
                    "entry zero's provider spelling was mapped to its catalogue kind",
                );
            }
            let slug = Slug::parse(&provider)?;
            let descriptor = catalogue::descriptor(kind.as_str());
            let base_url = zero
                .base_url
                .as_deref()
                .map(str::trim)
                .filter(|u| !u.is_empty())
                .or_else(|| descriptor.and_then(|d| d.default_endpoint))
                .unwrap_or("")
                .to_string();
            let mut record =
                ProviderRecord::new("prv_entry_zero", slug, provider.clone(), kind, base_url);
            record.origin = RecordOrigin::EntryZero;
            if let ModelOnRow::One(model) = model_on_row(&zero.models) {
                record.model = Some(ModelId::parse(&model)?);
            }
            record.validate()?;
            if out.config.contains(&record.slug) {
                out.loss.push(
                    "inference/config",
                    LossKind::Dropped,
                    "entry zero names a provider that also has an indexed row; the indexed row wins",
                );
            } else {
                out.loss.push(
                    "inference/config",
                    LossKind::Synthesised,
                    "entry zero became a read-only record",
                );
                out.config.providers.push(record);
            }
        }
    }

    managed_off |= snapshot
        .managed_enabled
        .as_deref()
        .is_some_and(|raw| raw.trim().eq_ignore_ascii_case("false"));
    if managed_off
        && let Some(descriptor) = catalogue::descriptors_in(ProviderGroup::Managed).next()
    {
        let slug = Slug::parse(descriptor.slug())?;
        let mut managed = ProviderRecord::new(
            "managed",
            slug,
            descriptor.label,
            descriptor.kind.clone(),
            "",
        );
        managed.enabled = false;
        managed.synthetic = true;
        managed.origin = RecordOrigin::Imported;
        out.config.providers.insert(0, managed);
    }

    if let Some(raw) = &snapshot.default {
        out.config.default = match parse_default(raw)? {
            DefaultChoice::ProviderOnly { provider } => DefaultChoice::ProviderOnly {
                provider: canon(&mut out.loss, &out.config, "inference/default", provider),
            },
            DefaultChoice::Full { provider, model } => DefaultChoice::Full {
                provider: canon(&mut out.loss, &out.config, "inference/default", provider),
                model,
            },
            other => other,
        };
    }

    let local_records = out
        .config
        .providers
        .iter()
        .filter(|p| catalogue::group_of(p.kind.as_str()) == ProviderGroup::Local)
        .count();
    for (tier, text) in &snapshot.routes {
        let mut route = legacy_oc::parse(text)?;
        if let RouteTarget::Provider(slug) = route.target.clone() {
            route.target = RouteTarget::Provider(canon(
                &mut out.loss,
                &out.config,
                &format!("inference/routes/{tier}"),
                slug,
            ));
        }
        if route.target == RouteTarget::Default {
            out.loss.push(
                format!("inference/routes/{tier}"),
                LossKind::Dropped,
                "a route to the default is the absence of a route",
            );
            continue;
        }
        if route.target == RouteTarget::Local(None) && local_records > 1 {
            out.loss.push(
                format!("inference/routes/{tier}"),
                LossKind::Ambiguous,
                "`local` names a category and more than one local runtime is configured; the first enabled one answers",
            );
        }
        if let RouteTarget::Provider(slug) = &route.target
            && !out.config.contains(slug)
            && catalogue::group_of(slug.as_str()) != ProviderGroup::Managed
        {
            out.loss.push(
                format!("inference/routes/{tier}"),
                LossKind::FailClosed,
                "the route names a provider that is not configured; the turn will fail closed",
            );
        }
        out.config
            .workload_routes
            .insert(WorkloadKey::new(tier.clone()), route);
    }

    if let Some(raw) = &snapshot.health {
        match serde_json::from_str::<BTreeMap<String, serde_json::Value>>(raw) {
            Ok(map) => {
                for (slug, entry) in map {
                    let (Ok(slug), Some(state)) = (
                        Slug::parse(&slug),
                        entry.get("state").and_then(|s| s.as_str()),
                    ) else {
                        continue;
                    };
                    let slug = canon(&mut out.loss, &out.config, "inference/health", slug);
                    out.health.insert(slug, health_state(state));
                }
                out.loss.push(
                    "inference/health",
                    LossKind::Dropped,
                    "health timestamps are not carried over; only the states are",
                );
            }
            Err(_) => out.loss.push(
                "inference/health",
                LossKind::Dropped,
                "the stored health was unreadable and is treated as empty",
            ),
        }
    }

    out.config.validate()?;
    Ok(out)
}
