//! Per-model capability records and pre-flight request validation.
//!
//! Media generation is billed on submit, so a request the model cannot honor
//! should fail locally before it costs anything. Capabilities come from the
//! provider's model listings (`GET /images/models`, `GET /videos/models`).
//! Every field is optional: `None` means "not advertised", and an unadvertised
//! field is never used to reject a request.

use serde_json::Value;

use crate::{Error, Result};

/// What a model advertises it accepts.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Accepted resolution tiers (`1K`, `720p`, …).
    pub resolutions: Option<Vec<String>>,
    /// Accepted aspect ratios (`16:9`, …).
    pub aspect_ratios: Option<Vec<String>>,
    /// Accepted clip durations in seconds (video only).
    pub durations: Option<Vec<u32>>,
    /// Accepted frame-image roles (`first_frame`, `last_frame`; video only).
    pub frame_images: Option<Vec<String>>,
    /// Inclusive range of images per request.
    pub n_range: Option<(u32, u32)>,
    /// Maximum number of reference assets.
    pub max_references: Option<u32>,
    /// Whether a deterministic seed is accepted.
    pub seed: Option<bool>,
    /// Whether audio generation is available (video only).
    pub generate_audio: Option<bool>,
}

impl ModelCapabilities {
    /// Reads an image-model record from `GET /images/models`.
    ///
    /// The record's `supported_parameters` map uses typed descriptors
    /// (`{"type":"enum","values":[…]}`, `{"type":"range","min":…,"max":…}`,
    /// `{"type":"boolean"}`); a key absent from a present map means the
    /// parameter is unsupported.
    #[must_use]
    pub fn from_image_model(record: &Value) -> Self {
        let Some(params) = record
            .get("supported_parameters")
            .and_then(Value::as_object)
        else {
            return Self::default();
        };
        // OpenRouter's contract: within a present `supported_parameters` map,
        // an absent key means the endpoint does not support that parameter.
        // So an omitted enum is "supports nothing" (`Some(vec![])`), which
        // rejects the field before a billed call instead of letting the
        // provider silently ignore it.
        let enum_values = |key: &str| {
            Some(
                params
                    .get(key)
                    .and_then(|descriptor| descriptor.get("values"))
                    .and_then(Value::as_array)
                    .map(|values| string_list(values))
                    .unwrap_or_default(),
            )
        };
        let range = |key: &str| {
            params.get(key).map(|descriptor| {
                let bound = |name: &str| {
                    descriptor
                        .get(name)
                        .and_then(Value::as_u64)
                        .and_then(|n| u32::try_from(n).ok())
                };
                (bound("min").unwrap_or(0), bound("max").unwrap_or(u32::MAX))
            })
        };
        Self {
            resolutions: enum_values("resolution"),
            aspect_ratios: enum_values("aspect_ratio"),
            durations: None,
            frame_images: None,
            n_range: Some(range("n").unwrap_or((1, 1))),
            max_references: Some(range("input_references").map_or(0, |(_, max)| max)),
            seed: Some(params.contains_key("seed")),
            generate_audio: None,
        }
    }

    /// Reads a video-model record from `GET /videos/models`
    /// (`supported_resolutions`, `supported_aspect_ratios`,
    /// `supported_durations`, `supported_frame_images`, `seed`,
    /// `generate_audio`). A `null` list means "not advertised".
    #[must_use]
    pub fn from_video_model(record: &Value) -> Self {
        let list = |key: &str| {
            record
                .get(key)
                .and_then(Value::as_array)
                .map(|values| string_list(values))
        };
        Self {
            resolutions: list("supported_resolutions"),
            aspect_ratios: list("supported_aspect_ratios"),
            durations: record
                .get("supported_durations")
                .and_then(Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_u64)
                        .filter_map(|n| u32::try_from(n).ok())
                        .collect()
                }),
            frame_images: list("supported_frame_images"),
            n_range: None,
            max_references: None,
            seed: record.get("seed").and_then(Value::as_bool),
            generate_audio: record.get("generate_audio").and_then(Value::as_bool),
        }
    }

    /// Rejects `value` for `field` when the model advertises a list that does
    /// not contain it.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] naming the field and the advertised values.
    pub fn check_one_of(
        model: &str,
        field: &str,
        value: &str,
        allowed: Option<&[String]>,
    ) -> Result<()> {
        let Some(allowed) = allowed else {
            return Ok(());
        };
        if allowed
            .iter()
            .any(|candidate| candidate.eq_ignore_ascii_case(value))
        {
            return Ok(());
        }
        Err(Error::Unsupported {
            model: model.to_owned(),
            field: field.to_owned(),
            value: value.to_owned(),
            allowed: allowed.to_vec(),
        })
    }

    /// Rejects a feature the model advertises as unavailable.
    ///
    /// # Errors
    ///
    /// [`Error::Unsupported`] when `supported` is `Some(false)`.
    pub fn check_flag(model: &str, field: &str, supported: Option<bool>) -> Result<()> {
        if supported == Some(false) {
            return Err(Error::Unsupported {
                model: model.to_owned(),
                field: field.to_owned(),
                value: "true".to_owned(),
                allowed: Vec::new(),
            });
        }
        Ok(())
    }
}

fn string_list(values: &[Value]) -> Vec<String> {
    values
        .iter()
        .filter_map(|value| match value {
            Value::String(text) => Some(text.clone()),
            Value::Number(number) => Some(number.to_string()),
            _ => None,
        })
        .collect()
}
