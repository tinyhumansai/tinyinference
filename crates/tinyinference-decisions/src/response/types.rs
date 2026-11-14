//! Stable response payload definitions.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One decoded System One response.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationResponse {
    /// Model that performed the evaluation.
    pub model: String,
    /// Answers keyed by the caller's question ids.
    pub answers: BTreeMap<String, Answer>,
    /// Provider-reported token counts.
    pub usage: Usage,
}

/// Provider-reported token usage.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct Usage {
    /// Input tokens, when reported.
    #[serde(default)]
    pub input_tokens: Option<u64>,
    /// Output tokens, when reported.
    #[serde(default)]
    pub output_tokens: Option<u64>,
    /// Cached input tokens, when the provider reports them.
    #[serde(
        default,
        alias = "cached_tokens",
        skip_serializing_if = "Option::is_none"
    )]
    pub cached_input_tokens: Option<u64>,
    /// Provider-reported USD charge in integer nanodollars. Invalid or
    /// unreported costs stay absent.
    #[serde(
        default,
        rename = "cost",
        deserialize_with = "deserialize_usd_nanos",
        serialize_with = "serialize_usd_nanos",
        skip_serializing_if = "Option::is_none"
    )]
    pub cost_usd_nanos: Option<i64>,
}

fn serialize_usd_nanos<S>(value: &Option<i64>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    match value {
        Some(nanos) => serializer.serialize_f64(*nanos as f64 / 1_000_000_000.0),
        None => serializer.serialize_none(),
    }
}

fn deserialize_usd_nanos<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    let Some(value) = value.and_then(|value| value.as_number()?.as_f64()) else {
        return Ok(None);
    };
    if !value.is_finite() || value < 0.0 || value > (i64::MAX as f64 / 1_000_000_000.0) {
        return Ok(None);
    }
    Ok(Some((value * 1_000_000_000.0).round() as i64))
}

/// One typed answer.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Answer {
    /// Closed-set selection and distribution.
    Choice(ChoiceAnswer),
    /// Ordered score and level distribution.
    Score(ScoreAnswer),
    /// Probability of yes.
    Noul(NoulAnswer),
    /// Where the asked-for thing is in the image.
    Box(BoxAnswer),
}

/// Answer to a Box question.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BoxAnswer {
    /// The box, as fractions of the image's width and height.
    #[serde(rename = "box")]
    pub edges: BoxEdges,
    /// Confidence in the box, from zero to one.
    pub confidence: f64,
}

/// The edges of a box, each a fraction (0 to 1) of the image measured from its
/// top-left corner: `top` and `bottom` of its height, `left` and `right` of
/// its width.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct BoxEdges {
    /// Top edge, a fraction of the height.
    pub top: f64,
    /// Bottom edge, a fraction of the height.
    pub bottom: f64,
    /// Left edge, a fraction of the width.
    pub left: f64,
    /// Right edge, a fraction of the width.
    pub right: f64,
}

impl BoxEdges {
    /// The centre of the box as `(x, y)` fractions of width and height.
    #[must_use]
    pub fn centre(&self) -> (f64, f64) {
        (
            f64::midpoint(self.left, self.right),
            f64::midpoint(self.top, self.bottom),
        )
    }
}

/// Answer to a Choice question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChoiceAnswer {
    /// Highest-probability option.
    pub choice: String,
    /// Probability for every requested option.
    pub probabilities: BTreeMap<String, f64>,
    /// Concentration of the distribution, not correctness probability.
    pub confidence: f64,
}

/// Answer to a Score question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ScoreAnswer {
    /// Probability-weighted position along the scale.
    pub score: f64,
    /// Requested level descriptions keyed by zero-based index.
    pub legend: BTreeMap<String, Value>,
    /// Probability for every level.
    pub probabilities: BTreeMap<String, f64>,
    /// Concentration of the distribution, not correctness probability.
    pub confidence: f64,
}

/// Answer to a Noul question.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub struct NoulAnswer {
    /// Probability that the requested condition is true.
    pub noul: f64,
}
