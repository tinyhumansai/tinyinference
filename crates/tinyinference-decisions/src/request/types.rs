//! Stable request payload definitions.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One complete System One evaluation request.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct EvaluationRequest {
    /// Unstructured text or structured application state visible to every question.
    pub state: Value,
    /// System One model identifier, normally `jev-latest`.
    pub model: String,
    /// Independently evaluated questions keyed by caller-owned ids.
    pub questions: BTreeMap<String, Question>,
    /// Images keyed by the reference string that `state` or a question uses
    /// to point at them. Empty for a text-only request, and then absent from
    /// the wire form, so text requests serialise exactly as before.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub assets: BTreeMap<String, Asset>,
}

impl EvaluationRequest {
    /// Build a request for an explicit model id, including a locally served
    /// OpenJEV, Decider, or other Jev-compatible decision model.
    ///
    /// The id is passed through unchanged and must match the response's model
    /// id for a self-hosted client. [`Self::validate`] checks the request before
    /// the client sends it.
    #[must_use]
    pub fn new(
        model: impl Into<String>,
        state: impl Into<Value>,
        questions: BTreeMap<String, Question>,
    ) -> Self {
        Self {
            state: state.into(),
            model: model.into(),
            questions,
            assets: BTreeMap::new(),
        }
    }

    /// Attach an image under `key`.
    ///
    /// `key` is the reference string: put it somewhere in `state` (for
    /// example `{"photo": key}`) or in a question's instructions so Jev knows
    /// where the image belongs. An image nobody references is rejected by
    /// [`Self::validate`]. Attaching any asset makes the client send
    /// `TypeSafe-Danger-Zone-Images: true`.
    #[must_use]
    pub fn with_image(mut self, key: impl Into<String>, image: ImageAsset) -> Self {
        self.assets.insert(key.into(), Asset::Image(image));
        self
    }

    /// Whether the request carries an image or asks a [`Question::Box`], the
    /// two things the provider admits only with its image opt-in header.
    #[must_use]
    pub fn needs_image_opt_in(&self) -> bool {
        !self.assets.is_empty()
            || self
                .questions
                .values()
                .any(|question| matches!(question, Question::Box(_)))
    }

    /// Build a request using the stable Jev alias.
    #[must_use]
    pub fn jev(state: impl Into<Value>, questions: BTreeMap<String, Question>) -> Self {
        Self::new("jev-latest", state, questions)
    }

    /// Build a request using OpenJEV's default model identifier, `openjev`.
    #[must_use]
    pub fn openjev(state: impl Into<Value>, questions: BTreeMap<String, Question>) -> Self {
        Self::new("openjev", state, questions)
    }
}

/// A typed question evaluated independently against shared state.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Question {
    /// Select exactly one option from a closed set.
    Choice(Choice),
    /// Place the state along an ordered descriptive scale.
    Score(Score),
    /// Estimate the probability that a yes/no condition holds.
    Noul(Noul),
    /// Locate one thing in an image (Jev 1.14 or later). The answer is a
    /// single box; ask a [`Noul`] alongside it to learn whether the thing is
    /// there at all, because a box is always returned.
    Box(BoxQuestion),
}

/// A question asking where something is in an attached image.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct BoxQuestion {
    /// What to find, and in which image: name the state field holding the
    /// image in backticks, for example ``"the search button in `page`"``.
    pub instructions: Value,
}

/// One attached asset.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Asset {
    /// A still PNG or JPEG image.
    Image(ImageAsset),
}

/// A still image sent inline as base64.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ImageAsset {
    /// Encoding of the image bytes.
    pub format: ImageFormat,
    /// Standard padded base64 of the image file, with no `data:` prefix.
    pub data_base64: String,
}

/// Image encodings the provider accepts.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    /// PNG.
    Png,
    /// JPEG, written `jpg` on the wire.
    #[serde(rename = "jpg", alias = "jpeg")]
    Jpg,
}

/// A closed-set selection question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Choice {
    /// Complete meaning of the decision to make.
    pub instructions: Value,
    /// Option name to optional distinguishing description.
    pub criteria: BTreeMap<String, Option<Value>>,
}

/// An ordered-scale question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Score {
    /// Dimension being rated.
    pub instructions: Value,
    /// Ordered, standalone level descriptions.
    pub criteria: Vec<Value>,
}

/// A yes/no probability question.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Noul {
    /// Condition whose probability of being true is requested.
    pub instructions: Value,
    /// Optional descriptions clarifying both outcomes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub criteria: Option<NoulCriteria>,
}

/// Descriptions of the true and false outcomes of a [`Noul`].
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct NoulCriteria {
    /// What counts as true.
    pub r#true: Value,
    /// What counts as false.
    pub r#false: Value,
}
