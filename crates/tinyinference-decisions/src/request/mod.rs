//! Typed System One request values and their local validation.

#[cfg(test)]
#[path = "mod_tests.rs"]
mod test;

pub(crate) mod image;
mod types;

pub use types::{
    Asset, BoxQuestion, Choice, EvaluationRequest, ImageAsset, ImageFormat, Noul, NoulCriteria,
    Question, Score,
};

use crate::{Error, Result};

impl EvaluationRequest {
    /// Validate this request before any network operation begins.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when the model, state, question ids,
    /// instructions, or criteria cannot form a valid System One request.
    pub fn validate(&self) -> Result<()> {
        if self.model.trim().is_empty() {
            return Err(Error::invalid_request("model must not be empty"));
        }
        if !matches!(
            self.state,
            serde_json::Value::String(_)
                | serde_json::Value::Array(_)
                | serde_json::Value::Object(_)
        ) {
            return Err(Error::invalid_request(
                "state must be a string, object, or array",
            ));
        }
        if self.questions.is_empty() {
            return Err(Error::invalid_request("questions must not be empty"));
        }
        for (id, question) in &self.questions {
            if id.trim().is_empty() {
                return Err(Error::invalid_request("question ids must not be empty"));
            }
            question.validate()?;
        }
        self.validate_assets()
    }

    fn validate_assets(&self) -> Result<()> {
        if self.assets.is_empty() {
            return Ok(());
        }
        // Every reference must appear somewhere the model reads, or the
        // provider rejects the request; the wire text of state and questions
        // is exactly where it looks. Search the decoded string values and
        // object keys, not the escaped JSON text, so keys containing `\`, `"`
        // or control characters match.
        let readable = serde_json::to_value((&self.state, &self.questions))
            .map_err(|_| Error::invalid_request("state and questions must serialise"))?;
        let mut total = 0_usize;
        for (key, asset) in &self.assets {
            if key.trim().is_empty() {
                return Err(Error::invalid_request("asset references must not be empty"));
            }
            if key.len() > image::MAX_REFERENCE_BYTES {
                return Err(Error::invalid_request(
                    "asset references must be at most 1024 bytes",
                ));
            }
            if !value_mentions(&readable, key) {
                return Err(Error::invalid_request(format!(
                    "asset `{key}` is not referenced by the state or any question"
                )));
            }
            let Asset::Image(image) = asset;
            total = total.saturating_add(image.checked_len()?);
        }
        if total > image::MAX_TOTAL_IMAGE_BYTES {
            return Err(Error::invalid_request(
                "images must total at most 32 MiB per request",
            ));
        }
        Ok(())
    }
}

/// Whether any decoded string value or object key inside `value` contains `needle`.
fn value_mentions(value: &serde_json::Value, needle: &str) -> bool {
    match value {
        serde_json::Value::String(text) => text.contains(needle),
        serde_json::Value::Array(items) => items.iter().any(|item| value_mentions(item, needle)),
        serde_json::Value::Object(map) => map
            .iter()
            .any(|(key, item)| key.contains(needle) || value_mentions(item, needle)),
        _ => false,
    }
}

impl Question {
    fn validate(&self) -> Result<()> {
        match self {
            Self::Choice(question) => question.validate(),
            Self::Score(question) => question.validate(),
            Self::Noul(question) => question.validate(),
            Self::Box(question) => validate_instructions(&question.instructions),
        }
    }
}

impl Choice {
    fn validate(&self) -> Result<()> {
        validate_instructions(&self.instructions)?;
        if !(2..=255).contains(&self.criteria.len()) {
            return Err(Error::invalid_request(
                "choice criteria must contain between 2 and 255 options",
            ));
        }
        if self.criteria.keys().any(|key| key.trim().is_empty()) {
            return Err(Error::invalid_request(
                "choice option names must not be empty",
            ));
        }
        Ok(())
    }
}

impl Score {
    fn validate(&self) -> Result<()> {
        validate_instructions(&self.instructions)?;
        if !(2..=10).contains(&self.criteria.len()) {
            return Err(Error::invalid_request(
                "score criteria must contain between 2 and 10 levels",
            ));
        }
        if self.criteria.iter().any(is_empty_text) {
            return Err(Error::invalid_request(
                "score level descriptions must not be empty",
            ));
        }
        Ok(())
    }
}

impl Noul {
    fn validate(&self) -> Result<()> {
        validate_instructions(&self.instructions)?;
        if let Some(criteria) = &self.criteria
            && (is_empty_text(&criteria.r#true) || is_empty_text(&criteria.r#false))
        {
            return Err(Error::invalid_request(
                "noul criteria descriptions must not be empty",
            ));
        }
        Ok(())
    }
}

fn validate_instructions(value: &serde_json::Value) -> Result<()> {
    if !matches!(
        value,
        serde_json::Value::String(_) | serde_json::Value::Array(_) | serde_json::Value::Object(_)
    ) || is_empty_text(value)
    {
        return Err(Error::invalid_request(
            "instructions must be a nonempty string, object, or array",
        ));
    }
    Ok(())
}

fn is_empty_text(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::String(text) => text.trim().is_empty(),
        serde_json::Value::Array(values) => values.is_empty(),
        serde_json::Value::Object(values) => values.is_empty(),
        _ => false,
    }
}
