//! Model-visible tool declaration and tool-call wire types.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The syntax used to expose a tool to a model.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ToolFormat {
    /// Native JSON/function-call style.
    #[default]
    Json,
    /// XML tag style.
    Xml,
    /// Compact ordered-parameter call syntax.
    PType {
        /// Ordered parameter names.
        parameters: Vec<String>,
    },
}

impl ToolFormat {
    /// Returns whether this is the default JSON format.
    pub fn is_json(&self) -> bool {
        matches!(self, Self::Json)
    }
}

/// A model-visible declaration of a callable tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolSchema {
    /// Canonical tool name.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// JSON Schema describing accepted arguments.
    pub parameters: Value,
    /// Preferred model-visible call format.
    #[serde(default, skip_serializing_if = "ToolFormat::is_json")]
    pub format: ToolFormat,
}

impl ToolSchema {
    /// Creates a JSON/function-call tool schema.
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
            format: ToolFormat::Json,
        }
    }

    /// Sets the model-visible call format.
    pub fn with_format(mut self, format: ToolFormat) -> Self {
        self.format = format;
        self
    }

    /// Validates a model-supplied call against this schema's structural subset.
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] when the tool name or arguments do
    /// not satisfy the declaration.
    pub fn validate_call(&self, call: &ToolCall) -> crate::Result<()> {
        if let Some(reason) = &call.invalid {
            return Err(crate::Error::Validation(format!(
                "tool call `{}` has malformed arguments: {reason}",
                call.name
            )));
        }
        if call.name != self.name {
            return Err(crate::Error::Validation(format!(
                "tool call `{}` does not match schema `{}`",
                call.name, self.name
            )));
        }
        validate_json_value(
            &self.parameters,
            &call.arguments,
            &format!("tool `{}` arguments", self.name),
        )
    }
}

/// A model request to invoke a tool.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Opaque signed-call replay context, scoped to its originating provider/model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replay: Option<ToolCallReplay>,
    /// Provider-assigned call identifier.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Parsed arguments, or the raw string when [`Self::invalid`] is set.
    #[serde(default)]
    pub arguments: Value,
    /// Parse error for malformed provider arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub invalid: Option<String>,
}

impl ToolCall {
    /// Creates a valid tool call.
    pub fn new(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: id.into(),
            replay: None,
            name: name.into(),
            arguments,
            invalid: None,
        }
    }

    /// Creates a malformed tool call while preserving its raw arguments.
    pub fn invalid(
        id: impl Into<String>,
        name: impl Into<String>,
        raw: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            replay: None,
            name: name.into(),
            arguments: Value::String(raw.into()),
            invalid: Some(reason.into()),
        }
    }

    /// Returns whether the provider emitted malformed arguments.
    pub fn is_invalid(&self) -> bool {
        self.invalid.is_some()
    }
}

/// Context required to replay a signed function call without changing its bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolCallReplay {
    /// Origin of the signature and argument representation.
    pub origin: crate::message::MessageOrigin,
    /// Provider output item ID, distinct from the call ID.
    pub item_id: Option<String>,
    /// Original argument JSON text, retained even when parsing succeeds.
    pub arguments: String,
    /// Opaque provider signature; never reconstruct or modify it.
    pub thought_signature: Option<String>,
}

/// Context for a native function result sent without replaying its preceding call.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultContext {
    /// Function name associated with the result's call ID.
    pub name: String,
    /// Original provider/model scope and signature when needed.
    pub replay: Option<ToolCallReplay>,
}

/// An incremental tool-call fragment emitted by a model stream.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolDelta {
    /// Call identifier this fragment belongs to.
    pub call_id: String,
    /// Incremental argument or content fragment.
    pub content: String,
    /// Tool name when the provider supplies it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    /// Position of this tool call's content block within the assistant
    /// message, when the provider reports block-indexed content (Anthropic's
    /// `content_block` index, or the OpenAI `tool_calls[].index`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_index: Option<usize>,
}

/// Validates `value` against the structural subset of JSON Schema that tool
/// arguments are held to: `type` (including unions), object `properties`,
/// `required`, `additionalProperties: false`, array `items`, and `enum`.
/// Unknown keywords are ignored and an empty or null schema imposes nothing.
///
/// `path` names the value in error messages — `tool \`lookup\` arguments`
/// produces `tool \`lookup\` arguments.limit must be integer, got string`.
///
/// # Errors
///
/// Returns [`crate::Error::Validation`] naming the first failing instance
/// path.
pub fn validate_json_value(schema: &Value, value: &Value, path: &str) -> crate::Result<()> {
    if schema.is_null() || schema.as_object().is_some_and(serde_json::Map::is_empty) {
        return Ok(());
    }
    if let Some(values) = schema.get("enum").and_then(Value::as_array)
        && !values.iter().any(|allowed| allowed == value)
    {
        return Err(crate::Error::Validation(format!(
            "{path} must be one of the declared enum values"
        )));
    }
    if let Some(type_spec) = schema.get("type") {
        validate_type_spec(type_spec, value, path)?;
    }
    if let Some(required) = schema.get("required").and_then(Value::as_array) {
        if let Some(object) = value.as_object() {
            for field in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(field) {
                    return Err(crate::Error::Validation(format!(
                        "{path}.{field} is required"
                    )));
                }
            }
        } else if schema.get("type").is_none() {
            return Err(crate::Error::Validation(format!(
                "{path} must be an object with the declared fields, got {}",
                json_value_kind(value)
            )));
        }
    }
    if let Some(properties) = schema.get("properties").and_then(Value::as_object) {
        if let Some(object) = value.as_object() {
            if schema.get("additionalProperties").and_then(Value::as_bool) == Some(false) {
                for field in object.keys() {
                    if !properties.contains_key(field) {
                        return Err(crate::Error::Validation(format!(
                            "{path}.{field} is not allowed"
                        )));
                    }
                }
            }
            for (field, field_schema) in properties {
                if let Some(field_value) = object.get(field) {
                    validate_json_value(field_schema, field_value, &format!("{path}.{field}"))?;
                }
            }
        } else if schema.get("type").is_none() {
            return Err(crate::Error::Validation(format!(
                "{path} must be an object with the declared fields, got {}",
                json_value_kind(value)
            )));
        }
    }
    if let Some(items_schema) = schema.get("items")
        && let Some(items) = value.as_array()
    {
        for (index, item) in items.iter().enumerate() {
            validate_json_value(items_schema, item, &format!("{path}[{index}]"))?;
        }
    }
    Ok(())
}

fn validate_type_spec(type_spec: &Value, value: &Value, path: &str) -> crate::Result<()> {
    if let Some(kind) = type_spec.as_str() {
        if json_value_matches_type(value, kind) {
            return Ok(());
        }
        return Err(crate::Error::Validation(format!(
            "{path} must be {kind}, got {}",
            json_value_kind(value)
        )));
    }
    if let Some(kinds) = type_spec.as_array() {
        let allowed: Vec<&str> = kinds.iter().filter_map(Value::as_str).collect();
        if allowed
            .iter()
            .any(|kind| json_value_matches_type(value, kind))
        {
            return Ok(());
        }
        return Err(crate::Error::Validation(format!(
            "{path} must be one of {}, got {}",
            allowed.join(", "),
            json_value_kind(value)
        )));
    }
    Ok(())
}

fn json_value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        _ => true,
    }
}

fn json_value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.as_i64().is_some() || number.as_u64().is_some() => {
            "integer"
        }
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}
