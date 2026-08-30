//! Public tool-schema validation contracts.

use serde_json::json;
use tinyinference::tool::{ToolCall, ToolSchema};

#[test]
fn invalid_provider_arguments_fail_even_with_permissive_schema() {
    let schema = ToolSchema::new("lookup", "lookup", json!({}));
    let call = ToolCall::invalid("call-1", "lookup", "{broken", "expected value");
    let error = schema.validate_call(&call).unwrap_err();
    assert!(error.to_string().contains("malformed arguments"));
}
