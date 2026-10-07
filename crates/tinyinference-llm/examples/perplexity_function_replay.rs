//! Run a Google function call and replay its result through Perplexity.
//!
//! Set `PERPLEXITY_API_KEY`, then run:
//! `cargo run -p tinyinference-llm --example perplexity_function_replay`.
//! This makes two billable requests and runs one local demonstration function.

use serde_json::json;
use tinyinference_llm::{
    ChatModel, ContentBlock, ExecutionStatus, Message, ModelRequest, PerplexityModel,
    PerplexitySelection, ToolSchema, message::ToolMessage, model::ToolChoice,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("PERPLEXITY_API_KEY")?;
    let model = PerplexityModel::new(
        key,
        PerplexitySelection::Model("google/gemini-3.1-flash-lite".into()),
    )?;
    let tool = ToolSchema::new(
        "lookup_code",
        "Return the verification code for a record",
        json!({
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "additionalProperties": false
        }),
    );
    let mut request = ModelRequest::new(vec![Message::user(
        "Use lookup_code for record 7, then tell me its verification code.",
    )])
    .with_tools(vec![tool])
    .with_tool_choice(ToolChoice::Tool("lookup_code".into()))
    .with_max_tokens(512);
    request.timeout_ms = Some(60_000);
    let first = model.invoke(&(), request.clone()).await?;
    let execution = first.execution.as_ref().ok_or("missing model identity")?;
    if execution.status != ExecutionStatus::Completed || first.message.tool_calls.is_empty() {
        return Err("expected a completed function call".into());
    }

    // Replay the original assistant call so its argument bytes and signature survive.
    request
        .messages
        .push(Message::Assistant(first.message.clone()));
    for call in &first.message.tool_calls {
        if call.invalid.is_some()
            || call.name != "lookup_code"
            || call.arguments.get("id").and_then(serde_json::Value::as_u64) != Some(7)
        {
            return Err("unexpected function or invalid arguments".into());
        }
        let result = ToolMessage::for_call(call, vec![ContentBlock::Text("CODE_7391".into())]);
        request.messages.push(Message::Tool(result));
    }
    request.model = Some(execution.model.clone());
    request.continuation_id = None;
    request.tool_choice = ToolChoice::None;
    let response = model.invoke(&(), request).await?;
    println!("{}", response.text());
    if !response.text().contains("CODE_7391") {
        return Err("the model did not return the demonstration function result".into());
    }
    Ok(())
}
