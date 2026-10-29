use crate::message::{ContentBlock, MessageOrigin};
use crate::model::*;
use crate::tool::{ToolCall, ToolCallReplay};
use crate::usage::{ChargedAmount, Usage};
use crate::{Error, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::collections::{BTreeMap, HashSet};

pub(super) struct DecodedJson {
    pub value: Value,
    pub usage: Option<Box<serde_json::value::RawValue>>,
}

impl From<Value> for DecodedJson {
    fn from(value: Value) -> Self {
        Self { value, usage: None }
    }
}

pub(super) fn decode(bytes: &[u8]) -> Result<DecodedJson> {
    let value: Value =
        serde_json::from_slice(bytes).map_err(|_| malformed("response is not valid JSON"))?;
    let path: &[&str] = if value.get("usage").is_some_and(Value::is_object) {
        &["usage"]
    } else if value
        .get("response")
        .and_then(|v| v.get("usage"))
        .is_some_and(Value::is_object)
    {
        &["response", "usage"]
    } else {
        return Ok(DecodedJson { value, usage: None });
    };
    let raw: Box<serde_json::value::RawValue> =
        serde_json::from_slice(bytes).map_err(|_| malformed("response is not valid JSON"))?;
    Ok(DecodedJson {
        value,
        usage: raw_at(Some(&raw), path),
    })
}

pub(super) fn parse(decoded: impl Into<DecodedJson>) -> Result<ModelResponse> {
    let DecodedJson {
        value,
        usage: exact_usage,
    } = decoded.into();
    let id = identity(&value, "id")?;
    let status = ExecutionStatus::from(identity(&value, "status")?);
    let model = if matches!(
        status,
        ExecutionStatus::Queued | ExecutionStatus::InProgress | ExecutionStatus::Cancelling
    ) {
        string(&value, "model")?
    } else {
        identity(&value, "model")?
    };
    let mut response = ModelResponse::assistant("");
    response.message.content.clear();
    response.raw = Some(value.clone());
    response.execution = Some(ModelExecution {
        id: id.clone(),
        model: model.clone(),
        status,
        incomplete_reason: value
            .get("incomplete_details")
            .and_then(|v| v.get("reason"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        sequence_number: optional(&value, "sequence_number")?,
        progress: Vec::new(),
        cost: None,
        tool_usage: BTreeMap::new(),
    });
    let decode = (|| -> Result<()> {
        let items = value
            .get("output")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("output must be an array"))?;
        let mut calls = HashSet::new();
        for (index, item) in items.iter().enumerate() {
            let item = parse_item(item, index, &id, &model)?;
            if let ModelOutputKind::FunctionCall { call } = &item.kind
                && !calls.insert(call.id.clone())
            {
                return Err(malformed("duplicate function call id"));
            }
            response.output.push(item);
        }
        if let Some(usage) = value.get("usage").filter(|v| !v.is_null()) {
            let (tokens, cost, tools) = parse_usage(usage, exact_usage.as_deref())?;
            response.usage = Some(tokens);
            if let Some(execution) = &mut response.execution {
                execution.cost = cost;
                execution.tool_usage = tools;
            }
        }
        Ok(())
    })();
    project(&mut response);
    if let Err(error) = decode {
        return Err(with_partial(error, response));
    }
    Ok(response)
}

pub(super) fn completed(response: ModelResponse) -> Result<ModelResponse> {
    let status = response.execution.as_ref().map(|v| &v.status);
    if matches!(
        status,
        Some(ExecutionStatus::Completed | ExecutionStatus::Incomplete)
    ) {
        if response
            .raw
            .as_ref()
            .and_then(|v| v.get("error"))
            .is_some_and(|v| !v.is_null())
        {
            return Err(failed_response(response, "provider_error"));
        }
        Ok(response)
    } else {
        let code = match status {
            Some(ExecutionStatus::Failed) => "provider_failed",
            Some(ExecutionStatus::Cancelled) => "cancelled",
            Some(
                ExecutionStatus::Queued | ExecutionStatus::InProgress | ExecutionStatus::Cancelling,
            ) => "not_completed",
            _ => "unknown_status",
        };
        Err(failed_response(response, code))
    }
}

fn failed_response(response: ModelResponse, code: &str) -> Error {
    let raw = response.raw.as_ref().and_then(|v| v.get("error"));
    let message = raw
        .and_then(|v| v.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("Perplexity run did not complete successfully");
    let error = Error::Provider(Box::new(ProviderError {
        provider: "perplexity".into(),
        code: Some(
            raw.and_then(|v| v.get("code"))
                .and_then(Value::as_str)
                .unwrap_or(code)
                .into(),
        ),
        message: tinyinference_core::sanitize::sanitize_api_error(message),
        ..Default::default()
    }));
    with_partial(error, response)
}

pub(super) fn with_partial(error: Error, response: ModelResponse) -> Error {
    let mut error = match error {
        Error::Provider(error) => *error,
        _ => ProviderError {
            provider: "perplexity".into(),
            code: Some("invalid_response".into()),
            message: "invalid Perplexity response".into(),
            ..Default::default()
        },
    };
    error.model = response.execution.as_ref().map(|e| e.model.clone());
    error.stop_reason = response.finish_reason.clone();
    error.partial_message = Some(response.message.clone());
    error.partial_response = Some(Box::new(response));
    Error::Provider(Box::new(error))
}

pub(super) fn project(response: &mut ModelResponse) {
    response.message.content.clear();
    response.message.tool_calls.clear();
    response.message.id = None;
    if let Some(execution) = &response.execution {
        response.message.origin = Some(origin(&execution.model));
        response.finish_reason = Some(match execution.status {
            ExecutionStatus::Incomplete => execution
                .incomplete_reason
                .clone()
                .unwrap_or_else(|| "incomplete".into()),
            ExecutionStatus::Completed => "stop".into(),
            _ => String::from(execution.status.clone()),
        });
    }
    for item in &response.output {
        match &item.kind {
            ModelOutputKind::Message { role, content } if role == "assistant" => {
                if response.message.id.is_none() {
                    response.message.id = item.id.clone();
                }
                response
                    .message
                    .content
                    .extend(content.iter().map(|part| part.block.clone()));
            }
            ModelOutputKind::Reasoning { content } => {
                response.message.content.extend(content.clone())
            }
            ModelOutputKind::FunctionCall { call } => {
                response.message.tool_calls.push(call.clone())
            }
            _ => {}
        }
    }
    if !response.message.tool_calls.is_empty()
        && response
            .execution
            .as_ref()
            .is_some_and(|e| e.status == ExecutionStatus::Completed)
    {
        response.finish_reason = Some("tool_calls".into());
    }
    response.message.usage = response.usage;
}

pub(super) fn parse_item(
    value: &Value,
    index: usize,
    response_id: &str,
    model: &str,
) -> Result<ModelOutputItem> {
    let item_type = identity(value, "type")?;
    let id = optional::<String>(value, "id")?;
    let status = optional::<String>(value, "status")?.map(ExecutionStatus::from);
    let kind = match item_type.as_str() {
        "message" => {
            let role = identity(value, "role")?;
            let content = required_array(value, "content")?
                .iter()
                .map(parse_content)
                .collect::<Result<Vec<_>>>()?;
            ModelOutputKind::Message { role, content }
        }
        "reasoning" => {
            let mut content = Vec::new();
            for part in optional_array(value, "summary")?
                .iter()
                .chain(optional_array(value, "content")?.iter())
            {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    content.push(ContentBlock::Thinking {
                        text: text.into(),
                        signature: None,
                    });
                } else {
                    content.push(ContentBlock::ProviderExtension(part.clone()));
                }
            }
            if let Some(data) = optional::<String>(value, "encrypted_content")? {
                content.push(ContentBlock::RedactedThinking { data });
            }
            ModelOutputKind::Reasoning { content }
        }
        "function_call" => {
            let args = string(value, "arguments")?;
            let call_id = identity(value, "call_id")?;
            let name = identity(value, "name")?;
            let mut call = match serde_json::from_str::<Value>(&args) {
                Ok(arguments) => ToolCall::new(call_id, name, arguments),
                Err(_) => ToolCall::invalid(call_id, name, &args, "malformed argument JSON"),
            };
            call.replay = Some(ToolCallReplay {
                origin: origin(model),
                item_id: id.clone(),
                arguments: args,
                thought_signature: optional(value, "thought_signature")?,
            });
            ModelOutputKind::FunctionCall { call }
        }
        "search_results" | "people_search_results" => ModelOutputKind::Search {
            tool: if item_type == "search_results" {
                "web_search"
            } else {
                "people_search"
            }
            .into(),
            queries: strings(value, "queries")?,
            results: required_array(value, "results")?
                .iter()
                .map(search_source)
                .collect::<Result<_>>()?,
        },
        "image_search_results" => ModelOutputKind::ImageSearch {
            queries: strings(value, "queries")?,
            results: array(value, "results")?,
            error: optional(value, "error")?,
        },
        "fetch_url_results" => ModelOutputKind::Fetch {
            contents: array(value, "contents")?,
        },
        "finance_results" => ModelOutputKind::Finance {
            categories: strings(value, "categories")?,
            tickers: strings(value, "tickers")?,
            results: array(value, "results")?,
        },
        "sandbox_results" => ModelOutputKind::Sandbox {
            code: optional(value, "code")?,
            stdout: optional(value, "stdout")?,
            stderr: optional(value, "stderr")?,
            exit_code: optional(value, "exit_code")?,
            duration_ms: optional(value, "duration_ms")?,
            status: identity(value, "status")?,
        },
        "mcp_list_tools" => ModelOutputKind::ToolDiscovery {
            server: identity(value, "server_label")?,
            connector_id: optional(value, "connector_id")?,
            error: optional(value, "error")?,
            tools: required_array(value, "tools")?
                .iter()
                .map(|tool| remote_tool(tool, "input_schema"))
                .collect::<Result<_>>()?,
        },
        "mcp_call" => ModelOutputKind::RemoteCall {
            server: identity(value, "server_label")?,
            connector_id: optional(value, "connector_id")?,
            name: identity(value, "name")?,
            arguments: string(value, "arguments")?,
            output: optional(value, "output")?,
            error: optional(value, "error")?,
        },
        "tool_search_output" => {
            let tools = required_array(value, "tools")?
                .iter()
                .map(|namespace| {
                    Ok(RemoteToolNamespace {
                        name: identity(namespace, "name")?,
                        description: optional(namespace, "description")?,
                        tools: required_array(namespace, "tools")?
                            .iter()
                            .map(|tool| remote_tool(tool, "parameters"))
                            .collect::<Result<_>>()?,
                    })
                })
                .collect::<Result<_>>()?;
            ModelOutputKind::ToolSearch {
                execution: identity(value, "execution")?,
                call_id: optional(value, "call_id")?,
                arguments: optional(value, "arguments")?,
                tools,
            }
        }
        "share_file" => ModelOutputKind::GeneratedFiles {
            response_id: response_id.into(),
            metadata: value.clone(),
        },
        _ => ModelOutputKind::Extension {
            provider: "perplexity".into(),
            item_type,
            data: value.clone(),
        },
    };
    Ok(ModelOutputItem {
        index,
        id,
        status,
        kind,
    })
}

fn parse_content(value: &Value) -> Result<AnnotatedContent> {
    let kind = identity(value, "type")?;
    let block = match kind.as_str() {
        "output_text" => ContentBlock::Text(string(value, "text")?),
        "reasoning_text" | "summary_text" => ContentBlock::thinking(string(value, "text")?),
        _ => ContentBlock::ProviderExtension(value.clone()),
    };
    let annotations = optional_array(value, "annotations")?
        .iter()
        .map(|v| {
            if v.get("type").and_then(Value::as_str) == Some("url_citation") {
                let start = optional::<u64>(v, "start_index")?;
                let end = optional::<u64>(v, "end_index")?;
                if start.zip(end).is_some_and(|(a, b)| a > b) {
                    return Err(malformed("invalid citation range"));
                }
                Ok(OutputAnnotation::UrlCitation {
                    offset_unit: "provider_characters".into(),
                    url: identity(v, "url")?,
                    title: optional(v, "title")?,
                    start,
                    end,
                })
            } else {
                Ok(OutputAnnotation::Extension {
                    provider: "perplexity".into(),
                    data: v.clone(),
                })
            }
        })
        .collect::<Result<_>>()?;
    Ok(AnnotatedContent { block, annotations })
}

fn search_source(value: &Value) -> Result<SearchSource> {
    let id = match value.get("id") {
        Some(Value::String(id)) if !id.is_empty() => id.clone(),
        Some(Value::Number(id)) if id.as_u64().is_some() => id.to_string(),
        _ => {
            return Err(malformed(
                "search result id must be a string or nonnegative integer",
            ));
        }
    };
    Ok(SearchSource {
        id,
        url: identity(value, "url")?,
        title: string(value, "title")?,
        snippet: string(value, "snippet")?,
        source: optional(value, "source")?,
        date: optional(value, "date")?,
        last_updated: optional(value, "last_updated")?,
    })
}

fn remote_tool(value: &Value, field: &str) -> Result<RemoteToolDefinition> {
    let parameters = match value.get(field) {
        Some(schema) if schema.is_object() => schema.clone(),
        None if field == "parameters" => Value::Null,
        _ => return Err(malformed("remote tool schema must be an object")),
    };
    Ok(RemoteToolDefinition {
        name: identity(value, "name")?,
        description: optional(value, "description")?,
        parameters,
    })
}

type UsageDetails = (
    Usage,
    Option<ReportedCost>,
    BTreeMap<String, HostedToolUsage>,
);

fn parse_usage(value: &Value, exact: Option<&serde_json::value::RawValue>) -> Result<UsageDetails> {
    let mut usage = Usage {
        input_tokens: integer(value, "input_tokens")?,
        output_tokens: integer(value, "output_tokens")?,
        total_tokens: integer(value, "total_tokens")?,
        cache_read_tokens: optional::<u64>(
            &value["input_tokens_details"],
            "cache_read_input_tokens",
        )?
        .or(optional(&value["input_tokens_details"], "cached_tokens")?)
        .unwrap_or(0),
        cache_creation_tokens: optional::<u64>(
            &value["input_tokens_details"],
            "cache_creation_input_tokens",
        )?
        .unwrap_or(0),
        reasoning_tokens: optional::<u64>(&value["output_tokens_details"], "reasoning_tokens")?
            .unwrap_or(0),
        ..Usage::default()
    };
    let cost = if let Some(value) = value.get("cost").filter(|v| !v.is_null()) {
        let currency = identity(value, "currency")?;
        let mut components = BTreeMap::new();
        for (wire, name) in [
            ("input_cost", "input"),
            ("output_cost", "output"),
            ("cache_read_cost", "cache_read"),
            ("cache_creation_cost", "cache_creation"),
            ("tool_calls_cost", "tools"),
            ("total_cost", "total"),
        ] {
            if let Some(amount) = value.get(wire).filter(|v| !v.is_null()) {
                components.insert(
                    name.into(),
                    decimal(amount, raw_at(exact, &["cost", wire]).as_deref())?,
                );
            }
        }
        if currency == "USD"
            && let Some(total) = components.get("total")
        {
            usage.charged_amount = Some(ChargedAmount::usd_micros(usd_micros(total)?));
        }
        let mut tools = BTreeMap::new();
        if let Some(object) = value
            .get("tool_calls_cost_details")
            .filter(|v| !v.is_null())
        {
            for (name, amount) in object
                .as_object()
                .ok_or_else(|| malformed("invalid tool cost details"))?
            {
                tools.insert(
                    name.clone(),
                    decimal(
                        amount,
                        raw_at(exact, &["cost", "tool_calls_cost_details", name]).as_deref(),
                    )?,
                );
            }
        }
        Some(ReportedCost {
            currency,
            components,
            tools,
        })
    } else {
        None
    };
    let mut tools = BTreeMap::new();
    if let Some(object) = value.get("tool_calls_details").filter(|v| !v.is_null()) {
        for (name, detail) in object
            .as_object()
            .ok_or_else(|| malformed("invalid tool usage details"))?
        {
            if !detail.is_object() {
                return Err(malformed("invalid tool usage entry"));
            }
            tools.insert(
                name.clone(),
                HostedToolUsage {
                    invocations: optional(detail, "invocation")?,
                    cost_usd: detail
                        .get("cost_usd")
                        .filter(|v| !v.is_null())
                        .map(|amount| {
                            decimal(
                                amount,
                                raw_at(exact, &["tool_calls_details", name, "cost_usd"]).as_deref(),
                            )
                        })
                        .transpose()?,
                },
            );
        }
    }
    Ok((usage, cost, tools))
}

fn raw_at(
    raw: Option<&serde_json::value::RawValue>,
    path: &[&str],
) -> Option<Box<serde_json::value::RawValue>> {
    let (key, tail) = path.split_first()?;
    let mut fields: BTreeMap<String, Box<serde_json::value::RawValue>> =
        serde_json::from_str(raw?.get()).ok()?;
    let value = fields.remove(*key)?;
    if tail.is_empty() {
        Some(value)
    } else {
        raw_at(Some(&value), tail)
    }
}

fn decimal(value: &Value, exact: Option<&serde_json::value::RawValue>) -> Result<String> {
    match value {
        Value::Number(number) if number.as_f64().is_some_and(|n| n.is_finite() && n >= 0.0) => {
            let text = exact.map_or_else(|| number.to_string(), |v| v.get().to_owned());
            let coefficient = text.split(['e', 'E']).next().unwrap_or_default();
            if text.starts_with('-') && coefficient.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
                return Err(malformed("reported cost must not be negative"));
            }
            Ok(text)
        }
        _ => Err(malformed(
            "reported cost must be a nonnegative finite number",
        )),
    }
}

fn usd_micros(decimal: &str) -> Result<i64> {
    let unsigned = decimal.trim_start_matches('-');
    let (coefficient, exponent) = match unsigned.split_once(['e', 'E']) {
        Some((coefficient, exponent)) => (
            coefficient,
            exponent
                .parse::<i32>()
                .map_err(|_| malformed("invalid cost exponent"))?,
        ),
        None => (unsigned, 0),
    };
    let fraction = coefficient
        .split_once('.')
        .map_or(0, |(_, digits)| digits.len());
    let digits = coefficient.replace('.', "");
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Ok(0);
    }
    let scale = i64::from(exponent) + 6 - fraction as i64;
    let cut = digits.len() as i64 + scale;
    if cut < 0 {
        return Ok(0);
    }
    if cut > 19 {
        return Err(malformed("reported cost overflows micro-units"));
    }
    let cut = cut as usize;
    let (whole, round_up) = if cut >= digits.len() {
        let whole = digits
            .parse::<i64>()
            .ok()
            .and_then(|v| v.checked_mul(10_i64.pow((cut - digits.len()) as u32)))
            .ok_or_else(|| malformed("reported cost overflows micro-units"))?;
        (whole, false)
    } else {
        let whole = if cut == 0 {
            0
        } else {
            digits[..cut]
                .parse::<i64>()
                .map_err(|_| malformed("reported cost overflows micro-units"))?
        };
        (whole, digits.as_bytes()[cut] >= b'5')
    };
    whole
        .checked_add(i64::from(round_up))
        .ok_or_else(|| malformed("reported cost overflows micro-units"))
}

fn origin(model: &str) -> MessageOrigin {
    MessageOrigin {
        provider: "perplexity".into(),
        api: "agent".into(),
        model: model.into(),
    }
}
fn integer(value: &Value, field: &str) -> Result<u64> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("token usage must contain nonnegative integers"))
}
fn identity(value: &Value, field: &str) -> Result<String> {
    let text = string(value, field)?;
    if text.is_empty() {
        Err(malformed("required identity must not be empty"))
    } else {
        Ok(text)
    }
}
fn string(value: &Value, field: &str) -> Result<String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| malformed("required string is missing or invalid"))
}
fn optional<T: DeserializeOwned>(value: &Value, field: &str) -> Result<Option<T>> {
    value
        .get(field)
        .filter(|v| !v.is_null())
        .map(|v| serde_json::from_value(v.clone()).map_err(|_| malformed("invalid optional field")))
        .transpose()
}
fn required_array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value]> {
    value
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| malformed("required array is missing or invalid"))
}
fn optional_array<'a>(value: &'a Value, field: &str) -> Result<&'a [Value]> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(&[]),
        Some(Value::Array(v)) => Ok(v),
        _ => Err(malformed("invalid optional array")),
    }
}
fn array<T: DeserializeOwned>(value: &Value, field: &str) -> Result<Vec<T>> {
    required_array(value, field)?
        .iter()
        .map(|v| {
            serde_json::from_value(v.clone()).map_err(|_| malformed("malformed known tool result"))
        })
        .collect()
}
fn strings(value: &Value, field: &str) -> Result<Vec<String>> {
    optional_array(value, field)?
        .iter()
        .map(|v| {
            v.as_str()
                .map(str::to_owned)
                .ok_or_else(|| malformed("expected string array"))
        })
        .collect()
}

pub(super) fn malformed(message: &str) -> Error {
    Error::Provider(Box::new(ProviderError {
        provider: "perplexity".into(),
        code: Some("invalid_response".into()),
        message: message.into(),
        ..Default::default()
    }))
}

#[cfg(test)]
#[path = "response_tests.rs"]
mod tests;
