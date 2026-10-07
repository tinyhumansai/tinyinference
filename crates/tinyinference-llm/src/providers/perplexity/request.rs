use std::collections::{BTreeMap, HashSet};

use super::config::*;
use crate::message::{ContentBlock, MediaRef, Message, replay_system_state};
use crate::model::{ModelRequest, ResponseFormat, ToolChoice};
use crate::tool::{ToolCall, ToolResultContext};
use crate::{Error, Result};
use base64::Engine;
use serde_json::{Map, Value, json};

pub(super) fn effective_options(
    config: &PerplexityConfig,
    request: &ModelRequest,
) -> Result<PerplexityOptions> {
    validate_options_before_serialization(&config.options)?;
    let mut value = serde_json::to_value(&config.options)?;
    if !request.provider_options.is_null() {
        let root = request
            .provider_options
            .as_object()
            .ok_or_else(|| invalid("provider options must be an object"))?;
        if root.keys().any(|key| key != "perplexity") {
            return Err(Error::Unsupported(
                "use typed PerplexityOptions for provider settings".into(),
            ));
        }
        if let Some(override_value) = root.get("perplexity") {
            if override_value.get("profile").is_some()
                || override_value.get("skills").is_some()
                || override_value
                    .get("tools")
                    .and_then(Value::as_array)
                    .is_some_and(|tools| {
                        tools
                            .iter()
                            .any(|tool| tool.get("require_approval").is_some())
                    })
            {
                return Err(Error::Unsupported("profiles, skills and remote approval controls are not supported by this adapter".into()));
            }
            let overrides: PerplexityOptions = serde_json::from_value(override_value.clone())
                .map_err(|_| invalid("unknown or invalid Perplexity option"))?;
            if let (Some(base), Some(overrides)) = (
                value.as_object_mut(),
                serde_json::to_value(overrides)?.as_object(),
            ) {
                base.extend(overrides.clone());
            }
        }
    }
    serde_json::from_value(value).map_err(|_| invalid("invalid Perplexity options"))
}

pub(super) fn build(
    config: &PerplexityConfig,
    request: &ModelRequest,
    streaming: bool,
    submit: bool,
) -> Result<Value> {
    if request.seed.is_some() || !request.stop_sequences.is_empty() {
        return Err(Error::Unsupported(
            "Perplexity does not support seed or stop sequences".into(),
        ));
    }
    if request
        .reasoning
        .as_ref()
        .is_some_and(|r| r.budget_tokens.is_some() || r.summary.is_some())
    {
        return Err(Error::Unsupported(
            "Perplexity does not support reasoning budgets or summaries in requests".into(),
        ));
    }
    if request.required_capabilities.is_some() {
        return Err(Error::Unsupported(
            "Perplexity model capabilities have not been resolved authoritatively".into(),
        ));
    }
    let options = effective_options(config, request)?;
    let mut selection = options
        .selection
        .clone()
        .unwrap_or_else(|| config.selection.clone());
    if let Some(model) = &request.model {
        match &mut selection {
            PerplexitySelection::Model(selected) => *selected = model.clone(),
            PerplexitySelection::Preset {
                model: selected,
                models: None,
                ..
            } => *selected = Some(model.clone()),
            _ => {
                return Err(invalid(
                    "a model override is ambiguous with a fallback chain",
                ));
            }
        }
    }
    selection.validate()?;
    let target = selected_model(&selection);
    let mut body = Map::new();
    match &selection {
        PerplexitySelection::Model(model) => {
            body.insert("model".into(), json!(model));
        }
        PerplexitySelection::Models(models) => {
            body.insert("models".into(), json!(models));
        }
        PerplexitySelection::Preset {
            name,
            model,
            models,
        } => {
            body.insert("preset".into(), json!(name));
            if let Some(model) = model {
                body.insert("model".into(), json!(model));
            }
            if let Some(models) = models {
                body.insert("models".into(), json!(models));
            }
        }
    }
    let cap = request.max_tokens.or(options.max_output_tokens);
    let needs_cap = match &selection {
        PerplexitySelection::Model(m) => m.starts_with("anthropic/"),
        PerplexitySelection::Models(ms) => ms.iter().any(|m| m.starts_with("anthropic/")),
        PerplexitySelection::Preset { model, models, .. } => {
            model.as_ref().is_some_and(|m| m.starts_with("anthropic/"))
                || models
                    .as_ref()
                    .is_some_and(|ms| ms.iter().any(|m| m.starts_with("anthropic/")))
        }
    };
    if cap == Some(0) || (needs_cap && cap.is_none()) {
        return Err(invalid(
            "an explicit positive output cap is required for Anthropic models",
        ));
    }
    if let Some(cap) = cap {
        body.insert("max_output_tokens".into(), json!(cap));
    }
    if let Some(steps) = options.max_steps {
        if !(1..=100).contains(&steps) {
            return Err(invalid("max_steps must be between 1 and 100"));
        }
        body.insert("max_steps".into(), json!(steps));
    }
    let background = submit || options.background == Some(true);
    if background && !streaming && !submit {
        return Err(Error::Unsupported(
            "use submit_background or stream for background runs".into(),
        ));
    }
    if background && options.store == Some(false) {
        return Err(invalid("background runs require retrievable storage"));
    }
    if background {
        body.insert("background".into(), json!(true));
    }
    body.insert("stream".into(), json!(streaming));
    if let Some(store) = options.store {
        body.insert("store".into(), json!(store));
    }
    if let Some(language) = &options.language_preference {
        if language.len() != 2 || !language.bytes().all(|b| b.is_ascii_alphabetic()) {
            return Err(invalid("language preference must be a two-letter code"));
        }
        body.insert("language_preference".into(), json!(language));
    }
    let effort = options.reasoning_effort.as_deref().or_else(|| {
        request
            .reasoning
            .as_ref()
            .and_then(|r| r.effort.map(|e| e.as_str()))
    });
    if let Some(effort) = effort {
        if !["minimal", "low", "medium", "high", "xhigh", "max"].contains(&effort) {
            return Err(Error::Unsupported(
                "unsupported Perplexity reasoning effort".into(),
            ));
        }
        body.insert("reasoning".into(), json!({"effort":effort}));
    }
    for (key, value, max) in [
        ("temperature", request.temperature, 2.0),
        ("top_p", request.top_p, 1.0),
    ] {
        if let Some(value) = value {
            if !value.is_finite() || !(0.0..=max).contains(&value) {
                return Err(invalid("sampling setting is outside provider bounds"));
            }
            body.insert(key.into(), json!(value));
        }
    }
    if let Some(format) = &request.response_format {
        let schema = match format {
            ResponseFormat::Text => None,
            ResponseFormat::JsonObject => Some(("response", json!({"type":"object"}))),
            ResponseFormat::JsonSchema { name, schema } | ResponseFormat::Auto { name, schema } => {
                Some((name.as_str(), schema.clone()))
            }
        };
        if let Some((name, schema)) = schema {
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                || !schema.is_object()
            {
                return Err(invalid("invalid JSON schema name or shape"));
            }
            body.insert(
                "response_format".into(),
                json!({"type":"json_schema","json_schema":{"name":name,"schema":schema}}),
            );
        }
    }
    let (instructions, system_tools) = replay_system_state(&request.messages);
    if !instructions.is_empty() {
        body.insert("instructions".into(), json!(instructions));
    }
    let mut tools = match &options.tools {
        Some(tools) => encode_hosted_tools(tools, &config.remote_credentials)?,
        None => Vec::new(),
    };
    let mut functions = BTreeMap::new();
    for tool in system_tools {
        functions.insert(tool.name.clone(), tool);
    }
    let mut explicit_names = HashSet::new();
    for tool in &request.tools {
        if tool.name.trim().is_empty() || !explicit_names.insert(&tool.name) {
            return Err(invalid("duplicate or blank function name"));
        }
        functions.insert(tool.name.clone(), tool.clone());
    }
    for tool in functions.values() {
        if !tool.parameters.is_object() || !tool.format.is_json() {
            return Err(Error::Unsupported(
                "Perplexity functions require native JSON schemas".into(),
            ));
        }
        tools.push(json!({"type":"function","name":tool.name,"description":tool.description,"parameters":tool.parameters}));
    }
    if options.tools.is_some() || !tools.is_empty() {
        body.insert("tools".into(), json!(tools));
    }
    let choice = options
        .tool_choice
        .as_ref()
        .map(|choice| match choice {
            PerplexityToolChoice::Auto => json!("auto"),
            PerplexityToolChoice::None => json!("none"),
            PerplexityToolChoice::Required => json!("required"),
            PerplexityToolChoice::Function(name) => json!({"type":"function","name":name}),
            PerplexityToolChoice::Hosted(kind) => json!({"type":kind}),
        })
        .or_else(|| match &request.tool_choice {
            ToolChoice::Auto => None,
            ToolChoice::None => Some(json!("none")),
            ToolChoice::Required => Some(json!("required")),
            ToolChoice::Tool(name) => Some(json!({"type":"function","name":name})),
        });
    if let Some(choice) = choice {
        if let Some(name) = choice.get("name").and_then(Value::as_str) {
            if !functions.contains_key(name) {
                return Err(invalid("selected function is not declared"));
            }
        } else if let Some(kind) = choice.get("type").and_then(Value::as_str) {
            if ![
                "web_search",
                "image_search",
                "fetch_url",
                "finance_search",
                "people_search",
                "sandbox",
            ]
            .contains(&kind)
            {
                return Err(invalid("unsupported forced hosted tool"));
            }
            if !matches!(selection, PerplexitySelection::Preset { .. })
                && !body
                    .get("tools")
                    .and_then(Value::as_array)
                    .is_some_and(|tools| {
                        tools
                            .iter()
                            .any(|tool| tool.get("type").and_then(Value::as_str) == Some(kind))
                    })
            {
                return Err(invalid("selected hosted tool is not declared"));
            }
        }
        body.insert("tool_choice".into(), choice);
    }
    if let Some(id) = &request.continuation_id {
        if id.trim().is_empty() {
            return Err(invalid("continuation id must not be blank"));
        }
        body.insert("previous_response_id".into(), json!(id));
    }
    body.insert("input".into(), json!(input_items(request, target)?));
    Ok(Value::Object(body))
}

fn selected_model(selection: &PerplexitySelection) -> Option<&str> {
    match selection {
        PerplexitySelection::Model(model) => Some(model),
        PerplexitySelection::Preset {
            model: Some(model), ..
        } => Some(model),
        _ => None,
    }
}

fn input_items(request: &ModelRequest, target: Option<&str>) -> Result<Vec<Value>> {
    let mut input = Vec::new();
    let mut calls: BTreeMap<&str, &ToolCall> = BTreeMap::new();
    let mut results = HashSet::new();
    let mut documents = 0;
    for message in &request.messages {
        match message {
            Message::System(_) | Message::Custom(_) => {},
            Message::User(user) => input.push(json!({"type":"message","role":"user","content":content(&user.content,true,&mut documents)?})),
            Message::Assistant(assistant) => {
                if request.continuation_id.is_some() { return Err(invalid("stateful continuation accepts new input, not replayed assistant history")); }
                if !assistant.content.is_empty() { input.push(json!({"type":"message","role":"assistant","content":content(&assistant.content,false,&mut documents)?})); }
                for call in &assistant.tool_calls {
                    if call.id.is_empty() || call.name.is_empty() || calls.insert(&call.id,call).is_some() { return Err(invalid("duplicate or missing function call identity")); }
                    let mut item = json!({"type":"function_call","call_id":call.id,"name":call.name,"arguments":call.arguments.to_string()});
                    if let Some(replay) = &call.replay {
                        validate_replay(replay,target)?;
                        let unchanged = if call.invalid.is_some() { call.arguments.as_str()==Some(replay.arguments.as_str()) }
                            else { serde_json::from_str::<Value>(&replay.arguments).ok().as_ref()==Some(&call.arguments) };
                        if !unchanged { return Err(invalid("replayed function arguments were changed")); }
                        item["arguments"] = json!(replay.arguments);
                        if let Some(signature) = &replay.thought_signature { item["thought_signature"]=json!(signature); }
                    } else if call.invalid.is_some() {
                        item["arguments"] = json!(call.arguments.as_str().ok_or_else(|| invalid("invalid function arguments lost their raw text"))?);
                    }
                    input.push(item);
                }
            },
            Message::Tool(result) => {
                if result.tool_call_id.is_empty() || !results.insert(&result.tool_call_id) { return Err(invalid("duplicate or missing function result id")); }
                let known = calls.get(result.tool_call_id.as_str());
                let inherited = known.map(|call| ToolResultContext {name:call.name.clone(),replay:call.replay.clone()});
                if let (Some(known),Some(context)) = (&inherited,&result.call_context)
                    && known!=context { return Err(invalid("function result context conflicts with its call")); }
                let context = result.call_context.as_ref().or(inherited.as_ref()).ok_or_else(|| invalid("function result has no matching call context"))?;
                if known.is_none() && request.continuation_id.is_none() { return Err(invalid("function result has no preceding call")); }
                let mut item = json!({"type":"function_call_output","call_id":result.tool_call_id,"name":context.name,
                    "output":content(&result.content,false,&mut documents)?});
                if let Some(replay) = &context.replay {
                    validate_replay(replay,target)?;
                    if let Some(signature)=&replay.thought_signature { item["thought_signature"]=json!(signature); }
                }
                input.push(item);
            },
        }
    }
    if input.is_empty() {
        return Err(invalid("Perplexity requires input messages"));
    }
    if documents > 30 {
        return Err(invalid("Perplexity accepts at most 30 input documents"));
    }
    Ok(input)
}

fn validate_replay(replay: &crate::tool::ToolCallReplay, target: Option<&str>) -> Result<()> {
    if replay.origin.provider != "perplexity"
        || replay.origin.api != "agent"
        || (replay.thought_signature.is_some() && target != Some(replay.origin.model.as_str()))
    {
        return Err(Error::Unsupported("signed function replay requires its originating Perplexity model; pin the returned model when using presets".into()));
    }
    Ok(())
}

fn content(blocks: &[ContentBlock], user: bool, documents: &mut usize) -> Result<Vec<Value>> {
    blocks.iter().map(|block| match block {
        ContentBlock::Text(text) => Ok(json!({"type":"input_text","text":text})),
        ContentBlock::Json(value) => Ok(json!({"type":"input_text","text":value.to_string()})),
        ContentBlock::Image(image) => {
            validate_image(&image.url)?;
            Ok(json!({"type":"input_image","image_url":image.url}))
        },
        ContentBlock::Document(reference) if user => {
            *documents+=1;
            match reference {
                MediaRef::Url {url,media_type} => {
                    require_https(url)?;
                    let mut part=json!({"type":"input_file","file_url":url});
                    if let Some(mime)=media_type { part["filename"]=json!(document_filename(mime)?); }
                    Ok(part)
                },
                MediaRef::Base64 {data,media_type} => {
                    base64::engine::general_purpose::STANDARD.decode(data).map_err(|_| invalid("invalid inline document base64"))?;
                    Ok(json!({"type":"input_file","file_data":data,"filename":document_filename(media_type)?}))
                },
                MediaRef::Path {..} => Err(Error::Unsupported("resolve local documents before calling Perplexity".into())),
            }
        },
        _ => Err(Error::Unsupported("this content block cannot be replayed through Perplexity; use server continuation for hosted output".into())),
    }).collect()
}

fn document_filename(mime: &str) -> Result<&'static str> {
    match mime {
        "application/pdf" => Ok("document.pdf"),
        "application/msword" => Ok("document.doc"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
            Ok("document.docx")
        }
        "text/plain" => Ok("document.txt"),
        "application/rtf" | "text/rtf" => Ok("document.rtf"),
        _ => Err(Error::Unsupported("unsupported document media type".into())),
    }
}

fn validate_image(url: &str) -> Result<()> {
    if let Some((prefix, data)) = url.split_once(",")
        && prefix.starts_with("data:image/")
        && prefix.ends_with(";base64")
    {
        base64::engine::general_purpose::STANDARD
            .decode(data)
            .map_err(|_| invalid("invalid inline image base64"))?;
        return Ok(());
    }
    require_https(url)
}

fn require_https(url: &str) -> Result<()> {
    if reqwest::Url::parse(url).is_ok_and(|u| {
        u.scheme() == "https"
            && u.host_str().is_some()
            && u.username().is_empty()
            && u.password().is_none()
    }) {
        Ok(())
    } else {
        Err(invalid(
            "remote media and MCP endpoints require HTTPS without embedded credentials",
        ))
    }
}

fn encode_hosted_tools(
    tools: &[PerplexityTool],
    credentials: &BTreeMap<String, PerplexityRemoteCredentials>,
) -> Result<Vec<Value>> {
    let mut identities = HashSet::new();
    let mut labels = HashSet::new();
    tools
        .iter()
        .map(|tool| {
            let mut value = serde_json::to_value(tool)?;
            let kind = value["type"].as_str().unwrap_or_default().to_owned();
            let label = match tool {
                PerplexityTool::Mcp { server_label, .. }
                | PerplexityTool::Connector { server_label, .. } => Some(server_label.as_str()),
                _ => None,
            };
            if !identities.insert((kind.clone(), label)) {
                return Err(invalid("duplicate hosted tool configuration"));
            }
            if let Some(label) = label
                && (label.is_empty()
                    || label.len() > 64
                    || !label
                        .bytes()
                        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
                    || !labels.insert(label))
            {
                return Err(invalid("invalid or duplicate remote server label"));
            }
            match tool {
                PerplexityTool::WebSearch { options } => validate_web_search(options)?,
                PerplexityTool::ImageSearch {
                    max_results,
                    filters,
                } => {
                    if max_results.is_some_and(|v| !(1..=30).contains(&v)) {
                        return Err(invalid("image search result limit must be 1..30"));
                    }
                    if let Some(filters) = filters {
                        validate_domains(filters.domain_filter.as_deref(), 10)?;
                        if filters.format_filter.as_ref().is_some_and(|formats| {
                            formats.len() > 10
                                || formats.iter().any(|v| {
                                    !["bmp", "gif", "jpeg", "png", "webp", "svg"]
                                        .contains(&v.as_str())
                                })
                        }) {
                            return Err(invalid("invalid image format filter"));
                        }
                    }
                }
                PerplexityTool::FetchUrl { max_urls }
                    if max_urls.is_some_and(|v| !(1..=10).contains(&v)) =>
                {
                    return Err(invalid("fetch URL limit must be 1..10"));
                }
                PerplexityTool::Mcp {
                    server_url,
                    server_label,
                    ..
                } => {
                    require_https(server_url)?;
                    if let Some(secret) = credentials.get(server_label) {
                        if reqwest::Url::parse(&secret.server_url).ok()
                            != reqwest::Url::parse(server_url).ok()
                        {
                            return Err(invalid(
                                "remote credentials belong to a different endpoint",
                            ));
                        }
                        if let Some(token) = &secret.authorization {
                            value["authorization"] = json!(token);
                        }
                        if !secret.headers.is_empty() {
                            value["headers"] = json!(secret.headers);
                        }
                    }
                }
                PerplexityTool::Connector { id, .. } if id.trim().is_empty() => {
                    return Err(invalid("connector id must not be blank"));
                }
                _ => {}
            }
            Ok(value)
        })
        .collect()
}

fn validate_web_search(options: &PerplexityWebSearch) -> Result<()> {
    if options.max_results.is_some_and(|v| !(1..=50).contains(&v))
        || options.max_tokens == Some(0)
        || options.max_tokens_per_page == Some(0)
        || options
            .search_type
            .as_deref()
            .is_some_and(|v| !["web", "fast"].contains(&v))
        || options
            .search_context_size
            .as_deref()
            .is_some_and(|v| !["low", "medium", "high"].contains(&v))
    {
        return Err(invalid("invalid web search budget or type"));
    }
    if let Some(location) = &options.user_location
        && (location.country.len() != 2
            || !location.country.bytes().all(|b| b.is_ascii_alphabetic())
            || location.latitude.is_some() != location.longitude.is_some()
            || location
                .latitude
                .is_some_and(|v| !v.is_finite() || !(-90.0..=90.0).contains(&v))
            || location
                .longitude
                .is_some_and(|v| !v.is_finite() || !(-180.0..=180.0).contains(&v)))
    {
        return Err(invalid("invalid search location"));
    }
    if let Some(filters) = &options.filters {
        validate_domains(filters.search_domain_filter.as_deref(), 20)?;
        if filters
            .search_recency_filter
            .as_deref()
            .is_some_and(|v| !["hour", "day", "week", "month", "year"].contains(&v))
        {
            return Err(invalid("invalid search recency"));
        }
        for date in [
            &filters.search_after_date_filter,
            &filters.search_before_date_filter,
            &filters.last_updated_after_filter,
            &filters.last_updated_before_filter,
        ]
        .into_iter()
        .flatten()
        {
            let parts = date
                .split('/')
                .map(str::parse::<u32>)
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|_| invalid("invalid search date"))?;
            let valid = if let [month, day, year] = parts.as_slice() {
                let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
                let days = match month {
                    1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
                    4 | 6 | 9 | 11 => 30,
                    2 => {
                        if leap {
                            29
                        } else {
                            28
                        }
                    }
                    _ => 0,
                };
                *year >= 1 && *day >= 1 && *day <= days
            } else {
                false
            };
            if !valid {
                return Err(invalid("invalid search date"));
            }
        }
    }
    Ok(())
}

fn validate_domains(domains: Option<&[String]>, max: usize) -> Result<()> {
    if let Some(domains) = domains {
        let exclude = domains.first().is_some_and(|s| s.starts_with('-'));
        if domains.len() > max
            || domains.iter().any(|s| {
                s.trim().is_empty() || s == "-" || s.len() > 253 || s.starts_with('-') != exclude
            })
        {
            return Err(invalid("invalid or mixed search domain filters"));
        }
    }
    Ok(())
}

fn invalid(message: &str) -> Error {
    Error::Validation(message.into())
}

pub(super) fn validate_options_before_serialization(options: &PerplexityOptions) -> Result<()> {
    if let Some(tools) = &options.tools {
        encode_hosted_tools(tools, &BTreeMap::new())?;
    }
    Ok(())
}

pub(super) fn validate_hook(original: &Value, changed: &Value) -> Result<()> {
    let old = original
        .as_object()
        .ok_or_else(|| invalid("invalid original request"))?;
    let new = changed
        .as_object()
        .ok_or_else(|| invalid("payload hook must leave an object"))?;
    for key in old.keys().chain(new.keys()) {
        if old.get(key) != new.get(key)
            && !["max_steps", "temperature", "top_p", "max_output_tokens"].contains(&key.as_str())
        {
            return Err(Error::Unsupported("payload hooks may adjust budgets/sampling only; use typed options for other settings".into()));
        }
    }
    for key in ["max_steps", "max_output_tokens"] {
        if let Some(value) = new.get(key) {
            let max = if key == "max_steps" {
                100
            } else {
                u32::MAX as u64
            };
            if value.as_u64().is_none_or(|v| v == 0 || v > max) {
                return Err(invalid("payload hook produced an invalid budget"));
            }
        }
    }
    if old.contains_key("max_output_tokens") && !new.contains_key("max_output_tokens") {
        return Err(invalid("payload hook removed the output budget"));
    }
    for (key, max) in [("temperature", 2.0), ("top_p", 1.0)] {
        if let Some(value) = new.get(key)
            && value
                .as_f64()
                .is_none_or(|v| !v.is_finite() || !(0.0..=max).contains(&v))
        {
            return Err(invalid("payload hook produced invalid sampling settings"));
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod tests;
