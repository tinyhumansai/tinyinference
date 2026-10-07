use super::{
    response,
    transport::{self, Transport},
};
use crate::message::MessageDelta;
use crate::model::*;
use crate::tool::ToolDelta;
use crate::{Error, Result};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, VecDeque},
    pin::Pin,
    sync::Arc,
};
use tokio::time::Instant;

pub(super) fn open(
    inner: Arc<Transport>,
    incoming: reqwest::Response,
    deadline: Instant,
    seed: Option<ModelExecution>,
    after: Option<u64>,
) -> Result<ModelStream> {
    if !incoming
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.to_ascii_lowercase().starts_with("text/event-stream"))
    {
        return Err(transport::failure(
            "invalid_stream",
            "expected an event stream",
            None,
        ));
    }
    let mut snapshot = ModelResponse::assistant("");
    snapshot.message.content.clear();
    snapshot.execution = seed;
    let state = State {
        bytes: Box::pin(incoming.bytes_stream().map(|r| {
            r.map_err(|_| {
                transport::failure(
                    "stream_interrupted",
                    "stream connection was interrupted",
                    None,
                )
            })
        })),
        inner,
        deadline,
        buffer: Vec::new(),
        data: Vec::new(),
        event_name: None,
        queue: VecDeque::from([ModelStreamItem::Started]),
        snapshot,
        items: BTreeMap::new(),
        text: BTreeMap::new(),
        arguments: BTreeMap::new(),
        sequence: after,
        resumed: after.is_some(),
        done: false,
        received: 0,
        progress: Vec::new(),
    };
    Ok(ModelStream::new(Box::pin(futures::stream::unfold(
        state, next,
    ))))
}

struct State {
    bytes: Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>,
    inner: Arc<Transport>,
    deadline: Instant,
    buffer: Vec<u8>,
    data: Vec<u8>,
    event_name: Option<String>,
    queue: VecDeque<ModelStreamItem>,
    snapshot: ModelResponse,
    items: BTreeMap<usize, Value>,
    text: BTreeMap<(usize, &'static str, usize), String>,
    arguments: BTreeMap<usize, String>,
    sequence: Option<u64>,
    resumed: bool,
    done: bool,
    received: usize,
    progress: Vec<ModelProgress>,
}

impl State {
    fn record_progress(&mut self, progress: ModelProgress) {
        self.progress.push(progress.clone());
        self.emit(ModelOutputEvent::Progress(progress));
    }
    fn emit(&mut self, event: ModelOutputEvent) {
        self.queue.push_back(ModelStreamItem::OutputEvent(event));
    }

    fn partial(&self) -> ModelResponse {
        let mut partial = self.snapshot.clone();
        if let Some(execution) = &mut partial.execution {
            execution.progress = self.progress.clone();
        }
        let run = partial.execution.as_ref();
        let id = run.map(|v| v.id.as_str()).unwrap_or_default();
        let model = run.map(|v| v.model.as_str()).unwrap_or_default();
        partial.output = self
            .items
            .iter()
            .filter_map(|(index, item)| response::parse_item(item, *index, id, model).ok())
            .collect();
        let mut raw = partial.raw.take().unwrap_or_else(|| json!({}));
        raw["output"] = Value::Array(self.items.values().cloned().collect());
        partial.raw = Some(raw);
        response::project(&mut partial);
        partial
    }

    fn fail(&mut self, error: Error) {
        let mut partial = self.partial();
        let error = match error {
            Error::Provider(mut error) => {
                if let Some(terminal) = error.partial_response.take() {
                    partial.execution = terminal.execution.or(partial.execution);
                    partial.usage = terminal.usage.or(partial.usage);
                    for item in terminal.output {
                        let compatible = terminal
                            .raw
                            .as_ref()
                            .and_then(|v| v.get("output"))
                            .and_then(|v| v.get(item.index))
                            .is_some_and(|raw| self.check_item(item.index, raw).is_ok());
                        if compatible {
                            if let Some(previous) =
                                partial.output.iter_mut().find(|v| v.index == item.index)
                            {
                                *previous = item;
                            } else {
                                partial.output.push(item);
                            }
                        }
                    }
                    partial.raw = terminal.raw.or(partial.raw);
                }
                Error::Provider(error)
            }
            error => error,
        };
        if let Some(execution) = &mut partial.execution {
            execution.progress = self.progress.clone();
            execution.sequence_number = self.sequence;
        }
        partial.output.sort_by_key(|item| item.index);
        response::project(&mut partial);
        let error = response::with_partial(error, partial);
        if let Error::Provider(error) = error {
            self.queue
                .push_back(ModelStreamItem::ProviderFailed(*error));
        }
        self.done = true;
        self.buffer.clear();
        self.data.clear();
    }

    fn frame(&mut self) -> Result<()> {
        if Instant::now() >= self.deadline {
            return Err(transport::timeout());
        }
        if self.data.is_empty() {
            self.event_name = None;
            return Ok(());
        }
        let bytes = std::mem::take(&mut self.data);
        if bytes.as_slice() == b"[DONE]" {
            return Err(response::malformed(
                "stream ended without an authoritative terminal response",
            ));
        }
        let decoded = response::decode(&bytes)?;
        let mut value = decoded.value;
        if !value.is_object() {
            return Err(response::malformed("stream event must be an object"));
        }
        if value.get("type").is_none()
            && let Some(name) = self.event_name.take()
        {
            value["type"] = json!(name);
        }
        self.event_name = None;
        self.event(self.inner.scrub_value(value), decoded.usage)
    }

    fn lines(&mut self) -> Result<()> {
        while let Some(end) = self.buffer.iter().position(|b| *b == b'\n') {
            let mut line = self.buffer.drain(..=end).collect::<Vec<_>>();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if line.is_empty() {
                self.frame()?;
                if self.done {
                    self.buffer.clear();
                    break;
                }
            } else if let Some(rest) = line.strip_prefix(b"data:") {
                let rest = rest.strip_prefix(b" ").unwrap_or(rest);
                if !self.data.is_empty() {
                    self.data.push(b'\n');
                }
                if rest.len()
                    > self
                        .inner
                        .config
                        .max_event_bytes
                        .saturating_sub(self.data.len())
                {
                    return Err(response::malformed("stream event exceeds byte limit"));
                }
                self.data.extend_from_slice(rest);
            } else if let Some(rest) = line.strip_prefix(b"event:") {
                self.event_name = Some(
                    std::str::from_utf8(rest)
                        .map_err(|_| response::malformed("invalid SSE event name"))?
                        .trim()
                        .into(),
                );
            }
        }
        if self.buffer.len().saturating_add(self.data.len()) > self.inner.config.max_event_bytes {
            return Err(response::malformed(
                "unterminated stream event exceeds byte limit",
            ));
        }
        Ok(())
    }

    fn event(
        &mut self,
        value: Value,
        usage: Option<Box<serde_json::value::RawValue>>,
    ) -> Result<()> {
        let event_type = value
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| response::malformed("stream event has no type"))?;
        let run_id = value
            .get("response")
            .and_then(|v| v.get("id"))
            .or_else(|| value.get("response_id"))
            .and_then(Value::as_str);
        if let (Some(expected), Some(actual)) = (self.snapshot.execution.as_ref(), run_id)
            && expected.id != actual
        {
            return Err(response::malformed("stream response identity changed"));
        }
        if let Some(sequence) = value.get("sequence_number") {
            let sequence = sequence
                .as_u64()
                .ok_or_else(|| response::malformed("invalid stream sequence number"))?;
            if self.sequence.is_some_and(|previous| sequence <= previous) {
                return Ok(());
            }
            self.sequence = Some(sequence);
        }
        if let Some(thought) = value.get("thought").filter(|v| !v.is_null()) {
            self.record_progress(ModelProgress::ReasoningText {
                text: thought
                    .as_str()
                    .ok_or_else(|| response::malformed("reasoning note must be text"))?
                    .into(),
            });
        }
        match event_type {
            "response.created" | "response.in_progress" => {
                let snapshot = response::parse(response::DecodedJson {
                    value: value
                        .get("response")
                        .cloned()
                        .ok_or_else(|| response::malformed("missing response snapshot"))?,
                    usage,
                })?;
                if let Some(raw) = &snapshot.raw
                    && let Some(items) = raw["output"].as_array()
                {
                    for (index, item) in items.iter().enumerate() {
                        self.items.insert(index, item.clone());
                    }
                }
                self.snapshot = snapshot;
            }
            "response.output_item.added" | "response.output_item.done" => {
                let index = self.index(&value)?;
                let item = value
                    .get("item")
                    .ok_or_else(|| response::malformed("missing output item"))?
                    .clone();
                self.check_item(index, &item)?;
                self.items.insert(index, item.clone());
                if event_type.ends_with(".added") {
                    self.emit(ModelOutputEvent::ItemStarted {
                        index,
                        id: item.get("id").and_then(Value::as_str).map(str::to_owned),
                        item_type: item
                            .get("type")
                            .and_then(Value::as_str)
                            .ok_or_else(|| response::malformed("missing item type"))?
                            .into(),
                    });
                } else {
                    self.complete_item(index, &item)?;
                }
            }
            "response.content_part.added" | "response.content_part.done" => {
                let index = self.index(&value)?;
                let part_index = content_index(&value)?;
                let part = value
                    .get("part")
                    .ok_or_else(|| response::malformed("missing content part"))?
                    .clone();
                self.part(index, part_index, &value)?;
                let mut item = self.items[&index].clone();
                item[content_area(&value)][part_index] = part;
                self.check_item(index, &item)?;
                self.items.insert(index, item);
                if event_type.ends_with(".done") {
                    self.complete_text(index, content_area(&value), part_index)?;
                }
            }
            "response.output_text.delta"
            | "response.reasoning_text.delta"
            | "response.reasoning_summary_text.delta" => {
                let index = self.index(&value)?;
                let part_index = content_index(&value)?;
                let delta = value
                    .get("delta")
                    .and_then(Value::as_str)
                    .ok_or_else(|| response::malformed("missing text delta"))?;
                let reasoning = event_type != "response.output_text.delta";
                let part = self.part(index, part_index, &value)?;
                if reasoning {
                    part["type"] = json!("reasoning_text");
                }
                let text = part
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned()
                    + delta;
                part["text"] = json!(text);
                self.text
                    .entry((index, content_area(&value), part_index))
                    .or_default()
                    .push_str(delta);
                self.queue
                    .push_back(ModelStreamItem::MessageDelta(if reasoning {
                        MessageDelta::reasoning(delta)
                    } else {
                        MessageDelta::text(delta)
                    }));
            }
            "response.output_text.done"
            | "response.reasoning_text.done"
            | "response.reasoning_summary_text.done" => {
                let index = self.index(&value)?;
                let part_index = content_index(&value)?;
                self.part(index, part_index, &value)?;
                let mut item = self.items[&index].clone();
                let part = &mut item[content_area(&value)][part_index];
                if event_type != "response.output_text.done" {
                    part["type"] = json!("reasoning_text");
                }
                part["text"] = json!(
                    value
                        .get("text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| response::malformed("missing final text"))?
                );
                self.check_item(index, &item)?;
                self.items.insert(index, item);
                self.complete_text(index, content_area(&value), part_index)?;
            }
            "response.output_text.annotation.added" => {
                let index = self.index(&value)?;
                let part_index = number(&value, "content_index")?;
                let annotation = value
                    .get("annotation")
                    .ok_or_else(|| response::malformed("missing annotation"))?
                    .clone();
                let part = self.part(index, part_index, &value)?;
                if part.get("annotations").is_none() {
                    part["annotations"] = json!([]);
                }
                part["annotations"]
                    .as_array_mut()
                    .ok_or_else(|| response::malformed("invalid annotations"))?
                    .push(annotation);
                let item = self.items[&index].clone();
                self.emit_normalized_item(index, &item)?;
            }
            "response.function_call_arguments.delta" => {
                let index = self.index(&value)?;
                let delta = value
                    .get("delta")
                    .and_then(Value::as_str)
                    .ok_or_else(|| response::malformed("missing argument delta"))?;
                let item = self.items.get_mut(&index).ok_or_else(|| {
                    response::malformed("function delta has no preceding call item")
                })?;
                item["arguments"] = json!(
                    item.get("arguments")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                        + delta
                );
                self.arguments.entry(index).or_default().push_str(delta);
                let call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| response::malformed("function delta has no call id"))?;
                self.queue
                    .push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                        call_id: call_id.into(),
                        content: delta.into(),
                        tool_name: item.get("name").and_then(Value::as_str).map(str::to_owned),
                        content_index: Some(index),
                    }));
            }
            "response.function_call_arguments.done" => {
                let index = self.index(&value)?;
                let args = value
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| response::malformed("missing final arguments"))?;
                let mut item = self
                    .items
                    .get(&index)
                    .ok_or_else(|| response::malformed("function completion has no call item"))?
                    .clone();
                item["arguments"] = json!(args);
                self.check_item(index, &item)?;
                self.items.insert(index, item.clone());
                self.complete_item(index, &item)?;
            }
            "response.reasoning.started" | "response.reasoning.stopped" => {
                self.record_progress(ModelProgress::ReasoningPhase {
                    active: event_type.ends_with(".started"),
                })
            }
            "response.reasoning.search_queries"
            | "response.reasoning.image_search_queries"
            | "response.reasoning.fetch_url_queries" => {
                let queries = serde_json::from_value(
                    value
                        .get(if event_type.contains("fetch_url") {
                            "urls"
                        } else {
                            "queries"
                        })
                        .cloned()
                        .ok_or_else(|| response::malformed("missing hosted queries"))?,
                )
                .map_err(|_| response::malformed("invalid hosted queries"))?;
                let tool = if event_type.contains("image_search") {
                    "image_search"
                } else if event_type.contains("fetch_url") {
                    "fetch_url"
                } else {
                    "web_search"
                };
                self.record_progress(ModelProgress::Queries {
                    tool: tool.into(),
                    queries,
                });
            }
            "response.completed"
            | "response.incomplete"
            | "response.cancelled"
            | "response.failed" => {
                let raw = value.get("response").cloned().ok_or_else(|| {
                    response::malformed("terminal event has no response snapshot")
                })?;
                let mut result = response::parse(response::DecodedJson {
                    value: raw.clone(),
                    usage,
                })?;
                if let Some(execution) = &mut result.execution {
                    execution.sequence_number = self.sequence;
                    execution.progress = self.progress.clone();
                }
                result = response::completed(result)?;
                self.snapshot.execution = result.execution.clone().map(|mut execution| {
                    execution.progress.clear();
                    execution
                });
                let items = raw["output"]
                    .as_array()
                    .ok_or_else(|| response::malformed("terminal output is not an array"))?;
                if !self.resumed && self.items.keys().any(|index| *index >= items.len()) {
                    return Err(response::malformed(
                        "terminal response dropped streamed items",
                    ));
                }
                for (index, item) in items.iter().enumerate() {
                    self.check_item(index, item)?;
                    self.items.insert(index, item.clone());
                    self.complete_item(index, item)?;
                }
                if let Some(usage) = result.usage {
                    self.queue.push_back(ModelStreamItem::UsageDelta(usage));
                }
                if let Some(observer) = &self.inner.config.on_response {
                    observer(&raw);
                }
                if Instant::now() >= self.deadline {
                    return Err(response::with_partial(transport::timeout(), result));
                }
                self.snapshot = result.clone();
                self.queue.push_back(ModelStreamItem::Completed(result));
                self.done = true;
            }
            "error" => {
                return Err(transport::failure(
                    "stream_failed",
                    &self.inner.scrub_text(
                        value
                            .get("message")
                            .and_then(Value::as_str)
                            .unwrap_or("provider stream error"),
                    ),
                    None,
                ));
            }
            "response.reasoning.search_results"
            | "response.reasoning.image_search_results"
            | "response.reasoning.fetch_url_results" => {
                let mut item = value.clone();
                item["type"] = json!(if event_type.contains("image_search") {
                    "image_search_results"
                } else if event_type.contains("fetch_url") {
                    "fetch_url_results"
                } else {
                    "search_results"
                });
                let run = self.snapshot.execution.as_ref().ok_or_else(|| {
                    response::malformed("hosted results arrived without response identity")
                })?;
                let result = response::parse_item(&item, 0, &run.id, &run.model)?;
                self.record_progress(ModelProgress::HostedActivity(Box::new(result.kind)));
            }
            _ => self.record_progress(ModelProgress::Extension {
                event_type: event_type.into(),
                data: value.clone(),
            }),
        }
        if !self.done
            && let Some(execution) = &mut self.snapshot.execution
        {
            execution.sequence_number = self.sequence;
            let execution = execution.clone();
            self.emit(ModelOutputEvent::Execution(Box::new(execution)));
        }
        Ok(())
    }

    fn index(&self, event: &Value) -> Result<usize> {
        if event.get("output_index").is_some() {
            return number(event, "output_index");
        }
        if let Some(id) = event.get("item_id").and_then(Value::as_str)
            && let Some((index, _)) = self
                .items
                .iter()
                .find(|(_, item)| item.get("id").and_then(Value::as_str) == Some(id))
        {
            return Ok(*index);
        }
        Err(response::malformed("event has no resolvable output index"))
    }

    fn check_item(&self, index: usize, item: &Value) -> Result<()> {
        if !self.resumed {
            for ((output, area, part), delivered) in &self.text {
                if *output == index
                    && !delivered.is_empty()
                    && !item
                        .get(*area)
                        .and_then(|v| v.get(*part))
                        .and_then(|v| v.get("text"))
                        .and_then(Value::as_str)
                        .is_some_and(|text| text.starts_with(delivered))
                {
                    return Err(response::malformed(
                        "final item conflicts with delivered text",
                    ));
                }
            }
            if let Some(delivered) = self.arguments.get(&index)
                && !delivered.is_empty()
                && !item
                    .get("arguments")
                    .and_then(Value::as_str)
                    .is_some_and(|args| args.starts_with(delivered))
            {
                return Err(response::malformed(
                    "final item conflicts with delivered arguments",
                ));
            }
        }
        if let Some(previous) = self.items.get(&index) {
            for key in ["id", "call_id", "type"] {
                if let (Some(a), Some(b)) = (previous.get(key), item.get(key))
                    && a != b
                {
                    return Err(response::malformed("output item identity changed"));
                }
            }
        }
        Ok(())
    }

    fn part(&mut self, index: usize, part_index: usize, event: &Value) -> Result<&mut Value> {
        if part_index > 4096 {
            return Err(response::malformed("content index exceeds bounds"));
        }
        let area = content_area(event);
        let reasoning = event
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|t| t.contains("reasoning"));
        let item = self.items.entry(index).or_insert_with(|| {
            if reasoning {
                json!({"type":"reasoning","id":event.get("item_id"),"content":[],"summary":[]})
            } else {
                json!({"type":"message","id":event.get("item_id"),"role":"assistant","content":[]})
            }
        });
        if let (Some(a), Some(b)) = (
            item.get("id").and_then(Value::as_str),
            event.get("item_id").and_then(Value::as_str),
        ) && a != b
        {
            return Err(response::malformed("text item identity changed"));
        }
        if matches!(
            item.get("type").and_then(Value::as_str),
            Some("message" | "reasoning")
        ) && item.get(area).is_none_or(Value::is_null)
        {
            item[area] = json!([]);
        }
        let content = item
            .get_mut(area)
            .and_then(Value::as_array_mut)
            .ok_or_else(|| response::malformed("text delta addressed a non-message item"))?;
        if part_index > content.len() && !self.resumed {
            return Err(response::malformed("content part index has a gap"));
        }
        while content.len() <= part_index {
            content.push(json!({"type":"output_text","text":"","annotations":[]}));
        }
        if !content[part_index].is_object() {
            return Err(response::malformed("content part must be an object"));
        }
        if content[part_index]
            .get("text")
            .is_some_and(|v| !v.is_string())
        {
            return Err(response::malformed("content text must be a string"));
        }
        Ok(&mut content[part_index])
    }

    fn complete_text(&mut self, index: usize, area: &'static str, part_index: usize) -> Result<()> {
        if self.resumed {
            return Ok(());
        }
        let part = &self.items[&index][area][part_index];
        let Some(text) = part.get("text").and_then(Value::as_str) else {
            return Ok(());
        };
        let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
        if !["output_text", "reasoning_text", "summary_text"].contains(&kind) {
            return Ok(());
        }
        let previous = self.text.entry((index, area, part_index)).or_default();
        let delta = text
            .strip_prefix(previous.as_str())
            .ok_or_else(|| response::malformed("final text conflicts with delivered text"))?;
        if !delta.is_empty() {
            self.queue
                .push_back(ModelStreamItem::MessageDelta(if kind == "output_text" {
                    MessageDelta::text(delta)
                } else {
                    MessageDelta::reasoning(delta)
                }));
        }
        *previous = text.into();
        Ok(())
    }

    fn emit_normalized_item(&mut self, index: usize, item: &Value) -> Result<()> {
        let run = self
            .snapshot
            .execution
            .as_ref()
            .ok_or_else(|| response::malformed("item arrived without response identity"))?;
        let item = response::parse_item(item, index, &run.id, &run.model)?;
        self.emit(ModelOutputEvent::Item(Box::new(item)));
        Ok(())
    }

    fn complete_item(&mut self, index: usize, item: &Value) -> Result<()> {
        for area in ["content", "summary"] {
            if let Some(parts) = item.get(area).and_then(Value::as_array) {
                for part_index in 0..parts.len() {
                    self.complete_text(index, area, part_index)?;
                }
            }
        }
        if !self.resumed && item.get("type").and_then(Value::as_str) == Some("function_call") {
            let args = item
                .get("arguments")
                .and_then(Value::as_str)
                .ok_or_else(|| response::malformed("missing function arguments"))?;
            let previous = self.arguments.entry(index).or_default();
            let delta = args.strip_prefix(previous.as_str()).ok_or_else(|| {
                response::malformed("final arguments conflict with delivered arguments")
            })?;
            if !delta.is_empty() {
                self.queue
                    .push_back(ModelStreamItem::ToolCallDelta(ToolDelta {
                        call_id: item
                            .get("call_id")
                            .and_then(Value::as_str)
                            .ok_or_else(|| response::malformed("missing function call id"))?
                            .into(),
                        tool_name: item.get("name").and_then(Value::as_str).map(str::to_owned),
                        content: delta.into(),
                        content_index: Some(index),
                    }));
            }
            *previous = args.into();
        }
        self.emit_normalized_item(index, item)
    }
}

fn content_area(value: &Value) -> &'static str {
    if value
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|kind| kind.contains("reasoning_summary"))
    {
        "summary"
    } else {
        "content"
    }
}

fn content_index(value: &Value) -> Result<usize> {
    number(
        value,
        if content_area(value) == "summary" {
            "summary_index"
        } else {
            "content_index"
        },
    )
}

fn number(value: &Value, key: &str) -> Result<usize> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| response::malformed("invalid output/content index"))
}

async fn next(mut state: State) -> Option<(ModelStreamItem, State)> {
    loop {
        if let Some(item) = state.queue.pop_front() {
            return Some((item, state));
        }
        if state.done {
            return None;
        }
        let idle = Instant::now()
            .checked_add(state.inner.config.idle_timeout)
            .unwrap_or(state.deadline)
            .min(state.deadline);
        match tokio::time::timeout_at(idle, state.bytes.next()).await {
            Err(_) => state.fail(transport::timeout()),
            Ok(Some(Err(error))) => state.fail(error),
            Ok(Some(Ok(chunk))) => {
                state.received = state.received.saturating_add(chunk.len());
                if state.received > state.inner.config.max_body_bytes {
                    state.fail(response::malformed("stream exceeds total byte limit"));
                    continue;
                }
                state.buffer.extend_from_slice(&chunk);
                if let Err(error) = state.lines() {
                    state.fail(error);
                }
            }
            Ok(None) => {
                if !state.buffer.is_empty() {
                    state.buffer.push(b'\n');
                    if let Err(error) = state.lines() {
                        state.fail(error);
                        continue;
                    }
                }
                if !state.done
                    && !state.data.is_empty()
                    && let Err(error) = state.frame()
                {
                    state.fail(error);
                    continue;
                }
                if !state.done {
                    state.fail(transport::failure(
                        "truncated_stream",
                        "stream ended before completion",
                        None,
                    ));
                }
            }
        }
    }
}
