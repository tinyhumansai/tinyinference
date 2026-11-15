//! Box questions emulated with Noul questions over image crops, for providers
//! that do not answer Box natively.
//!
//! Each Box question's image is cut into an `n` by `n` grid. Every cell is
//! upscaled to the provider's minimum side, encoded as PNG, and asked about as
//! a Noul probe ("does this tile show the target?"). The box is the bounding
//! rectangle of the positive cells connected to the best one, and an optional
//! refinement round grids that box again. Probes never reach the caller: the
//! response carries exactly the caller's question ids.

pub(crate) mod crop;
pub(crate) mod geometry;

use std::collections::{BTreeMap, BTreeSet};

use image::DynamicImage;
use serde_json::{Value, json};

pub(crate) use geometry::Pick;
use geometry::Rect;

use crate::request::image::MAX_IMAGE_BYTES;
use crate::request::value_mentions;
use crate::{
    Answer, Asset, BoxAnswer, Error, EvaluationRequest, EvaluationResponse, ImageAsset, Noul,
    NoulCriteria, Question, Result, Usage,
};

/// Probe question ids and asset keys start with this; callers may not use it.
pub(crate) const PROBE_PREFIX: &str = "__tinyinference_box";

/// One Box question being emulated.
struct Target {
    /// The caller's question id.
    id: String,
    /// The question's instructions, as text for the probe questions.
    instructions: String,
    /// Asset key of the image the box is drawn on.
    source: String,
    width: u32,
    height: u32,
}

/// Plans probe rounds for a request's Box questions and turns their answers
/// back into Box answers.
pub(crate) struct BoxEmulator<'a> {
    request: &'a EvaluationRequest,
    grid: u32,
    targets: Vec<Target>,
    images: BTreeMap<String, DynamicImage>,
}

/// One Noul question about one grid cell of a target's image.
pub(crate) struct Probe {
    /// Index of the target this probe belongs to.
    pub(crate) target: usize,
    /// The cell, in pixels of the target's whole image.
    pub(crate) cell: Rect,
    /// Question id, also the key of the probe's own crop asset.
    pub(crate) id: String,
    pub(crate) question: Question,
    pub(crate) asset: ImageAsset,
    /// Size of the crop file.
    pub(crate) bytes: usize,
}

/// Answers, usage, and metadata accumulated across sub-requests.
#[derive(Debug, Default)]
pub(crate) struct Responses {
    model: Option<String>,
    request_id: Option<String>,
    answers: BTreeMap<String, Answer>,
    usage: Usage,
}

impl Responses {
    /// Record one validated sub-response. The first model and the first
    /// request id seen are kept; usage is summed.
    pub(crate) fn add(&mut self, response: EvaluationResponse, request_id: Option<String>) {
        self.model.get_or_insert(response.model);
        if self.request_id.is_none() {
            self.request_id = request_id;
        }
        self.answers.extend(response.answers);
        self.usage = add_usage(self.usage, response.usage);
    }
}

impl<'a> BoxEmulator<'a> {
    /// Resolve and decode the image of every Box question in `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when a caller question id or asset
    /// key uses [`PROBE_PREFIX`], a Box question does not identify exactly one
    /// image, or an image cannot be decoded.
    pub(crate) fn new(request: &'a EvaluationRequest, grid: u32) -> Result<Self> {
        if request
            .questions
            .keys()
            .chain(request.assets.keys())
            .any(|key| key.starts_with(PROBE_PREFIX))
        {
            return Err(Error::invalid_request(format!(
                "question ids and asset keys starting with `{PROBE_PREFIX}` are reserved for \
                 emulated Box questions"
            )));
        }
        let mut images = BTreeMap::new();
        let mut targets = Vec::new();
        for (id, question) in &request.questions {
            let Question::Box(question) = question else {
                continue;
            };
            let source = source_asset(request, id, &question.instructions)?;
            if !images.contains_key(&source) {
                let Asset::Image(asset) = &request.assets[&source];
                images.insert(source.clone(), crop::decode(&source, asset)?);
            }
            let image = &images[&source];
            targets.push(Target {
                id: id.clone(),
                instructions: as_text(&question.instructions),
                width: image.width(),
                height: image.height(),
                source,
            });
        }
        Ok(Self {
            request,
            grid,
            targets,
            images,
        })
    }

    /// Probes over the whole image of every target.
    pub(crate) fn first_round(&self) -> Result<Vec<Probe>> {
        let regions: Vec<Option<Rect>> = self
            .targets
            .iter()
            .map(|target| Some(Rect::image(target.width, target.height)))
            .collect();
        self.round(0, &regions)
    }

    /// Probes over each first-round box worth refining; empty when none is.
    pub(crate) fn refinement_round(&self, picks: &[Pick]) -> Result<Vec<Probe>> {
        let regions: Vec<Option<Rect>> = self
            .targets
            .iter()
            .zip(picks)
            .map(|(target, pick)| {
                geometry::refinement_region(
                    pick.rect,
                    Rect::image(target.width, target.height),
                    self.grid,
                )
            })
            .collect();
        self.round(1, &regions)
    }

    fn round(&self, round: u32, regions: &[Option<Rect>]) -> Result<Vec<Probe>> {
        let mut probes = Vec::new();
        for (index, (target, region)) in self.targets.iter().zip(regions).enumerate() {
            let Some(region) = region else {
                continue;
            };
            let image = &self.images[&target.source];
            for (cell_index, cell) in geometry::grid(*region, self.grid).into_iter().enumerate() {
                let cell_index = u32::try_from(cell_index).unwrap_or(u32::MAX);
                let id = format!(
                    "{PROBE_PREFIX}{index}_round{round}_r{}c{}",
                    cell_index / self.grid,
                    cell_index % self.grid
                );
                let (asset, bytes) = crop::crop(image, cell)?;
                probes.push(Probe {
                    target: index,
                    cell,
                    question: probe_question(&id, target),
                    id,
                    asset,
                    bytes,
                });
            }
        }
        Ok(probes)
    }

    /// Requests that ask every probe, each carrying at most `limit` bytes of
    /// images.
    ///
    /// With `with_base`, the caller's other questions and the assets they or
    /// the state reference are sent too. Probes join that request unless it
    /// carries one of their source images, so a tile is never judged next to
    /// the whole picture it was cut from. Other requests carry the caller's
    /// state and model, the probes, and their crops only.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidRequest`] when one crop alone exceeds `limit`
    /// or the provider's per-image limit.
    pub(crate) fn batches(
        &self,
        probes: &[Probe],
        with_base: bool,
        limit: usize,
    ) -> Result<Vec<EvaluationRequest>> {
        let empty = || EvaluationRequest {
            state: self.request.state.clone(),
            model: self.request.model.clone(),
            questions: BTreeMap::new(),
            assets: BTreeMap::new(),
        };
        let mut batches = Vec::new();
        let mut current = empty();
        let mut total = 0_usize;
        if with_base {
            let base = self.base()?;
            if !base.questions.is_empty() {
                let carries_source = probes
                    .iter()
                    .any(|probe| base.assets.contains_key(&self.targets[probe.target].source));
                if carries_source {
                    batches.push(base);
                } else {
                    total = image_bytes(&base)?;
                    current = base;
                }
            }
        }
        for probe in probes {
            if probe.bytes > limit.min(MAX_IMAGE_BYTES) {
                return Err(Error::invalid_request(
                    "an emulated Box image crop exceeds the provider's image size limit",
                ));
            }
            if total.saturating_add(probe.bytes) > limit && !current.questions.is_empty() {
                batches.push(std::mem::replace(&mut current, empty()));
                total = 0;
            }
            current
                .questions
                .insert(probe.id.clone(), probe.question.clone());
            current
                .assets
                .insert(probe.id.clone(), Asset::Image(probe.asset.clone()));
            total = total.saturating_add(probe.bytes);
        }
        if !current.questions.is_empty() {
            batches.push(current);
        }
        Ok(batches)
    }

    /// The caller's request without its Box questions or the images only they
    /// reference.
    fn base(&self) -> Result<EvaluationRequest> {
        let questions: BTreeMap<String, Question> = self
            .request
            .questions
            .iter()
            .filter(|(_, question)| !matches!(question, Question::Box(_)))
            .map(|(id, question)| (id.clone(), question.clone()))
            .collect();
        let readable = serde_json::to_value((&self.request.state, &questions))
            .map_err(|_| Error::invalid_request("state and questions must serialise"))?;
        let assets = self
            .request
            .assets
            .iter()
            .filter(|(key, _)| value_mentions(&readable, key))
            .map(|(key, asset)| (key.clone(), asset.clone()))
            .collect();
        Ok(EvaluationRequest {
            state: self.request.state.clone(),
            model: self.request.model.clone(),
            questions,
            assets,
        })
    }

    /// The box each target's probes select, or `None` for a target this round
    /// did not probe.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidResponse`] when a probe was not answered with a
    /// Noul.
    pub(crate) fn picks(
        &self,
        probes: &[Probe],
        responses: &Responses,
    ) -> Result<Vec<Option<Pick>>> {
        let mut cells: Vec<(Vec<Rect>, Vec<f64>)> =
            vec![(Vec::new(), Vec::new()); self.targets.len()];
        for probe in probes {
            let Some(Answer::Noul(answer)) = responses.answers.get(&probe.id) else {
                return Err(Error::invalid_response(format!(
                    "emulated Box probe `{}` was not answered with a noul",
                    probe.id
                )));
            };
            let (rects, probabilities) = &mut cells[probe.target];
            rects.push(probe.cell);
            probabilities.push(answer.noul);
        }
        Ok(cells
            .into_iter()
            .map(|(rects, probabilities)| {
                (!rects.is_empty()).then(|| geometry::pick(&rects, &probabilities, self.grid))
            })
            .collect())
    }

    /// The caller-facing response: the caller's non-Box answers plus one Box
    /// answer per target, with every probe answer dropped. Also returns the
    /// first sub-request id.
    pub(crate) fn response(
        &self,
        responses: Responses,
        picks: &[Pick],
    ) -> (EvaluationResponse, Option<String>) {
        let mut answers: BTreeMap<String, Answer> = responses
            .answers
            .into_iter()
            .filter(|(id, _)| self.request.questions.contains_key(id))
            .collect();
        for (target, pick) in self.targets.iter().zip(picks) {
            answers.insert(
                target.id.clone(),
                Answer::Box(BoxAnswer {
                    edges: pick.rect.edges(target.width, target.height),
                    confidence: pick.confidence,
                }),
            );
        }
        let response = EvaluationResponse {
            model: responses.model.unwrap_or_default(),
            answers,
            usage: responses.usage,
        };
        (response, responses.request_id)
    }
}

/// First-round picks with each one replaced by its refinement when that found
/// the target; a refinement with no positive cell keeps the first-round box.
pub(crate) fn refine(first: Vec<Pick>, second: Vec<Option<Pick>>) -> Vec<Pick> {
    first
        .into_iter()
        .zip(second)
        .map(|(first, second)| match second {
            Some(second) if second.found => second,
            _ => first,
        })
        .collect()
}

/// The asset key of the image a Box question is about.
///
/// In order: the one asset key its instructions mention (a key contained in a
/// longer mentioned key does not count); else the one asset held by a
/// top-level state field its instructions name in backticks; else the
/// request's only image.
fn source_asset(request: &EvaluationRequest, id: &str, instructions: &Value) -> Result<String> {
    let ambiguous = || {
        Error::invalid_request(format!(
            "box question `{id}` names more than one image; emulating Box needs exactly one"
        ))
    };
    let mentioned: Vec<&String> = request
        .assets
        .keys()
        .filter(|key| value_mentions(instructions, key))
        .collect();
    let mentioned: Vec<&String> = mentioned
        .iter()
        .filter(|key| {
            !mentioned
                .iter()
                .any(|other| other != *key && other.contains(key.as_str()))
        })
        .copied()
        .collect();
    match mentioned.as_slice() {
        [key] => return Ok((*key).clone()),
        [] => {}
        _ => return Err(ambiguous()),
    }
    let mut fields = Vec::new();
    backticked(instructions, &mut fields);
    let held: BTreeSet<&String> = fields
        .iter()
        .filter_map(|field| request.state.get(field)?.as_str())
        .filter_map(|value| request.assets.get_key_value(value).map(|(key, _)| key))
        .collect();
    match held.len() {
        1 => return Ok(held.into_iter().next().cloned().unwrap_or_default()),
        0 => {}
        _ => return Err(ambiguous()),
    }
    if let [(key, _)] = request.assets.iter().collect::<Vec<_>>().as_slice() {
        return Ok((*key).clone());
    }
    Err(Error::invalid_request(format!(
        "box question `{id}` must identify one image to emulate Box: mention its asset key, or \
         name a state field holding the key in backticks"
    )))
}

/// Every span between backticks in the string values of `value`.
fn backticked(value: &Value, spans: &mut Vec<String>) {
    match value {
        Value::String(text) => spans.extend(
            text.split('`')
                .skip(1)
                .step_by(2)
                .filter(|span| !span.is_empty())
                .map(str::to_owned),
        ),
        Value::Array(items) => items.iter().for_each(|item| backticked(item, spans)),
        Value::Object(map) => map.values().for_each(|item| backticked(item, spans)),
        _ => {}
    }
}

fn as_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn probe_question(id: &str, target: &Target) -> Question {
    Question::Noul(Noul {
        instructions: Value::String(format!(
            "`{id}` is a tile cut from the image `{}`. Judge only the tile `{id}`. Does it show \
             all or part of the following? {}",
            target.source, target.instructions
        )),
        criteria: Some(NoulCriteria {
            r#true: json!("The tile shows all or part of it."),
            r#false: json!("The tile shows none of it."),
        }),
    })
}

fn image_bytes(request: &EvaluationRequest) -> Result<usize> {
    request
        .assets
        .values()
        .try_fold(0_usize, |total, Asset::Image(image)| {
            Ok(total.saturating_add(image.checked_len()?))
        })
}

fn add_usage(left: Usage, right: Usage) -> Usage {
    fn sum<T: Copy + SaturatingAdd>(left: Option<T>, right: Option<T>) -> Option<T> {
        match (left, right) {
            (Some(left), Some(right)) => Some(left.saturating(right)),
            (value, None) | (None, value) => value,
        }
    }
    Usage {
        input_tokens: sum(left.input_tokens, right.input_tokens),
        output_tokens: sum(left.output_tokens, right.output_tokens),
        cached_input_tokens: sum(left.cached_input_tokens, right.cached_input_tokens),
        cost_usd_nanos: sum(left.cost_usd_nanos, right.cost_usd_nanos),
    }
}

trait SaturatingAdd {
    fn saturating(self, other: Self) -> Self;
}

impl SaturatingAdd for u64 {
    fn saturating(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl SaturatingAdd for i64 {
    fn saturating(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
