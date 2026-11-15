//! Box questions emulated with Noul questions over image crops, for providers
//! that do not answer Box natively.

mod crop;
mod geometry;

use std::collections::BTreeMap;

use geometry::{Pick, Rect};

use crate::{
    Answer, Error, EvaluationRequest, EvaluationResponse, ImageAsset, ImageFormat, Question,
    Result, Usage,
};

pub(crate) struct BoxEmulator<'a> {
    request: &'a EvaluationRequest,
}

pub(crate) struct Probe {
    pub(crate) target: usize,
    pub(crate) cell: Rect,
    pub(crate) id: String,
    pub(crate) question: Question,
    pub(crate) asset: ImageAsset,
    pub(crate) bytes: usize,
}

#[derive(Default)]
pub(crate) struct Responses {
    answers: BTreeMap<String, Answer>,
}

impl Responses {
    pub(crate) fn add(&mut self, _response: EvaluationResponse, _request_id: Option<String>) {
        todo!()
    }
}

impl<'a> BoxEmulator<'a> {
    pub(crate) fn new(_request: &'a EvaluationRequest, _grid: u32) -> Result<Self> {
        todo!()
    }
    pub(crate) fn first_round(&self) -> Result<Vec<Probe>> {
        todo!()
    }
    pub(crate) fn refinement_round(&self, _picks: &[Pick]) -> Result<Vec<Probe>> {
        todo!()
    }
    pub(crate) fn batches(
        &self,
        _probes: &[Probe],
        _with_base: bool,
        _limit: usize,
    ) -> Result<Vec<EvaluationRequest>> {
        todo!()
    }
    pub(crate) fn picks(&self, _probes: &[Probe], _responses: &Responses) -> Result<Vec<Option<Pick>>> {
        todo!()
    }
    pub(crate) fn response(&self, _responses: Responses, _picks: &[Pick]) -> (EvaluationResponse, Option<String>) {
        todo!()
    }
}

pub(crate) fn refine(_first: Vec<Pick>, _second: Vec<Option<Pick>>) -> Vec<Pick> {
    todo!()
}

fn source_asset(_request: &EvaluationRequest, _id: &str, _instructions: &serde_json::Value) -> Result<String> {
    todo!()
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod tests;
