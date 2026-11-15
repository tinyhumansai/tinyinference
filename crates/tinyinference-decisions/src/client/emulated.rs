//! Evaluating a request whose Box questions the provider cannot answer.

use super::{BoxEmulation, Client};
use crate::{EvaluationRequest, EvaluationResponse, Result};

impl Client {
    pub(super) async fn evaluate_emulated(
        &self,
        _request: &EvaluationRequest,
        _settings: BoxEmulation,
        _attempts: &mut u32,
    ) -> Result<(EvaluationResponse, Option<String>)> {
        todo!()
    }
}

#[cfg(test)]
#[path = "emulated_tests.rs"]
mod tests;
