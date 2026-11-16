//! Evaluating a request whose Box questions the provider cannot answer.
//!
//! [`crate::emulation`] plans the probe requests and builds the answers; this
//! module only sends them, one after another, through the same retry and
//! response validation as any other request.

use super::{BoxEmulation, Client};
use crate::emulation::{BoxEmulator, Probe, Responses, refine};
use crate::request::image::MAX_TOTAL_IMAGE_BYTES;
use crate::{EvaluationRequest, EvaluationResponse, Result};

impl Client {
    /// Answer `request`, emulating its Box questions as `settings` says.
    ///
    /// Every HTTP attempt of every sub-request is added to `attempts`. The
    /// returned request id is the first one a sub-request reported.
    pub(super) async fn evaluate_emulated(
        &self,
        request: &EvaluationRequest,
        settings: BoxEmulation,
        attempts: &mut u32,
    ) -> Result<(EvaluationResponse, Option<String>)> {
        let emulator = BoxEmulator::new(request, settings.grid)?;
        let mut responses = Responses::default();
        let first = emulator.first_round()?;
        self.ask(&emulator, &first, true, &mut responses, attempts)
            .await?;
        let mut picks: Vec<_> = emulator
            .picks(&first, &responses)?
            .into_iter()
            .flatten()
            .collect();
        if settings.refine {
            let second = emulator.refinement_round(&picks)?;
            if !second.is_empty() {
                self.ask(&emulator, &second, false, &mut responses, attempts)
                    .await?;
                picks = refine(picks, emulator.picks(&second, &responses)?);
            }
        }
        let (response, request_id) = emulator.response(responses, &picks);
        self.validate_response(&response, request)?;
        Ok((response, request_id))
    }

    async fn ask(
        &self,
        emulator: &BoxEmulator<'_>,
        probes: &[Probe],
        with_base: bool,
        responses: &mut Responses,
        attempts: &mut u32,
    ) -> Result<()> {
        for batch in emulator.batches(probes, with_base, MAX_TOTAL_IMAGE_BYTES)? {
            batch.validate()?;
            let (response, request_id) = self.send_validated(&batch, attempts).await?;
            responses.add(response, request_id);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "emulated_tests.rs"]
mod tests;
