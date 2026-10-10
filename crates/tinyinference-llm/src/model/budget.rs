//! Atomic spend admission around physical provider calls, including retries.
//!
//! Wrap each concrete provider below any retry, cache or fallback adapter.
//! Cloned budgets share one ledger; child budgets share the same lock and
//! debit every ancestor. Missing usage retains the reservation as charged:
//! a timed-out request may have been billed even when its response is lost.

use super::{ChatModel, ModelRequest, ModelResponse, ModelStream, ModelStreamItem};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// A token count and USD charge in integer microdollars.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spend {
    /// Input plus output tokens (reasoning is already included in output).
    pub tokens: u64,
    /// USD multiplied by one million, rounded up.
    pub cost_micros: u64,
}
impl Spend {
    fn add(self, rhs: Self) -> Self {
        Self {
            tokens: self.tokens.saturating_add(rhs.tokens),
            cost_micros: self.cost_micros.saturating_add(rhs.cost_micros),
        }
    }
    fn sub(self, rhs: Self) -> Self {
        Self {
            tokens: self.tokens.saturating_sub(rhs.tokens),
            cost_micros: self.cost_micros.saturating_sub(rhs.cost_micros),
        }
    }
}
/// Independent optional ceilings. Zero refuses all calls on that dimension.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpendLimits {
    /// Token ceiling across all calls.
    pub tokens: Option<u64>,
    /// Cost ceiling in microdollars across all calls.
    pub cost_micros: Option<u64>,
}
/// Snapshot separates known or conservatively charged spend from live reservations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    /// Completed calls, including retained reservations for unknown usage.
    pub spent: Spend,
    /// Calls still in flight.
    pub reserved: Spend,
}
/// Typed admission refusal; no provider call was started.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("budget exceeded: spent {snapshot:?}, requested {requested:?}, limits {limits:?}")]
pub struct BudgetExceeded {
    /// Ledger that refused the call (the turn or any ancestor run).
    pub snapshot: BudgetSnapshot,
    /// Amount the call attempted to reserve.
    pub requested: Spend,
    /// Ceiling that refused the reservation.
    pub limits: SpendLimits,
}
#[derive(Debug)]
struct Entry {
    limits: SpendLimits,
    snapshot: BudgetSnapshot,
    ancestors: Vec<usize>,
    refusal: Option<(usize, BudgetExceeded)>,
}
/// Shareable run budget. Child ledgers debit parent ledgers atomically.
#[derive(Debug, Clone)]
pub struct Budget {
    ledger: Arc<Mutex<Vec<Entry>>>,
    id: usize,
}
impl Budget {
    /// Start an independent run ledger.
    pub fn new(limits: SpendLimits) -> Self {
        Self {
            ledger: Arc::new(Mutex::new(vec![Entry {
                limits,
                snapshot: BudgetSnapshot::default(),
                ancestors: vec![],
                refusal: None,
            }])),
            id: 0,
        }
    }
    /// Add a per-turn ceiling, still charged to this run and all ancestors.
    pub fn child(&self, limits: SpendLimits) -> Self {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ancestors = ledger[self.id].ancestors.clone();
        ancestors.push(self.id);
        let id = ledger.len();
        ledger.push(Entry {
            limits,
            snapshot: BudgetSnapshot::default(),
            ancestors,
            refusal: None,
        });
        Self {
            ledger: self.ledger.clone(),
            id,
        }
    }
    /// Read this ledger's spend, including its children.
    pub fn snapshot(&self) -> BudgetSnapshot {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)[self.id]
            .snapshot
    }
    /// Most recent admission refusal made directly through this ledger.
    pub fn refusal(&self) -> Option<BudgetExceeded> {
        let ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        ledger[self.id]
            .refusal
            .as_ref()
            .map(|(refusing_id, error)| {
                let mut error = error.clone();
                error.snapshot = ledger[*refusing_id].snapshot;
                error
            })
    }

    /// Atomically admit a physical call against this ledger and its ancestors.
    pub fn reserve(&self, requested: Spend) -> Result<Reservation, BudgetExceeded> {
        let mut ledger = self
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut ids = ledger[self.id].ancestors.clone();
        ids.push(self.id);
        for &id in &ids {
            let entry = &ledger[id];
            let committed = entry.snapshot.spent.add(entry.snapshot.reserved);

            if entry.limits.tokens.is_some_and(|max| {
                committed.tokens >= max || requested.tokens > max.saturating_sub(committed.tokens)
            }) || entry.limits.cost_micros.is_some_and(|max| {
                committed.cost_micros >= max
                    || requested.cost_micros > max.saturating_sub(committed.cost_micros)
            }) {
                let error = BudgetExceeded {
                    snapshot: entry.snapshot,
                    requested,
                    limits: entry.limits,
                };
                ledger[self.id].refusal = Some((id, error.clone()));
                return Err(error);
            }
        }
        for &id in &ids {
            ledger[id].snapshot.reserved = ledger[id].snapshot.reserved.add(requested);
        }
        Ok(Reservation {
            budget: self.clone(),
            ids,
            requested,
            settled: false,
        })
    }
}
/// An admitted call. Dropping without settlement charges the full reservation.
#[derive(Debug)]
pub struct Reservation {
    budget: Budget,
    ids: Vec<usize>,
    requested: Spend,
    settled: bool,
}
impl Reservation {
    /// Reconcile known usage. Unknown dimensions must use the reserved value.
    pub fn settle(mut self, actual: Spend) {
        self.finish(actual);
    }
    fn finish(&mut self, actual: Spend) {
        if self.settled {
            return;
        }
        let mut ledger = self
            .budget
            .ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for &id in &self.ids {
            let snapshot = &mut ledger[id].snapshot;
            snapshot.reserved = snapshot.reserved.sub(self.requested);
            snapshot.spent = snapshot.spent.add(actual);
        }
        self.settled = true;
    }
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.finish(self.requested);
    }
}
/// Host-supplied conservative bound for any one call on the wrapped route.
///
/// The host must include all input modalities and provider pricing when choosing
/// these bounds. They are admission reservations, not provider billing caps.
#[derive(Debug, Clone, Copy)]
pub struct CallBudget {
    /// Conservative maximum input token count. Text requests exceeding this
    /// in serialized request bytes are refused; non-text modalities are refused.
    pub input_tokens: u64,
    /// Output cap imposed on every physical provider request.
    pub output_tokens: u32,
    /// Conservative cost bound including prompt, cache, reasoning and output.
    pub cost_micros: u64,
}
// Concrete transports call this before sending: adapters may retry request
// shape internally, below ChatModel. The first attempt owns the outer permit;
// each extra attempt is conservatively charged before dispatch.
struct AttemptScope {
    budget: Budget,
    requested: Spend,
    attempts: std::sync::atomic::AtomicUsize,
}
tokio::task_local! { static PHYSICAL_CALL: Arc<AttemptScope>; }
/// Admit an extra physical attempt made internally by a concrete provider.
///
/// Outside a budgeted invocation this is a no-op. The first attempt uses the
/// invocation reservation; later attempts must atomically reserve again.
pub(crate) fn before_physical_attempt() -> crate::Result<()> {
    PHYSICAL_CALL
        .try_with(|scope| {
            if scope
                .attempts
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                == 0
            {
                return Ok(());
            }
            let reservation = scope
                .budget
                .reserve(scope.requested)
                .map_err(crate::Error::BudgetExceeded)?;
            drop(reservation);
            Ok(())
        })
        .unwrap_or(Ok(()))
}

/// Applies a shared budget to a concrete provider, before every invocation.
pub struct BudgetedModel<State: Send + Sync> {
    inner: Arc<dyn ChatModel<State>>,
    budget: Budget,
    call: CallBudget,
}
impl<State: Send + Sync> std::fmt::Debug for BudgetedModel<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BudgetedModel")
            .field("budget", &self.budget)
            .field("call", &self.call)
            .finish_non_exhaustive()
    }
}
impl<State: Send + Sync> BudgetedModel<State> {
    /// Wrap a provider below retries/fallbacks so each attempt reserves separately.
    pub fn new(inner: Arc<dyn ChatModel<State>>, budget: Budget, call: CallBudget) -> Self {
        Self {
            inner,
            budget,
            call,
        }
    }
    fn admit(&self, request: &mut ModelRequest) -> crate::Result<Reservation> {
        if ["max_tokens", "max_completion_tokens", "max_output_tokens"]
            .iter()
            .any(|key| request.provider_options.get(*key).is_some())
        {
            return Err(crate::Error::Validation(
                "budgeted model output caps must use typed max_tokens".into(),
            ));
        }
        // Count the complete typed request: response schemas and provider
        // instructions also become model input. Framing remains host-bounded.
        let bytes = serde_json::to_vec(&request)?.len() as u64;
        let text_only = request.messages.iter().all(|message| {
            use crate::message::{ContentBlock, Message};
            let blocks = match message {
                Message::User(message) => &message.content,
                Message::Assistant(message) => &message.content,
                Message::Tool(message) => &message.content,
                Message::System(_) | Message::Custom(_) => return true,
            };
            blocks.iter().all(|block| {
                matches!(
                    block,
                    ContentBlock::Text(_) | ContentBlock::Json(_) | ContentBlock::Thinking { .. }
                )
            })
        });
        if bytes > self.call.input_tokens || !text_only {
            return Err(crate::Error::Validation(
                "budgeted model requires text input within its conservative input-token bound"
                    .into(),
            ));
        }
        request.max_tokens = Some(
            request
                .max_tokens
                .unwrap_or(self.call.output_tokens)
                .min(self.call.output_tokens),
        );
        let amount = Spend {
            tokens: self
                .call
                .input_tokens
                .saturating_add(request.max_tokens.unwrap_or_default() as u64),
            cost_micros: self.call.cost_micros,
        };
        self.budget
            .reserve(amount)
            .map_err(crate::Error::BudgetExceeded)
    }
    fn settle(reservation: Reservation, response: &ModelResponse) {
        let reserved = reservation.requested;
        let spent = if response.served_from_cache {
            Spend::default()
        } else {
            Spend {
                tokens: response.usage.map_or(reserved.tokens, |usage| {
                    usage
                        .input_tokens
                        .saturating_add(usage.output_tokens)
                        .max(usage.total_tokens)
                }),
                cost_micros: response
                    .usage
                    .and_then(|usage| usage.charged_amount)
                    .and_then(|amount| u64::try_from(amount.micros).ok())
                    .unwrap_or(reserved.cost_micros),
            }
        };
        reservation.settle(spent);
    }
}
#[async_trait]
impl<State: Send + Sync> ChatModel<State> for BudgetedModel<State> {
    fn profile(&self) -> Option<&super::ModelProfile> {
        self.inner.profile()
    }
    fn supports_input(
        &self,
        _modality: super::InputModality,
        _mime: &str,
        _source: super::InputSource,
    ) -> bool {
        false
    }
    fn cache_identity(&self) -> Option<String> {
        self.inner.cache_identity()
    }
    async fn invoke(
        &self,
        state: &State,
        mut request: ModelRequest,
    ) -> crate::Result<ModelResponse> {
        let reservation = self.admit(&mut request)?;
        let scope = Arc::new(AttemptScope {
            budget: self.budget.clone(),
            requested: reservation.requested,
            attempts: std::sync::atomic::AtomicUsize::new(0),
        });
        let response = PHYSICAL_CALL
            .scope(scope, self.inner.invoke(state, request))
            .await?;
        Self::settle(reservation, &response);
        Ok(response)
    }
    async fn stream(&self, state: &State, mut request: ModelRequest) -> crate::Result<ModelStream> {
        let reservation = self.admit(&mut request)?;
        let scope = Arc::new(AttemptScope {
            budget: self.budget.clone(),
            requested: reservation.requested,
            attempts: std::sync::atomic::AtomicUsize::new(0),
        });
        let stream = PHYSICAL_CALL
            .scope(Arc::clone(&scope), self.inner.stream(state, request))
            .await?;
        let metadata = stream.metadata().clone();
        let mut stream = Box::pin(stream);
        let mut reservation = Some(reservation);
        // Lazy providers can send/retry while polled; retain the same attempt
        // counter and admission scope over both construction and consumption.
        let stream = futures::stream::poll_fn(move |cx| {
            PHYSICAL_CALL.sync_scope(Arc::clone(&scope), || {
                let item = futures::Stream::poll_next(stream.as_mut(), cx);
                match &item {
                    std::task::Poll::Ready(Some(ModelStreamItem::Completed(response))) => {
                        if let Some(permit) = reservation.take() {
                            Self::settle(permit, response);
                        }
                    }
                    std::task::Poll::Ready(Some(
                        ModelStreamItem::Failed(_)
                        | ModelStreamItem::ProviderFailed(_)
                        | ModelStreamItem::Deferred(_),
                    ))
                    | std::task::Poll::Ready(None) => drop(reservation.take()),
                    _ => (),
                }
                item
            })
        });
        Ok(ModelStream::new(Box::pin(stream)).with_metadata(metadata))
    }
}
#[cfg(test)]
#[path = "budget_tests.rs"]
mod tests;
