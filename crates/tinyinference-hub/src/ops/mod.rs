//! The operations, as `impl Hub` blocks grouped by what they touch.
//!
//! * `add`: `add` and `connect`.
//! * `edit`: `edit`.
//! * `remove`: `remove`, `set_enabled`, `set_key`, `clear_key`.
//! * `default`: `set_default`, `clear_default`, `pin_agent`,
//!   `set_workload_route`.
//! * `query`: `probe_draft`, `test`, `list_models`, `health`, `status`,
//!   `record_outcome`, `retest_down`.

mod add;
mod default;
mod edit;
mod edit_move;
mod query;
mod remove;

#[cfg(test)]
#[path = "guards_test.rs"]
mod guards_tests;
#[cfg(test)]
#[path = "interleave_test.rs"]
mod interleave_tests;
#[cfg(test)]
#[path = "query_test.rs"]
mod query_tests;
#[cfg(test)]
#[path = "race_test.rs"]
mod race_tests;
#[cfg(test)]
#[path = "round3_test.rs"]
mod round3_tests;
#[cfg(test)]
#[path = "test.rs"]
mod tests;
