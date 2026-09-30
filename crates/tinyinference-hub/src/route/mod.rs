//! Routes: the structured replacement for OpenCompany's route strings and
//! OpenHuman's provider strings, and resolving a turn to one provider.
//!
//! * `types`: [`ProviderRoute`], [`RouteTarget`], [`TurnQuery`],
//!   [`ResolvedTurn`].
//! * `legacy_oc` and `legacy_oh`: the two string grammars, read and written, so
//!   an adapter can keep speaking strings until its UI moves.
//! * `resolve`: [`Hub::resolve_for_turn`](crate::Hub::resolve_for_turn).

pub mod legacy_oc;
pub mod legacy_oh;
mod resolve;
mod types;

pub use resolve::check_route;
pub use types::{ProviderRoute, ResolvedTurn, ResolvedVia, RouteTarget, Temperature, TurnQuery};

#[cfg(test)]
#[path = "resolve_test.rs"]
mod resolve_tests;
#[cfg(test)]
#[path = "test.rs"]
mod tests;
