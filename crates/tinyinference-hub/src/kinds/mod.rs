//! Kind drivers: how each provider kind is reached.
//!
//! A [`KindDriver`] turns a [`Target`] and a [`DriverContext`] into the three
//! things the hub asks of a provider: validate a key ([`KindDriver::key_check`]),
//! list its models ([`KindDriver::list_models`]), and prove a completion works
//! ([`KindDriver::completion_ping`]). Everything a driver sends goes through the
//! [`Http`](crate::ports::Http) port, so a driver is testable without a socket.
//!
//! The built-in drivers cover every catalogue kind:
//!
//! * [`OpenAiCompatDriver`]: the hosted OpenAI-compatible vendors and custom
//!   endpoints, including OpenRouter's account-scoped listing and `/key` check;
//! * [`AnthropicDriver`]: the native paged `/models` and `/messages` calls;
//! * [`ManagedDriver`]: the TinyHumans catalog with prices and the typed
//!   signed-out state;
//! * [`LocalDriver`]: local runtimes, with the native-listing fallbacks;
//! * [`CliDriver`]: no HTTP at all.
//!
//! A [`DriverRegistry`] maps a kind to its driver; a host adds its own kinds
//! with [`DriverRegistry::register`].

mod anthropic;
mod cli;
mod context;
mod local;
mod managed;
mod openai_compat;
mod ping;
mod registry;
mod traits;

pub use anthropic::AnthropicDriver;
pub use cli::CliDriver;
pub use context::{DriverContext, Target};
pub use local::LocalDriver;
pub use managed::ManagedDriver;
pub use openai_compat::OpenAiCompatDriver;
pub use registry::DriverRegistry;
pub use traits::KindDriver;

#[cfg(test)]
#[path = "contract_test.rs"]
mod contract_tests;
#[cfg(test)]
#[path = "drivers_test.rs"]
mod drivers_tests;
#[cfg(test)]
#[path = "test.rs"]
mod tests;
