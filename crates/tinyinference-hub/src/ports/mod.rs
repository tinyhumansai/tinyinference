//! The ports: everything the hub needs from the host that it cannot own.
//!
//! A host implements the four required ports ([`CredentialStore`],
//! [`ConfigStore`], [`Http`], [`Clock`]) and may replace the defaulted ones
//! ([`HealthStore`], [`EventSink`], [`EnvSource`], with in-memory or no-op
//! defaults in [`memory`]) or supply the optional ones only when they apply
//! ([`TokenSource`] for a rotating token, [`Detector`] for first-run detection). In-memory implementations live in [`memory`]; the no-socket
//! test doubles live in `testkit` (feature `testing`).
//!
//! Every port is object-safe, `Send + Sync`, and has a redacting `Debug`, so
//! a host can hold each as an `Arc<dyn ...>` and log the hub without leaking a
//! credential.
//!
//! # Contracts worth reading twice
//!
//! * [`CredentialStore`]: `Err` means *unreadable*, never *absent*. A caller
//!   that treated an outage as "no key" would fall through to the managed
//!   provider and spend the operator's account.
//! * [`ConfigStore::save`] is a compare-and-swap. In-process mutexes are not
//!   enough for a hub embedded in several processes.
//! * [`Http::send`] must apply the endpoint policy on every redirect hop and
//!   connect to the address it checked. [`follow_redirects`] implements the
//!   hop loop so a host does not re-derive it.

mod clock;
mod config;
mod credential;
mod detect;
mod env;
mod error;
mod events;
mod health;
mod http;
mod interleave;
pub mod memory;
#[cfg(feature = "cli")]
mod process;
mod redirect;
#[cfg(feature = "http-reqwest")]
mod reqwest_http;
mod token;
mod usage;

pub use clock::{Clock, SystemClock};
pub use config::{ConfigStore, Version};
pub use credential::CredentialStore;
pub use detect::{DetectOptions, Detector};
pub use env::EnvSource;
pub use error::PortError;
pub use events::{EventSink, HubEvent};
pub use health::HealthStore;
pub use http::{Http, HttpError, HubRequest, HubResponse, Method};
#[cfg(feature = "cli")]
pub use process::{CliCommand, CliOutput, ProcessSpawner};
pub use redirect::follow_redirects;
#[cfg(feature = "http-reqwest")]
pub use reqwest_http::{
    Executor, Pin, ReqwestExecutor, ReqwestHttp, Resolver, SystemResolver, build_client,
    build_request, map_error_parts, pin_addresses, read_response,
};
pub use token::TokenSource;
pub use usage::UsageQuery;

#[cfg(test)]
#[path = "test.rs"]
mod tests;
