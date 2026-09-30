//! The credential chain (D2): an ordered list of sources behind one port.
//!
//! A host builds the chain that matches how it obtains keys and the hub runs it
//! on **every** request, never caching the answer, so a rotated key or token is
//! used on the next call. The chain reports which source answered
//! ([`CredentialOrigin`]) so a UI can say "Using company key" instead of
//! guessing.
//!
//! * OpenCompany: pasted provider key, then company account key, then instance
//!   identity.
//! * OpenHuman: the session token.
//! * Anything else: an environment variable, the keychain, a static key.
//!
//! Two rules from OpenCompany hold for every chain:
//!
//! * a source that **errors** stops the chain ([`crate::HubError::StoreUnreadable`]),
//!   because falling through from an unreadable store to the managed account
//!   would spend the operator's money on a store outage;
//! * an empty or whitespace-only value is "not here", never a credential.

mod chain;
mod origin;
mod sources;

pub use chain::{CredentialChain, CredentialSource};
pub use origin::CredentialOrigin;
pub use sources::{EnvVarSource, LegacyFlatSlot, StaticSource, StoreSource, TokenSourceAdapter};

#[cfg(test)]
#[path = "test.rs"]
mod tests;
