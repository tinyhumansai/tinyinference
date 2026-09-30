//! [`CredentialOrigin`]: which source supplied a credential.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Where the credential in use came from.
///
/// Carries only names, never a value, so it is safe in a status payload and a
/// log line.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "origin", content = "name", rename_all = "snake_case")]
pub enum CredentialOrigin {
    /// A key pasted for this provider.
    ProviderKey,
    /// The company's account key (OpenCompany's managed fan-out).
    AccountKey,
    /// The instance's own identity (a hosted tenant's platform token).
    InstanceIdentity,
    /// A signed-in session token (OpenHuman).
    SessionJwt,
    /// An environment variable, by name.
    Env(String),
    /// The OS keychain.
    Keychain,
    /// A key fixed at construction.
    Static,
    /// A browser login (feature `oauth`).
    OAuth,
}

impl CredentialOrigin {
    /// A short sentence for a status row (`docs`: managed row, D5).
    pub fn describe(&self) -> String {
        match self {
            Self::ProviderKey => "Using the key you added".to_string(),
            Self::AccountKey => "Using company key".to_string(),
            Self::InstanceIdentity => "Instance identity".to_string(),
            Self::SessionJwt => "Signed in".to_string(),
            Self::Env(name) => format!("Using the {name} environment variable"),
            Self::Keychain => "Using the system keychain".to_string(),
            Self::Static => "Using a fixed key".to_string(),
            Self::OAuth => "Signed in with a browser login".to_string(),
        }
    }
}

impl fmt::Display for CredentialOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.describe())
    }
}
