//! CLI-login readiness (feature `cli`).
//!
//! A CLI login (`claude`, `codex`) has no endpoint: whether it works is decided
//! by **running the binary** through the host's
//! [`ProcessSpawner`](crate::ports::ProcessSpawner). The Claude subscription is
//! used only this way (D12): the hub never mints or replays its token.
//!
//! **`Unknown` is not `SignedOut`.** A timeout, a crash, output the hub does not
//! recognise: none of them says the user is signed out, and telling a signed-in
//! user to sign in is worse than saying nothing.

use std::time::Duration;

use crate::error::{HubError, Operation};
use crate::hub::Hub;
use crate::ids::KindId;
use crate::ports::{CliCommand, CliOutput};
use crate::taxonomy::CliKind;

/// How long a readiness command may run.
pub const READINESS_TIMEOUT: Duration = Duration::from_secs(10);

/// What running a CLI login's status command showed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CliReadiness {
    /// The binary ran and reports a logged-in user.
    Ready,
    /// The binary could not be launched.
    NotInstalled,
    /// The binary ran and reports no logged-in user.
    SignedOut,
    /// Nothing conclusive: a timeout, a crash, unrecognised output.
    Unknown,
}

/// The status command for a CLI kind.
pub fn readiness_command(kind: CliKind) -> CliCommand {
    match kind {
        CliKind::ClaudeCode => CliCommand {
            program: "claude".to_string(),
            args: vec!["auth".into(), "status".into(), "--json".into()],
        },
        CliKind::Codex => CliCommand {
            program: "codex".to_string(),
            args: vec!["login".into(), "status".into()],
        },
    }
}

/// Reads what a status command printed.
pub fn interpret(kind: CliKind, output: &CliOutput) -> CliReadiness {
    if output.status.is_none() {
        // Killed (a timeout): not evidence of anything.
        return CliReadiness::Unknown;
    }
    let text = format!("{}\n{}", output.stdout, output.stderr).to_ascii_lowercase();
    match kind {
        CliKind::ClaudeCode => {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(output.stdout.trim())
                && let Some(logged_in) = value
                    .get("loggedIn")
                    .or_else(|| value.get("logged_in"))
                    .and_then(serde_json::Value::as_bool)
            {
                return if logged_in {
                    CliReadiness::Ready
                } else {
                    CliReadiness::SignedOut
                };
            }
            if text.contains("not logged in") {
                CliReadiness::SignedOut
            } else {
                CliReadiness::Unknown
            }
        }
        CliKind::Codex => {
            if text.contains("not logged in") {
                CliReadiness::SignedOut
            } else if output.status == Some(0) && text.contains("logged in") {
                CliReadiness::Ready
            } else {
                CliReadiness::Unknown
            }
        }
    }
}

impl Hub {
    /// Whether a CLI login is usable, by running its status command.
    ///
    /// # Errors
    ///
    /// [`HubError::Unsupported`] when the builder was given no
    /// [`ProcessSpawner`](crate::ports::ProcessSpawner).
    pub async fn cli_readiness(&self, kind: CliKind) -> Result<CliReadiness, HubError> {
        let Some(spawner) = &self.inner.spawner else {
            return Err(HubError::Unsupported {
                op: Operation::CliReadiness,
                kind: KindId::new(kind.option_slug()),
            });
        };
        Ok(
            match spawner
                .run(&readiness_command(kind), READINESS_TIMEOUT)
                .await
            {
                Ok(output) => interpret(kind, &output),
                Err(_) => CliReadiness::NotInstalled,
            },
        )
    }
}

#[cfg(test)]
#[path = "test.rs"]
mod tests;
