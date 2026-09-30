//! [`ScriptedSpawner`]: a [`ProcessSpawner`] that answers from a script
//! (feature `cli`), so CLI-login readiness is testable without launching
//! anything.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use async_trait::async_trait;

use crate::ports::{CliCommand, CliOutput, PortError, ProcessSpawner};

/// What a scripted binary does when run.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Spawned {
    /// It runs and exits with this status, printing this.
    Exit {
        /// The exit status.
        status: i32,
        /// Standard output.
        stdout: String,
        /// Standard error.
        stderr: String,
    },
    /// It cannot be launched (not installed, not executable).
    NotInstalled,
    /// It hangs until the timeout kills it.
    Hang,
}

impl Spawned {
    /// A binary that exits 0 printing `stdout`.
    pub fn ok(stdout: impl Into<String>) -> Self {
        Self::Exit {
            status: 0,
            stdout: stdout.into(),
            stderr: String::new(),
        }
    }

    /// A binary that exits non-zero printing `stderr`.
    pub fn failed(status: i32, stderr: impl Into<String>) -> Self {
        Self::Exit {
            status,
            stdout: String::new(),
            stderr: stderr.into(),
        }
    }
}

/// A spawner with one script per program name. A program with no script is
/// [`Spawned::NotInstalled`], which is what a machine without it does.
#[derive(Debug, Default)]
pub struct ScriptedSpawner {
    scripts: Mutex<HashMap<String, Spawned>>,
    log: Mutex<Vec<(CliCommand, Duration)>>,
}

impl ScriptedSpawner {
    /// A spawner with nothing installed.
    pub fn new() -> Self {
        Self::default()
    }

    fn scripts(&self) -> MutexGuard<'_, HashMap<String, Spawned>> {
        self.scripts.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Scripts what `program` does.
    pub fn script(&self, program: impl Into<String>, spawned: Spawned) {
        self.scripts().insert(program.into(), spawned);
    }

    /// Every command run so far, with the timeout it was given.
    pub fn runs(&self) -> Vec<(CliCommand, Duration)> {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

#[async_trait]
impl ProcessSpawner for ScriptedSpawner {
    async fn run(&self, command: &CliCommand, timeout: Duration) -> Result<CliOutput, PortError> {
        self.log
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push((command.clone(), timeout));
        let script = self
            .scripts()
            .get(&command.program)
            .cloned()
            .unwrap_or(Spawned::NotInstalled);
        match script {
            Spawned::Exit {
                status,
                stdout,
                stderr,
            } => Ok(CliOutput {
                status: Some(status),
                stdout,
                stderr,
            }),
            Spawned::NotInstalled => {
                Err(PortError::unavailable("the program could not be launched"))
            }
            Spawned::Hang => Ok(CliOutput {
                status: None,
                stdout: String::new(),
                stderr: String::new(),
            }),
        }
    }
}
