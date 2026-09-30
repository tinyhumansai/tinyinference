//! Tests for CLI-login readiness.

use super::*;
use crate::hub::fixtures::Bed;
use crate::ports::CliOutput;
use crate::testkit::{ScriptedSpawner, Spawned};

fn output(status: Option<i32>, stdout: &str, stderr: &str) -> CliOutput {
    CliOutput {
        status,
        stdout: stdout.into(),
        stderr: stderr.into(),
    }
}

#[test]
fn cli_the_status_commands_are_the_documented_ones() {
    let claude = readiness_command(CliKind::ClaudeCode);
    assert_eq!(
        (claude.program.as_str(), claude.args.join(" ")),
        ("claude", "auth status --json".to_string())
    );
    let codex = readiness_command(CliKind::Codex);
    assert_eq!(
        (codex.program.as_str(), codex.args.join(" ")),
        ("codex", "login status".to_string())
    );
}

#[test]
fn cli_readiness_table_and_unknown_is_never_signed_out() {
    let cases: Vec<(CliKind, CliOutput, CliReadiness)> = vec![
        (
            CliKind::ClaudeCode,
            output(Some(0), r#"{"loggedIn":true,"email":"x"}"#, ""),
            CliReadiness::Ready,
        ),
        (
            CliKind::ClaudeCode,
            output(Some(0), r#"{"logged_in":true}"#, ""),
            CliReadiness::Ready,
        ),
        (
            CliKind::ClaudeCode,
            output(Some(1), r#"{"loggedIn":false}"#, ""),
            CliReadiness::SignedOut,
        ),
        (
            CliKind::ClaudeCode,
            output(Some(1), "", "Not logged in"),
            CliReadiness::SignedOut,
        ),
        (
            CliKind::ClaudeCode,
            output(Some(0), "welcome", ""),
            CliReadiness::Unknown,
        ),
        (
            CliKind::ClaudeCode,
            output(Some(2), "", "segfault"),
            CliReadiness::Unknown,
        ),
        (
            CliKind::ClaudeCode,
            output(None, "", ""),
            CliReadiness::Unknown,
        ),
        (
            CliKind::ClaudeCode,
            output(Some(0), r#"{"loggedIn":"yes"}"#, ""),
            CliReadiness::Unknown,
        ),
        (
            CliKind::Codex,
            output(Some(0), "Logged in using ChatGPT", ""),
            CliReadiness::Ready,
        ),
        (
            CliKind::Codex,
            output(Some(1), "", "Not logged in"),
            CliReadiness::SignedOut,
        ),
        (
            CliKind::Codex,
            output(Some(1), "", "boom"),
            CliReadiness::Unknown,
        ),
        (
            CliKind::Codex,
            output(Some(0), "hello", ""),
            CliReadiness::Unknown,
        ),
        (
            CliKind::Codex,
            output(None, "Logged in", ""),
            CliReadiness::Unknown,
        ),
    ];
    for (kind, out, want) in cases {
        assert_eq!(interpret(kind, &out), want, "{kind:?} {out:?}");
    }
}

#[tokio::test]
async fn sim_cli_readiness_ok_missing_hang_logged_out() {
    let spawner = std::sync::Arc::new(ScriptedSpawner::new());
    let bed = Bed::with(|b| b.process_spawner(spawner.clone()));
    // Nothing installed.
    assert_eq!(
        bed.hub.cli_readiness(CliKind::ClaudeCode).await.unwrap(),
        CliReadiness::NotInstalled
    );
    spawner.script("claude", Spawned::ok(r#"{"loggedIn":true}"#));
    assert_eq!(
        bed.hub.cli_readiness(CliKind::ClaudeCode).await.unwrap(),
        CliReadiness::Ready
    );
    spawner.script("claude", Spawned::Hang);
    assert_eq!(
        bed.hub.cli_readiness(CliKind::ClaudeCode).await.unwrap(),
        CliReadiness::Unknown,
        "a hang is unknown, not signed out"
    );
    spawner.script("codex", Spawned::failed(1, "Not logged in"));
    assert_eq!(
        bed.hub.cli_readiness(CliKind::Codex).await.unwrap(),
        CliReadiness::SignedOut
    );
    let runs = spawner.runs();
    assert_eq!(runs.len(), 4);
    assert!(
        runs.iter()
            .all(|(_, timeout)| *timeout == READINESS_TIMEOUT)
    );
}

#[tokio::test]
async fn cli_without_a_spawner_readiness_is_unsupported() {
    let bed = Bed::new();
    assert!(matches!(
        bed.hub.cli_readiness(CliKind::Codex).await,
        Err(HubError::Unsupported {
            op: Operation::CliReadiness,
            ..
        })
    ));
}

#[test]
fn cli_scripted_spawner_defaults_to_not_installed_and_exit_helpers() {
    assert_eq!(
        Spawned::ok("x"),
        Spawned::Exit {
            status: 0,
            stdout: "x".into(),
            stderr: String::new()
        }
    );
    assert_eq!(
        Spawned::failed(3, "e"),
        Spawned::Exit {
            status: 3,
            stdout: String::new(),
            stderr: "e".into()
        }
    );
}
