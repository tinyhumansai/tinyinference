//! The two CLI-login kinds (D9): a subprocess transport, no endpoint.
//!
//! Readiness is decided by launching the binary (feature `cli`, a later
//! milestone). The Claude subscription is used **only** by running the `claude`
//! binary, never by replaying a token (D12).

use crate::descriptor::ProviderDescriptor;
use crate::ids::KindId;
use crate::taxonomy::{AuthStyle, CatalogShape, CliKind, Protocol, ProviderGroup, Transport};

pub(super) fn descriptors() -> Vec<ProviderDescriptor> {
    [
        (CliKind::ClaudeCode, "Claude Code"),
        (CliKind::Codex, "Codex"),
    ]
    .into_iter()
    .map(|(cli, label)| ProviderDescriptor {
        kind: KindId::new(cli.option_slug()),
        label,
        group: ProviderGroup::Cli,
        transport: Transport::Subprocess,
        protocol: Protocol::CliStream,
        auth: AuthStyle::None,
        catalog: CatalogShape::None,
        default_endpoint: None,
        endpoint_editable: false,
        needs_key: false,
        key_placeholder: None,
        local_runtime: None,
        cli: Some(cli),
        aliases: &[],
        // A CLI has no listing or completion to test; readiness replaces
        // the test depths.
        test_depths: &[],
        free_text_models: false,
        extra_headers: &[],
        quirks: &[],
    })
    .collect()
}
