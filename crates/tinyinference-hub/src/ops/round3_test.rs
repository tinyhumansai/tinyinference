//! Round 3 regressions: guard G3 over the whole credential chain, and an add
//! that cannot confirm itself is undone.

use crate::config::ProviderDraft;
use crate::credential::StaticSource;
use crate::error::{HubError, InvalidInput};
use crate::hub::ProviderPatch;
use crate::hub::fixtures::{Bed, slug};
use crate::secret::Secret;

#[tokio::test]
async fn guard_g3_a_chain_credential_does_not_follow_an_endpoint_to_another_origin() {
    // No stored key at all: the credential comes from a host-supplied source.
    let bed = Bed::with(|b| {
        b.credential_source(
            "custom",
            StaticSource::new(Secret::new("sk-not-a-real-key")),
        )
    });
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("custom")
                .with_label("Acme")
                .with_base_url("https://llm.acme.test/v1"),
        )
        .await
        .unwrap();
    let error = bed
        .hub
        .edit(
            &bed.scope,
            &slug("acme"),
            ProviderPatch::new().base_url("https://other.test/v1"),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::Invalid(InvalidInput::Malformed { .. })),
        "{error:?}"
    );
}

/// A config store whose first read after a save fails once (a flaky backend).
struct FlakyAfterSave {
    inner: std::sync::Arc<crate::ports::memory::MemoryConfig>,
    armed: std::sync::atomic::AtomicBool,
}

impl std::fmt::Debug for FlakyAfterSave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FlakyAfterSave")
    }
}

#[async_trait::async_trait]
impl crate::ports::ConfigStore for FlakyAfterSave {
    async fn load(
        &self,
        scope: &crate::ids::ScopeKey,
    ) -> Result<Option<(crate::config::HubConfig, crate::ports::Version)>, crate::ports::PortError>
    {
        if self.armed.swap(false, std::sync::atomic::Ordering::SeqCst) {
            return Err(crate::ports::PortError::unavailable("flaky"));
        }
        self.inner.load(scope).await
    }

    async fn save(
        &self,
        scope: &crate::ids::ScopeKey,
        config: &crate::config::HubConfig,
        expect: Option<crate::ports::Version>,
    ) -> Result<crate::ports::Version, crate::ports::PortError> {
        let saved = self.inner.save(scope, config, expect).await?;
        self.armed.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok(saved)
    }
}

#[tokio::test]
async fn ops_an_add_whose_recheck_cannot_read_the_config_is_undone_not_left_half_done() {
    let ports = crate::testkit::MemoryPorts::new();
    let flaky = std::sync::Arc::new(FlakyAfterSave {
        inner: ports.config.clone(),
        armed: std::sync::atomic::AtomicBool::new(false),
    });
    let hub = ports.builder().config_arc(flaky).build().unwrap();
    let me = crate::hub::fixtures::scope("company:acme");
    let error = hub
        .add(
            &me,
            ProviderDraft::new("groq").with_key(Secret::new("gsk-not-a-real-key")),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::StoreUnreadable { .. }),
        "{error:?}"
    );
    // The undo's own save re-armed the flaky store; spend that failure.
    let _ = hub.status(&me).await;
    assert!(hub.status(&me).await.unwrap().providers.is_empty());
    assert!(ports.credentials.is_empty(), "no key was left behind");
}

#[tokio::test]
async fn ops_a_label_rename_does_not_need_the_credential_store_to_be_readable() {
    let bed = Bed::new();
    bed.hub
        .add(
            &bed.scope,
            ProviderDraft::new("custom")
                .with_label("Acme")
                .with_base_url("https://llm.acme.test/v1")
                .with_key(Secret::new("sk-not-a-real-key")),
        )
        .await
        .unwrap();
    bed.ports
        .credentials
        .inject(crate::ports::memory::CredentialFault::Read);
    bed.hub
        .edit(
            &bed.scope,
            &slug("acme"),
            ProviderPatch::new().label("Acme Two"),
        )
        .await
        .expect("no endpoint moved, so no credential is read");
}

#[tokio::test]
async fn ops_an_import_quarantined_cloud_row_cannot_be_switched_on() {
    let bed = Bed::new();
    let imported = crate::import::oc::import(
        &serde_json::from_value(serde_json::json!({"providers": [
            {"id": "a", "slug": "openai", "kind": "openai", "base_url": "https://evil.test/v1"}
        ]}))
        .unwrap(),
    )
    .unwrap();
    bed.ports
        .config
        .put_raw(&bed.scope, serde_json::to_string(&imported.config).unwrap());
    let error = bed
        .hub
        .set_enabled(&bed.scope, &slug("openai"), true, crate::hub::Confirm::no())
        .await
        .unwrap_err();
    assert!(
        matches!(error, HubError::Invalid(InvalidInput::Malformed { .. })),
        "{error:?}"
    );
}
