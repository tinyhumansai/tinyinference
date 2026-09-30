//! Test fixtures shared by the hub's unit tests.

use serde_json::{Value, json};

use crate::config::ProviderDraft;
use crate::hub::{Hub, HubBuilder};
use crate::ids::{ModelId, ScopeKey, Slug};
use crate::ports::CredentialStore;
use crate::secret::Secret;
use crate::testkit::{Match, MemoryPorts, Scripted};

/// A hub over in-memory ports and one scope.
pub(crate) struct Bed {
    pub(crate) ports: MemoryPorts,
    pub(crate) hub: Hub,
    pub(crate) scope: ScopeKey,
}

pub(crate) fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

pub(crate) fn model(id: &str) -> ModelId {
    ModelId::parse(id).unwrap()
}

pub(crate) fn scope(name: &str) -> ScopeKey {
    ScopeKey::new(name)
}

pub(crate) fn models_body(ids: &[&str]) -> Value {
    json!({"data": ids.iter().map(|i| json!({"id": i})).collect::<Vec<_>>()})
}

pub(crate) const KEY: &str = "sk-not-a-real-key";

impl Bed {
    pub(crate) fn new() -> Self {
        Self::with(|builder| builder)
    }

    pub(crate) fn with(tweak: impl FnOnce(HubBuilder) -> HubBuilder) -> Self {
        let ports = MemoryPorts::new();
        let hub = tweak(ports.builder()).build().unwrap();
        Self {
            ports,
            hub,
            scope: scope("company:acme"),
        }
    }

    /// `GET https://api.openai.com/v1/models` answers with these ids.
    pub(crate) fn openai_lists(&self, ids: &[&str]) {
        self.ports.http.route(
            Match::prefix("https://api.openai.com/v1/models"),
            Scripted::json(200, &models_body(ids)),
        );
    }

    pub(crate) fn openai_rejects_key(&self) {
        self.ports.http.route(
            Match::prefix("https://api.openai.com/v1/"),
            Scripted::json(
                401,
                &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
            ),
        );
    }

    pub(crate) fn openai_chat_ok(&self) {
        self.ports.http.route(
            Match::post("https://api.openai.com/v1/chat/completions"),
            Scripted::json(200, &json!({"choices": [{"message": {"content": "ok"}}]})),
        );
    }

    pub(crate) fn openai_draft(&self) -> ProviderDraft {
        ProviderDraft::new("openai")
            .with_key(Secret::new(KEY))
            .with_model(model("gpt-x"))
    }

    pub(crate) async fn key_of(&self, name: &str) -> Option<String> {
        self.ports
            .credentials
            .get(&self.scope, &slug(name).key_slot())
            .await
            .unwrap()
            .map(|s| s.expose().to_string())
    }

    pub(crate) async fn store_key(&self, name: &str, key: &str) {
        self.ports
            .credentials
            .set(&self.scope, &slug(name).key_slot(), Secret::new(key))
            .await
            .unwrap();
    }

    /// Whether the hub is holding no re-test stamps.
    pub(crate) fn inner_retests_empty(&self) -> bool {
        self.hub
            .inner
            .retests
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }

    /// The stored document, as JSON.
    pub(crate) fn stored(&self) -> Value {
        serde_json::from_str(
            &self
                .ports
                .config
                .raw(&self.scope)
                .expect("a stored document"),
        )
        .unwrap()
    }
}
