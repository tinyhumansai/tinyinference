//! The world the scenario runner plays in: which providers exist, how each one
//! answers in each mode, and the platform token.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde_json::json;

use crate::ids::ScopeKey;
use crate::ports::{PortError, TokenSource};
use crate::secret::Secret;
use crate::testkit::{FakeClock, Match, Scripted, ScriptedHttp};

/// One provider of the simulated world.
#[derive(Clone, Copy, Debug)]
pub struct World {
    /// The routing slug the runner expects the provider to have.
    pub slug: &'static str,
    /// The catalogue kind.
    pub kind: &'static str,
    /// The display name a `custom` draft needs.
    pub label: Option<&'static str>,
    /// The endpoint a draft types (custom kinds); `None` uses the kind's.
    pub draft_base: Option<&'static str>,
    /// The endpoint the saved record has.
    pub base: &'static str,
    /// A second origin the provider can be moved to (an editable endpoint).
    /// Both answer identically; what the runner checks is which credential
    /// each was sent.
    pub alt_base: Option<&'static str>,
    /// Whether the provider takes a key.
    pub keyed: bool,
    /// Whether it is the managed provider.
    pub managed: bool,
}

/// The providers of the world, in the order actions index them.
pub const WORLD: &[World] = &[
    World {
        slug: "openai",
        kind: "openai",
        label: None,
        draft_base: None,
        base: "https://api.openai.com/v1",
        alt_base: None,
        keyed: true,
        managed: false,
    },
    World {
        slug: "groq",
        kind: "groq",
        label: None,
        draft_base: None,
        base: "https://api.groq.com/openai/v1",
        alt_base: None,
        keyed: true,
        managed: false,
    },
    World {
        slug: "mistral",
        kind: "mistral",
        label: None,
        draft_base: None,
        base: "https://api.mistral.ai/v1",
        alt_base: None,
        keyed: true,
        managed: false,
    },
    World {
        slug: "acme",
        kind: "custom",
        label: Some("Acme"),
        draft_base: Some("https://llm.acme.test/v1"),
        base: "https://llm.acme.test/v1",
        alt_base: Some("https://llm.acme-two.test/v1"),
        keyed: true,
        managed: false,
    },
    World {
        slug: "ollama",
        kind: "ollama",
        label: None,
        draft_base: None,
        base: "http://localhost:11434/v1",
        alt_base: None,
        keyed: false,
        managed: false,
    },
    World {
        slug: "tinyhumans",
        kind: "tinyhumans",
        label: None,
        draft_base: None,
        base: MANAGED_BASE,
        alt_base: None,
        keyed: true,
        managed: true,
    },
];

/// The managed backend the runner's hub is configured with.
pub const MANAGED_BASE: &str = "https://api.tinyhumans.test/agent-integrations/openrouter";

/// How a provider answers right now.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Everything works.
    Healthy,
    /// Every request is a `401`.
    AuthFail,
    /// Every request is a spend-cap `429`.
    Quota,
    /// Every request is a `503`.
    Down,
    /// Every request times out.
    Timeout,
    /// Nothing is listening.
    Refused,
    /// A `200` whose body is not a listing.
    Malformed,
}

impl Mode {
    /// Every mode, for random choice.
    pub const ALL: [Mode; 7] = [
        Mode::Healthy,
        Mode::AuthFail,
        Mode::Quota,
        Mode::Down,
        Mode::Timeout,
        Mode::Refused,
        Mode::Malformed,
    ];

    /// Whether a listing read can succeed in this mode.
    pub fn serves(self) -> bool {
        self == Mode::Healthy
    }
}

fn listing(world: &World, id: &str) -> Scripted {
    if world.managed {
        Scripted::json(
            200,
            &json!({"success": true, "data": {"object": "list", "total": 1, "limit": 500, "offset": 0,
                "data": [{"id": id}]}}),
        )
    } else {
        Scripted::json(200, &json!({"data": [{"id": id}]}))
    }
}

/// The scripted answer for a mode. `id` is the model a healthy listing names.
pub(crate) fn answer(world: &World, mode: Mode, id: &str, chat: bool) -> Scripted {
    match mode {
        Mode::Healthy if chat => {
            Scripted::json(200, &json!({"choices": [{"message": {"content": "ok"}}]}))
        }
        Mode::Healthy => listing(world, id),
        Mode::AuthFail => Scripted::json(
            401,
            &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
        ),
        Mode::Quota => Scripted::text(
            429,
            "You exceeded your current quota, please check your plan and billing details",
        ),
        Mode::Down => Scripted::text(503, "service unavailable"),
        Mode::Timeout => Scripted::Timeout,
        Mode::Refused => Scripted::ConnectRefused,
        Mode::Malformed => Scripted::Malformed(b"<html>not a listing</html>".to_vec()),
    }
}

/// Installs the rules for one provider: a listing and a chat rule per known
/// key (matched on the credential the request carries), or one of each for a
/// provider that takes no key or answers every credential the same.
pub(crate) fn install(http: &ScriptedHttp, world: &World, mode: Mode, keys: &[(String, usize)]) {
    let models = format!("{}/models", world.base);
    let chat = format!("{}/chat/completions", world.base);
    if !world.keyed {
        http.route(Match::prefix(models), answer(world, mode, "llama3", false));
        http.route(Match::prefix(chat), answer(world, mode, "llama3", true));
        // Ollama's native listing, which the driver falls back to when the
        // OpenAI-compatible one is unreadable.
        let tags = match mode {
            Mode::Healthy => Scripted::json(200, &json!({"models": [{"name": "llama3"}]})),
            other => answer(world, other, "llama3", false),
        };
        http.route(Match::prefix("http://localhost:11434/api/tags"), tags);
        return;
    }
    if world.managed {
        // Whatever token or pasted key arrives; the runner checks *which*.
        http.route(
            Match::prefix(models).with_header_present("authorization"),
            answer(world, mode, "managed-model", false),
        );
        http.route(
            Match::prefix(chat).with_header_present("authorization"),
            answer(world, mode, "managed-model", true),
        );
        return;
    }
    // Every origin the provider can be at answers the same way for every key:
    // which key went where is the runner's invariant to check, not the world's.
    for base in std::iter::once(world.base).chain(world.alt_base) {
        let models = format!("{base}/models");
        let chat = format!("{base}/chat/completions");
        if world.kind == "custom" {
            // A custom endpoint takes an optional key: with none it is asked
            // without one. Installed first so a per-key rule below wins.
            http.route(
                Match::prefix(models.clone()).without_header("authorization"),
                answer(world, mode, "model-none", false),
            );
            http.route(
                Match::prefix(chat.clone()).without_header("authorization"),
                answer(world, mode, "model-none", true),
            );
        }
        for (secret, id) in keys {
            let bearer = format!("Bearer {secret}");
            let model = format!("model-{id}");
            http.route(
                Match::prefix(models.clone()).with_header("authorization", bearer.clone()),
                answer(world, mode, &model, false),
            );
            http.route(
                Match::prefix(chat.clone()).with_header("authorization", bearer),
                answer(world, mode, &model, true),
            );
        }
    }
}

/// The platform token: a new one every minute of fake time, or none when
/// signed out.
#[derive(Debug)]
pub struct SimToken {
    clock: FakeClock,
    signed_out: Arc<AtomicBool>,
}

impl SimToken {
    /// A token source on `clock`, signed out while `signed_out` is set.
    pub fn new(clock: FakeClock, signed_out: Arc<AtomicBool>) -> Self {
        Self { clock, signed_out }
    }

    /// The token valid `minutes` after the start.
    pub fn at_minute(minutes: u64) -> String {
        format!("platform-token-{minutes}")
    }
}

#[async_trait]
impl TokenSource for SimToken {
    async fn token(&self, _scope: &ScopeKey) -> Result<Option<Secret>, PortError> {
        if self.signed_out.load(Ordering::SeqCst) {
            return Ok(None);
        }
        Ok(Some(Secret::new(Self::at_minute(
            self.clock.elapsed().as_secs() / 60,
        ))))
    }
}
