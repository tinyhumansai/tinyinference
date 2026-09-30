//! A minimal host: everything a program needs to plug `tinyinference-hub` in.
//!
//! It uses only the in-memory ports and the scripted transport from `testkit`,
//! so it runs with no network, no keychain and no files:
//!
//! ```sh
//! cargo run -p tinyinference-hub --example minimal_host --features testing
//! ```
//!
//! A real host replaces the four required ports (`CredentialStore`,
//! `ConfigStore`, `Http`, `Clock`) with its own (the `http-reqwest` feature
//! ships a reference `Http`), and keeps everything else.

use serde_json::json;
use tinyinference_hub::testkit::{Match, MemoryPorts, Scripted};
use tinyinference_hub::{
    ConnectOptions, HubError, ModelChoice, ModelId, ProviderDraft, ScopeKey, Secret, Slug,
    TurnQuery,
};

/// Runs the walk-through and returns what it printed.
pub(crate) async fn transcript() -> Result<String, HubError> {
    let mut out = String::new();
    let mut say = |line: String| {
        out.push_str(&line);
        out.push('\n');
    };

    // 1. The ports. (A real host implements its own; these are in memory.)
    let ports = MemoryPorts::new();
    let hub = ports.hub();
    let me = ScopeKey::new("user:local");

    // 2. Something for the "network" to say.
    ports.http.route(
        Match::get("https://api.openai.com/v1/models"),
        Scripted::json(200, &json!({"data": [{"id": "gpt-x"}, {"id": "gpt-y"}]})),
    );

    // 3. Connect a provider: the hub checks the key by reading the catalog, and
    //    keeps the row because the check passed.
    let draft = ProviderDraft::new("openai")
        .with_key(Secret::new("sk-not-a-real-key"))
        .with_model(ModelId::parse("gpt-x").expect("a valid model id"));
    let added = hub.connect(&me, draft, ConnectOptions::default()).await?;
    say(format!("connect: {:?} ({})", added.status, added.note));

    // 4. List models and choose the default. (The first provider added is made
    //    the default automatically; this shows the explicit call.)
    let slug = Slug::parse("openai").expect("a valid slug");
    let models = hub.list_models(&me, &slug, false).await?;
    say(format!("models: {:?}", models.ids()));
    hub.set_default(
        &me,
        ModelChoice::new(slug.clone(), models.models[1].id.clone()),
    )
    .await?;

    // 5. Resolve what a turn should call. The result carries no credential and
    //    its Debug is safe to log.
    let turn = hub.resolve_for_turn(&me, &TurnQuery::new()).await?;
    say(format!("turn: {turn:?}"));

    // 6. The key stops working upstream. The next check says so, and the health
    //    a UI shows follows it.
    ports.http.route(
        Match::get("https://api.openai.com/v1/models"),
        Scripted::json(
            401,
            &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}}),
        ),
    );
    let report = hub
        .test(&me, &slug, tinyinference_hub::TestDepth::Catalog, None)
        .await?;
    let failure = report.failure.expect("the provider now rejects the key");
    say(format!(
        "test: {} (retry {:?})",
        failure.reason, failure.retry
    ));
    say(format!(
        "health: {:?}",
        hub.health(&me, &slug).await?.health
    ));

    // 7. A rejected key rolls a *new* add back instead of saving a broken row.
    let other = ScopeKey::new("user:other");
    let refused = hub
        .connect(
            &other,
            ProviderDraft::new("openai").with_key(Secret::new("sk-not-a-real-key")),
            ConnectOptions::default(),
        )
        .await;
    say(format!(
        "connect with a rejected key: {} ({} providers kept)",
        refused.map_or_else(|e| e.reason().to_string(), |_| "saved".to_string()),
        hub.status(&other).await?.providers.len()
    ));

    // The key never appeared anywhere that prints.
    assert!(!out.contains("sk-not-a-real-key"));
    Ok(out)
}

fn main() -> Result<(), HubError> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .map_err(|_| HubError::Conflict)?;
    print!("{}", runtime.block_on(transcript())?);
    Ok(())
}
