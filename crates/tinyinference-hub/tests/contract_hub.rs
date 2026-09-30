//! One operation set for every kind (D6): the same `Hub` calls, driven through
//! every catalogue kind, behave the same way, and a kind that cannot do an
//! operation says so with a typed `Unsupported`, never a silent no-op.
#![cfg(feature = "testing")]

use serde_json::json;

use tinyinference_hub::catalogue;
use tinyinference_hub::testkit::{ContractFixture, Match, MemoryPorts, Scripted};
use tinyinference_hub::{
    Confirm, ConnectOptions, EndpointPolicy, HubError, ManagedConfig, Operation, ProviderDraft,
    ProviderGroup, ProviderPatch, ReasonCode, ScopeKey, Secret, TestDepth,
};

fn draft_for(f: &ContractFixture, kind: &str, key: Option<&str>) -> ProviderDraft {
    let mut draft = ProviderDraft::new(kind);
    // A kind with no preset (OMLX) or an endpoint that is normalised on add
    // is given the fixture's own base.
    if f.group == ProviderGroup::Local {
        draft = draft.with_base_url(f.base_url.clone());
    }
    match key {
        Some(key) => draft.with_key(Secret::new(key)),
        None => draft,
    }
}

fn script_listing(ports: &MemoryPorts, f: &ContractFixture, ids: &[&str]) {
    let body = (f.listing_body)(ids);
    for url in &f.listing_urls {
        ports
            .http
            .route(Match::prefix(url.clone()), Scripted::json(200, &body));
    }
}

#[tokio::test]
async fn contract_hub_every_kind_answers_the_same_operations() {
    let scope = ScopeKey::new("company:contract");
    let mut exercised = 0;
    for descriptor in catalogue::descriptors() {
        let kind = descriptor.kind.as_str();
        let f = ContractFixture::for_builtin(descriptor);
        let ports = MemoryPorts::new();
        let mut builder = ports.builder().policy(if f.group == ProviderGroup::Local {
            EndpointPolicy::desktop()
        } else {
            EndpointPolicy::hosted()
        });
        if f.group == ProviderGroup::Managed {
            builder = builder.managed(ManagedConfig::new(f.base_url.clone()));
        }
        let hub = builder.build().unwrap();
        let slug = f.slug.clone();

        match f.group {
            ProviderGroup::Cli => {
                for op in [Operation::Add, Operation::Connect, Operation::ProbeDraft] {
                    let result = match op {
                        Operation::Add => {
                            hub.add(&scope, ProviderDraft::new(kind)).await.map(|_| ())
                        }
                        Operation::Connect => hub
                            .connect(&scope, ProviderDraft::new(kind), ConnectOptions::default())
                            .await
                            .map(|_| ()),
                        _ => hub
                            .probe_draft(&scope, &ProviderDraft::new(kind), TestDepth::Catalog)
                            .await
                            .map(|_| ()),
                    };
                    assert!(
                        matches!(result, Err(HubError::Unsupported { op: o, .. }) if o == op),
                        "[{kind}] {op}: {result:?}"
                    );
                }
                exercised += 1;
                continue;
            }
            ProviderGroup::Managed => {
                let error = hub.add(&scope, ProviderDraft::new(kind)).await.unwrap_err();
                assert!(
                    matches!(
                        error,
                        HubError::Unsupported {
                            op: Operation::Add,
                            ..
                        }
                    ),
                    "[{kind}] {error:?}"
                );
                let error = hub
                    .remove(&scope, &slug, Confirm::in_use())
                    .await
                    .unwrap_err();
                assert!(
                    matches!(
                        error,
                        HubError::Unsupported {
                            op: Operation::Remove,
                            ..
                        }
                    ),
                    "[{kind}] {error:?}"
                );
                // Signed out until a key exists: typed, not an empty list.
                assert!(matches!(
                    hub.list_models(&scope, &slug, false).await,
                    Err(HubError::SignedOut { .. })
                ));
                hub.set_key(&scope, &slug, Secret::new("th-fake"))
                    .await
                    .unwrap();
            }
            _ => {
                let key = f.key.as_ref().map(|k| k.expose().to_string());
                hub.add(&scope, draft_for(&f, kind, key.as_deref()))
                    .await
                    .unwrap_or_else(|e| panic!("[{kind}] add failed: {e:?}"));
            }
        }

        // The happy path: list, health, status.
        script_listing(&ports, &f, &["m1", "m2"]);
        let listed = hub
            .list_models(&scope, &slug, false)
            .await
            .unwrap_or_else(|e| panic!("[{kind}] list failed: {e:?}"));
        let mut ids: Vec<String> = listed.ids().iter().map(|s| (*s).to_string()).collect();
        ids.sort();
        assert_eq!(ids, ["m1", "m2"], "[{kind}]");
        assert!(hub.health(&scope, &slug).await.is_ok(), "[{kind}]");
        let status = hub.status(&scope).await.unwrap();
        assert!(
            status.providers.iter().any(|p| p.view.record.slug == slug),
            "[{kind}]"
        );

        // Test depths: a supported depth runs, an unsupported one is typed.
        for depth in [
            TestDepth::KeyOnly,
            TestDepth::Catalog,
            TestDepth::Completion,
        ] {
            if depth == TestDepth::KeyOnly
                && let Some(url) = &f.key_check_url
            {
                ports.http.route(
                    Match::prefix(url.clone()),
                    Scripted::json(200, &json!({"data": {"label": "x"}})),
                );
            }
            if depth == TestDepth::Completion {
                for path in ["/chat/completions", "/messages"] {
                    ports.http.route(
                        Match::post(format!("{}{path}", f.base_url.trim_end_matches('/'))),
                        Scripted::json(200, &json!({"choices": [{"message": {"content": "ok"}}], "content": [{"type": "text", "text": "ok"}]})),
                    );
                }
            }
            let model = tinyinference_hub::ModelId::parse("contract-model").unwrap();
            let result = hub.test(&scope, &slug, depth, Some(&model)).await;
            if descriptor.supports_depth(depth) {
                let report = result.unwrap_or_else(|e| panic!("[{kind}] {depth}: {e:?}"));
                assert!(report.ok(), "[{kind}] {depth}: {report:?}");
            } else {
                assert!(
                    matches!(result, Err(HubError::Unsupported { op: Operation::Test(d), .. }) if d == depth),
                    "[{kind}] {depth} must be Unsupported: {result:?}"
                );
            }
        }

        // A rejected key is the same typed failure everywhere it can be observed.
        if f.key.is_some() && f.group != ProviderGroup::Local {
            for url in &f.listing_urls {
                ports.http.route(Match::prefix(url.clone()), Scripted::json(401, &json!({"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}})));
            }
            let error = hub.list_models(&scope, &slug, true).await.unwrap_err();
            assert_eq!(error.reason(), ReasonCode::Auth, "[{kind}] {error:?}");
        }

        // Rotation and removal work on every kind that can be removed.
        hub.set_key(&scope, &slug, Secret::new("sk-rotated"))
            .await
            .unwrap_or_else(|e| panic!("[{kind}] {e:?}"));
        assert!(
            hub.health(&scope, &slug).await.unwrap().health
                == tinyinference_hub::health::ProviderHealth::Unknown
                || f.group == ProviderGroup::Managed,
            "[{kind}] rotation resets health"
        );
        if f.group != ProviderGroup::Managed {
            hub.edit(&scope, &slug, ProviderPatch::new().label("Renamed"))
                .await
                .unwrap();
            hub.clear_default(&scope).await.unwrap();
            hub.remove(&scope, &slug, Confirm::no())
                .await
                .unwrap_or_else(|e| panic!("[{kind}] {e:?}"));
            assert!(
                hub.status(&scope).await.unwrap().providers.is_empty(),
                "[{kind}]"
            );
            assert!(
                ports.credentials.is_empty(),
                "[{kind}] removal deletes the key"
            );
        }
        exercised += 1;
    }
    assert_eq!(exercised, 34, "26 cloud + managed + 5 local + 2 CLI");
}
