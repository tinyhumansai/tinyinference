//! Public API checks for provider overrides as consumed by an external host.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tinyinference_embeddings::{EmbeddingModel, MockEmbeddingModel};
use tinyinference_llm::model::ChatModel;
use tinyinference_llm::providers::MockModel;
use tinyinference_llm::providers::overrides::{
    CatalogEntry, CatalogLimitsFuture, CatalogLimitsRequest, CatalogRequest, ChatModelEntry,
    ChatModelRequest, EmbeddingFactoryFuture, EmbeddingRequest, ProviderModelMetadata,
    ProviderOverrides, ReadinessRequest,
};

struct HostContext<'a> {
    borrowed_config: &'a str,
    // Deliberately non-Clone, like a host context that borrows configuration.
    _ownership_marker: std::marker::PhantomData<&'a mut ()>,
}

#[tokio::test]
async fn asynchronous_factories_accept_owned_non_clone_host_contexts() {
    let context = HostContext {
        borrowed_config: "borrowed-host-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let overrides = ProviderOverrides::new(
        |id| {
            (id == "hosted").then(|| ChatModelEntry {
                build: Arc::new(|request: &ChatModelRequest<'_, HostContext<'_>>| {
                    assert_eq!(request.context.borrowed_config, "borrowed-host-config");
                    Ok((
                        Arc::new(MockModel::constant("borrowed")) as Arc<dyn ChatModel<()>>,
                        "borrowed-model".into(),
                    ))
                }),
                probe: None,
                models: None,
            })
        },
        |id| {
            (id == "hosted").then(|| {
                Arc::new(
                    |request: EmbeddingRequest<HostContext<'static>>| -> EmbeddingFactoryFuture {
                        Box::pin(async move {
                            assert_eq!(request.context.borrowed_config, "borrowed-host-config");
                            Ok(Box::new(MockEmbeddingModel::new(request.dimensions))
                                as Box<dyn EmbeddingModel>)
                        })
                    },
                )
                    as tinyinference_llm::providers::overrides::EmbeddingFactory<
                        HostContext<'static>,
                    >
            })
        },
        |_| None,
    );

    let chat = overrides.build_chat(
        "hosted",
        &ChatModelRequest {
            role: "chat",
            model_override: None,
            default_model: None,
            native_tool_calling: false,
            thread_id: None,
            context: &context,
        },
        |_| panic!("registered chat factory should be used"),
    );
    assert_eq!(chat.unwrap().1, "borrowed-model");

    let embedding = overrides
        .build_embedding(
            "hosted",
            EmbeddingRequest {
                model: "embed".into(),
                dimensions: 3,
                context,
            },
            |_| panic!("registered embedding factory should be used"),
        )
        .await;
    assert_eq!(embedding.unwrap().dimensions(), 3);
}

#[tokio::test]
async fn independent_live_family_lookups_override_only_their_builtin() {
    let global_context = HostContext {
        borrowed_config: "global-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let scoped_context = HostContext {
        borrowed_config: "scoped-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let chat_enabled = Arc::new(AtomicBool::new(true));
    let chat_enabled_lookup = Arc::clone(&chat_enabled);
    let overrides = ProviderOverrides::new(
        move |id| {
            (id == "shared" && chat_enabled_lookup.load(Ordering::SeqCst)).then(|| ChatModelEntry {
                build: Arc::new(|request: &ChatModelRequest<'_, HostContext<'_>>| {
                    assert_eq!(request.context.borrowed_config, "global-config");
                    if request.model_override == Some("fail") {
                        return Err(tinyinference_llm::Error::Validation(
                            "override construction failed".into(),
                        ));
                    }
                    Ok((
                        Arc::new(MockModel::constant("chat override")) as Arc<dyn ChatModel<()>>,
                        "chat-model".into(),
                    ))
                }),
                probe: Some(Arc::new(
                    |request: ReadinessRequest<HostContext<'static>>| {
                        Box::pin(async move {
                            assert_eq!(request.role, "chat");
                            assert_eq!(request.model, "chat-model");
                            assert_eq!(request.context.borrowed_config, "readiness-config");
                            Ok(())
                        })
                    },
                )),
                models: None,
            })
        },
        |id| {
            (id == "shared").then(|| {
                Arc::new(
                    |request: EmbeddingRequest<HostContext<'static>>| -> EmbeddingFactoryFuture {
                        Box::pin(async move {
                            assert_eq!(request.context.borrowed_config, "scoped-config");
                            Ok(Box::new(MockEmbeddingModel::new(request.dimensions))
                                as Box<dyn EmbeddingModel>)
                        })
                    },
                )
                    as tinyinference_llm::providers::overrides::EmbeddingFactory<
                        HostContext<'static>,
                    >
            })
        },
        |id| {
            (id == "shared").then(|| CatalogEntry {
                list: Arc::new(|request: CatalogRequest<HostContext<'static>>| {
                    Box::pin(async move {
                        assert_eq!(request.context.borrowed_config, "catalog-config");
                        Ok(serde_json::json!({"models": []}))
                    })
                }),
                limits: Some(Arc::new(
                    |request: CatalogLimitsRequest<HostContext<'static>>| -> CatalogLimitsFuture {
                        Box::pin(async move {
                            (request.model == "catalog-model"
                                && request.context.borrowed_config == "catalog-config")
                                .then(|| {
                                    tinyinference_llm::model::discover::DiscoveryRequest::new(
                                        "https://provider.invalid/v1",
                                        request.model,
                                    )
                                })
                        })
                    },
                )),
            })
        },
    );
    let _cloned_overrides = overrides.clone();

    let chat = overrides.build_chat(
        "shared",
        &ChatModelRequest {
            role: "chat",
            model_override: Some("chosen-model"),
            default_model: None,
            native_tool_calling: true,
            thread_id: Some("thread"),
            context: &global_context,
        },
        |_| panic!("chat override must run before builtin"),
    );
    assert_eq!(chat.unwrap().1, "chat-model");

    let override_error = overrides.build_chat(
        "shared",
        &ChatModelRequest {
            role: "chat",
            model_override: Some("fail"),
            default_model: None,
            native_tool_calling: true,
            thread_id: None,
            context: &global_context,
        },
        |_| panic!("a failed override must not fall through to builtin"),
    );
    assert!(override_error.is_err());

    let embedding = overrides
        .build_embedding(
            "shared",
            EmbeddingRequest {
                model: "embed-model".into(),
                dimensions: 8,
                context: scoped_context,
            },
            |_| panic!("embedding override must run before builtin"),
        )
        .await;
    assert_eq!(embedding.unwrap().dimensions(), 8);

    let fallback_context = HostContext {
        borrowed_config: "fallback-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let fallback = overrides
        .build_embedding(
            "unregistered",
            EmbeddingRequest {
                model: "embed-model".into(),
                dimensions: 4,
                context: fallback_context,
            },
            |_| {
                Box::pin(async {
                    Ok(Box::new(MockEmbeddingModel::new(4)) as Box<dyn EmbeddingModel>)
                })
            },
        )
        .await;
    assert_eq!(fallback.unwrap().dimensions(), 4);

    chat_enabled.store(false, Ordering::SeqCst);
    let chat_fallback = overrides.build_chat(
        "shared",
        &ChatModelRequest {
            role: "chat",
            model_override: None,
            default_model: Some("builtin-model"),
            native_tool_calling: false,
            thread_id: None,
            context: &HostContext {
                borrowed_config: "fallback-config",
                _ownership_marker: std::marker::PhantomData,
            },
        },
        |_| {
            Ok((
                Arc::new(MockModel::constant("builtin chat")) as Arc<dyn ChatModel<()>>,
                "builtin-model".into(),
            ))
        },
    );
    assert_eq!(chat_fallback.unwrap().1, "builtin-model");
    chat_enabled.store(true, Ordering::SeqCst);

    let chat_entry = overrides.chat_entry("shared").unwrap();
    let probe = chat_entry.probe.unwrap();
    (probe)(ReadinessRequest {
        role: "chat".into(),
        model: "chat-model".into(),
        context: HostContext {
            borrowed_config: "readiness-config",
            _ownership_marker: std::marker::PhantomData,
        },
    })
    .await
    .unwrap();

    let catalog_context = HostContext {
        borrowed_config: "catalog-config",
        _ownership_marker: std::marker::PhantomData,
    };
    let limits = overrides.catalog_entry("shared").unwrap().limits.unwrap();
    assert!(
        limits(CatalogLimitsRequest {
            model: "catalog-model".into(),
            context: catalog_context,
        })
        .await
        .is_some()
    );
    let listing = overrides
        .list_catalog(
            "shared",
            CatalogRequest {
                context: HostContext {
                    borrowed_config: "catalog-config",
                    _ownership_marker: std::marker::PhantomData,
                },
            },
            |_| panic!("catalog override must run before builtin"),
        )
        .await
        .unwrap();
    assert_eq!(listing["models"], serde_json::json!([]));

    let builtin_listing = overrides
        .list_catalog(
            "unregistered",
            CatalogRequest {
                context: HostContext {
                    borrowed_config: "builtin-catalog-config",
                    _ownership_marker: std::marker::PhantomData,
                },
            },
            |_| Box::pin(async { Ok(serde_json::json!({"builtin": true})) }),
        )
        .await
        .unwrap();
    assert_eq!(builtin_listing["builtin"], true);
}

#[test]
fn registered_chat_entry_exposes_all_provider_model_metadata() {
    let overrides = ProviderOverrides::new(
        |id| {
            (id == "hosted").then(|| ChatModelEntry {
                build: Arc::new(|_| unreachable!("metadata test does not build a model")),
                probe: None,
                models: Some(ProviderModelMetadata {
                    default_model: Arc::new(|| Some("provider-default".into())),
                    role_for_model: Arc::new(|model| {
                        (model == "chat-alias").then(|| "chat".into())
                    }),
                    is_alias: Arc::new(|model| model == "chat-alias"),
                    is_raw_model: Arc::new(|model| model.starts_with("raw/")),
                    supports_vision: Arc::new(|model| model == "vision-model"),
                    resolve_model: Arc::new(|role, context: &HostContext<'_>, pin| {
                        if role == "chat" {
                            Some(pin.unwrap_or(context.borrowed_config).to_owned())
                        } else {
                            None
                        }
                    }),
                    context_window: Arc::new(|model| (model == "vision-model").then_some(131_072)),
                    default_pricing: Arc::new(|| Some(("provider-default".into(), 0.25, 1.5))),
                    without_reasoning: Arc::new(|mut request| {
                        request.metadata["reasoning_disabled"] = serde_json::Value::Bool(true);
                        request
                    }),
                }),
            })
        },
        |_| None,
        |_| None,
    );

    let entry = overrides.chat_entry("hosted").expect("registered entry");
    let metadata = entry.models.expect("provider metadata");
    let context = HostContext {
        borrowed_config: "context-default",
        _ownership_marker: std::marker::PhantomData,
    };

    assert_eq!((metadata.default_model)(), Some("provider-default".into()));
    assert_eq!((metadata.role_for_model)("chat-alias"), Some("chat".into()));
    assert!((metadata.is_alias)("chat-alias"));
    assert!((metadata.is_raw_model)("raw/model"));
    assert!((metadata.supports_vision)("vision-model"));
    assert_eq!(
        (metadata.resolve_model)("chat", &context, None),
        Some("context-default".into())
    );
    assert_eq!(
        (metadata.resolve_model)("chat", &context, Some("explicit-model")),
        Some("explicit-model".into())
    );
    assert_eq!((metadata.context_window)("vision-model"), Some(131_072));
    assert_eq!(
        (metadata.default_pricing)(),
        Some(("provider-default".into(), 0.25, 1.5))
    );
    let request = (metadata.without_reasoning)(tinyinference_llm::model::ModelRequest {
        model: Some("request-model".into()),
        metadata: serde_json::json!({"caller": "retained"}),
        ..Default::default()
    });
    assert_eq!(request.model.as_deref(), Some("request-model"));
    assert_eq!(request.metadata["caller"], "retained");
    assert_eq!(request.metadata["reasoning_disabled"], true);
}
