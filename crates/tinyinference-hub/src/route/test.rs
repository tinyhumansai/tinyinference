//! Tests for the two route grammars and the route types.

use std::collections::BTreeMap;

use super::legacy_oc;
use super::legacy_oh::{self, OhContext};
use super::*;
use crate::ids::{ModelId, Slug};
use crate::import::LossKind;
use crate::taxonomy::{CliKind, LocalRuntime};

fn slug(name: &str) -> Slug {
    Slug::parse(name).unwrap()
}

fn model(id: &str) -> ModelId {
    ModelId::parse(id).unwrap()
}

// ---- types -------------------------------------------------------------------------------

#[test]
fn route_a_temperature_is_finite_by_construction() {
    assert!(Temperature::new(0.7).is_some());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(Temperature::new(bad).is_none());
    }
    assert!(serde_json::from_str::<Temperature>("0.5").is_ok());
    assert_eq!(f32::from(Temperature::new(0.25).unwrap()), 0.25);
    assert!(Temperature::try_from(f32::NAN).is_err());
    let route = ProviderRoute::provider(slug("openai"))
        .with_model(model("m"))
        .with_temperature(Temperature::new(0.5).unwrap());
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&route, &mut hasher);
    assert_eq!(route.clone(), route);
}

#[test]
fn route_wire_form_is_stable_and_round_trips() {
    let route = ProviderRoute::provider(slug("openai")).with_model(model("gpt-5"));
    let text = serde_json::to_string(&route).unwrap();
    assert_eq!(
        text,
        r#"{"target":{"kind":"provider","value":"openai"},"model":"gpt-5"}"#
    );
    assert_eq!(serde_json::from_str::<ProviderRoute>(&text).unwrap(), route);
    for target in [
        RouteTarget::Default,
        RouteTarget::Managed,
        RouteTarget::Local(None),
        RouteTarget::Local(Some(LocalRuntime::LmStudio)),
        RouteTarget::Cli(CliKind::Codex),
        RouteTarget::Ephemeral,
    ] {
        let route = ProviderRoute::new(target);
        let back: ProviderRoute =
            serde_json::from_str(&serde_json::to_string(&route).unwrap()).unwrap();
        assert_eq!(back, route);
    }
    assert_eq!(ProviderRoute::default_route().provider_slug(), None);
    assert_eq!(route.provider_slug(), Some(&slug("openai")));
}

#[test]
fn route_check_rejects_a_default_or_ephemeral_target() {
    assert!(check_route(&ProviderRoute::provider(slug("a"))).is_ok());
    assert!(check_route(&ProviderRoute::default_route()).is_err());
    assert!(check_route(&ProviderRoute::new(RouteTarget::Ephemeral)).is_err());
}

#[test]
fn route_a_turn_query_builds_from_its_parts() {
    let query = TurnQuery::new()
        .with_agent("a".into())
        .with_workload("w".into())
        .with_override(ProviderRoute::default_route());
    assert!(query.agent.is_some() && query.workload.is_some() && query.override_route.is_some());
}

#[test]
fn route_a_resolved_turn_debug_redacts_the_endpoint_and_has_no_secret() {
    let turn = ResolvedTurn {
        slug: slug("acme"),
        kind: "custom".into(),
        group: crate::taxonomy::ProviderGroup::Custom,
        base_url: "https://user:hunter2@llm.acme.test/v1".into(),
        model: Some(model("m")),
        protocol: crate::taxonomy::Protocol::OpenAiChat,
        auth: crate::taxonomy::AuthStyle::Bearer,
        via: ResolvedVia::Default,
        origin: None,
        temperature: None,
        cli: None,
    };
    assert!(!format!("{turn:?}").contains("hunter2"));
}

// ---- OpenCompany grammar -----------------------------------------------------------------

#[test]
fn import_oc_route_grammar_table() {
    let cases: Vec<(&str, RouteTarget, Option<&str>, Option<&str>)> = vec![
        ("", RouteTarget::Default, None, Some("")),
        ("  ", RouteTarget::Default, None, Some("")),
        ("default", RouteTarget::Default, None, Some("")),
        ("DEFAULT", RouteTarget::Default, None, Some("")),
        ("managed", RouteTarget::Managed, None, Some("managed")),
        ("local", RouteTarget::Local(None), None, Some("local")),
        (
            "local:llama3",
            RouteTarget::Local(None),
            Some("llama3"),
            Some("local:llama3"),
        ),
        (
            "claude-code",
            RouteTarget::Cli(CliKind::ClaudeCode),
            None,
            Some("claude-code"),
        ),
        (
            "claude-code:sonnet",
            RouteTarget::Cli(CliKind::ClaudeCode),
            Some("sonnet"),
            Some("claude-code:sonnet"),
        ),
        (
            "openai:gpt-5",
            RouteTarget::Provider(slug("openai")),
            Some("gpt-5"),
            Some("openai:gpt-5"),
        ),
        (
            "acme",
            RouteTarget::Provider(slug("acme")),
            None,
            Some("acme"),
        ),
        (
            "ollama:llama3:8b",
            RouteTarget::Provider(slug("ollama")),
            Some("llama3:8b"),
            Some("ollama:llama3:8b"),
        ),
        (
            "acme:",
            RouteTarget::Provider(slug("acme")),
            None,
            Some("acme"),
        ),
    ];
    for (raw, target, model_id, back) in cases {
        let route = legacy_oc::parse(raw).unwrap_or_else(|e| panic!("{raw:?}: {e}"));
        assert_eq!(route.target, target, "{raw:?}");
        assert_eq!(
            route.model.as_ref().map(ModelId::as_str),
            model_id,
            "{raw:?}"
        );
        assert_eq!(legacy_oc::to_string(&route).as_deref(), back, "{raw:?}");
    }
}

#[test]
fn import_oc_route_grammar_refuses_what_is_not_a_slug_or_model() {
    for raw in [
        "Bad Slug",
        "openai:has space",
        ":model",
        "a/b:m",
        "openai:\u{7}bell",
    ] {
        assert!(legacy_oc::parse(raw).is_err(), "{raw:?}");
    }
}

#[test]
fn import_oc_to_string_says_none_for_what_the_grammar_cannot_say() {
    let hot = ProviderRoute::provider(slug("a")).with_temperature(Temperature::new(0.1).unwrap());
    assert_eq!(legacy_oc::to_string(&hot), None);
    for target in [
        RouteTarget::Ephemeral,
        RouteTarget::Local(Some(LocalRuntime::Vllm)),
        RouteTarget::Cli(CliKind::Codex),
    ] {
        assert_eq!(legacy_oc::to_string(&ProviderRoute::new(target)), None);
    }
    assert_eq!(
        legacy_oc::to_string(&ProviderRoute::default_route().with_model(model("m"))),
        None,
        "a default with a model is not expressible"
    );
}

// ---- OpenHuman grammar (06 section 3) ----------------------------------------------------

fn oh(raw: &str) -> legacy_oh::OhParsed {
    legacy_oh::parse(raw, &OhContext::new())
}

fn kinds(parsed: &legacy_oh::OhParsed) -> Vec<LossKind> {
    parsed.loss.iter().map(|e| e.kind).collect()
}

#[test]
fn oh_temp_suffix_is_split_at_the_last_at_only_when_it_is_a_finite_number() {
    // Case 1, the five cases named in the plan.
    let plain = oh("openai:m@0.7");
    let route = plain.route.clone().unwrap();
    assert_eq!(route.model.as_ref().unwrap().as_str(), "m");
    assert_eq!(route.temperature.unwrap().get(), 0.7);
    assert!(plain.loss.is_empty(), "{:?}", plain.loss);

    let not_a_number = oh("openai:m@x");
    assert_eq!(
        not_a_number.route.clone().unwrap().model.unwrap().as_str(),
        "m@x"
    );
    assert_eq!(kinds(&not_a_number), [LossKind::Ambiguous]);

    let two = oh("openai:a@b@0.2");
    let route = two.route.clone().unwrap();
    assert_eq!(
        (
            route.model.unwrap().as_str(),
            route.temperature.unwrap().get()
        ),
        ("a@b", 0.2)
    );
    assert_eq!(kinds(&two), [LossKind::Ambiguous]);

    let nan = oh("openai:m@NaN");
    assert_eq!(nan.route.clone().unwrap().model.unwrap().as_str(), "m@NaN");
    assert_eq!(kinds(&nan), [LossKind::Ambiguous]);

    let cli = oh("claude-code:m@0.3");
    let route = cli.route.clone().unwrap();
    assert_eq!(route.target, RouteTarget::Cli(CliKind::ClaudeCode));
    assert_eq!(
        (route.model.unwrap().as_str(), route.temperature),
        ("m", None)
    );
    assert_eq!(kinds(&cli), [LossKind::Dropped]);
}

#[test]
fn oh_ollama_prefix_overloaded_reads_as_the_configured_local_runtime() {
    // Case 2.
    let context = OhContext::new().with_local_ai_runtime(LocalRuntime::LmStudio);
    let parsed = legacy_oh::parse("ollama:qwen", &context);
    assert_eq!(
        parsed.route.clone().unwrap().target,
        RouteTarget::Local(Some(LocalRuntime::LmStudio))
    );
    assert_eq!(kinds(&parsed), [LossKind::Ambiguous]);
    // No local_ai, or local_ai really is Ollama: read literally.
    assert_eq!(
        oh("ollama:qwen").route.clone().unwrap().target,
        RouteTarget::Local(Some(LocalRuntime::Ollama))
    );
    let literal = legacy_oh::parse(
        "ollama:qwen",
        &OhContext::new().with_local_ai_runtime(LocalRuntime::Ollama),
    );
    assert!(literal.loss.is_empty());
}

#[test]
fn oh_bare_strings_follow_the_backend_and_unknown_ones_fail_closed() {
    // Case 3.
    let vllm = oh("vllm");
    assert_eq!(
        vllm.route.clone().unwrap().target,
        RouteTarget::Local(Some(LocalRuntime::Vllm))
    );
    let openai = oh("openai");
    assert_eq!(
        openai.route.clone().unwrap().target,
        RouteTarget::Local(Some(LocalRuntime::OpenAiCompatible))
    );
    assert_eq!(kinds(&openai), [LossKind::Ambiguous]);
    for unknown in ["anthropic", "gibberish", "some-provider"] {
        let parsed = oh(unknown);
        assert!(parsed.route.is_none(), "{unknown}");
        assert_eq!(kinds(&parsed), [LossKind::FailClosed]);
    }
    // The prefixed form is the hosted row, never a local runtime.
    assert_eq!(
        oh("openai:gpt-5").route.clone().unwrap().target,
        RouteTarget::Provider(slug("openai"))
    );
    assert_eq!(
        oh("lmstudio").route.clone().unwrap().target,
        RouteTarget::Local(Some(LocalRuntime::LmStudio))
    );
    assert_eq!(
        oh("claude-code").route.clone().unwrap().target,
        RouteTarget::Cli(CliKind::ClaudeCode)
    );
    assert_eq!(
        oh("claude_agent_sdk").route.clone().unwrap().target,
        RouteTarget::Cli(CliKind::ClaudeCode)
    );
    assert_eq!(
        oh("ephemeral-route").route.clone().unwrap().target,
        RouteTarget::Ephemeral
    );
    assert_eq!(
        oh("byok-inference").route.clone().unwrap().target,
        RouteTarget::Provider(slug("byok-inference"))
    );
}

#[test]
fn oh_empty_route_variants() {
    // Case 4.
    for raw in ["", "  ", "cloud", "CLOUD"] {
        let parsed = oh(raw);
        assert_eq!(
            parsed.route.clone().unwrap().target,
            RouteTarget::Default,
            "{raw:?}"
        );
    }
    assert_eq!(
        oh("openhuman").route.clone().unwrap().target,
        RouteTarget::Managed
    );
    let incomplete = oh(legacy_oh::BYOK_INCOMPLETE);
    assert!(incomplete.route.is_none());
    assert_eq!(kinds(&incomplete), [LossKind::FailClosed]);
}

#[test]
fn oh_hint_models_are_replaced_by_the_rows_default_or_fail_closed() {
    // Case 5.
    let context = OhContext::new().with_default_model(slug("anthropic"), model("claude-x"));
    let replaced = legacy_oh::parse("anthropic:hint:reasoning", &context);
    assert_eq!(
        replaced.route.clone().unwrap().model.unwrap().as_str(),
        "claude-x"
    );
    assert_eq!(kinds(&replaced), [LossKind::Normalised]);
    let none = legacy_oh::parse("groq:hint:fast", &context);
    let route = none.route.clone().unwrap();
    assert_eq!(route.model, None, "the route stands without a model");
    assert_eq!(kinds(&none), [LossKind::FailClosed]);
}

#[test]
fn oh_synthetic_routes_and_local_aliases() {
    // Case 7 and the alias table.
    let byok = oh("byok-inference:m1");
    assert_eq!(
        byok.route.as_ref().unwrap().provider_slug(),
        Some(&slug("byok-inference"))
    );
    assert!(byok.loss.is_empty());
    let ephemeral = oh("ephemeral-route:m2");
    assert_eq!(
        ephemeral.route.clone().unwrap().target,
        RouteTarget::Ephemeral
    );
    let sdk = oh("claude_agent_sdk:opus");
    assert_eq!(
        sdk.route.clone().unwrap().target,
        RouteTarget::Cli(CliKind::ClaudeCode)
    );
    assert_eq!(
        kinds(&sdk),
        [LossKind::Normalised],
        "claude_agent_sdk is written back as claude-code"
    );
    for (prefix, runtime) in [
        ("lm-studio", LocalRuntime::LmStudio),
        ("lm_studio", LocalRuntime::LmStudio),
        ("mlx-server", LocalRuntime::Mlx),
        ("mlx_lm", LocalRuntime::Mlx),
        ("omlx", LocalRuntime::Omlx),
        ("custom-openai", LocalRuntime::OpenAiCompatible),
        ("llamacpp", LocalRuntime::LlamaCpp),
        ("llama.cpp", LocalRuntime::LlamaCpp),
        ("vllm", LocalRuntime::Vllm),
    ] {
        let parsed = oh(&format!("{prefix}:m"));
        assert_eq!(
            parsed.route.clone().unwrap().target,
            RouteTarget::Local(Some(runtime)),
            "{prefix}"
        );
        // Written back with a prefix OpenHuman reads the same way, and this one
        // is stable.
        let written = legacy_oh::to_string(&parsed.route.unwrap()).unwrap();
        assert_eq!(
            oh(&written).route.unwrap().target,
            RouteTarget::Local(Some(runtime)),
            "{prefix} via {written}"
        );
    }
}

#[test]
fn oh_an_unwritable_model_or_slug_fails_closed_instead_of_guessing() {
    for raw in ["Bad Slug:m", "acme:has space", "acme:\u{7}"] {
        let parsed = oh(raw);
        assert!(parsed.route.is_none(), "{raw:?}");
        assert_eq!(kinds(&parsed), [LossKind::FailClosed], "{raw:?}");
    }
}

#[test]
fn oh_a_round_trip_reproduces_the_input_or_says_why_not() {
    let corpus = [
        "",
        "cloud",
        "openhuman",
        "openai:gpt-5",
        "openai:gpt-5@0.4",
        "acme:m",
        "ollama:llama3",
        "lmstudio:qwen",
        "lm-studio:qwen",
        "vllm:m",
        "mlx:m",
        "omlx:m",
        "local-openai:m",
        "claude-code:m",
        "claude_agent_sdk:m",
        "ephemeral-route:m",
        "byok-inference:m",
        "vllm",
        "openai",
        "ollama",
        "openai:m@x",
        "openai:a@b@0.2",
        "openai:m@1.0",
        "claude-code",
    ];
    for raw in corpus {
        let parsed = oh(raw);
        let route = parsed.route.unwrap_or_else(|| panic!("{raw:?}"));
        let written = legacy_oh::to_string(&route).unwrap_or_else(|| panic!("{raw:?} unwritable"));
        assert!(
            written == raw.trim() || !parsed.loss.is_empty(),
            "{raw:?} became {written:?} with no loss entry"
        );
        // Writing then reading again is stable.
        let again = legacy_oh::parse(&written, &OhContext::new());
        assert_eq!(
            again.route.as_ref(),
            Some(&route),
            "{raw:?} via {written:?}"
        );
    }
}

#[test]
fn oh_to_string_says_none_for_what_the_grammar_cannot_say() {
    assert_eq!(
        legacy_oh::to_string(&ProviderRoute::default_route().with_model(model("m"))),
        None
    );
    assert_eq!(
        legacy_oh::to_string(&ProviderRoute::new(RouteTarget::Managed).with_model(model("m"))),
        None
    );
    assert_eq!(
        legacy_oh::to_string(&ProviderRoute::provider(slug("openrouter"))),
        None,
        "a bare provider name does not read back as that provider"
    );
    assert_eq!(
        legacy_oh::to_string(
            &ProviderRoute::new(RouteTarget::Cli(CliKind::Codex)).with_model(model("m"))
        ),
        None,
        "another CLI would read back as claude-code"
    );
    assert_eq!(
        legacy_oh::to_string(&ProviderRoute::provider(slug("ollama")).with_model(model("m"))),
        None,
        "a slug that is a local runtime prefix would read back as the runtime"
    );
    assert_eq!(
        legacy_oh::to_string(&ProviderRoute::provider(slug("openrouter")).with_model(model("m")))
            .as_deref(),
        Some("openrouter:m")
    );
    assert_eq!(
        legacy_oh::to_string(&ProviderRoute::new(RouteTarget::Local(None))).as_deref(),
        Some("ollama")
    );
}

#[test]
fn oh_the_two_deliberately_inexact_spellings_are_documented_behaviour() {
    // Finding 4.7. `Local(None)`: written as OpenHuman's own spelling for "the
    // configured local runtime", read back as Ollama, or as the configured
    // runtime when the context names one.
    let any_local = ProviderRoute::new(RouteTarget::Local(None)).with_model(model("llama3"));
    let written = legacy_oh::to_string(&any_local).unwrap();
    assert_eq!(written, "ollama:llama3");
    assert_eq!(
        legacy_oh::parse(&written, &OhContext::new()).route,
        Some(
            ProviderRoute::new(RouteTarget::Local(Some(LocalRuntime::Ollama)))
                .with_model(model("llama3"))
        ),
        "with no context it reads back as Ollama, not as `Local(None)`"
    );
    let context = OhContext::new().with_local_ai_runtime(LocalRuntime::Mlx);
    let parsed = legacy_oh::parse(&written, &context);
    assert_eq!(
        parsed.route.unwrap().target,
        RouteTarget::Local(Some(LocalRuntime::Mlx)),
        "with the configured runtime it reads back as that runtime"
    );
    assert!(
        parsed.loss.iter().any(|e| e.kind == LossKind::Ambiguous),
        "and the report says the spelling is ambiguous"
    );

    // A CLI route with a temperature: the temperature is not written, and the
    // string reads back as the same route without it, with a `Dropped` note only
    // when the input had one.
    let mut cli =
        ProviderRoute::new(RouteTarget::Cli(CliKind::ClaudeCode)).with_model(model("sonnet"));
    cli.temperature = Temperature::new(0.5);
    let written = legacy_oh::to_string(&cli).unwrap();
    assert_eq!(written, "claude-code:sonnet");
    let back = legacy_oh::parse(&written, &OhContext::new());
    cli.temperature = None;
    assert_eq!(back.route, Some(cli));
    assert!(back.loss.is_empty(), "{:?}", back.loss);
    let with_temperature = legacy_oh::parse("claude-code:sonnet@0.5", &OhContext::new());
    assert!(
        with_temperature
            .loss
            .iter()
            .any(|e| e.kind == LossKind::Dropped)
    );
}

mod route_props {
    use proptest::prelude::*;

    use super::*;

    proptest! {
        /// Neither grammar ever panics on arbitrary text, and what parses
        /// writes back to something that parses to the same route.
        #[test]
        fn route_prop_parsers_are_total_and_round_trip(raw in "\\PC{0,60}") {
            let _ = legacy_oc::parse(&raw);
            let parsed = legacy_oh::parse(&raw, &OhContext::new());
            if let Some(route) = parsed.route
                && let Some(written) = legacy_oh::to_string(&route)
            {
                let again = legacy_oh::parse(&written, &OhContext::new());
                prop_assert_eq!(again.route, Some(route));
            }
        }

        #[test]
        fn route_prop_oc_strings_round_trip_exactly(
            head in "[a-z0-9][a-z0-9_-]{0,12}",
            m in proptest::option::of("[a-zA-Z0-9./:_-]{1,20}"),
        ) {
            let raw = match &m { Some(m) => format!("{head}:{m}"), None => head.clone() };
            let route = legacy_oc::parse(&raw).unwrap();
            let written = legacy_oc::to_string(&route).unwrap();
            prop_assert_eq!(legacy_oc::parse(&written).unwrap(), route);
        }

        #[test]
        fn route_prop_serde_round_trips(
            head in "[a-z0-9][a-z0-9_-]{0,12}",
            m in proptest::option::of("[a-zA-Z0-9./:_-]{1,20}"),
            t in proptest::option::of(-5.0f32..5.0),
        ) {
            let mut route = ProviderRoute::provider(Slug::parse(&head).unwrap());
            route.model = m.map(|m| ModelId::parse(&m).unwrap());
            route.temperature = t.and_then(Temperature::new);
            let back: ProviderRoute = serde_json::from_str(&serde_json::to_string(&route).unwrap()).unwrap();
            prop_assert_eq!(back, route);
        }
    }

    #[test]
    fn route_the_property_map_type_is_used() {
        let _: BTreeMap<String, String> = BTreeMap::new();
    }
}
