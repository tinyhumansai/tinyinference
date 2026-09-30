//! Tests for the error types, `HubError`'s accessors, the copy, and the
//! conversion from llm's error.

use std::time::Duration;

use serde_json::json;
use tinyinference_llm::model::ProviderError;

use super::*;
use crate::ids::{AgentKey, KindId, ModelId, Slug, WorkloadKey};
use crate::policy::EndpointRefusal;
use crate::secret::LogOnly;
use crate::taxonomy::{ProviderGroup, TestDepth};

fn slug(s: &str) -> Slug {
    Slug::parse(s).unwrap()
}

fn every_error() -> Vec<HubError> {
    vec![
        HubError::Provider(ProviderFailure::new(ReasonCode::Auth, Retry::Never)),
        HubError::SignedOut {
            provider: slug("tinyhumans"),
        },
        HubError::Unsupported {
            op: Operation::Remove,
            kind: KindId::new("tinyhumans"),
        },
        HubError::Policy(PolicyViolation::Endpoint(EndpointRefusal::LinkLocal)),
        HubError::Invalid(InvalidInput::Empty(InputField::Slug)),
        HubError::NotFound(NotFound::Provider(slug("acme"))),
        HubError::AlreadyExists { slug: slug("acme") },
        HubError::InUse(UsedBy {
            default_choice: true,
            ..UsedBy::default()
        }),
        HubError::Conflict,
        HubError::StoreUnreadable {
            port: PortName::Credentials,
            detail: LogOnly::new("disk on fire".into()),
        },
        HubError::Unresolved(Unresolved::NoProvider),
    ]
}

// ---- ReasonCode ----------------------------------------------------------------

#[test]
fn reason_codes_have_stable_unique_snake_case_wire_strings() {
    let mut seen = std::collections::HashSet::new();
    for code in ReasonCode::ALL {
        let wire = code.as_str();
        assert!(seen.insert(wire), "duplicate {wire}");
        assert!(
            wire.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
            "{wire}"
        );
        assert_eq!(serde_json::to_value(code).unwrap(), json!(wire));
        assert_eq!(
            serde_json::from_value::<ReasonCode>(json!(wire)).unwrap(),
            code
        );
        assert_eq!(code.to_string(), wire);
    }
    assert_eq!(seen.len(), 17);
    // The provider-facing D6 codes plus the D12 split.
    for expected in [
        "auth",
        "model",
        "quota",
        "rate_limited",
        "endpoint",
        "timeout",
        "signed_out",
        "unsupported",
        "unknown",
    ] {
        assert!(seen.contains(expected), "{expected}");
    }
}

#[test]
fn exactly_one_reason_code_destroys_a_credential() {
    let destructive: Vec<_> = ReasonCode::ALL
        .iter()
        .filter(|c| c.destroys_credential())
        .collect();
    assert_eq!(destructive, vec![&ReasonCode::Auth]);
}

// ---- HubError ----------------------------------------------------------------------

#[test]
fn every_error_maps_to_a_reason_and_a_retry() {
    let expected = [
        (ReasonCode::Auth, Retry::Never),
        (ReasonCode::SignedOut, Retry::Never),
        (ReasonCode::Unsupported, Retry::Never),
        (ReasonCode::Policy, Retry::Never),
        (ReasonCode::Invalid, Retry::Never),
        (ReasonCode::NotFound, Retry::Never),
        (ReasonCode::AlreadyExists, Retry::Never),
        (ReasonCode::InUse, Retry::Never),
        (ReasonCode::Conflict, Retry::Now),
        (ReasonCode::StoreUnreadable, Retry::Later(None)),
        (ReasonCode::Unresolved, Retry::Never),
    ];
    let errors = every_error();
    assert_eq!(errors.len(), expected.len());
    for (error, (reason, retry)) in errors.iter().zip(expected) {
        assert_eq!(error.reason(), reason, "{error}");
        assert_eq!(error.retry(), retry, "{error}");
    }
}

#[test]
fn every_reason_code_is_produced_by_some_error() {
    let mut produced: std::collections::HashSet<_> =
        every_error().iter().map(HubError::reason).collect();
    for reason in [
        ReasonCode::Model,
        ReasonCode::Quota,
        ReasonCode::RateLimited,
        ReasonCode::Endpoint,
        ReasonCode::Timeout,
        ReasonCode::Unknown,
    ] {
        produced.insert(HubError::Provider(ProviderFailure::new(reason, Retry::Never)).reason());
    }
    for code in ReasonCode::ALL {
        assert!(produced.contains(&code), "{code}");
    }
}

#[test]
fn display_never_contains_raw_upstream_text_or_store_detail() {
    let failure = ProviderFailure::new(ReasonCode::Auth, Retry::Never)
        .with_status(401)
        .with_provider_code("invalid_api_key")
        .with_raw("Bearer sk-not-a-real-key rejected");
    let error = HubError::Provider(failure);
    let shown = format!("{error} | {error:?}");
    assert!(shown.contains("401") && shown.contains("invalid_api_key"));
    assert!(!shown.contains("sk-not"), "{shown}");
    let store = HubError::StoreUnreadable {
        port: PortName::Config,
        detail: LogOnly::new("password=hunter2".into()),
    };
    let shown = format!("{store} | {store:?}");
    assert!(!shown.contains("hunter2"), "{shown}");
    assert!(shown.contains("config store unreadable"));
}

#[test]
fn errors_display_the_details_a_caller_needs() {
    let texts: Vec<String> = every_error().iter().map(ToString::to_string).collect();
    assert!(texts[0].contains("provider refused"));
    assert!(texts[1].contains("signed out of tinyhumans"));
    assert!(texts[2].contains("remove is not supported by tinyhumans"));
    assert!(texts[3].contains("link-local"));
    assert!(texts[4].contains("slug"));
    assert!(texts[5].contains("provider `acme`"));
    assert!(texts[6].contains("acme already connected"));
    assert!(texts[7].contains("the default"));
    assert!(texts[8].contains("concurrently"));
    assert!(texts[9].contains("credential store unreadable"));
    assert!(texts[10].contains("no provider is configured"));
}

#[test]
fn only_provider_failures_roll_back_and_by_group() {
    let auth = HubError::Provider(ProviderFailure::new(ReasonCode::Auth, Retry::Never));
    let endpoint = HubError::Provider(ProviderFailure::new(
        ReasonCode::Endpoint,
        Retry::Later(None),
    ));
    let timeout = HubError::Provider(ProviderFailure::new(
        ReasonCode::Timeout,
        Retry::Later(None),
    ));
    let quota = HubError::Provider(ProviderFailure::new(ReasonCode::Quota, Retry::Never));
    for group in [
        ProviderGroup::Cloud,
        ProviderGroup::Local,
        ProviderGroup::Cli,
        ProviderGroup::Custom,
        ProviderGroup::Managed,
    ] {
        assert!(auth.rolls_back(group), "auth rolls back for {group}");
        assert!(!quota.rolls_back(group));
    }
    // The local exception: a runtime that is not running is not worth keeping.
    for local_only in [&endpoint, &timeout] {
        assert!(local_only.rolls_back(ProviderGroup::Local));
        assert!(!local_only.rolls_back(ProviderGroup::Cloud));
        assert!(!local_only.rolls_back(ProviderGroup::Custom));
    }
    for other in every_error().iter().skip(1) {
        assert!(!other.rolls_back(ProviderGroup::Local), "{other}");
    }
}

#[test]
fn conversions_wrap_into_the_matching_variant() {
    let failure = ProviderFailure::new(ReasonCode::Model, Retry::Never);
    assert!(matches!(HubError::from(failure), HubError::Provider(_)));
    assert!(matches!(
        HubError::from(InvalidInput::Empty(InputField::Endpoint)),
        HubError::Invalid(_)
    ));
    assert!(matches!(
        HubError::from(PolicyViolation::CrossOriginRedirect),
        HubError::Policy(_)
    ));
    let r: Result<u8> = Err(HubError::Conflict);
    assert!(r.is_err());
}

// ---- small types ---------------------------------------------------------------------

#[test]
fn provider_failure_builders_set_each_field() {
    let f = ProviderFailure::new(ReasonCode::Quota, Retry::Never)
        .with_status(429)
        .with_provider_code("c")
        .with_request_id("r")
        .with_truncated(true)
        .with_raw("raw");
    assert_eq!(f.status, Some(429));
    assert_eq!(f.provider_code.as_deref(), Some("c"));
    assert_eq!(f.request_id.as_deref(), Some("r"));
    assert!(f.truncated);
    assert_eq!(f.raw.expose(), "raw");
    assert_eq!(f.to_string(), "quota (status 429) [c]");
    assert_eq!(
        ProviderFailure::new(ReasonCode::Unknown, Retry::Never).to_string(),
        "unknown"
    );
}

#[test]
fn used_by_counts_and_describes_references() {
    let mut used = UsedBy::default();
    assert!(used.is_empty());
    assert_eq!(used.count(), 0);
    assert_eq!(used.to_string(), "nothing");
    used.default_choice = true;
    used.agents.push(AgentKey::new("a1"));
    used.agents.push(AgentKey::new("a2"));
    used.workloads.push(WorkloadKey::new("w"));
    used.other.push("a schedule".into());
    assert!(!used.is_empty());
    assert_eq!(used.count(), 5);
    let said = used.to_string();
    assert!(said.contains("the default") && said.contains("2 agent pin(s)"));
    assert!(said.contains("1 workload route(s)") && said.contains("1 other reference(s)"));
    let only_other = UsedBy {
        other: vec!["x".into()],
        ..UsedBy::default()
    };
    assert!(!only_other.is_empty());
}

#[test]
fn input_and_not_found_errors_read_as_sentences() {
    assert_eq!(
        InvalidInput::Empty(InputField::ModelId).to_string(),
        "a model id cannot be empty"
    );
    assert_eq!(
        InvalidInput::TooLong {
            field: InputField::Slug,
            max: 80
        }
        .to_string(),
        "a provider slug can be at most 80 characters"
    );
    assert!(
        InvalidInput::ControlCharacters(InputField::Endpoint)
            .to_string()
            .contains("control")
    );
    assert!(
        InvalidInput::Whitespace(InputField::ModelId)
            .to_string()
            .contains("spaces")
    );
    assert!(
        InvalidInput::BadCharacters(InputField::Kind)
            .to_string()
            .contains("not allowed")
    );
    assert!(
        InvalidInput::Reserved {
            field: InputField::ProviderName,
            value: "groq".into()
        }
        .to_string()
        .contains("`groq` is reserved")
    );
    assert!(
        InvalidInput::Malformed {
            field: InputField::Key,
            reason: "nope"
        }
        .to_string()
        .contains("nope")
    );
    for field in [
        InputField::Slug,
        InputField::ProviderName,
        InputField::ModelId,
        InputField::Endpoint,
        InputField::Kind,
        InputField::Key,
    ] {
        assert!(!field.to_string().is_empty());
    }
    let model = ModelId::parse("gpt-5").unwrap();
    assert_eq!(
        NotFound::Model {
            provider: slug("openai"),
            model
        }
        .to_string(),
        "model `gpt-5` on `openai`"
    );
    assert_eq!(NotFound::Agent(AgentKey::new("a")).to_string(), "agent `a`");
    assert_eq!(
        NotFound::Workload(WorkloadKey::new("w")).to_string(),
        "workload `w`"
    );
    assert_eq!(
        NotFound::Kind(KindId::new("zzz")).to_string(),
        "provider kind `zzz`"
    );
}

#[test]
fn policy_violations_and_unresolved_reasons_read_as_sentences() {
    assert!(
        PolicyViolation::CredentialInEndpoint
            .to_string()
            .contains("username")
    );
    assert!(
        PolicyViolation::TooManyRedirects { max: 3 }
            .to_string()
            .contains("3 redirects")
    );
    assert!(
        PolicyViolation::CrossOriginRedirect
            .to_string()
            .contains("another origin")
    );
    assert!(
        Unresolved::ProviderOnlyDefault(slug("a"))
            .to_string()
            .contains("no model")
    );
    assert!(
        Unresolved::Missing(slug("a"))
            .to_string()
            .contains("does not exist")
    );
    assert!(
        Unresolved::Disabled(slug("a"))
            .to_string()
            .contains("disabled")
    );
    assert!(Unresolved::NoKey(slug("a")).to_string().contains("no key"));
}

#[test]
fn ports_and_operations_have_names() {
    for port in [
        PortName::Credentials,
        PortName::Config,
        PortName::Health,
        PortName::Http,
        PortName::Token,
        PortName::Process,
    ] {
        assert!(!port.to_string().is_empty());
    }
    let ops = [
        (Operation::Connect, "connect"),
        (Operation::Add, "add"),
        (Operation::Edit, "edit"),
        (Operation::Remove, "remove"),
        (Operation::SetEnabled, "set_enabled"),
        (Operation::SetKey, "set_key"),
        (Operation::ClearKey, "clear_key"),
        (Operation::ProbeDraft, "probe_draft"),
        (Operation::Test(TestDepth::KeyOnly), "test(key_only)"),
        (Operation::ListModels, "list_models"),
        (Operation::Health, "health"),
        (Operation::RecordOutcome, "record_outcome"),
        (Operation::SetDefault, "set_default"),
        (Operation::PinAgent, "pin_agent"),
        (Operation::ResolveForTurn, "resolve_for_turn"),
        (Operation::ChatModel, "chat_model"),
        (Operation::LocalStatus, "local_status"),
        (Operation::CliReadiness, "cli_readiness"),
        (Operation::OAuth, "oauth"),
    ];
    for (op, name) in ops {
        assert_eq!(op.to_string(), name);
    }
}

// ---- copy ---------------------------------------------------------------------------------

#[test]
fn saved_copy_says_saved_for_every_class_that_kept_the_record() {
    for reason in [
        ReasonCode::Model,
        ReasonCode::Quota,
        ReasonCode::RateLimited,
        ReasonCode::Endpoint,
        ReasonCode::Timeout,
        ReasonCode::Unknown,
    ] {
        assert!(
            describe(reason, "Acme").starts_with("Saved"),
            "{reason}: {}",
            describe(reason, "Acme")
        );
    }
    assert!(describe(ReasonCode::Auth, "Acme").starts_with("Could not reach Acme"));
    assert!(describe(ReasonCode::Endpoint, "Acme").contains("Acme"));
    assert!(describe(ReasonCode::RateLimited, "Acme").contains("rate limiting"));
}

#[test]
fn every_reason_code_has_a_non_empty_sentence_in_both_registers() {
    for reason in ReasonCode::ALL {
        assert!(!describe(reason, "Acme").is_empty(), "{reason}");
        assert!(!describe_refusal(reason, "Acme").is_empty(), "{reason}");
    }
    assert!(describe(ReasonCode::SignedOut, "TinyHumans").contains("Sign in"));
    assert!(describe(ReasonCode::Conflict, "x").contains("Try again"));
}

#[test]
fn a_refusal_never_says_saved_and_names_the_next_step() {
    for reason in ReasonCode::ALL {
        let said = describe_refusal(reason, "Ollama");
        assert!(!said.contains("Saved"), "{reason}: {said}");
    }
    assert!(describe_refusal(ReasonCode::Endpoint, "Ollama").contains("Start it"));
    assert!(describe_refusal(ReasonCode::Auth, "Groq").contains("rejected the credential"));
    assert!(describe_refusal(ReasonCode::Timeout, "Ollama").contains("did not answer in time"));
    assert!(describe_refusal(ReasonCode::Quota, "Ollama").starts_with("Could not verify"));
}

#[test]
fn user_message_switches_register_on_the_context_and_never_echoes_raw() {
    let raw = "401 Unauthorized: Bearer sk-not-a-real-key rejected";
    let error = HubError::Provider(crate::classify(401, &[], raw));
    let saved = error.user_message(CopyContext::saved("Acme"));
    let undone = error.user_message(CopyContext::undone("Acme"));
    for said in [&saved, &undone] {
        assert!(
            !said.contains("sk-not-a-real-key") && !said.contains(raw),
            "{said}"
        );
    }
    let quota = HubError::Provider(ProviderFailure::new(ReasonCode::Quota, Retry::Never));
    assert!(
        quota
            .user_message(CopyContext::saved("Acme"))
            .starts_with("Saved")
    );
    assert!(
        quota
            .user_message(CopyContext::undone("Acme"))
            .starts_with("Could not verify")
    );
    assert_eq!(
        CopyContext::saved("a"),
        CopyContext {
            subject: "a".into(),
            undone: false
        }
    );
    assert!(CopyContext::undone("a").undone);
}

// ---- From<llm::Error> ---------------------------------------------------------------------

#[test]
fn a_structured_provider_error_keeps_its_status_code_and_delay() {
    let error = tinyinference_llm::Error::Provider(Box::new(ProviderError {
        provider: "openai".into(),
        status: Some(429),
        code: Some("rate_limit_exceeded".into()),
        message: "Rate limit reached".into(),
        retryable: true,
        retry_after_ms: Some(2_500),
        ..ProviderError::default()
    }));
    let HubError::Provider(failure) = HubError::from(error) else {
        panic!("provider")
    };
    assert_eq!(failure.reason, ReasonCode::RateLimited);
    assert_eq!(
        failure.retry,
        Retry::Later(Some(Duration::from_millis(2_500)))
    );
    assert_eq!(failure.status, Some(429));
    assert_eq!(
        failure.provider_code.as_deref(),
        Some("rate_limit_exceeded")
    );
    assert_eq!(failure.raw.expose(), "Rate limit reached");
}

#[test]
fn a_structured_spend_cap_is_quota_even_when_llm_marked_it_retryable() {
    let error = tinyinference_llm::Error::Provider(Box::new(ProviderError {
        provider: "anthropic".into(),
        status: Some(429),
        message: "You have reached your specified API usage limits.".into(),
        retryable: true,
        ..ProviderError::default()
    }));
    let HubError::Provider(failure) = HubError::from(error) else {
        panic!("provider")
    };
    assert_eq!(failure.reason, ReasonCode::Quota);
    assert_eq!(failure.retry, Retry::Never);
}

#[test]
fn an_unknown_but_retryable_provider_error_becomes_retry_later() {
    let error = tinyinference_llm::Error::Provider(Box::new(ProviderError {
        provider: "x".into(),
        message: "something odd".into(),
        retryable: true,
        code: Some("bad code with spaces".into()),
        ..ProviderError::default()
    }));
    let HubError::Provider(failure) = HubError::from(error) else {
        panic!("provider")
    };
    assert_eq!(failure.reason, ReasonCode::Unknown);
    assert_eq!(failure.retry, Retry::Later(None));
    assert_eq!(failure.provider_code, None, "an unsafe code is dropped");
    assert_eq!(failure.status, None);
}

#[test]
fn a_provider_error_for_openrouter_uses_the_kind_specific_rule() {
    let error = tinyinference_llm::Error::Provider(Box::new(ProviderError {
        provider: "openrouter".into(),
        status: Some(403),
        message: "User not found.".into(),
        ..ProviderError::default()
    }));
    let HubError::Provider(failure) = HubError::from(error) else {
        panic!("provider")
    };
    assert_eq!(failure.reason, ReasonCode::Auth);
}

#[test]
fn free_text_model_errors_are_classified_after_their_urls_are_stripped() {
    // The transport's own text carries the URL, which always mentions /models.
    let error = tinyinference_llm::Error::Model(
        "error sending request for url (https://api.acme.test/v1/models): connection refused"
            .into(),
    );
    let HubError::Provider(failure) = HubError::from(error) else {
        panic!("provider")
    };
    assert_eq!(failure.reason, ReasonCode::Endpoint);
    assert_eq!(failure.retry, Retry::Later(None));
    let error =
        tinyinference_llm::Error::Model("model error (429): Retry-After: 4 slow down".into());
    let HubError::Provider(failure) = HubError::from(error) else {
        panic!("provider")
    };
    assert_eq!(failure.reason, ReasonCode::RateLimited);
    assert_eq!(failure.status, Some(429));
    assert_eq!(failure.retry, Retry::Later(Some(Duration::from_secs(4))));
}

#[test]
fn the_other_llm_variants_map_to_typed_hub_errors() {
    let HubError::Provider(f) =
        HubError::from(tinyinference_llm::Error::Catalog("bad envelope".into()))
    else {
        panic!()
    };
    assert_eq!(f.reason, ReasonCode::Unknown);
    assert_eq!(f.raw.expose(), "bad envelope");
    let serde_err = serde_json::from_str::<u8>("x").unwrap_err();
    let HubError::Provider(f) = HubError::from(tinyinference_llm::Error::Serialization(serde_err))
    else {
        panic!()
    };
    assert_eq!(f.reason, ReasonCode::Unknown);
    let HubError::Provider(f) =
        HubError::from(tinyinference_llm::Error::Unsupported("tools".into()))
    else {
        panic!()
    };
    assert_eq!(f.reason, ReasonCode::Unsupported);
    let invalid = HubError::from(tinyinference_llm::Error::Validation(
        "bad temperature sk-not-a-real-key".into(),
    ));
    // The validation text can echo input: kept log-only, never displayed.
    let HubError::Invalid(InvalidInput::Rejected(detail)) = &invalid else {
        panic!("rejected")
    };
    assert!(detail.expose().contains("sk-not-a-real-key"));
    assert!(!format!("{invalid} {invalid:?}").contains("sk-not"));
    assert_eq!(invalid.reason(), ReasonCode::Invalid);
}

#[test]
fn a_structured_code_is_classified_with_the_message() {
    // Regression (review finding): only the message was classified, so a
    // spend-cap *code* with a generic message was retried as a cooldown.
    for (code, message, expected) in [
        (
            "enforced_spend_limit_reached",
            "Request failed",
            ReasonCode::Quota,
        ),
        ("insufficient_quota", "Request failed", ReasonCode::Quota),
        ("invalid_api_key", "Request failed", ReasonCode::Auth),
        ("model_not_found", "Request failed", ReasonCode::Model),
    ] {
        let error = tinyinference_llm::Error::Provider(Box::new(ProviderError {
            provider: "openai".into(),
            status: Some(429),
            code: Some(code.into()),
            message: message.into(),
            retryable: true,
            ..ProviderError::default()
        }));
        let HubError::Provider(failure) = HubError::from(error) else {
            panic!("provider")
        };
        assert_eq!(failure.reason, expected, "{code}");
        assert_eq!(failure.provider_code.as_deref(), Some(code));
        if expected != ReasonCode::RateLimited {
            assert_eq!(failure.retry, Retry::Never, "{code}");
        }
    }
    // An empty code is ignored.
    let error = tinyinference_llm::Error::Provider(Box::new(ProviderError {
        provider: "openai".into(),
        status: Some(429),
        code: Some(String::new()),
        message: "slow down".into(),
        ..ProviderError::default()
    }));
    let HubError::Provider(failure) = HubError::from(error) else {
        panic!("provider")
    };
    assert_eq!(failure.reason, ReasonCode::RateLimited);
}

#[test]
fn an_already_taken_slug_has_its_own_code_and_copy() {
    let error = HubError::AlreadyExists { slug: slug("acme") };
    assert_eq!(error.reason(), ReasonCode::AlreadyExists);
    assert_eq!(error.retry(), Retry::Never);
    assert_eq!(
        serde_json::to_value(error.reason()).unwrap(),
        json!("already_exists")
    );
    // Regression (review finding): it used to share `in_use`, so a host keyed
    // on that code for its "confirm removal" flow misfired on a duplicate name.
    assert_ne!(error.reason(), HubError::InUse(UsedBy::default()).reason());
    assert!(
        error
            .user_message(CopyContext::saved("Acme"))
            .contains("already connected")
    );
}

#[test]
fn credential_and_rejected_input_errors_read_as_sentences() {
    let said = InvalidInput::CredentialField {
        name: "api_key".into(),
    }
    .to_string();
    assert!(
        said.contains("api_key") && said.contains("credential"),
        "{said}"
    );
    assert!(
        InvalidInput::Rejected(LogOnly::new("secret".into()))
            .to_string()
            .contains("rejected")
    );
}

#[test]
fn an_endpoint_refusal_converts_to_the_matching_policy_violation() {
    assert_eq!(
        PolicyViolation::from(EndpointRefusal::CredentialInUrl),
        PolicyViolation::CredentialInEndpoint
    );
    assert_eq!(
        PolicyViolation::from(EndpointRefusal::LinkLocal),
        PolicyViolation::Endpoint(EndpointRefusal::LinkLocal)
    );
}

#[test]
fn a_status_guessed_from_free_text_is_reported_but_never_authoritative() {
    // Regression (review round 2): llm's extractor takes any three digits after
    // a `(`, and the hub used to treat that guess as the HTTP status, skipping
    // the token rules that text-only errors get.
    let HubError::Provider(guess) = HubError::from(tinyinference_llm::Error::Model(
        "stream stalled (500 items sent) after a timeout".into(),
    )) else {
        panic!("provider")
    };
    assert_eq!(guess.status, Some(500), "reported for the log");
    assert_eq!(
        guess.reason,
        ReasonCode::Timeout,
        "but the class comes from the text rules"
    );
    // A URL cannot supply the number either.
    let HubError::Provider(url) = HubError::from(tinyinference_llm::Error::Model(
        "request to https://x.test/v1/(401/models failed: connection refused".into(),
    )) else {
        panic!("provider")
    };
    assert_eq!(url.reason, ReasonCode::Endpoint);
    assert_eq!(url.status, None);
}

#[test]
fn an_undone_add_keeps_its_own_sentence_for_hub_internal_errors() {
    // Regression (review round 2): every non-provider error read "Could not
    // verify X, so it was not connected" when the add was undone.
    let invalid = HubError::Invalid(InvalidInput::Empty(InputField::Slug));
    let said = invalid.user_message(CopyContext::undone("Acme"));
    assert_eq!(said, "That input is not valid.");
    let policy = HubError::Policy(PolicyViolation::CrossOriginRedirect);
    assert!(
        policy
            .user_message(CopyContext::undone("Acme"))
            .contains("not allowed")
    );
    let taken = HubError::AlreadyExists { slug: slug("acme") };
    assert!(
        taken
            .user_message(CopyContext::undone("Acme"))
            .contains("already connected")
    );
    // Provider-facing classes still switch register.
    let endpoint = HubError::Provider(ProviderFailure::new(
        ReasonCode::Endpoint,
        Retry::Later(None),
    ));
    assert!(
        endpoint
            .user_message(CopyContext::undone("Acme"))
            .contains("Start it")
    );
}
