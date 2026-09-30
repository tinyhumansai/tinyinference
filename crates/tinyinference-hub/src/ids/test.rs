//! Identifier and validator tests. The validator corpus is ported from
//! OpenCompany's `store_tests_providers.rs` slug/model-id cases and extended
//! with properties.

use proptest::prelude::*;
use serde_json::json;

use super::*;
use crate::error::{HubError, InputField, InvalidInput};

fn not_reserved(_: &str) -> bool {
    false
}

// ---- slugify ---------------------------------------------------------------

#[test]
fn slugify_lowercases_and_collapses_separators() {
    assert_eq!(slugify("My Gateway"), "my-gateway");
    assert_eq!(slugify("  Acme -- Corp!! "), "acme-corp");
    assert_eq!(slugify("ALLCAPS42"), "allcaps42");
    assert_eq!(slugify("a_b.c/d"), "a-b-c-d");
    assert_eq!(slugify("---"), "");
    assert_eq!(slugify(""), "");
    assert_eq!(slugify("  "), "");
    // Non-ASCII letters are separators, exactly as OpenCompany derives them.
    assert_eq!(slugify("Café Münster"), "caf-m-nster");
}

#[test]
fn a_slugified_name_parses_as_a_slug() {
    for name in ["My Gateway", "Acme -- Corp", "x", "42", "A.B"] {
        let slug = slugify(name);
        assert!(Slug::parse(&slug).is_ok(), "{name:?} -> {slug:?}");
    }
}

// ---- provider names and slugs ----------------------------------------------

#[test]
fn a_provider_name_must_be_present_and_bounded() {
    assert_eq!(check_provider_name("Acme"), Ok(()));
    assert_eq!(check_provider_name("   "), Err(SlugError::Empty));
    assert_eq!(check_provider_name(""), Err(SlugError::Empty));
    let longest = "a".repeat(MAX_PROVIDER_NAME_CHARS);
    assert_eq!(check_provider_name(&longest), Ok(()));
    assert_eq!(
        check_provider_name(&format!("{longest}a")),
        Err(SlugError::TooLong)
    );
    // Counted in characters, not bytes.
    assert_eq!(
        check_provider_name(&"é".repeat(MAX_PROVIDER_NAME_CHARS)),
        Ok(())
    );
}

#[test]
fn check_slug_reports_empty_long_taken_and_reserved_in_that_order() {
    let existing = ["acme", "beta"];
    let reserved = |s: &str| s == "groq";
    assert_eq!(check_slug(existing, "gamma", reserved), Ok(()));
    assert_eq!(check_slug(existing, "  ", reserved), Err(SlugError::Empty));
    assert_eq!(
        check_slug(existing, &"a".repeat(81), reserved),
        Err(SlugError::TooLong)
    );
    assert_eq!(
        check_slug(existing, " acme ", reserved),
        Err(SlugError::Taken)
    );
    assert_eq!(
        check_slug(existing, "groq", reserved),
        Err(SlugError::Reserved)
    );
    // Taken wins over reserved when both hold.
    assert_eq!(
        check_slug(["groq"], "groq", reserved),
        Err(SlugError::Taken)
    );
    // Owned strings and slugs work as `existing` too.
    let owned = vec![Slug::parse("acme").unwrap()];
    assert_eq!(
        check_slug(&owned, "acme", not_reserved),
        Err(SlugError::Taken)
    );
}

#[test]
fn slug_errors_read_as_sentences_and_convert_to_hub_errors() {
    for (error, needle) in [
        (SlugError::Empty, "needs a name"),
        (SlugError::Taken, "already has"),
        (SlugError::Reserved, "built-in"),
        (SlugError::TooLong, "80"),
    ] {
        assert!(error.to_string().contains(needle), "{error:?}");
        let boxed: Box<dyn std::error::Error> = Box::new(error);
        assert_eq!(boxed.to_string(), error.to_string());
    }
    assert!(matches!(
        SlugError::Empty.into_hub_error(""),
        HubError::Invalid(InvalidInput::Empty(InputField::Slug))
    ));
    assert!(matches!(
        SlugError::TooLong.into_hub_error("x"),
        HubError::Invalid(InvalidInput::TooLong {
            field: InputField::Slug,
            max: 80
        })
    ));
    match SlugError::Reserved.into_hub_error(" groq ") {
        HubError::Invalid(InvalidInput::Reserved { value, .. }) => assert_eq!(value, "groq"),
        other => panic!("{other:?}"),
    }
    match SlugError::Taken.into_hub_error("acme") {
        HubError::AlreadyExists { slug } => assert_eq!(slug.as_str(), "acme"),
        other => panic!("{other:?}"),
    }
    // A taken slug that is not even a valid slug degrades to invalid input.
    assert!(matches!(
        SlugError::Taken.into_hub_error("Not A Slug"),
        HubError::Invalid(InvalidInput::BadCharacters(InputField::Slug))
    ));
}

// ---- model ids -------------------------------------------------------------

#[test]
fn a_model_id_is_trimmed_and_checked() {
    assert_eq!(check_model_id("  gpt-5  ", &[]).unwrap(), "gpt-5");
    assert_eq!(
        check_model_id("anthropic/claude-3.5:beta", &[]).unwrap(),
        "anthropic/claude-3.5:beta"
    );
    assert_eq!(
        check_model_id("", &[]),
        Err(InvalidInput::Empty(InputField::ModelId))
    );
    assert_eq!(
        check_model_id("   ", &[]),
        Err(InvalidInput::Empty(InputField::ModelId))
    );
    assert_eq!(
        check_model_id("a\u{0007}b", &[]),
        Err(InvalidInput::ControlCharacters(InputField::ModelId))
    );
    assert_eq!(
        check_model_id("a\nb", &[]),
        Err(InvalidInput::ControlCharacters(InputField::ModelId))
    );
    assert_eq!(
        check_model_id("a b", &[]),
        Err(InvalidInput::Whitespace(InputField::ModelId))
    );
    assert_eq!(
        check_model_id("a\u{00a0}b", &[]),
        Err(InvalidInput::Whitespace(InputField::ModelId))
    );
    let longest = "m".repeat(MAX_MODEL_ID_CHARS);
    assert!(check_model_id(&longest, &[]).is_ok());
    assert_eq!(
        check_model_id(&format!("{longest}m"), &[]),
        Err(InvalidInput::TooLong {
            field: InputField::ModelId,
            max: 256
        })
    );
}

#[test]
fn reserved_words_are_host_supplied_and_the_hub_reserves_none() {
    // D7: tier vocabulary is not a hub concept.
    assert!(check_model_id("chat-v1", &[]).is_ok());
    assert_eq!(
        check_model_id("chat-v1", &["chat-v1", "vision-v1"]),
        Err(InvalidInput::Reserved {
            field: InputField::ModelId,
            value: "chat-v1".into()
        })
    );
    assert!(check_model_id("chat-v2", &["chat-v1"]).is_ok());
    assert!(ModelId::parse_with_reserved("chat-v1", &["chat-v1"]).is_err());
    assert!(ModelId::parse("chat-v1").is_ok());
}

#[test]
fn model_ids_validate_on_deserialisation() {
    let ok: ModelId = serde_json::from_value(json!("gpt-5")).unwrap();
    assert_eq!(ok.as_str(), "gpt-5");
    assert_eq!(ok.to_string(), "gpt-5");
    assert_eq!(ok.as_ref(), "gpt-5");
    assert_eq!(serde_json::to_value(&ok).unwrap(), json!("gpt-5"));
    assert_eq!(String::from(ok.clone()), "gpt-5");
    assert!(serde_json::from_value::<ModelId>(json!("has space")).is_err());
    assert!(serde_json::from_value::<ModelId>(json!("")).is_err());
    assert_eq!(ModelId::try_from("x".to_string()).unwrap().as_str(), "x");
}

// ---- Slug ------------------------------------------------------------------

#[test]
fn slug_accepts_lowercase_alphanumerics_dashes_and_underscores() {
    for ok in [
        "openai",
        "vercel-ai-gateway",
        "my_gateway",
        "a",
        "0x",
        "a-b_c9",
    ] {
        let slug = Slug::parse(ok).unwrap();
        assert_eq!(slug.as_str(), ok);
        assert_eq!(slug.to_string(), ok);
        assert_eq!(slug.as_ref(), ok);
    }
    assert_eq!(Slug::parse("  openai ").unwrap().as_str(), "openai");
}

#[test]
fn slug_rejects_everything_else() {
    assert_eq!(Slug::parse(""), Err(InvalidInput::Empty(InputField::Slug)));
    assert_eq!(
        Slug::parse("  "),
        Err(InvalidInput::Empty(InputField::Slug))
    );
    assert_eq!(
        Slug::parse(&"a".repeat(81)),
        Err(InvalidInput::TooLong {
            field: InputField::Slug,
            max: 80
        })
    );
    for bad in [
        "OpenAI", "a b", "a/b", "a.b", "-lead", "_lead", "trail!", "é", "a\nb",
    ] {
        assert_eq!(
            Slug::parse(bad),
            Err(InvalidInput::BadCharacters(InputField::Slug)),
            "{bad:?}"
        );
    }
    assert!(Slug::parse(&"a".repeat(80)).is_ok());
}

#[test]
fn slug_deserialisation_validates_and_serialises_as_a_string() {
    let slug: Slug = serde_json::from_value(json!("groq")).unwrap();
    assert_eq!(serde_json::to_value(&slug).unwrap(), json!("groq"));
    assert_eq!(String::from(slug.clone()), "groq");
    assert!(serde_json::from_value::<Slug>(json!("Bad Slug")).is_err());
    assert_eq!(Slug::try_from("ok".to_string()).unwrap().as_str(), "ok");
}

#[test]
fn a_slug_addresses_its_key_slot_the_way_opencompany_does() {
    assert_eq!(Slug::parse("groq").unwrap().key_slot(), "provider/groq/key");
}

// ---- KindId and the opaque keys ---------------------------------------------

#[test]
fn kind_ids_are_trimmed_and_lowercased() {
    assert_eq!(KindId::new("  OpenAI ").as_str(), "openai");
    assert_eq!(KindId::from("Custom"), KindId::new("custom"));
    assert_eq!(KindId::new("x").to_string(), "x");
    assert_eq!(
        serde_json::to_value(KindId::new("Groq")).unwrap(),
        json!("groq")
    );
    let back: KindId = serde_json::from_value(json!("groq")).unwrap();
    assert_eq!(back, KindId::new("groq"));
    // A stored spelling with capitals still resolves (deserialisation
    // normalises, it does not bypass `KindId::new`).
    let loaded: KindId = serde_json::from_value(json!(" Custom ")).unwrap();
    assert_eq!(loaded.as_str(), "custom");
    assert_eq!(String::from(loaded), "custom");
    assert_eq!(KindId::from("X".to_string()).as_str(), "x");
}

#[test]
fn opaque_keys_carry_whatever_the_host_says() {
    let scope = ScopeKey::new("company:acme/harness:x");
    assert_eq!(scope.as_str(), "company:acme/harness:x");
    assert_eq!(scope.to_string(), "company:acme/harness:x");
    assert_eq!(scope.as_ref(), "company:acme/harness:x");
    assert_eq!(ScopeKey::from("a"), ScopeKey::from("a".to_string()));
    assert_eq!(
        serde_json::to_value(&scope).unwrap(),
        json!("company:acme/harness:x")
    );
    let agent = AgentKey::from("agent one");
    let workload = WorkloadKey::from("chat-v1");
    assert_eq!(agent.as_str(), "agent one");
    assert_eq!(workload.as_str(), "chat-v1");
    assert!(ScopeKey::new("a") < ScopeKey::new("b"));
    let round: WorkloadKey = serde_json::from_value(json!("w")).unwrap();
    assert_eq!(round.to_string(), "w");
}

// ---- properties -------------------------------------------------------------

proptest! {
    #[test]
    fn slugify_output_always_parses_unless_empty(label in "\\PC{0,60}") {
        let slug = slugify(&label);
        if slug.is_empty() {
            prop_assert!(!label.chars().any(|c| c.is_ascii_alphanumeric()));
        } else if slug.chars().count() <= MAX_PROVIDER_NAME_CHARS {
            prop_assert!(Slug::parse(&slug).is_ok(), "{slug:?}");
            prop_assert!(!slug.starts_with('-') && !slug.ends_with('-'));
            prop_assert!(!slug.contains("--"));
        }
    }

    #[test]
    fn slugify_is_idempotent(label in "\\PC{0,60}") {
        let once = slugify(&label);
        prop_assert_eq!(slugify(&once), once.clone());
    }

    #[test]
    fn a_derived_slug_passes_check_slug_unless_reserved(label in "[A-Za-z0-9 _.-]{1,60}") {
        let slug = slugify(&label);
        prop_assume!(!slug.is_empty());
        prop_assert_eq!(check_slug(Vec::<String>::new(), &slug, |_| false), Ok(()));
        prop_assert_eq!(
            check_slug(Vec::<String>::new(), &slug, |s| s == slug),
            Err(SlugError::Reserved)
        );
    }

    #[test]
    fn a_valid_model_id_is_returned_trimmed_and_unchanged(id in "[A-Za-z0-9/:._-]{1,100}") {
        let padded = format!("  {id}\t");
        prop_assert_eq!(check_model_id(&padded, &[]).unwrap(), id.clone());
    }

    #[test]
    fn a_model_id_with_whitespace_inside_is_never_accepted(
        left in "[a-z0-9]{1,10}", right in "[a-z0-9]{1,10}", gap in prop::sample::select(vec![' ', '\t', '\u{00a0}', '\u{3000}']),
    ) {
        let id = format!("{left}{gap}{right}");
        prop_assert!(check_model_id(&id, &[]).is_err());
    }

    #[test]
    fn slug_parse_never_panics_and_accepted_slugs_round_trip(raw in "\\PC{0,100}") {
        if let Ok(slug) = Slug::parse(&raw) {
            let json = serde_json::to_string(&slug).unwrap();
            let back: Slug = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(back, slug);
        }
    }
}

#[test]
fn check_slug_enforces_the_slug_alphabet_too() {
    // Regression (review round 4): `check_slug` accepted `../x`, which
    // `Slug::parse` refuses, and the slug addresses a secret.
    for bad in ["../x", "A B/c", "a/b", "a b", "UPPER", "-lead", "é", "a.b"] {
        assert_eq!(
            check_slug(Vec::<&str>::new(), bad, |_| false),
            Err(SlugError::Invalid),
            "{bad}"
        );
    }
    assert!(SlugError::Invalid.to_string().contains("lowercase"));
    assert!(matches!(
        SlugError::Invalid.into_hub_error("../x"),
        HubError::Invalid(InvalidInput::BadCharacters(InputField::Slug))
    ));
    // The order is empty, too long, invalid, taken, reserved.
    assert_eq!(
        check_slug(["a/b"], "a/b", |_| true),
        Err(SlugError::Invalid)
    );
    assert_eq!(check_slug(["ok"], "ok", |_| true), Err(SlugError::Taken));
}

proptest! {
    #[test]
    fn check_slug_agrees_with_slug_parse(raw in "\\PC{0,40}") {
        let ok = check_slug(Vec::<&str>::new(), &raw, |_| false).is_ok();
        prop_assert_eq!(ok, Slug::parse(&raw).is_ok());
    }
}
