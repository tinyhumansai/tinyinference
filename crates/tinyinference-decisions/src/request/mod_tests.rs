//! Request wire and validation tests.

use std::collections::BTreeMap;

use serde_json::json;

use super::*;

fn questions() -> BTreeMap<String, Question> {
    BTreeMap::from([
        (
            "route".to_owned(),
            Question::Choice(Choice {
                instructions: json!("Which agent should answer?"),
                criteria: BTreeMap::from([
                    ("planner".to_owned(), Some(json!("plans work"))),
                    ("reviewer".to_owned(), None),
                ]),
            }),
        ),
        (
            "quality".to_owned(),
            Question::Score(Score {
                instructions: json!("How strong is the evidence?"),
                criteria: vec![json!("unsupported"), json!("direct")],
            }),
        ),
        (
            "unsafe".to_owned(),
            Question::Noul(Noul {
                instructions: json!("Does this violate a stated constraint?"),
                criteria: Some(NoulCriteria {
                    r#true: json!("a constraint is violated"),
                    r#false: json!("all constraints are satisfied"),
                }),
            }),
        ),
    ])
}

#[test]
fn every_primitive_pins_its_wire_shape() {
    let request = EvaluationRequest::jev(json!({"message": "review this"}), questions());
    assert_eq!(
        serde_json::to_value(request).unwrap(),
        json!({
            "state": {"message": "review this"},
            "model": "jev-latest",
            "questions": {
                "quality": {
                    "type": "score",
                    "instructions": "How strong is the evidence?",
                    "criteria": ["unsupported", "direct"]
                },
                "route": {
                    "type": "choice",
                    "instructions": "Which agent should answer?",
                    "criteria": {"planner": "plans work", "reviewer": null}
                },
                "unsafe": {
                    "type": "noul",
                    "instructions": "Does this violate a stated constraint?",
                    "criteria": {
                        "true": "a constraint is violated",
                        "false": "all constraints are satisfied"
                    }
                }
            }
        })
    );
}

#[test]
fn openjev_constructor_uses_openjev_default_model() {
    let request = EvaluationRequest::openjev(json!({"state": "test"}), questions());
    assert_eq!(request.model, "openjev");
}

#[test]
fn explicit_model_constructor_preserves_local_model_and_question_types() {
    let request = EvaluationRequest::new(
        "ci-models-gemma__ci-rune-26b-a4b",
        json!({"ticket": "charged twice"}),
        questions(),
    );
    request.validate().unwrap();
    let wire = serde_json::to_value(&request).unwrap();
    assert_eq!(wire["model"], "ci-models-gemma__ci-rune-26b-a4b");
    assert_eq!(wire["state"], json!({"ticket": "charged twice"}));
    assert_eq!(wire["questions"]["route"]["type"], "choice");
    assert_eq!(wire["questions"]["quality"]["type"], "score");
    assert_eq!(wire["questions"]["unsafe"]["type"], "noul");
    assert_eq!(
        serde_json::from_value::<EvaluationRequest>(wire).unwrap(),
        request
    );
    assert!(
        EvaluationRequest::new(" ", "state", questions())
            .validate()
            .is_err()
    );
}

#[test]
fn public_request_types_round_trip_with_and_without_noul_criteria() {
    let request = EvaluationRequest::jev(json!({"message": "review this"}), questions());
    let value = serde_json::to_value(&request).unwrap();
    assert_eq!(value["questions"]["route"]["type"], "choice");
    assert_eq!(value["questions"]["quality"]["type"], "score");
    assert_eq!(value["questions"]["unsafe"]["type"], "noul");
    assert_eq!(
        serde_json::from_value::<EvaluationRequest>(value).unwrap(),
        request
    );

    let request = EvaluationRequest::jev(
        "state",
        BTreeMap::from([(
            "safe".to_owned(),
            Question::Noul(Noul {
                instructions: json!("Is the operation safe?"),
                criteria: None,
            }),
        )]),
    );
    let value = serde_json::to_value(&request).unwrap();
    assert!(value["questions"]["safe"].get("criteria").is_none());
    assert_eq!(
        serde_json::from_value::<EvaluationRequest>(value).unwrap(),
        request
    );
}

#[test]
fn accepts_string_object_and_array_state() {
    for state in [json!("text"), json!({"field": true}), json!([1, 2])] {
        EvaluationRequest::jev(state, questions())
            .validate()
            .unwrap();
    }
}

#[test]
fn rejects_empty_or_wrongly_shaped_inputs() {
    let cases = [
        EvaluationRequest::jev(json!(null), questions()),
        EvaluationRequest::jev(json!(true), questions()),
        EvaluationRequest::jev("state", BTreeMap::new()),
    ];
    for request in cases {
        assert!(request.validate().is_err());
    }
}

#[test]
fn enforces_choice_and_score_bounds() {
    let choice = EvaluationRequest::jev(
        "state",
        BTreeMap::from([(
            "choice".to_owned(),
            Question::Choice(Choice {
                instructions: json!("choose"),
                criteria: BTreeMap::from([("only".to_owned(), None)]),
            }),
        )]),
    );
    let score = EvaluationRequest::jev(
        "state",
        BTreeMap::from([(
            "score".to_owned(),
            Question::Score(Score {
                instructions: json!("score"),
                criteria: vec![json!("only")],
            }),
        )]),
    );
    assert!(choice.validate().is_err());
    assert!(score.validate().is_err());
}

#[test]
fn rejects_blank_model_ids_instructions_and_criteria() {
    let mut model = EvaluationRequest::jev("state", questions());
    model.model = " ".into();
    assert!(model.validate().is_err());

    let mut id = EvaluationRequest::jev("state", questions());
    let question = id.questions.remove("route").unwrap();
    id.questions.insert(" ".into(), question);
    assert!(id.validate().is_err());

    let blank_choice = EvaluationRequest::jev(
        "state",
        BTreeMap::from([(
            "choice".into(),
            Question::Choice(Choice {
                instructions: json!(" "),
                criteria: BTreeMap::from([("a".into(), None), ("b".into(), None)]),
            }),
        )]),
    );
    assert!(blank_choice.validate().is_err());

    for instructions in [json!({}), json!([])] {
        let empty_structured = EvaluationRequest::jev(
            "state",
            BTreeMap::from([(
                "choice".into(),
                Question::Choice(Choice {
                    instructions,
                    criteria: BTreeMap::from([("a".into(), None), ("b".into(), None)]),
                }),
            )]),
        );
        assert!(empty_structured.validate().is_err());
    }

    let empty_option = EvaluationRequest::jev(
        "state",
        BTreeMap::from([(
            "choice".into(),
            Question::Choice(Choice {
                instructions: json!("choose"),
                criteria: BTreeMap::from([(String::new(), None), ("b".into(), None)]),
            }),
        )]),
    );
    assert!(empty_option.validate().is_err());
}

#[test]
fn rejects_blank_score_and_noul_descriptions() {
    let score = EvaluationRequest::jev(
        "state",
        BTreeMap::from([(
            "score".into(),
            Question::Score(Score {
                instructions: json!("rate"),
                criteria: vec![json!("low"), json!(" ")],
            }),
        )]),
    );
    assert!(score.validate().is_err());

    let noul = EvaluationRequest::jev(
        "state",
        BTreeMap::from([(
            "noul".into(),
            Question::Noul(Noul {
                instructions: json!("is it safe?"),
                criteria: Some(NoulCriteria {
                    r#true: json!(" "),
                    r#false: json!("not safe"),
                }),
            }),
        )]),
    );
    assert!(noul.validate().is_err());
}

fn image_request(state: serde_json::Value) -> EvaluationRequest {
    EvaluationRequest::new(
        "jev-preview",
        state,
        BTreeMap::from([(
            "found".to_owned(),
            Question::Noul(Noul {
                instructions: json!("Does `photo` show a bus?"),
                criteria: None,
            }),
        )]),
    )
    .with_image(
        "a3f1-photo",
        ImageAsset::png(&super::image::tests::png_header(400, 300)),
    )
}

#[test]
fn a_text_request_has_no_assets_on_the_wire() {
    let request = EvaluationRequest::jev(json!({"message": "review this"}), questions());
    let wire = serde_json::to_value(&request).unwrap();
    assert!(wire.get("assets").is_none());
    assert!(!request.needs_image_opt_in());
    assert_eq!(
        serde_json::from_value::<EvaluationRequest>(wire).unwrap(),
        request
    );
}

#[test]
fn an_image_request_pins_the_documented_asset_shape() {
    let request = image_request(json!({"photo": "a3f1-photo"}));
    request.validate().unwrap();
    assert!(request.needs_image_opt_in());
    let wire = serde_json::to_value(&request).unwrap();
    let asset = &wire["assets"]["a3f1-photo"];
    assert_eq!(asset["type"], "image");
    assert_eq!(asset["format"], "png");
    assert!(
        asset["data_base64"]
            .as_str()
            .unwrap()
            .starts_with("iVBORw0KGgo")
    );
    assert_eq!(
        serde_json::from_value::<EvaluationRequest>(wire).unwrap(),
        request
    );
    let jpeg: ImageFormat = serde_json::from_value(json!("jpeg")).unwrap();
    assert_eq!(jpeg, ImageFormat::Jpg);
    assert_eq!(serde_json::to_value(jpeg).unwrap(), json!("jpg"));
}

#[test]
fn an_unreferenced_asset_is_rejected() {
    let error = image_request(json!({"photo": "something else"}))
        .validate()
        .unwrap_err();
    assert!(error.to_string().contains("not referenced"), "{error}");
}

#[test]
fn asset_references_must_be_nonempty_and_short() {
    let mut request = image_request(json!({"photo": "a3f1-photo"}));
    let image = request.assets.remove("a3f1-photo").unwrap();
    request.assets.insert(" ".into(), image.clone());
    assert!(
        request
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must not be empty")
    );

    let long = "k".repeat(1025);
    let mut request = image_request(json!({ "photo": long.clone() }));
    request.assets.clear();
    request.assets.insert(long, image);
    assert!(
        request
            .validate()
            .unwrap_err()
            .to_string()
            .contains("1024 bytes")
    );
}

#[test]
fn images_may_not_total_more_than_the_request_limit() {
    let mut big = super::image::tests::png_header(400, 400);
    big.resize(12 * 1024 * 1024, 0);
    let mut request = image_request(json!({"a": "img-a", "b": "img-b", "c": "img-c"}));
    request.assets.clear();
    for key in ["img-a", "img-b", "img-c"] {
        request = request.with_image(key, ImageAsset::png(&big));
    }
    let error = request.validate().unwrap_err();
    assert!(error.to_string().contains("32 MiB"), "{error}");
}

#[test]
fn a_box_question_needs_the_opt_in_and_nonempty_instructions() {
    let request = EvaluationRequest::new(
        "jev-preview",
        json!({"page": "p1"}),
        BTreeMap::from([(
            "button".to_owned(),
            Question::Box(BoxQuestion {
                instructions: json!("the verify button in `page`"),
            }),
        )]),
    );
    assert!(request.needs_image_opt_in());
    request.validate().unwrap();
    assert_eq!(
        serde_json::to_value(&request).unwrap()["questions"]["button"],
        json!({"type": "box", "instructions": "the verify button in `page`"})
    );
    let mut empty = request;
    empty.questions.insert(
        "button".into(),
        Question::Box(BoxQuestion {
            instructions: json!(" "),
        }),
    );
    assert!(empty.validate().is_err());
}
