//! Response wire and request-relative validation tests.

use std::collections::BTreeMap;

use serde_json::json;

use super::*;
use crate::{Choice, EvaluationRequest, Noul, Question, Score};

fn request() -> EvaluationRequest {
    EvaluationRequest::jev(
        "state",
        BTreeMap::from([
            (
                "route".to_owned(),
                Question::Choice(Choice {
                    instructions: json!("route"),
                    criteria: BTreeMap::from([("a".to_owned(), None), ("b".to_owned(), None)]),
                }),
            ),
            (
                "quality".to_owned(),
                Question::Score(Score {
                    instructions: json!("quality"),
                    criteria: vec![json!("low"), json!("high")],
                }),
            ),
            (
                "safe".to_owned(),
                Question::Noul(Noul {
                    instructions: json!("safe"),
                    criteria: None,
                }),
            ),
        ]),
    )
}

fn response() -> EvaluationResponse {
    serde_json::from_value(json!({
        "model": "jev-latest",
        "answers": {
            "route": {
                "type": "choice",
                "choice": "b",
                "probabilities": {"a": 0.25, "b": 0.75},
                "confidence": 0.5
            },
            "quality": {
                "type": "score",
                "score": 0.8,
                "legend": {"0": "low", "1": "high"},
                "probabilities": {"0": 0.2, "1": 0.8},
                "confidence": 0.6
            },
            "safe": {"type": "noul", "noul": 0.9}
        },
        "usage": {"input_tokens": 42, "output_tokens": 3}
    }))
    .unwrap()
}

#[test]
fn validates_all_three_answer_types() {
    response().validate_for(&request()).unwrap();
    let mut rounded = response();
    let Answer::Score(answer) = rounded.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    answer.score = 0.81;
    rounded.validate_for(&request()).unwrap();
}

#[test]
fn accepts_a_choice_that_rounding_left_a_hundredth_below_another() {
    // The provider picks the maximum before rounding each probability to two
    // decimals: two options at 0.335 can come back as 0.33 and 0.34 with the
    // first still chosen. This shape came back from the live service.
    let request = EvaluationRequest::jev(
        "state",
        BTreeMap::from([(
            "route".to_owned(),
            Question::Choice(Choice {
                instructions: json!("route"),
                criteria: ["a", "b", "c"]
                    .into_iter()
                    .map(|option| (option.to_owned(), None))
                    .collect(),
            }),
        )]),
    );
    let answer = |a: f64, b: f64, c: f64| -> EvaluationResponse {
        serde_json::from_value(json!({
            "model": "jev-latest",
            "answers": {"route": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": a, "b": b, "c": c},
                "confidence": 0.1
            }},
            "usage": {}
        }))
        .unwrap()
    };
    answer(0.33, 0.34, 0.33).validate_for(&request).unwrap();
    assert!(answer(0.30, 0.40, 0.30).validate_for(&request).is_err());
}

#[test]
fn usage_fields_remain_optional() {
    let usage: Usage = serde_json::from_value(json!({})).unwrap();
    assert_eq!(usage, Usage::default());
}

#[test]
fn usage_preserves_cached_tokens_and_reported_openrouter_cost() {
    let usage: Usage = serde_json::from_value(json!({
        "input_tokens": 100,
        "cached_input_tokens": 40,
        "output_tokens": 15,
        "cost": 0.0000125
    }))
    .unwrap();
    assert_eq!(usage.input_tokens, Some(100));
    assert_eq!(usage.cached_input_tokens, Some(40));
    assert_eq!(usage.output_tokens, Some(15));
    assert_eq!(usage.cost_usd_nanos, Some(12_500));
    let encoded = serde_json::to_value(usage).unwrap();
    assert_eq!(encoded["cost"], json!(0.000_012_5));
    let decoded: Usage = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded.cost_usd_nanos, Some(12_500));

    let unknown: Usage = serde_json::from_value(json!({
        "input_tokens": 2,
        "output_tokens": 1,
        "cost": "unknown"
    }))
    .unwrap();
    assert_eq!(unknown.cost_usd_nanos, None);
    assert_eq!(unknown.cached_input_tokens, None);
}

#[test]
fn rejects_missing_extra_or_wrongly_typed_answers() {
    let mut missing = response();
    missing.answers.remove("safe");
    assert!(missing.validate_for(&request()).is_err());

    let mut wrong = response();
    wrong.answers.insert(
        "safe".to_owned(),
        Answer::Choice(ChoiceAnswer {
            choice: "a".to_owned(),
            probabilities: BTreeMap::from([("a".to_owned(), 1.0)]),
            confidence: 1.0,
        }),
    );
    assert!(wrong.validate_for(&request()).is_err());
}

#[test]
fn rejects_invalid_distributions_and_inconsistent_scores() {
    let mut distribution = response();
    let Answer::Choice(choice) = distribution.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.probabilities.insert("a".to_owned(), 0.75);
    assert!(distribution.validate_for(&request()).is_err());

    let mut score = response();
    let Answer::Score(answer) = score.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    answer.score = 0.1;
    assert!(score.validate_for(&request()).is_err());
}

#[test]
fn rejects_empty_model_extra_ids_and_nonmaximal_choice() {
    let mut empty_model = response();
    empty_model.model.clear();
    assert!(empty_model.validate_for(&request()).is_err());

    let mut wrong_model = response();
    wrong_model.model = "jev-other".into();
    assert!(wrong_model.validate_for(&request()).is_err());

    let mut extra = response();
    extra
        .answers
        .insert("extra".into(), Answer::Noul(NoulAnswer { noul: 0.5 }));
    assert!(extra.validate_for(&request()).is_err());

    let mut nonmaximal = response();
    let Answer::Choice(choice) = nonmaximal.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.choice = "a".into();
    assert!(nonmaximal.validate_for(&request()).is_err());
}

#[test]
fn openrouter_accepts_resolved_jev_models_only() {
    let mut resolved = response();
    resolved.model = "typesafe/jev-1.13-20260917".into();
    resolved.validate_for_openrouter(&request()).unwrap();

    let mut namespaced_latest = request();
    namespaced_latest.model = "~typesafe/jev-latest".into();
    resolved
        .validate_for_openrouter(&namespaced_latest)
        .unwrap();

    let mut unrelated = resolved;
    unrelated.model = "typesafe/other-1".into();
    assert!(unrelated.validate_for_openrouter(&request()).is_err());
}

#[test]
fn openrouter_accepts_a_resolved_release_of_a_specific_jev_model() {
    let mut specific_model = request();
    specific_model.model = "typesafe/jev-1.13".into();
    let mut resolved = response();
    resolved.model = "typesafe/jev-1.13-20260917".into();

    resolved.validate_for_openrouter(&specific_model).unwrap();
}

#[test]
fn rejects_out_of_range_empty_and_mismatched_probability_payloads() {
    let mut confidence = response();
    let Answer::Choice(choice) = confidence.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.confidence = 1.1;
    assert!(confidence.validate_for(&request()).is_err());

    let mut empty = response();
    let Answer::Choice(choice) = empty.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.probabilities.clear();
    assert!(empty.validate_for(&request()).is_err());

    let mut labels = response();
    let Answer::Choice(choice) = labels.answers.get_mut("route").unwrap() else {
        panic!("fixture answer should be a choice")
    };
    choice.probabilities.remove("a");
    choice.probabilities.insert("c".into(), 0.25);
    assert!(labels.validate_for(&request()).is_err());

    let mut noul = response();
    let Answer::Noul(answer) = noul.answers.get_mut("safe").unwrap() else {
        panic!("fixture answer should be a noul")
    };
    answer.noul = f64::NAN;
    assert!(noul.validate_for(&request()).is_err());
}

#[test]
fn rejects_nonfinite_score_and_mismatched_legend() {
    let mut nonfinite = response();
    let Answer::Score(score) = nonfinite.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.score = f64::INFINITY;
    assert!(nonfinite.validate_for(&request()).is_err());

    let mut outside = response();
    let Answer::Score(score) = outside.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.probabilities = BTreeMap::from([("0".into(), 0.98), ("1".into(), 0.02)]);
    score.score = -0.01;
    assert!(outside.validate_for(&request()).is_err());

    let mut legend = response();
    let Answer::Score(score) = legend.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.legend.remove("1");
    assert!(legend.validate_for(&request()).is_err());

    let mut reversed = response();
    let Answer::Score(score) = reversed.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.legend.insert("0".into(), json!("high"));
    score.legend.insert("1".into(), json!("low"));
    assert!(reversed.validate_for(&request()).is_err());
}

#[test]
fn accepts_the_exact_score_rounding_boundary() {
    let mut boundary = response();
    let Answer::Score(score) = boundary.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.probabilities = BTreeMap::from([("0".into(), 0.97), ("1".into(), 0.03)]);
    score.score = 0.05;
    boundary.validate_for(&request()).unwrap();
}

#[test]
fn accepts_multilevel_score_when_rounded_probabilities_shift_weighted_value() {
    let criteria = (0..21).map(|level| json!(level)).collect::<Vec<_>>();
    let mut request = request();
    let Question::Score(question) = request.questions.get_mut("quality").unwrap() else {
        panic!("fixture question should be a score")
    };
    question.criteria = criteria;
    let mut response = response();
    let Answer::Score(score) = response.answers.get_mut("quality").unwrap() else {
        panic!("fixture answer should be a score")
    };
    score.legend = (0..21)
        .map(|level| (level.to_string(), json!(level)))
        .collect();
    score.probabilities = (0..20)
        .map(|level| (level.to_string(), 0.04))
        .chain(std::iter::once(("20".to_owned(), 0.19)))
        .collect();
    score.score = 11.43;

    response.validate_for(&request).unwrap();
}

#[test]
fn many_option_distributions_tolerate_per_option_rounding() {
    // 21 options rounded to two decimals summing to 0.99: what the OpenRouter
    // endpoint answers. Rejected before the size-aware tolerance.
    let mut probabilities = std::collections::BTreeMap::new();
    for i in 0..20 {
        probabilities.insert(format!("o{i}"), 0.04);
    }
    probabilities.insert("o20".to_owned(), 0.19);
    assert!((probabilities.values().sum::<f64>() - 0.99).abs() < 1e-9);
    assert!(validate_distribution(&probabilities, "choice").is_ok());

    // Seven independently rounded two-decimal values can miss one by 0.03.
    let seven: std::collections::BTreeMap<_, _> = (0..6)
        .map(|index| (format!("o{index}"), 0.14))
        .chain(std::iter::once(("o6".to_owned(), 0.13)))
        .collect();
    assert!((seven.values().sum::<f64>() - 0.97).abs() < 1e-9);
    assert!(validate_distribution(&seven, "choice").is_ok());

    // Two options allow up to a 0.01 rounding difference, but reject larger errors.
    let two = std::collections::BTreeMap::from([("a".to_owned(), 0.6), ("b".to_owned(), 0.37)]);
    assert!(validate_distribution(&two, "choice").is_err());

    let all_zero = (0..255)
        .map(|index| (format!("option-{index}"), 0.0))
        .collect();
    assert!(validate_distribution(&all_zero, "choice").is_err());
}

#[test]
fn response_answer_types_round_trip_with_optional_usage() {
    let original = response();
    let value = serde_json::to_value(&original).unwrap();
    assert_eq!(value["answers"]["route"]["type"], "choice");
    assert_eq!(value["answers"]["quality"]["type"], "score");
    assert_eq!(value["answers"]["safe"]["type"], "noul");
    assert_eq!(
        serde_json::from_value::<EvaluationResponse>(value).unwrap(),
        original
    );

    let mut without_usage = original;
    without_usage.usage = Usage::default();
    let value = serde_json::to_value(&without_usage).unwrap();
    assert_eq!(
        serde_json::from_value::<EvaluationResponse>(value).unwrap(),
        without_usage
    );
}

fn box_request() -> EvaluationRequest {
    EvaluationRequest::new(
        "jev-preview",
        json!({"page": "p1"}),
        BTreeMap::from([(
            "button".to_owned(),
            Question::Box(crate::BoxQuestion {
                instructions: json!("the verify button in `page`"),
            }),
        )]),
    )
}

fn box_response(edges: serde_json::Value, confidence: f64) -> EvaluationResponse {
    serde_json::from_value(json!({
        "model": "jev-preview",
        "answers": {"button": {"type": "box", "box": edges, "confidence": confidence}},
        "usage": {"input_tokens": 900, "output_tokens": 0}
    }))
    .unwrap()
}

#[test]
fn a_box_answer_decodes_and_reports_its_centre() {
    let response = box_response(
        json!({"top": 0.8, "bottom": 0.9, "left": 0.7, "right": 0.9}),
        0.93,
    );
    response.validate_for(&box_request()).unwrap();
    let Answer::Box(answer) = &response.answers["button"] else {
        panic!("expected a box answer");
    };
    let (x, y) = answer.edges.centre();
    assert!((x - 0.8).abs() < 1e-9 && (y - 0.85).abs() < 1e-9);
}

#[test]
fn a_box_answer_with_inverted_or_out_of_range_edges_is_rejected() {
    let inverted = box_response(
        json!({"top": 0.9, "bottom": 0.1, "left": 0.1, "right": 0.2}),
        0.5,
    );
    assert!(inverted.validate_for(&box_request()).is_err());
    let outside = box_response(
        json!({"top": 0.1, "bottom": 1.2, "left": 0.1, "right": 0.2}),
        0.5,
    );
    assert!(outside.validate_for(&box_request()).is_err());
    let unsure = box_response(
        json!({"top": 0.1, "bottom": 0.2, "left": 0.1, "right": 0.2}),
        1.5,
    );
    assert!(unsure.validate_for(&box_request()).is_err());
}

#[test]
fn a_box_answer_to_a_noul_question_is_a_type_mismatch() {
    let mut request = box_request();
    request.questions.insert(
        "button".into(),
        Question::Noul(Noul {
            instructions: json!("Is there a button?"),
            criteria: None,
        }),
    );
    let response = box_response(
        json!({"top": 0.1, "bottom": 0.2, "left": 0.1, "right": 0.2}),
        0.5,
    );
    assert!(response.validate_for(&request).is_err());
}
