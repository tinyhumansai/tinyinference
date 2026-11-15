//! Source selection, probe generation, batching, and answer synthesis.

use std::collections::BTreeMap;

use image::ImageFormat as Codec;
use serde_json::json;

use super::crop::tests::{encode, picture};
use super::*;
use crate::{BoxQuestion, ImageFormat, Noul, NoulAnswer};

fn rect(left: u32, top: u32, right: u32, bottom: u32) -> Rect {
    Rect {
        left,
        top,
        right,
        bottom,
    }
}

fn png(width: u32, height: u32) -> ImageAsset {
    ImageAsset::png(&encode(
        &picture(width, height, rect(0, 0, 1, 1)),
        Codec::Png,
    ))
}

fn boxed(instructions: serde_json::Value) -> Question {
    Question::Box(BoxQuestion { instructions })
}

fn noul(instructions: &str) -> Question {
    Question::Noul(Noul {
        instructions: json!(instructions),
        criteria: None,
    })
}

fn request(state: serde_json::Value, questions: Vec<(&str, Question)>) -> EvaluationRequest {
    EvaluationRequest::new(
        "jev-1.14",
        state,
        questions
            .into_iter()
            .map(|(id, question)| (id.to_owned(), question))
            .collect(),
    )
}

fn source_of(request: &EvaluationRequest) -> Result<String> {
    let (id, question) = request.questions.iter().next().unwrap();
    let Question::Box(question) = question else {
        panic!("first question must be a Box");
    };
    source_asset(request, id, &question.instructions)
}

#[test]
fn the_source_is_an_asset_key_named_in_the_instructions() {
    let request = request(
        json!({"a": "shot-a", "b": "shot-b"}),
        vec![("button", boxed(json!("the Verify button in shot-b")))],
    )
    .with_image("shot-a", png(300, 300))
    .with_image("shot-b", png(300, 300));
    assert_eq!(source_of(&request).unwrap(), "shot-b");
}

#[test]
fn a_key_contained_in_a_longer_named_key_is_not_ambiguous() {
    let request = request(
        json!({"a": "img-1", "b": "img-10"}),
        vec![("button", boxed(json!("the Verify button in img-10")))],
    )
    .with_image("img-1", png(300, 300))
    .with_image("img-10", png(300, 300));
    assert_eq!(source_of(&request).unwrap(), "img-10");
}

#[test]
fn independently_named_prefix_keys_stay_ambiguous() {
    let request = request(
        json!({"a": "img-1", "b": "img-10"}),
        vec![(
            "button",
            boxed(json!({"source": "img-1", "comparison": "img-10"})),
        )],
    )
    .with_image("img-1", png(300, 300))
    .with_image("img-10", png(300, 300));
    assert!(matches!(
        source_of(&request),
        Err(Error::InvalidRequest { .. })
    ));
}

#[test]
fn a_short_key_inside_ordinary_words_is_not_a_reference() {
    let request = request(
        json!({"page": "b", "other": "a"}),
        vec![("button", boxed(json!("the red square in `page`")))],
    )
    .with_image("a", png(300, 300))
    .with_image("b", png(300, 300));
    assert_eq!(source_of(&request).unwrap(), "b");
}

#[test]
fn the_source_is_the_asset_held_by_a_backticked_state_field() {
    let request = request(
        json!({"page": "shot-a", "logo": "shot-b"}),
        vec![(
            "button",
            boxed(json!({"find": "the Verify button in `page`"})),
        )],
    )
    .with_image("shot-a", png(300, 300))
    .with_image("shot-b", png(300, 300));
    assert_eq!(source_of(&request).unwrap(), "shot-a");
}

#[test]
fn the_source_is_the_sole_image_otherwise() {
    let request = request(
        json!({"page": "shot-a"}),
        vec![("button", boxed(json!("the Verify button")))],
    )
    .with_image("shot-a", png(300, 300));
    assert_eq!(source_of(&request).unwrap(), "shot-a");
}

#[test]
fn an_unnamed_or_ambiguous_source_is_a_request_error() {
    let two = |instructions: &str| {
        request(
            json!({"a": "shot-a", "b": "shot-b"}),
            vec![("button", boxed(json!(instructions)))],
        )
        .with_image("shot-a", png(300, 300))
        .with_image("shot-b", png(300, 300))
    };
    for instructions in [
        "the Verify button",
        "the button in shot-a or shot-b",
        "`a` or `b`",
    ] {
        let error = source_of(&two(instructions)).unwrap_err();
        assert!(
            matches!(&error, Error::InvalidRequest { reason } if reason.contains("`button`")),
            "{instructions}: {error:?}"
        );
    }
    let none = request(json!("no image"), vec![("button", boxed(json!("x")))]);
    assert!(matches!(
        source_of(&none),
        Err(Error::InvalidRequest { .. })
    ));
}

#[test]
fn reserved_probe_ids_and_keys_are_rejected() {
    let clash = request(
        json!({"page": "shot"}),
        vec![
            ("button", boxed(json!("the button"))),
            ("__tinyinference_box_mine", noul("anything")),
        ],
    )
    .with_image("shot", png(300, 300));
    assert!(matches!(
        BoxEmulator::new(&clash, 4),
        Err(Error::InvalidRequest { .. })
    ));
}

fn screenshot_request() -> EvaluationRequest {
    request(
        json!({"page": "shot"}),
        vec![
            ("button", boxed(json!("the Verify button in `page`"))),
            ("present", noul("Is there a Verify button in `page`?")),
            ("polite", noul("Is the page polite?")),
        ],
    )
    .with_image("shot", png(800, 400))
}

#[test]
fn the_first_round_asks_one_noul_per_grid_cell_with_its_own_crop() {
    let request = screenshot_request();
    let emulator = BoxEmulator::new(&request, 4).unwrap();
    let probes = emulator.first_round().unwrap();
    assert_eq!(probes.len(), 16);
    assert_eq!(probes[0].cell, rect(0, 0, 200, 100));
    assert_eq!(probes[15].cell, rect(600, 300, 800, 400));
    assert_eq!(probes[5].id, "__tinyinference_box0_round0_r1c1");
    let Question::Noul(question) = &probes[5].question else {
        panic!("probes must be Noul questions");
    };
    let text = question.instructions.as_str().unwrap();
    assert!(
        text.contains("`__tinyinference_box0_round0_r1c1`"),
        "{text}"
    );
    assert!(text.contains("the Verify button in `page`"), "{text}");
    assert_eq!(probes[5].asset.format, ImageFormat::Png);
    assert_eq!(probes[5].bytes, probes[5].asset.checked_len().unwrap());
}

#[test]
fn probes_join_a_base_request_that_does_not_carry_their_source() {
    let request = request(
        json!("a text-only state"),
        vec![
            ("button", boxed(json!("the Verify button in shot"))),
            ("polite", noul("Is the page polite?")),
        ],
    )
    .with_image("shot", png(800, 400));
    let emulator = BoxEmulator::new(&request, 2).unwrap();
    let probes = emulator.first_round().unwrap();
    let batches = emulator.batches(&probes, true, 32 * 1024 * 1024).unwrap();
    assert_eq!(batches.len(), 1);
    request.validate().unwrap();
    let batch = &batches[0];
    batch.validate().unwrap();
    assert!(batch.needs_image_opt_in());
    assert_eq!(batch.state, request.state);
    assert_eq!(batch.model, "jev-1.14");
    assert_eq!(batch.questions.len(), 5);
    assert!(batch.questions.contains_key("polite"));
    assert!(!batch.questions.contains_key("button"));
    assert!(!batch.assets.contains_key("shot"));
    assert!(
        !batch
            .questions
            .values()
            .any(|question| matches!(question, Question::Box(_)))
    );
}

#[test]
fn a_base_request_carrying_the_source_image_is_sent_apart_from_its_crops() {
    let request = screenshot_request();
    let emulator = BoxEmulator::new(&request, 2).unwrap();
    let probes = emulator.first_round().unwrap();
    let batches = emulator.batches(&probes, true, 32 * 1024 * 1024).unwrap();
    assert_eq!(batches.len(), 2);
    let ids: Vec<&String> = batches[0].questions.keys().collect();
    assert_eq!(ids, ["polite", "present"]);
    assert_eq!(batches[0].assets.keys().collect::<Vec<_>>(), ["shot"]);
    assert_eq!(batches[1].questions.len(), 4);
    assert!(!batches[1].assets.contains_key("shot"));
    for batch in &batches {
        batch.validate().unwrap();
    }
}

#[test]
fn a_request_of_only_box_questions_sends_only_probe_batches() {
    let request = request(
        json!({"page": "shot"}),
        vec![("button", boxed(json!("the Verify button in `page`")))],
    )
    .with_image("shot", png(800, 400));
    let emulator = BoxEmulator::new(&request, 2).unwrap();
    let probes = emulator.first_round().unwrap();
    let batches = emulator.batches(&probes, true, 32 * 1024 * 1024).unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].questions.len(), 4);
}

#[test]
fn batches_split_to_respect_the_image_byte_limit() {
    let request = request(
        json!({"page": "shot"}),
        vec![("button", boxed(json!("the Verify button in `page`")))],
    )
    .with_image("shot", png(800, 400));
    let emulator = BoxEmulator::new(&request, 4).unwrap();
    let probes = emulator.first_round().unwrap();
    let largest = probes.iter().map(|probe| probe.bytes).max().unwrap();
    let limit = largest * 3;
    let batches = emulator.batches(&probes, true, limit).unwrap();
    assert!(batches.len() > 1, "{} batches", batches.len());
    let mut asked = 0;
    for batch in &batches {
        let total: usize = batch
            .assets
            .values()
            .map(|crate::Asset::Image(image)| image.checked_len().unwrap())
            .sum();
        assert!(total <= limit);
        asked += batch.questions.len();
        batch.validate().unwrap();
    }
    assert_eq!(asked, 16);

    let error = emulator.batches(&probes, true, largest - 1).unwrap_err();
    assert!(matches!(error, Error::InvalidRequest { .. }));
}

fn answers_for(probes: &[Probe], positive: &[&str]) -> Responses {
    let mut responses = Responses::default();
    let answers = probes
        .iter()
        .map(|probe| {
            let noul = if positive.iter().any(|id| probe.id.ends_with(id)) {
                0.9
            } else {
                0.1
            };
            (probe.id.clone(), Answer::Noul(NoulAnswer { noul }))
        })
        .collect();
    responses.add(
        EvaluationResponse {
            model: "typesafe/jev-1.14-20261014".to_owned(),
            answers,
            usage: Usage {
                input_tokens: Some(100),
                output_tokens: Some(10),
                cached_input_tokens: None,
                cost_usd_nanos: Some(1_000),
            },
        },
        Some("first".to_owned()),
    );
    responses
}

#[test]
fn box_answers_are_synthesised_and_probes_stripped() {
    let request = screenshot_request();
    let emulator = BoxEmulator::new(&request, 4).unwrap();
    let probes = emulator.first_round().unwrap();
    let mut responses = answers_for(&probes, &["r1c1", "r1c2"]);
    responses.add(
        EvaluationResponse {
            model: "typesafe/jev-1.14-20261014".to_owned(),
            answers: BTreeMap::from([
                ("present".to_owned(), Answer::Noul(NoulAnswer { noul: 0.8 })),
                ("polite".to_owned(), Answer::Noul(NoulAnswer { noul: 0.4 })),
            ]),
            usage: Usage {
                input_tokens: Some(50),
                output_tokens: None,
                cached_input_tokens: None,
                cost_usd_nanos: Some(500),
            },
        },
        Some("second".to_owned()),
    );
    let picks: Vec<Pick> = emulator
        .picks(&probes, &responses)
        .unwrap()
        .into_iter()
        .map(Option::unwrap)
        .collect();
    let (response, request_id) = emulator.response(responses, &picks);
    assert_eq!(request_id.as_deref(), Some("first"));
    assert_eq!(response.model, "typesafe/jev-1.14-20261014");
    assert_eq!(
        response.answers.keys().collect::<Vec<_>>(),
        ["button", "polite", "present"]
    );
    let Answer::Box(answer) = response.answers["button"] else {
        panic!("button must be answered with a Box");
    };
    assert_eq!(answer.confidence, 0.9);
    assert_eq!(answer.edges.left, 0.25);
    assert_eq!(answer.edges.right, 0.75);
    assert_eq!(answer.edges.top, 0.25);
    assert_eq!(answer.edges.bottom, 0.5);
    assert_eq!(response.usage.input_tokens, Some(150));
    assert_eq!(response.usage.output_tokens, Some(10));
    assert_eq!(response.usage.cost_usd_nanos, Some(1_500));
    response.validate_for_openrouter(&request).unwrap();
}

#[test]
fn a_missing_probe_answer_is_a_response_error() {
    let request = screenshot_request();
    let emulator = BoxEmulator::new(&request, 2).unwrap();
    let probes = emulator.first_round().unwrap();
    let responses = answers_for(&probes[..3], &[]);
    assert!(matches!(
        emulator.picks(&probes, &responses),
        Err(Error::InvalidResponse { .. })
    ));
}

#[test]
fn refinement_grids_the_chosen_region_and_replaces_only_a_found_box() {
    let request = request(
        json!({"page": "shot"}),
        vec![("button", boxed(json!("the Verify button in `page`")))],
    )
    .with_image("shot", png(800, 800));
    let emulator = BoxEmulator::new(&request, 4).unwrap();
    let first = Pick {
        rect: rect(200, 200, 400, 400),
        confidence: 0.9,
        found: true,
    };
    let probes = emulator.refinement_round(&[first]).unwrap();
    assert_eq!(probes.len(), 16);
    assert_eq!(probes[0].cell, rect(200, 200, 250, 250));
    assert_eq!(probes[0].id, "__tinyinference_box0_round1_r0c0");

    let found = answers_for(&probes, &["r0c0"]);
    let refined = refine(vec![first], emulator.picks(&probes, &found).unwrap());
    assert_eq!(refined[0].rect, rect(200, 200, 250, 250));
    assert_eq!(refined[0].confidence, 0.9);

    let missed = answers_for(&probes, &[]);
    let kept = refine(vec![first], emulator.picks(&probes, &missed).unwrap());
    assert_eq!(kept[0], first);
}

#[test]
fn refinement_skips_a_box_that_already_covers_the_image() {
    let request = screenshot_request();
    let emulator = BoxEmulator::new(&request, 4).unwrap();
    let whole = Pick {
        rect: Rect::image(800, 400),
        confidence: 0.9,
        found: true,
    };
    assert!(emulator.refinement_round(&[whole]).unwrap().is_empty());
}
