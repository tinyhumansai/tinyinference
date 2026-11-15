//! Box emulation through the client: request rewriting, synthesis, metering.

use std::sync::{Arc, Mutex};

use base64::Engine as _;
use reqwest::StatusCode;
use serde_json::{Value, json};

use super::super::{
    BoxSupport, ClientConfig, Failure, HttpTransport, IMAGE_OPT_IN_HEADER, Provider,
    TransportResponse,
};
use super::*;
use crate::emulation::crop::tests::{encode, is_red, picture};
use crate::emulation::geometry::Rect;
use crate::{Answer, BoxQuestion, Error, ImageAsset, Noul, Question};

/// Answers every question from the request itself: a probe is positive when
/// its crop shows red, a Box is answered with a fixed box, a Noul with 0.7.
struct EchoTransport {
    model: String,
    bodies: Mutex<Vec<(Value, bool, String)>>,
}

#[async_trait::async_trait]
impl HttpTransport for EchoTransport {
    async fn send(
        &self,
        request: reqwest::Request,
        _response_limit: usize,
    ) -> std::result::Result<TransportResponse, Failure> {
        let body: Value =
            serde_json::from_slice(request.body().and_then(reqwest::Body::as_bytes).unwrap())
                .unwrap();
        let opted_in = request.headers().get(IMAGE_OPT_IN_HEADER).is_some();
        let mut answers = serde_json::Map::new();
        for (id, question) in body["questions"].as_object().unwrap() {
            let answer = match question["type"].as_str().unwrap() {
                "box" => json!({
                    "type": "box",
                    "box": {"top": 0.1, "bottom": 0.2, "left": 0.3, "right": 0.4},
                    "confidence": 0.8
                }),
                _ if id.starts_with(crate::emulation::PROBE_PREFIX) => {
                    let data = body["assets"][id]["data_base64"].as_str().unwrap();
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(data)
                        .unwrap();
                    let crop = image::load_from_memory(&bytes).unwrap().to_rgba8();
                    let noul = if crop.pixels().any(is_red) {
                        0.95
                    } else {
                        0.05
                    };
                    json!({"type": "noul", "noul": noul})
                }
                _ => json!({"type": "noul", "noul": 0.7}),
            };
            answers.insert(id.clone(), answer);
        }
        let mut bodies = self.bodies.lock().unwrap();
        bodies.push((body, opted_in, request.url().path().to_owned()));
        let response = json!({
            "model": self.model,
            "answers": answers,
            "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.000_001},
            "id": "gen-dec-1",
            "provider": "TypeSafe"
        });
        Ok(TransportResponse {
            status: StatusCode::OK,
            request_id: Some(format!("req-{}", bodies.len())),
            retry_after: None,
            body: response.to_string().into_bytes(),
        })
    }
}

fn echo_client(config: ClientConfig, model: &str) -> (Client, Arc<EchoTransport>) {
    let transport = Arc::new(EchoTransport {
        model: model.to_owned(),
        bodies: Mutex::new(Vec::new()),
    });
    let client = Client::with_transport(config, Arc::clone(&transport) as _).unwrap();
    (client, transport)
}

fn surplus() -> (Client, Arc<EchoTransport>) {
    echo_client(ClientConfig::surplus("key"), "typesafe/jev-1.14-20261014")
}

/// An 800 by 800 page with a red square at x 420..470, y 220..260.
fn page_request() -> EvaluationRequest {
    let red = Rect {
        left: 420,
        top: 220,
        right: 470,
        bottom: 260,
    };
    let png = encode(&picture(800, 800, red), image::ImageFormat::Png);
    EvaluationRequest::new(
        "jev-1.14",
        json!({"page": "shot"}),
        [
            (
                "square".to_owned(),
                Question::Box(BoxQuestion {
                    instructions: json!("the red square in `page`"),
                }),
            ),
            (
                "present".to_owned(),
                Question::Noul(Noul {
                    instructions: json!("Is there a red square in `page`?"),
                    criteria: None,
                }),
            ),
        ]
        .into(),
    )
    .with_image("shot", ImageAsset::png(&png))
}

fn sent_box(transport: &EchoTransport) -> bool {
    transport.bodies.lock().unwrap().iter().any(|(body, _, _)| {
        body["questions"]
            .as_object()
            .unwrap()
            .values()
            .any(|question| question["type"] == "box")
    })
}

#[tokio::test]
async fn surplus_answers_box_questions_from_refined_crop_probes() {
    let (client, transport) = surplus();
    let result = client.evaluate(&page_request()).await.unwrap();

    assert!(!sent_box(&transport));
    let bodies = transport.bodies.lock().unwrap();
    // The page and its Noul, the first-round crops, the refinement crops.
    assert_eq!(bodies.len(), 3);
    for (_, opted_in, path) in bodies.iter() {
        assert!(opted_in);
        assert_eq!(path, "/v1/decisions");
    }
    assert!(bodies[0].0["assets"].get("shot").is_some());
    assert_eq!(bodies[0].0["questions"].as_object().unwrap().len(), 1);
    assert_eq!(bodies[1].0["questions"].as_object().unwrap().len(), 16);
    assert_eq!(bodies[2].0["questions"].as_object().unwrap().len(), 16);
    drop(bodies);

    let response = &result.response;
    assert_eq!(response.model, "typesafe/jev-1.14-20261014");
    assert_eq!(
        response.answers.keys().collect::<Vec<_>>(),
        ["present", "square"]
    );
    let Answer::Box(answer) = response.answers["square"] else {
        panic!("square must be answered with a Box");
    };
    assert_eq!(answer.confidence, 0.95);
    assert_eq!(
        (
            answer.edges.left,
            answer.edges.right,
            answer.edges.top,
            answer.edges.bottom
        ),
        (0.5, 0.625, 0.25, 0.375)
    );
    assert_eq!(response.usage.input_tokens, Some(30));
    assert_eq!(response.usage.cost_usd_nanos, Some(3_000));
    assert_eq!(result.attempts, 3);
    assert_eq!(result.request_id.as_deref(), Some("req-1"));
}

#[tokio::test]
async fn without_refinement_the_box_is_the_first_round_cell() {
    let config =
        ClientConfig::surplus("key").with_box_support(BoxSupport::Emulated(BoxEmulation {
            grid: 4,
            refine: false,
        }));
    let (client, transport) = echo_client(config, "typesafe/jev-1.14");
    let result = client.evaluate(&page_request()).await.unwrap();
    assert_eq!(transport.bodies.lock().unwrap().len(), 2);
    let Answer::Box(answer) = result.response.answers["square"] else {
        panic!("square must be answered with a Box");
    };
    assert_eq!(
        (
            answer.edges.left,
            answer.edges.right,
            answer.edges.top,
            answer.edges.bottom
        ),
        (0.5, 0.75, 0.25, 0.5)
    );
}

#[tokio::test]
async fn native_providers_send_box_questions_unchanged() {
    let (client, transport) = echo_client(ClientConfig::new("key"), "jev-1.14");
    let result = client.evaluate(&page_request()).await.unwrap();
    assert!(sent_box(&transport));
    assert_eq!(transport.bodies.lock().unwrap().len(), 1);
    assert!(matches!(result.response.answers["square"], Answer::Box(_)));
}

#[tokio::test]
async fn box_support_can_be_overridden_either_way() {
    let native = ClientConfig::surplus("key").with_box_support(BoxSupport::Native);
    assert_eq!(native.box_support(), BoxSupport::Native);
    let (client, transport) = echo_client(native, "typesafe/jev-1.14");
    client.evaluate(&page_request()).await.unwrap();
    assert!(sent_box(&transport));

    let emulated =
        ClientConfig::new("key").with_box_support(BoxSupport::Emulated(BoxEmulation::default()));
    let (client, transport) = echo_client(emulated, "jev-1.14");
    client.evaluate(&page_request()).await.unwrap();
    assert!(!sent_box(&transport));
}

#[test]
fn box_support_defaults_follow_the_provider() {
    assert_eq!(
        ClientConfig::surplus("key").box_support(),
        BoxSupport::Emulated(BoxEmulation {
            grid: 4,
            refine: true
        })
    );
    assert_eq!(ClientConfig::new("key").box_support(), BoxSupport::Native);
    let mut retargeted = ClientConfig::new("key");
    retargeted.provider = Provider::Surplus;
    assert!(matches!(retargeted.box_support(), BoxSupport::Emulated(_)));
}

#[test]
fn the_emulation_grid_is_bounded() {
    for grid in [0, 1, 9] {
        let config = ClientConfig::surplus("key")
            .with_box_support(BoxSupport::Emulated(BoxEmulation { grid, refine: true }));
        assert!(
            matches!(Client::new(config), Err(Error::InvalidConfig { .. })),
            "grid {grid}"
        );
    }
}

#[tokio::test]
async fn a_box_question_without_an_image_fails_before_any_request() {
    let (client, transport) = surplus();
    let request = EvaluationRequest::new(
        "jev-1.14",
        json!("no image here"),
        [(
            "square".to_owned(),
            Question::Box(BoxQuestion {
                instructions: json!("the red square"),
            }),
        )]
        .into(),
    );
    let Err(Error::EvaluationFailure(failure)) = client.evaluate(&request).await else {
        panic!("expected an evaluation failure");
    };
    assert!(matches!(*failure.error, Error::InvalidRequest { .. }));
    assert_eq!(failure.attempts, 0);
    assert!(transport.bodies.lock().unwrap().is_empty());
}

#[tokio::test]
async fn a_sub_response_from_another_model_fails_with_its_attempts() {
    let (client, transport) = echo_client(ClientConfig::surplus("key"), "typesafe/jev-1.13");
    let Err(Error::EvaluationFailure(failure)) = client.evaluate(&page_request()).await else {
        panic!("expected an evaluation failure");
    };
    assert!(matches!(*failure.error, Error::InvalidResponse { .. }));
    assert_eq!(failure.attempts, 1);
    assert_eq!(transport.bodies.lock().unwrap().len(), 1);
}
