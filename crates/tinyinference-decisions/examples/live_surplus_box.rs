//! Live check of Box emulation on Surplus Intelligence, using a synthetic
//! image.
//!
//! Run with `SURPLUS_API_KEY` set (and optionally `SURPLUS_JEV_MODEL`,
//! default `jev-1.14`). Surplus does not accept Box questions, so the client
//! answers this one by cropping the picture and asking a Noul per cell. Each
//! run spends a few cents of the account's balance at most.

use std::collections::BTreeMap;
use std::io::Cursor;

use image::{ImageFormat as Encoding, Rgb, RgbImage};
use serde_json::json;
use tinyinference_decisions::{
    Answer, BoxQuestion, Client, ClientConfig, EvaluationRequest, ImageAsset, Question,
};

/// The red square drawn on a white 640×480 picture, as pixel bounds.
const SQUARE: (u32, u32, u32, u32) = (400, 80, 520, 200);

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let key = std::env::var("SURPLUS_API_KEY")?;
    let model = std::env::var("SURPLUS_JEV_MODEL").unwrap_or_else(|_| "jev-1.14".to_owned());
    let client = Client::new(ClientConfig::surplus(key))?;

    let (left, top, right, bottom) = SQUARE;
    let picture = RgbImage::from_fn(640, 480, |x, y| {
        if (left..right).contains(&x) && (top..bottom).contains(&y) {
            Rgb([220, 20, 20])
        } else {
            Rgb([255, 255, 255])
        }
    });
    let mut png = Vec::new();
    picture.write_to(&mut Cursor::new(&mut png), Encoding::Png)?;

    let questions = BTreeMap::from([(
        "square".to_owned(),
        Question::Box(BoxQuestion {
            instructions: json!({ "find": "the red square in `picture`" }),
        }),
    )]);
    let request = EvaluationRequest::new(model, json!({ "picture": "picture-image" }), questions)
        .with_image("picture-image", ImageAsset::png(&png));

    let result = client.evaluate(&request).await?;
    println!("model: {}", result.response.model);
    println!(
        "expected: left {:.3} right {:.3} top {:.3} bottom {:.3}",
        f64::from(left) / 640.0,
        f64::from(right) / 640.0,
        f64::from(top) / 480.0,
        f64::from(bottom) / 480.0,
    );
    match result.response.answers.get("square") {
        Some(Answer::Box(found)) => println!(
            "box:      left {:.3} right {:.3} top {:.3} bottom {:.3} (confidence {:.2})",
            found.edges.left, found.edges.right, found.edges.top, found.edges.bottom, found.confidence,
        ),
        other => println!("unexpected answer: {other:?}"),
    }
    Ok(())
}
