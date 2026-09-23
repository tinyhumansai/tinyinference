//! Live smoke test: generate an image through OpenRouter and save it.
//!
//! Network- and credential-gated. Reads `OPENROUTER_API_KEY` from the
//! environment, falling back to the workspace `.env` file; exits cleanly
//! (status 0, "skipped") when neither provides one.
//!
//! ```sh
//! cargo run -p tinyinference-image --example live_openrouter_image
//! # image-to-image: pass a reference image (path or URL)
//! LIVE_REFERENCE=path/to/ref.png cargo run -p tinyinference-image --example live_openrouter_image
//! # optional overrides
//! LIVE_IMAGE_MODEL=google/gemini-3.1-flash-lite-image LIVE_PROMPT="…" cargo run …
//! ```
//!
//! Output lands in `target/live-media/`.

use std::path::PathBuf;
use std::time::Instant;

use tinyinference_image::{
    ImageGenerator, ImageRequest, MediaAuth, MediaReference, OpenRouterImageGenerator,
};

fn api_key() -> Option<String> {
    if let Ok(key) = std::env::var("OPENROUTER_API_KEY")
        && !key.trim().is_empty()
    {
        return Some(key.trim().to_owned());
    }
    let env_file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.env");
    std::fs::read_to_string(env_file).ok()?.lines().find_map(|line| {
        let value = line.trim().strip_prefix("OPENROUTER_API_KEY=")?;
        let value = value.trim().trim_matches('"').trim_matches('\'');
        (!value.is_empty()).then(|| value.to_owned())
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Some(key) = api_key() else {
        println!("skipped: OPENROUTER_API_KEY is not set (env or workspace .env)");
        return Ok(());
    };
    let generator = OpenRouterImageGenerator::new(MediaAuth::ApiKey(key));
    let model = std::env::var("LIVE_IMAGE_MODEL").unwrap_or_else(|_| generator.default_model().to_owned());
    let prompt = std::env::var("LIVE_PROMPT").unwrap_or_else(|_| {
        "A four-panel anime comic of two cheerful engineers shaking hands in front of a glowing \
         server, bright colors, thick ink outlines"
            .to_owned()
    });

    let mut request = ImageRequest::new(prompt)
        .with_model(&model)
        .with_aspect_ratio("landscape")
        .with_seed(42);
    if let Ok(reference) = std::env::var("LIVE_REFERENCE") {
        println!("image-to-image with reference {reference}");
        request = request.with_reference(MediaReference::parse(&reference));
    }

    let started = Instant::now();
    let response = generator.generate(request).await?;
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/live-media");
    for (index, image) in response.images.iter().enumerate() {
        let path = image
            .persist(&out_dir, &format!("image-{}-{index}", model.replace('/', "_")), "png")
            .await?;
        println!(
            "saved {} ({} bytes, {})",
            path.display(),
            image.data.len(),
            image.media_type
        );
    }
    println!(
        "model={} images={} cost_usd={:?} elapsed={:.1}s",
        response.model,
        response.images.len(),
        response.cost_usd,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
