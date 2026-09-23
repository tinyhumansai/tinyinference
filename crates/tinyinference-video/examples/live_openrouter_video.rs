//! Live smoke test: generate a short video through OpenRouter and save it.
//!
//! Network- and credential-gated, and billed. Reads `OPENROUTER_API_KEY` from
//! the environment, falling back to the workspace `.env` file; exits cleanly
//! (status 0, "skipped") when neither provides one.
//!
//! ```sh
//! cargo run -p tinyinference-video --example live_openrouter_video
//! # image-to-video: pass a first frame (path or URL)
//! LIVE_FIRST_FRAME=target/live-media/image-….png cargo run -p tinyinference-video --example live_openrouter_video
//! # resume a job that timed out, without paying again
//! LIVE_RESUME_JOB=gen-vid-… cargo run -p tinyinference-video --example live_openrouter_video
//! ```
//!
//! Output lands in `target/live-media/`.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tinyinference_video::{
    MediaAuth, MediaReference, OpenRouterVideoGenerator, VideoGenerator, VideoJobStatus,
    VideoRequest, WaitPolicy, wait_for_job,
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
    let generator = OpenRouterVideoGenerator::new(MediaAuth::ApiKey(key));
    let model = std::env::var("LIVE_VIDEO_MODEL")
        .unwrap_or_else(|_| generator.default_model().to_owned());
    let started = Instant::now();
    let wait = WaitPolicy::new(Duration::from_secs(5), Duration::from_secs(900)).with_progress(
        Arc::new(move |status: &VideoJobStatus| {
            println!(
                "  [{:>5.1}s] {} state={} outputs={}",
                started.elapsed().as_secs_f64(),
                status.id,
                status.state,
                status.outputs
            );
        }),
    );

    let response = if let Ok(job_id) = std::env::var("LIVE_RESUME_JOB") {
        println!("resuming job {job_id}");
        wait_for_job(&generator, &job_id, &model, &wait).await?
    } else {
        let prompt = std::env::var("LIVE_PROMPT").unwrap_or_else(|_| {
            "Anime style: two cheerful engineers shake hands in front of a glowing server as \
             confetti falls, gentle camera push-in"
                .to_owned()
        });
        let mut request = VideoRequest::new(prompt)
            .with_model(&model)
            .with_duration(4)
            .with_resolution("480p")
            .with_aspect_ratio("16:9");
        if let Ok(frame) = std::env::var("LIVE_FIRST_FRAME") {
            println!("image-to-video with first frame {frame}");
            request = request.with_first_frame(MediaReference::parse(&frame));
        }
        generator.generate(request, &wait).await?
    };

    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/live-media");
    for (index, video) in response.videos.iter().enumerate() {
        let path = video
            .persist(&out_dir, &format!("video-{}-{index}", response.job_id), "mp4")
            .await?;
        println!(
            "saved {} ({} bytes, {})",
            path.display(),
            video.data.len(),
            video.media_type
        );
    }
    println!(
        "job={} model={} videos={} cost_usd={:?} elapsed={:.1}s",
        response.job_id,
        response.model,
        response.videos.len(),
        response.cost_usd,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
