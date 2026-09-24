//! Tests for media reference and output-shape standards.

use std::path::Path;

use bytes::Bytes;

use crate::media::GeneratedMedia;
use crate::mock::TINY_PNG;
use crate::reference::{
    MediaReference, ReferenceKind, content_part, extension_for_media_type, media_type_for_path,
    normalize_aspect_ratio, normalize_image_resolution, normalize_size, normalize_video_resolution,
};
use crate::{Error, ImageGenerator, ImageRequest, MockImageGenerator};

#[test]
fn aspect_ratio_spellings_normalize() {
    for (input, expected) in [
        ("16:9", "16:9"),
        ("16x9", "16:9"),
        ("16/9", "16:9"),
        (" 9 × 16 ", "9:16"),
        ("Landscape", "16:9"),
        ("portrait", "9:16"),
        ("square", "1:1"),
        ("auto", "auto"),
        ("2.35:1", "2.35:1"),
    ] {
        assert_eq!(
            normalize_aspect_ratio(input).as_deref(),
            Some(expected),
            "{input}"
        );
    }
    for input in ["wide-ish", "16:0", "1:2:3", ""] {
        assert_eq!(normalize_aspect_ratio(input), None, "{input}");
    }
}

#[test]
fn resolution_spellings_normalize() {
    assert_eq!(normalize_image_resolution("2k").as_deref(), Some("2K"));
    assert_eq!(normalize_image_resolution("1024").as_deref(), Some("1K"));
    assert_eq!(normalize_image_resolution("4K UHD").as_deref(), Some("4K"));
    assert_eq!(normalize_image_resolution("720p"), None);
    assert_eq!(normalize_video_resolution("720").as_deref(), Some("720p"));
    assert_eq!(
        normalize_video_resolution("Full HD").as_deref(),
        Some("1080p")
    );
    assert_eq!(normalize_video_resolution("4k").as_deref(), Some("4K"));
    assert_eq!(normalize_video_resolution("hd").as_deref(), Some("720p"));
    assert_eq!(normalize_video_resolution("8k"), None);
}

#[test]
fn sizes_normalize() {
    assert_eq!(normalize_size("1536x1024").as_deref(), Some("1536x1024"));
    assert_eq!(normalize_size(" 1024 × 768 ").as_deref(), Some("1024x768"));
    assert_eq!(normalize_size("2k").as_deref(), Some("2K"));
    assert_eq!(normalize_size("0x10"), None);
    assert_eq!(normalize_size("big"), None);
}

#[test]
fn references_classify_and_infer_kind() {
    assert!(matches!(
        MediaReference::parse("https://x.test/a.png"),
        MediaReference::Url(_)
    ));
    assert!(matches!(
        MediaReference::parse("data:image/png;base64,AA=="),
        MediaReference::DataUrl(_)
    ));
    assert!(matches!(
        MediaReference::parse("./frames/first.jpg"),
        MediaReference::Path(_)
    ));

    assert_eq!(
        MediaReference::parse("https://x.test/clip.mp4?sig=1").kind(),
        ReferenceKind::Video
    );
    assert_eq!(
        MediaReference::parse("data:audio/wav;base64,AA==").kind(),
        ReferenceKind::Audio
    );
    assert_eq!(
        MediaReference::parse("photo.jpeg").kind(),
        ReferenceKind::Image
    );
}

#[test]
fn content_parts_match_the_wire_shape() {
    let image = content_part(ReferenceKind::Image, "https://x.test/a.png");
    assert_eq!(image["type"], "image_url");
    assert_eq!(image["image_url"]["url"], "https://x.test/a.png");
    let video = content_part(ReferenceKind::Video, "https://x.test/a.mp4");
    assert_eq!(video["type"], "video_url");
    assert_eq!(video["video_url"]["url"], "https://x.test/a.mp4");
}

#[tokio::test]
async fn local_files_inline_as_data_urls_within_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ref.png");
    std::fs::write(&path, TINY_PNG).unwrap();

    let reference = MediaReference::Path(path.clone());
    let url = reference.resolve(1024).await.unwrap();
    assert!(url.starts_with("data:image/png;base64,"), "{url}");

    let error = reference.resolve(8).await.unwrap_err();
    assert!(matches!(error, Error::TooLarge { limit: 8 }), "{error:?}");

    let missing = MediaReference::Path(dir.path().join("missing.png"));
    assert!(matches!(missing.resolve(1024).await, Err(Error::Io(_))));
}

#[tokio::test]
async fn malformed_and_empty_references_are_rejected() {
    assert!(matches!(
        MediaReference::DataUrl("data:image/png;base64".into())
            .resolve(1024)
            .await,
        Err(Error::Validation(_))
    ));
    assert!(matches!(
        MediaReference::Bytes {
            media_type: "image/png".into(),
            data: Bytes::new()
        }
        .resolve(1024)
        .await,
        Err(Error::Validation(_))
    ));
}

#[test]
fn debug_output_hides_payloads_and_signatures() {
    let debug = format!(
        "{:?} {:?}",
        MediaReference::Url("https://x.test/a.png?X-Amz-Signature=secret".into()),
        MediaReference::DataUrl("data:image/png;base64,SECRETPAYLOAD".into())
    );
    assert!(
        !debug.contains("secret") && !debug.contains("SECRETPAYLOAD"),
        "{debug}"
    );
}

#[test]
fn media_types_and_extensions_round_trip() {
    assert_eq!(media_type_for_path(Path::new("a.WEBP")), "image/webp");
    assert_eq!(media_type_for_path(Path::new("a.mov")), "video/quicktime");
    assert_eq!(extension_for_media_type("image/jpeg", "bin"), "jpg");
    assert_eq!(
        extension_for_media_type("video/mp4; codecs=avc1", "bin"),
        "mp4"
    );
    assert_eq!(
        extension_for_media_type("application/x-unknown", "bin"),
        "bin"
    );
}

#[tokio::test]
async fn persist_sanitizes_the_stem_and_refuses_empty_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let media = GeneratedMedia::new("image/png", TINY_PNG);
    let path = media
        .persist(dir.path(), "../../etc/passwd", "bin")
        .await
        .unwrap();
    assert_eq!(path.parent().unwrap(), dir.path());
    assert_eq!(path.file_name().unwrap(), "______etc_passwd.png");
    assert_eq!(std::fs::read(&path).unwrap(), TINY_PNG);

    let empty = GeneratedMedia::new("image/png", Bytes::new());
    assert!(matches!(
        empty.persist(dir.path(), "x", "png").await,
        Err(Error::Validation(_))
    ));
}

#[tokio::test]
async fn mock_generator_records_requests_and_simulates_no_media() {
    let mock = MockImageGenerator::new();
    let response = mock
        .generate(ImageRequest::new("x").with_n(2))
        .await
        .unwrap();
    assert_eq!(response.images.len(), 2);
    assert_eq!(mock.requests().len(), 1);

    let failing = MockImageGenerator::returning_no_media();
    assert!(matches!(
        failing.generate(ImageRequest::new("x")).await,
        Err(Error::NoMedia { .. })
    ));
}

/// A `Typed` reference that names a local file is inlined (never sent as a
/// raw path), and an extensionless file takes its media type from the stated
/// kind.
#[tokio::test]
async fn typed_local_paths_are_inlined_with_the_stated_kind() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("frame-without-extension");
    std::fs::write(&path, TINY_PNG).unwrap();
    let reference = MediaReference::Typed {
        kind: ReferenceKind::Video,
        url: path.display().to_string(),
    };
    let url = reference.resolve(1024).await.unwrap();
    assert!(url.starts_with("data:video/mp4;base64,"), "{url}");
    assert_eq!(reference.kind(), ReferenceKind::Video);

    let remote = MediaReference::Typed {
        kind: ReferenceKind::Audio,
        url: "https://cdn.test/opaque-id".into(),
    };
    assert_eq!(
        remote.resolve(1024).await.unwrap(),
        "https://cdn.test/opaque-id"
    );
}

#[tokio::test]
async fn oversized_data_urls_are_rejected_before_decoding() {
    let payload = "A".repeat(4_000);
    let reference = MediaReference::DataUrl(format!("data:image/png;base64,{payload}"));
    assert!(matches!(
        reference.resolve(100).await,
        Err(Error::TooLarge { limit: 100 })
    ));
}
