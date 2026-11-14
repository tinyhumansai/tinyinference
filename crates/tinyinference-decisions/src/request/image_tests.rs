//! Image asset construction, header reading, and limit checks.

use super::*;

/// The first bytes of a PNG file of the given size: enough for the header.
pub(crate) fn png_header(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = PNG_SIGNATURE.to_vec();
    bytes.extend_from_slice(&13_u32.to_be_bytes());
    bytes.extend_from_slice(b"IHDR");
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&[8, 6, 0, 0, 0]);
    bytes
}

/// The first bytes of a baseline JPEG of the given size, with an APP0 segment
/// before the frame so the reader has to walk a segment.
fn jpeg_header(width: u16, height: u16) -> Vec<u8> {
    let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    bytes.extend_from_slice(b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
    bytes.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
    bytes.extend_from_slice(&height.to_be_bytes());
    bytes.extend_from_slice(&width.to_be_bytes());
    bytes.extend_from_slice(&[0x03, 0x01, 0x22, 0x00]);
    bytes
}

#[test]
fn png_asset_is_plain_base64_and_reads_its_size() {
    let asset = ImageAsset::png(&png_header(400, 300));
    assert_eq!(asset.format, ImageFormat::Png);
    assert!(!asset.data_base64.starts_with("data:"));
    assert_eq!(asset.checked_len().unwrap(), png_header(400, 300).len());
}

#[test]
fn jpeg_asset_reads_its_size_past_other_segments() {
    let bytes = jpeg_header(1024, 768);
    assert_eq!(dimensions(ImageFormat::Jpg, &bytes).unwrap(), (1024, 768));
    assert!(ImageAsset::jpeg(&bytes).checked_len().is_ok());
}

#[test]
fn jpeg_reader_skips_fill_bytes_and_standalone_markers() {
    let mut bytes = vec![0xFF, 0xD8, 0xFF, 0xFF, 0xD0];
    bytes.extend_from_slice(&jpeg_header(300, 300)[2..]);
    assert_eq!(dimensions(ImageFormat::Jpg, &bytes).unwrap(), (300, 300));
}

#[test]
fn rejects_sides_outside_the_documented_range() {
    for (width, height) in [(279, 400), (400, 279), (1569, 400), (400, 1569)] {
        let error = ImageAsset::png(&png_header(width, height))
            .checked_len()
            .unwrap_err();
        assert!(error.to_string().contains("280 to 1568"), "{error}");
    }
    assert!(
        ImageAsset::png(&png_header(280, 1568))
            .checked_len()
            .is_ok()
    );
}

#[test]
fn rejects_bytes_that_do_not_match_the_declared_format() {
    let error = ImageAsset::png(&jpeg_header(400, 400))
        .checked_len()
        .unwrap_err();
    assert!(error.to_string().contains("not a readable PNG"));
    let error = ImageAsset::jpeg(&png_header(400, 400))
        .checked_len()
        .unwrap_err();
    assert!(error.to_string().contains("not a readable JPEG"));
    assert!(dimensions(ImageFormat::Jpg, &[0xFF, 0xD8, 0x00]).is_err());
    assert!(dimensions(ImageFormat::Jpg, &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x01]).is_err());
    assert!(dimensions(ImageFormat::Png, &PNG_SIGNATURE[..4]).is_err());
}

#[test]
fn rejects_invalid_base64_and_oversized_files() {
    let asset = ImageAsset {
        format: ImageFormat::Png,
        data_base64: "data:image/png;base64,AAAA".into(),
    };
    assert!(
        asset
            .checked_len()
            .unwrap_err()
            .to_string()
            .contains("base64")
    );

    let mut huge = png_header(400, 400);
    huge.resize(MAX_IMAGE_BYTES + 1, 0);
    let error = ImageAsset::png(&huge).checked_len().unwrap_err();
    assert!(error.to_string().contains("16 MiB"));
}
