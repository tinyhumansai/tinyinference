//! Decoding source images and encoding upscaled crops.

use image::{ImageFormat as Codec, Rgb, RgbImage};

use super::*;
use crate::request::image::{MAX_SIDE, MIN_SIDE};

/// A white image with a red square covering `red`.
pub(crate) fn picture(width: u32, height: u32, red: Rect) -> RgbImage {
    RgbImage::from_fn(width, height, |x, y| {
        if (red.left..red.right).contains(&x) && (red.top..red.bottom).contains(&y) {
            Rgb([255, 0, 0])
        } else {
            Rgb([255, 255, 255])
        }
    })
}

pub(crate) fn encode(image: &RgbImage, codec: Codec) -> Vec<u8> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image.write_to(&mut bytes, codec).unwrap();
    bytes.into_inner()
}

pub(crate) fn is_red(pixel: &image::Rgba<u8>) -> bool {
    pixel[0] > 200 && pixel[1] < 80 && pixel[2] < 80
}

fn rect(left: u32, top: u32, right: u32, bottom: u32) -> Rect {
    Rect {
        left,
        top,
        right,
        bottom,
    }
}

fn decoded(asset: &ImageAsset) -> image::DynamicImage {
    assert_eq!(asset.format, ImageFormat::Png);
    decode("crop", asset).unwrap()
}

#[test]
fn a_small_crop_is_upscaled_to_the_minimum_side_and_keeps_its_content() {
    let source = picture(400, 300, rect(120, 80, 140, 100));
    let image = decode("page", &ImageAsset::png(&encode(&source, Codec::Png))).unwrap();
    let (asset, bytes) = crop(&image, rect(100, 75, 200, 150)).unwrap();
    let crop = decoded(&asset);
    assert_eq!((crop.width(), crop.height()), (374, MIN_SIDE));
    assert_eq!(bytes, asset.checked_len().unwrap());
    assert!(crop.to_rgba8().pixels().any(is_red));

    let (elsewhere, _) = crop_of(&image, rect(200, 150, 300, 225));
    assert!(!elsewhere.to_rgba8().pixels().any(is_red));
}

fn crop_of(image: &image::DynamicImage, rect: Rect) -> (image::DynamicImage, usize) {
    let (asset, bytes) = crop(image, rect).unwrap();
    (decoded(&asset), bytes)
}

#[test]
fn a_crop_already_large_enough_is_not_resized() {
    let source = picture(1000, 800, rect(0, 0, 1, 1));
    let image = decode("page", &ImageAsset::png(&encode(&source, Codec::Png))).unwrap();
    let (crop, _) = crop_of(&image, rect(100, 100, 400, 390));
    assert_eq!((crop.width(), crop.height()), (300, 290));
}

#[test]
fn an_extreme_aspect_crop_is_clamped_to_the_maximum_side() {
    let source = picture(1000, 800, rect(0, 0, 1, 1));
    let image = decode("page", &ImageAsset::png(&encode(&source, Codec::Png))).unwrap();
    let (crop, _) = crop_of(&image, rect(0, 0, 400, 50));
    assert_eq!((crop.width(), crop.height()), (MAX_SIDE, MIN_SIDE));
}

#[test]
fn jpeg_sources_are_decoded_and_cropped_to_png() {
    let source = picture(600, 400, rect(0, 0, 300, 200));
    let image = decode("page", &ImageAsset::jpeg(&encode(&source, Codec::Jpeg))).unwrap();
    assert_eq!((image.width(), image.height()), (600, 400));
    let (crop, _) = crop_of(&image, rect(0, 0, 300, 200));
    assert!(crop.to_rgba8().pixels().any(is_red));
}

#[test]
fn an_undecodable_image_names_its_asset() {
    let asset = ImageAsset::png(&crate::request::image::tests::png_header(400, 300));
    let error = decode("page", &asset).unwrap_err();
    assert!(
        matches!(&error, Error::InvalidRequest { reason } if reason.contains("`page`")),
        "{error:?}"
    );
}

#[test]
fn an_empty_crop_is_a_request_error_not_a_panic() {
    let source = picture(400, 300, rect(0, 0, 1, 1));
    let image = decode("page", &ImageAsset::png(&encode(&source, Codec::Png))).unwrap();
    assert!(matches!(
        crop(&image, rect(10, 10, 10, 50)),
        Err(Error::InvalidRequest { .. })
    ));
}
