//! Decoding a source image and encoding upscaled crops of it.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use image::{DynamicImage, ImageFormat as Codec, imageops::FilterType};

use super::geometry::Rect;
use crate::request::image::{MAX_SIDE, MIN_SIDE};
use crate::{Error, ImageAsset, ImageFormat, Result};

/// Decode the image attached under `key`.
pub(crate) fn decode(key: &str, asset: &ImageAsset) -> Result<DynamicImage> {
    let bytes = STANDARD
        .decode(&asset.data_base64)
        .map_err(|_| Error::invalid_request(format!("image `{key}` must be padded base64")))?;
    let codec = match asset.format {
        ImageFormat::Png => Codec::Png,
        ImageFormat::Jpg => Codec::Jpeg,
    };
    image::load_from_memory_with_format(&bytes, codec).map_err(|_| {
        Error::invalid_request(format!(
            "image `{key}` could not be decoded to emulate a Box question"
        ))
    })
}

/// Cut `rect` out of `image` as a PNG asset the provider accepts, returning it
/// with its file size.
///
/// A crop whose shorter side is under the provider's minimum is upscaled
/// (bilinear) until it reaches it, keeping its aspect ratio unless that would
/// break the maximum side, which then clamps the longer side.
pub(crate) fn crop(image: &DynamicImage, rect: Rect) -> Result<(ImageAsset, usize)> {
    let (width, height) = (rect.width(), rect.height());
    if width == 0 || height == 0 {
        return Err(Error::invalid_request(
            "an emulated Box image crop must not be empty",
        ));
    }
    let mut crop = image.crop_imm(rect.left, rect.top, width, height);
    let shorter = width.min(height);
    if shorter < MIN_SIDE {
        let scale = |side: u32| {
            u32::try_from((u64::from(side) * u64::from(MIN_SIDE)).div_ceil(u64::from(shorter)))
                .unwrap_or(MAX_SIDE)
                .clamp(MIN_SIDE, MAX_SIDE)
        };
        crop = crop.resize_exact(scale(width), scale(height), FilterType::Triangle);
    }
    let mut bytes = std::io::Cursor::new(Vec::new());
    crop.write_to(&mut bytes, Codec::Png)
        .map_err(|_| Error::invalid_request("an image crop could not be encoded as PNG"))?;
    let bytes = bytes.into_inner();
    Ok((ImageAsset::png(&bytes), bytes.len()))
}

#[cfg(test)]
#[path = "crop_tests.rs"]
pub(crate) mod tests;
