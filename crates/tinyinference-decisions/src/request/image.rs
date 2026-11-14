//! Building inline image assets and checking them against the provider's
//! documented image limits before anything is sent.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;

use super::types::{ImageAsset, ImageFormat};
use crate::{Error, Result};

/// Smallest accepted width or height, in pixels.
pub(crate) const MIN_SIDE: u32 = 280;
/// Largest accepted width or height, in pixels.
pub(crate) const MAX_SIDE: u32 = 1568;
/// Largest accepted image file.
pub(crate) const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;
/// Largest accepted total of all image files in one request.
pub(crate) const MAX_TOTAL_IMAGE_BYTES: usize = 32 * 1024 * 1024;
/// Longest accepted asset reference string, in bytes.
pub(crate) const MAX_REFERENCE_BYTES: usize = 1024;

const PNG_SIGNATURE: &[u8] = b"\x89PNG\r\n\x1a\n";

impl ImageAsset {
    /// An image from the bytes of a PNG file.
    #[must_use]
    pub fn png(bytes: &[u8]) -> Self {
        Self {
            format: ImageFormat::Png,
            data_base64: STANDARD.encode(bytes),
        }
    }

    /// An image from the bytes of a JPEG file.
    #[must_use]
    pub fn jpeg(bytes: &[u8]) -> Self {
        Self {
            format: ImageFormat::Jpg,
            data_base64: STANDARD.encode(bytes),
        }
    }

    /// Decode the image and check it against the provider's limits, returning
    /// its size in bytes.
    pub(crate) fn checked_len(&self) -> Result<usize> {
        let bytes = STANDARD
            .decode(&self.data_base64)
            .map_err(|_| Error::invalid_request("image data_base64 must be padded base64"))?;
        if bytes.len() > MAX_IMAGE_BYTES {
            return Err(Error::invalid_request("image must be at most 16 MiB"));
        }
        let (width, height) = dimensions(self.format, &bytes)?;
        if !(MIN_SIDE..=MAX_SIDE).contains(&width) || !(MIN_SIDE..=MAX_SIDE).contains(&height) {
            return Err(Error::invalid_request(format!(
                "image sides must be {MIN_SIDE} to {MAX_SIDE} pixels, got {width}x{height}"
            )));
        }
        Ok(bytes.len())
    }
}

/// Width and height read from the image header.
pub(crate) fn dimensions(format: ImageFormat, bytes: &[u8]) -> Result<(u32, u32)> {
    match format {
        ImageFormat::Png => png_dimensions(bytes),
        ImageFormat::Jpg => jpeg_dimensions(bytes),
    }
    .ok_or_else(|| match format {
        ImageFormat::Png => Error::invalid_request("image is not a readable PNG"),
        ImageFormat::Jpg => Error::invalid_request("image is not a readable JPEG"),
    })
}

fn png_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    // The signature is followed by the IHDR chunk: length, "IHDR", width, height.
    if !bytes.starts_with(PNG_SIGNATURE) || bytes.get(12..16)? != b"IHDR" {
        return None;
    }
    let width = u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?);
    let height = u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?);
    Some((width, height))
}

fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    if !bytes.starts_with(&[0xFF, 0xD8]) {
        return None;
    }
    let mut at = 2;
    loop {
        // Skip fill bytes before a marker.
        while *bytes.get(at)? == 0xFF && *bytes.get(at + 1)? == 0xFF {
            at += 1;
        }
        if *bytes.get(at)? != 0xFF {
            return None;
        }
        let marker = *bytes.get(at + 1)?;
        // Markers without a length field.
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            at += 2;
            continue;
        }
        let length = usize::from(u16::from_be_bytes([*bytes.get(at + 2)?, *bytes.get(at + 3)?]));
        // Start-of-frame markers carry the size; C4, C8 and CC do not.
        if matches!(marker, 0xC0..=0xCF) && !matches!(marker, 0xC4 | 0xC8 | 0xCC) {
            let height = u16::from_be_bytes([*bytes.get(at + 5)?, *bytes.get(at + 6)?]);
            let width = u16::from_be_bytes([*bytes.get(at + 7)?, *bytes.get(at + 8)?]);
            return Some((u32::from(width), u32::from(height)));
        }
        if length < 2 {
            return None;
        }
        at += 2 + length;
    }
}

#[cfg(test)]
#[path = "image_tests.rs"]
pub(crate) mod tests;
