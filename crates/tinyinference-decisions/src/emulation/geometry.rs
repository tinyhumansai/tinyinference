//! Pixel rectangles, grids over them, and selecting a box from cell scores.

use crate::BoxEdges;

/// A cell counts as containing the target from this probability up.
pub(crate) const POSITIVE: f64 = 0.5;
/// Refinement cells must be at least this many pixels on their shorter side;
/// smaller crops carry too little of the original image to be worth a round.
const MIN_REFINE_CELL_PX: u32 = 32;
/// A refinement region is widened or heightened until neither side exceeds
/// this multiple of the other, so its cells can be upscaled to the provider's
/// minimum side without breaking its maximum.
const MAX_REFINE_ASPECT: u32 = 4;

/// A pixel rectangle; `right` and `bottom` are exclusive.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Rect {
    pub(crate) left: u32,
    pub(crate) top: u32,
    pub(crate) right: u32,
    pub(crate) bottom: u32,
}

impl Rect {
    /// The whole of a `width` by `height` image.
    pub(crate) fn image(width: u32, height: u32) -> Self {
        Self {
            left: 0,
            top: 0,
            right: width,
            bottom: height,
        }
    }

    pub(crate) fn width(self) -> u32 {
        self.right - self.left
    }

    pub(crate) fn height(self) -> u32 {
        self.bottom - self.top
    }

    /// This rectangle as fractions of a `width` by `height` image.
    pub(crate) fn edges(self, width: u32, height: u32) -> BoxEdges {
        let (width, height) = (f64::from(width), f64::from(height));
        BoxEdges {
            top: f64::from(self.top) / height,
            bottom: f64::from(self.bottom) / height,
            left: f64::from(self.left) / width,
            right: f64::from(self.right) / width,
        }
    }

    fn union(self, other: Self) -> Self {
        Self {
            left: self.left.min(other.left),
            top: self.top.min(other.top),
            right: self.right.max(other.right),
            bottom: self.bottom.max(other.bottom),
        }
    }
}

/// The region selected from a grid of cell probabilities.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Pick {
    /// Bounding rectangle of the selected cells.
    pub(crate) rect: Rect,
    /// Highest cell probability.
    pub(crate) confidence: f64,
    /// Whether any cell reached [`POSITIVE`].
    pub(crate) found: bool,
}

/// An `n` by `n` grid over `region`, row-major. Cell boundaries are floored,
/// so the cells tile the region exactly.
pub(crate) fn grid(region: Rect, n: u32) -> Vec<Rect> {
    let split = |start: u32, length: u32, index: u32| {
        let offset = u64::from(length) * u64::from(index) / u64::from(n);
        start + u32::try_from(offset).unwrap_or(length)
    };
    let (width, height) = (region.width(), region.height());
    (0..n)
        .flat_map(|row| (0..n).map(move |column| (row, column)))
        .map(|(row, column)| Rect {
            left: split(region.left, width, column),
            top: split(region.top, height, row),
            right: split(region.left, width, column + 1),
            bottom: split(region.top, height, row + 1),
        })
        .collect()
}

/// Select a box from the row-major `n` by `n` `cells` and their
/// `probabilities`.
///
/// The best cell (the first, on ties) anchors the box. When it reaches
/// [`POSITIVE`], the box is the bounding rectangle of every positive cell
/// connected to it, diagonals included; other positive cells are taken to be
/// another instance. Otherwise the box is the best cell alone.
pub(crate) fn pick(cells: &[Rect], probabilities: &[f64], n: u32) -> Pick {
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    let best = probabilities
        .iter()
        .enumerate()
        .fold(0, |best, (index, probability)| {
            if *probability > probabilities[best] {
                index
            } else {
                best
            }
        });
    let confidence = probabilities[best];
    let found = confidence >= POSITIVE;
    let mut rect = cells[best];
    if found {
        let mut seen = vec![false; cells.len()];
        let mut pending = vec![best];
        seen[best] = true;
        while let Some(index) = pending.pop() {
            rect = rect.union(cells[index]);
            let (row, column) = (index / n, index % n);
            for neighbour_row in row.saturating_sub(1)..=(row + 1).min(n - 1) {
                for neighbour_column in column.saturating_sub(1)..=(column + 1).min(n - 1) {
                    let neighbour = neighbour_row * n + neighbour_column;
                    if !seen[neighbour] && probabilities[neighbour] >= POSITIVE {
                        seen[neighbour] = true;
                        pending.push(neighbour);
                    }
                }
            }
        }
    }
    Pick {
        rect,
        confidence,
        found,
    }
}

/// The region a refinement round grids, or `None` when refining `chosen`
/// would only repeat the first round (it is the whole `image`) or cut cells
/// smaller than [`MIN_REFINE_CELL_PX`].
///
/// A region more than [`MAX_REFINE_ASPECT`] times longer than it is wide (or
/// the reverse) is grown about its centre, shifted to stay inside the image.
pub(crate) fn refinement_region(chosen: Rect, image: Rect, n: u32) -> Option<Rect> {
    let mut region = chosen;
    let (width, height) = (region.width(), region.height());
    if width > height.saturating_mul(MAX_REFINE_ASPECT) {
        (region.top, region.bottom) = grow(
            region.top,
            height,
            width.div_ceil(MAX_REFINE_ASPECT),
            image.bottom,
        );
    } else if height > width.saturating_mul(MAX_REFINE_ASPECT) {
        (region.left, region.right) = grow(
            region.left,
            width,
            height.div_ceil(MAX_REFINE_ASPECT),
            image.right,
        );
    }
    let too_small = region.width().min(region.height()) < MIN_REFINE_CELL_PX.saturating_mul(n);
    (region != image && !too_small).then_some(region)
}

/// Grow the span `start..start + length` to `target` (at most `limit`) about
/// its centre, keeping it inside `0..limit`.
fn grow(start: u32, length: u32, target: u32, limit: u32) -> (u32, u32) {
    let target = target.min(limit);
    let start = start
        .saturating_sub((target - length) / 2)
        .min(limit - target);
    (start, start + target)
}

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod tests;
