//! Grid cells, cell selection, and refinement regions.

use super::*;

fn rect(left: u32, top: u32, right: u32, bottom: u32) -> Rect {
    Rect {
        left,
        top,
        right,
        bottom,
    }
}

#[test]
fn grid_covers_the_region_row_major_without_gaps() {
    let cells = grid(rect(0, 0, 10, 7), 3);
    assert_eq!(cells.len(), 9);
    assert_eq!(cells[0], rect(0, 0, 3, 2));
    assert_eq!(cells[1], rect(3, 0, 6, 2));
    assert_eq!(cells[2], rect(6, 0, 10, 2));
    assert_eq!(cells[3], rect(0, 2, 3, 4));
    assert_eq!(cells[8], rect(6, 4, 10, 7));
    let area: u32 = cells.iter().map(|cell| cell.width() * cell.height()).sum();
    assert_eq!(area, 70);
}

#[test]
fn grid_is_offset_by_the_region_origin() {
    let cells = grid(rect(100, 50, 300, 250), 2);
    assert_eq!(
        cells,
        vec![
            rect(100, 50, 200, 150),
            rect(200, 50, 300, 150),
            rect(100, 150, 200, 250),
            rect(200, 150, 300, 250),
        ]
    );
}

#[test]
fn edges_are_fractions_of_the_whole_image() {
    let edges = rect(100, 50, 300, 250).edges(400, 500);
    assert_eq!(edges.left, 0.25);
    assert_eq!(edges.right, 0.75);
    assert_eq!(edges.top, 0.1);
    assert_eq!(edges.bottom, 0.5);
}

#[test]
fn pick_unions_the_positive_component_around_the_best_cell() {
    let cells = grid(rect(0, 0, 400, 400), 4);
    #[rustfmt::skip]
    let probabilities = [
        0.1, 0.6, 0.0, 0.0,
        0.0, 0.9, 0.7, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.8,
    ];
    let pick = pick(&cells, &probabilities, 4);
    // The 0.8 corner is positive but not connected to the 0.9 cell.
    assert_eq!(pick.rect, rect(100, 0, 300, 200));
    assert_eq!(pick.confidence, 0.9);
    assert!(pick.found);
}

#[test]
fn pick_joins_diagonal_neighbours() {
    let cells = grid(rect(0, 0, 400, 400), 4);
    #[rustfmt::skip]
    let probabilities = [
        0.9, 0.0, 0.0, 0.0,
        0.0, 0.5, 0.0, 0.0,
        0.0, 0.0, 0.49, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ];
    let pick = pick(&cells, &probabilities, 4);
    assert_eq!(pick.rect, rect(0, 0, 200, 200));
}

#[test]
fn pick_falls_back_to_the_first_best_cell_when_none_is_positive() {
    let cells = grid(rect(0, 0, 400, 400), 4);
    let mut probabilities = [0.1; 16];
    probabilities[6] = 0.3;
    probabilities[9] = 0.3;
    let pick = pick(&cells, &probabilities, 4);
    assert_eq!(pick.rect, cells[6]);
    assert_eq!(pick.confidence, 0.3);
    assert!(!pick.found);
}

#[test]
fn refinement_keeps_a_squarish_region_and_skips_the_whole_image() {
    let image = rect(0, 0, 800, 600);
    assert_eq!(
        refinement_region(rect(200, 150, 400, 300), image, 4),
        Some(rect(200, 150, 400, 300))
    );
    assert_eq!(refinement_region(image, image, 4), None);
}

#[test]
fn refinement_widens_a_thin_region_inside_the_image() {
    let image = rect(0, 0, 1200, 800);
    // 50 wide, 400 tall: widened to 100 about its centre.
    assert_eq!(
        refinement_region(rect(600, 0, 650, 400), image, 2),
        Some(rect(575, 0, 675, 400))
    );
    // Against the left edge the widening shifts right instead.
    assert_eq!(
        refinement_region(rect(0, 0, 50, 400), image, 2),
        Some(rect(0, 0, 100, 400))
    );
}

#[test]
fn refinement_is_skipped_when_cells_would_be_too_small() {
    let image = rect(0, 0, 1000, 1000);
    assert_eq!(refinement_region(rect(0, 0, 100, 100), image, 4), None);
    assert!(refinement_region(rect(0, 0, 128, 128), image, 4).is_some());
}
