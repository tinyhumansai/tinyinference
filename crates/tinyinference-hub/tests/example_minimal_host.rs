//! The example host's output is a golden file: a change to what the walk-through
//! prints is a change to the documented behaviour and must be deliberate.
#![cfg(feature = "testing")]

#[allow(dead_code)] // the example's `main` is not run by this test
#[path = "../examples/minimal_host.rs"]
mod example;

const GOLDEN: &str = include_str!("golden/example_minimal_host.txt");

#[tokio::test]
async fn golden_example_minimal_host_transcript() {
    let got = example::transcript().await.expect("the walk-through runs");
    assert_eq!(
        got, GOLDEN,
        "update tests/golden/example_minimal_host.txt if this change is intended"
    );
}
