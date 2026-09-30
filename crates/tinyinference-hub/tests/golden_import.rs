//! Golden tests for the migration readers (06-migration-mapping): a realistic
//! OpenCompany company and OpenHuman config go in, the exact configuration and
//! loss report come out. A change to either file is a change to what an
//! adopting host will see, so it must be deliberate.

use serde_json::{Value, json};

use tinyinference_hub::import::{Imported, oc, oh};

fn shape(imported: &Imported) -> Value {
    json!({
        "config": serde_json::to_value(&imported.config).unwrap(),
        "loss": serde_json::to_value(&imported.loss).unwrap(),
        "credentials": imported.credentials.iter().map(|c| c.slug.to_string()).collect::<Vec<_>>(),
        "overrides": imported.overrides.iter().map(|o| json!({
            "model": o.model.as_str(),
            "input_per_1m": o.input_per_1m,
            "output_per_1m": o.output_per_1m,
            "context_window": o.context_window,
            "temperature": o.temperature.map(|t| format!("{t:?}")),
        })).collect::<Vec<_>>(),
        "health": imported.health.iter().map(|(s, h)| (s.to_string(), serde_json::to_value(h).unwrap())).collect::<serde_json::Map<_, _>>(),
    })
}

fn assert_golden(actual: &Value, expected: &str, name: &str) {
    let expected: Value = serde_json::from_str(expected).unwrap();
    assert_eq!(
        actual,
        &expected,
        "the {name} import changed; if intended, replace tests/golden/import/{name}.expected.json with:\n{}",
        serde_json::to_string_pretty(actual).unwrap()
    );
}

#[test]
fn golden_import_oc_realistic_company() {
    let snapshot: oc::OcSnapshot =
        serde_json::from_str(include_str!("golden/import/oc_realistic.json")).unwrap();
    let imported = oc::import(&snapshot).unwrap();
    assert_golden(
        &shape(&imported),
        include_str!("golden/import/oc_realistic.expected.json"),
        "oc_realistic",
    );
}

#[test]
fn golden_import_oh_realistic_config() {
    let snapshot: oh::OhSnapshot =
        serde_json::from_str(include_str!("golden/import/oh_realistic.json")).unwrap();
    let imported = oh::import(&snapshot).unwrap();
    // The keys OpenHuman stored never reach the configuration, and never print.
    let text = serde_json::to_string(&shape(&imported)).unwrap();
    assert!(!text.contains("sk-not-a-real-key") && !text.contains("sk-byok-fake"));
    assert_golden(
        &shape(&imported),
        include_str!("golden/import/oh_realistic.expected.json"),
        "oh_realistic",
    );
}
