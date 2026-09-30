//! Golden test: the hub's cloud rows must equal OpenCompany's `CLOUD_PROVIDERS`
//! table, scraped independently into `tests/golden/oc_cloud_rows.tsv` (a regex
//! over the OpenCompany source, not output of hub code), so a typo in a ported
//! endpoint fails here rather than in production.

use tinyinference_hub::{AuthStyle, ProviderGroup, catalogue};

const GOLDEN: &str = include_str!("golden/oc_cloud_rows.tsv");

fn rows() -> Vec<Vec<&'static str>> {
    GOLDEN
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split('\t').collect())
        .collect()
}

#[test]
fn every_opencompany_cloud_row_is_in_the_hub_verbatim() {
    let rows = rows();
    assert_eq!(rows.len(), 27, "26 cloud rows plus tinyhumans");
    for row in rows.iter().filter(|r| r[0] != "tinyhumans") {
        let d = catalogue::descriptor(row[0]).unwrap_or_else(|| panic!("{} missing", row[0]));
        assert_eq!(d.group, ProviderGroup::Cloud, "{}", row[0]);
        assert_eq!(d.label, row[1], "{} label", row[0]);
        assert_eq!(d.default_endpoint, Some(row[2]), "{} endpoint", row[0]);
        let auth = match d.auth {
            AuthStyle::Bearer => "bearer",
            AuthStyle::Anthropic => "anthropic",
            ref other => panic!("{}: unexpected {other:?}", row[0]),
        };
        assert_eq!(auth, row[3], "{} auth", row[0]);
        let placeholder = d.key_placeholder.unwrap_or("-");
        assert_eq!(placeholder, row[4], "{} placeholder", row[0]);
    }
}

#[test]
fn the_hub_has_no_cloud_row_opencompany_lacks() {
    let golden: Vec<&str> = rows().iter().map(|r| r[0]).collect();
    for d in catalogue::descriptors_in(ProviderGroup::Cloud) {
        assert!(
            golden.contains(&d.slug()),
            "{} is not in OpenCompany's table",
            d.slug()
        );
    }
}

#[test]
fn the_managed_row_keeps_opencompanys_label_and_placeholder_but_no_url() {
    let row = rows().into_iter().find(|r| r[0] == "tinyhumans").unwrap();
    let d = catalogue::descriptor("tinyhumans").unwrap();
    assert_eq!(d.label, row[1]);
    assert_eq!(d.key_placeholder, Some("th-..."));
    assert_eq!(
        d.default_endpoint, None,
        "Q3: the host supplies the managed endpoint"
    );
    assert!(row[2].ends_with(catalogue::MANAGED_PROXY_PATH));
}

#[test]
fn the_three_disputed_endpoints_take_opencompanys_values() {
    // Open question Q1 (default applied): OpenCompany records each as a fix.
    assert_eq!(
        catalogue::descriptor("deepseek").unwrap().default_endpoint,
        Some("https://api.deepseek.com")
    );
    assert_eq!(
        catalogue::descriptor("together").unwrap().default_endpoint,
        Some("https://api.together.ai/v1")
    );
    assert_eq!(
        catalogue::descriptor("stepfun").unwrap().default_endpoint,
        Some("https://api.stepfun.ai/v1")
    );
}
