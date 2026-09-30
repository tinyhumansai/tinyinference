//! A guard for the scenario catalogue (08-test-plan section 4): every named
//! scenario exists as a test, in the file that drives it. Renaming or deleting
//! one fails here instead of silently shrinking the suite.

const HUB_SCENARIOS: &str = include_str!("sim_hub.rs");
const RANDOM_SCENARIOS: &str = include_str!("sim_random.rs");

const CATALOGUE: &[&str] = &[
    "sim_desktop_onboarding",
    "sim_cli_oneshot_env_only",
    "sim_connect_rollback_on_auth",
    "sim_connect_add_anyway",
    "sim_local_rollback_on_timeout",
    "sim_key_rotation_next_call",
    "sim_platform_token_rotation",
    "sim_managed_origin_switch",
    "sim_signed_out",
    "sim_tenant_isolation",
    "sim_refresh_bypass",
    "sim_ssrf_literal_ips",
    "sim_ssrf_redirect_chain",
    "sim_dns_rebinding_scripted",
    "sim_cleartext_credential",
    "sim_offline_local_only",
    "sim_partial_outage",
    "sim_quota_vs_rate",
    "sim_concurrent_writers",
    "sim_delete_vs_pin",
    "sim_disable_fail_closed",
    "sim_store_unreadable",
    "sim_cli_readiness",
    "sim_oauth_disabled",
    "sim_import_oc_then_ops",
    "sim_import_oh_then_resolve",
    "sim_detect_excludes_self",
    "sim_detect_off_when_hosted",
    "sim_slow_stream_completion",
    "sim_malformed_catalogs",
    "sim_paged_catalog",
    "sim_random_regressions",
];

#[test]
fn catalogue_all_thirty_two_scenarios_exist() {
    assert_eq!(CATALOGUE.len(), 32);
    for name in CATALOGUE {
        let declared = format!("async fn {name}(");
        assert!(
            HUB_SCENARIOS.contains(&declared) || RANDOM_SCENARIOS.contains(&declared),
            "the scenario `{name}` is missing"
        );
    }
}

#[test]
fn catalogue_the_guards_have_named_tests() {
    // Every guard of 04-operation-matrix section 2 except G10 (tier routing
    // stays in OpenCompany) has at least one `guard_gNN_` test in the crate.
    let sources = [
        include_str!("../src/ops/test.rs"),
        include_str!("../src/ops/guards_test.rs"),
        include_str!("../src/route/resolve_test.rs"),
        include_str!("../src/import/test.rs"),
        include_str!("guards_foundations.rs"),
        include_str!("guards_engine.rs"),
        include_str!("../src/policy/test.rs"),
        include_str!("../src/ids/test.rs"),
        include_str!("../src/error/test.rs"),
        include_str!("../src/catalog/cache_test.rs"),
        include_str!("../src/catalogue/test.rs"),
        include_str!("../src/health/test.rs"),
    ];
    let all = sources.concat();
    let mut missing = Vec::new();
    for n in (1..=27).filter(|n| *n != 10) {
        let marker = format!("guard_g{n}_");
        let marker_padded = format!("guard_g{n:02}_");
        if !all.contains(&marker) && !all.contains(&marker_padded) {
            missing.push(n);
        }
    }
    assert!(
        missing.is_empty(),
        "guards with no `guard_gNN_` test: {missing:?}"
    );
}

/// The test names 09-use-cases gives every P0 use case (`*` = any suffix).
const P0_TEST_NAMES: &[&str] = &[
    "sim_desktop_onboarding",
    "sim_tenant_isolation",
    "sim_cli_oneshot_env_only",
    "credential_env_source_*",
    "client_builds_for_each_protocol",
    "testkit_scripted_http_*",
    "golden_hubconfig_v1",
    "sim_platform_token_rotation",
    "sim_detect_local_fingerprint*",
    "detect_env_never_persists*",
    "sim_connect_rollback_on_auth",
    "guard_g11_add_anyway_keeps_row",
    "sim_key_rotation_next_call",
    "sim_managed_origin_switch",
    "resolve_pin_precedence",
    "workload_keys_are_opaque",
    "catalog_context_window_sources",
    "guard_g26_product_header",
    "guard_g9_first_provider_default_race",
    "list_models_error_is_typed",
    "classify_quota_vs_rate_vs_auth",
    "custom_provider_*",
    "guard_g4_multi_instance_policy",
    "import_oc_*",
    "import_oh_*",
    "cache_ttl_with_fake_clock",
    "config_schema_version_*",
    "catalogue_golden",
    "catalog_parser_*",
    "cache_single_flight_*",
    "catalog_tolerant_*",
    "catalog_body_cap",
    "reason_code_strings_stable",
    "secret_*",
    "hubconfig_has_no_secret_fields",
    "credential_shaped_values_rejected",
    "raw_never_in_user_message",
    "cache_key_never_contains_secret",
    "slot_bound_to_record",
    "cache_failure_memo_60s",
    "guard_g27_unknown_kind_fails",
    "model_ids_verbatim",
    "store_err_is_not_none",
    "policy_hosted_refuses_loopback",
    "sim_ssrf_literal_ips",
    "sim_dns_rebinding_scripted",
    "sim_concurrent_writers",
    "builder_registers_custom_kind*",
    "resolve_is_allocation_light*",
    "contract_*",
    "guard_g6_*",
    "guard_g7_*",
    "guard_g22_*",
];

fn collect(dir: &std::path::Path, out: &mut String) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push_str(&std::fs::read_to_string(&path).unwrap());
            out.push('\n');
        }
    }
}

#[test]
fn catalogue_every_p0_use_case_has_its_named_test() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut source = String::new();
    collect(&root.join("src"), &mut source);
    collect(&root.join("tests"), &mut source);
    let mut missing = Vec::new();
    for pattern in P0_TEST_NAMES {
        let found = match pattern.strip_suffix('*') {
            Some(prefix) => source.contains(&format!("fn {prefix}")),
            None => source.contains(&format!("fn {pattern}(")),
        };
        if !found {
            missing.push(*pattern);
        }
    }
    assert!(
        missing.is_empty(),
        "P0 use cases with no test of the catalogued name: {missing:?}"
    );
}
