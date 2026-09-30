//! The contract suite, run over every built-in kind.

use crate::catalogue::{custom_descriptor, descriptors};
use crate::kinds::{DriverRegistry, KindDriver, OpenAiCompatDriver};
use crate::testkit::{ContractFixture, run_contract};

#[tokio::test]
async fn contract_every_catalogue_kind_passes_the_same_suite_through_the_registry() {
    let registry = DriverRegistry::with_builtin();
    let mut ran = 0;
    for descriptor in descriptors() {
        let driver = registry
            .get(&descriptor.kind)
            .unwrap_or_else(|| panic!("no driver for {}", descriptor.kind));
        run_contract(driver.as_ref(), &ContractFixture::for_builtin(descriptor)).await;
        ran += 1;
    }
    assert_eq!(ran, 34, "26 cloud + managed + 5 local + 2 CLI");
}

#[tokio::test]
async fn contract_a_custom_endpoint_passes_it_too() {
    let driver = OpenAiCompatDriver::custom();
    let fixture = ContractFixture::openai_shaped("acme-gateway", "https://gateway.acme.test/v1");
    run_contract(&driver, &fixture).await;
    assert_eq!(
        driver.descriptor().kind.as_str(),
        custom_descriptor().kind.as_str()
    );
}

#[tokio::test]
async fn contract_a_host_defined_kind_registers_and_passes() {
    use std::sync::Arc;

    let mut registry = DriverRegistry::with_builtin();
    let mut descriptor = custom_descriptor();
    descriptor.kind = crate::ids::KindId::new("acme-cloud");
    descriptor.group = crate::taxonomy::ProviderGroup::Cloud;
    let replaced = registry.register(Arc::new(OpenAiCompatDriver::for_descriptor(descriptor)));
    assert!(replaced.is_none(), "a new kind replaces nothing");
    let driver = registry
        .resolve("ACME-Cloud")
        .expect("registered kinds resolve case-insensitively");
    run_contract(
        driver.as_ref(),
        &ContractFixture::openai_shaped("acme-cloud", "https://api.acme-cloud.test/v1"),
    )
    .await;
}
