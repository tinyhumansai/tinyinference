//! The kind contract suite through the public API only: every built-in kind, and
//! a kind a host defines itself, pass the same checks.
#![cfg(feature = "testing")]

use std::sync::Arc;

use tinyinference_hub::catalogue::{self, custom_descriptor};
use tinyinference_hub::kinds::{DriverRegistry, KindDriver, OpenAiCompatDriver};
use tinyinference_hub::testkit::{ContractFixture, run_contract};
use tinyinference_hub::{KindId, ProviderGroup};

#[tokio::test]
async fn contract_every_built_in_kind_passes_from_outside_the_crate() {
    let registry = DriverRegistry::with_builtin();
    for descriptor in catalogue::descriptors() {
        let driver = registry.get(&descriptor.kind).expect("a driver per kind");
        run_contract(driver.as_ref(), &ContractFixture::for_builtin(descriptor)).await;
    }
}

#[tokio::test]
async fn contract_a_host_kind_registered_through_the_public_api_passes() {
    let mut descriptor = custom_descriptor();
    descriptor.kind = KindId::new("host-kind");
    descriptor.group = ProviderGroup::Cloud;
    let mut registry = DriverRegistry::new();
    registry.register(Arc::new(OpenAiCompatDriver::for_descriptor(descriptor)));
    let driver = registry.resolve("host-kind").expect("registered");
    run_contract(
        driver.as_ref() as &dyn KindDriver,
        &ContractFixture::openai_shaped("host-kind", "https://api.host.test/v1"),
    )
    .await;
}
