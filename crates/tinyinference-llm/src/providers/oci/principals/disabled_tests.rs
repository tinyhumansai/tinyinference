use super::*;

#[test]
fn flows_report_the_feature_name_when_disabled() {
    for result in [instance_source(), oke_source()] {
        match result {
            Err(error) => assert!(error.to_string().contains("oci-principals"), "{error}"),
            Ok(_) => panic!("expected the flow to be unsupported"),
        }
    }
}
