use super::*;

/// One endpoint for the tests that do not exercise scoping. Every test uses
/// its own model id, because the learned store is process-wide.
const AT: &str = "https://api.example/v1";

#[test]
fn a_rejection_blames_only_a_parameter_we_actually_sent() {
    let sent = ["temperature", "max_tokens"];
    assert_eq!(
        parameter_blamed_by(
            "Unsupported value: 'temperature' does not support 0.2 with this model.",
            &sent
        ),
        Some("temperature")
    );
    // A field we did not send is not ours to drop.
    assert_eq!(
        parameter_blamed_by("Unsupported parameter: 'logit_bias'", &sent),
        None
    );
    // And a 400 that is not about the request's shape teaches us nothing.
    assert_eq!(
        parameter_blamed_by("the temperature in Paris is 19 degrees", &sent),
        None
    );
}

#[test]
fn a_rejection_names_the_wire_name_that_actually_went_out() {
    // After a rename the wire name is not the caller's name, so the blame is
    // matched against what was sent.
    let sent = ["temperature", "max_completion_tokens"];
    assert_eq!(
        parameter_blamed_by(
            "400: Unsupported parameter: 'max_completion_tokens' is not supported",
            &sent
        ),
        Some("max_completion_tokens")
    );
    // Nothing we sent is named, so there is nothing to drop and no retry.
    assert_eq!(
        parameter_blamed_by(
            "400: Extra inputs are not permitted: 'reasoning_effort'",
            &sent
        ),
        None
    );
}

#[test]
fn the_longest_matching_name_is_blamed() {
    // `max_completion_tokens` does not contain `max_tokens`, but a future pair
    // that nests would otherwise blame the shorter name.
    let sent = ["top", "top_p"];
    assert_eq!(
        parameter_blamed_by("Unsupported parameter: 'top_p'", &sent),
        Some("top_p")
    );
}

#[test]
fn nothing_is_omitted_until_a_rejection_is_remembered() {
    let model = "omission-learning-test-model-v1";
    assert!(!is_omitted(AT, model, "top_p"));
    remember_omit(AT, model, "top_p");
    assert!(is_omitted(AT, model, "top_p"));
    assert!(
        !is_omitted(AT, model, "temperature"),
        "only the rejected parameter is dropped"
    );
}

#[test]
fn a_learned_omission_ignores_model_id_case() {
    let model = "Omission-Case-Test-Model";
    remember_omit(AT, model, "seed");
    assert!(is_omitted(AT, &model.to_ascii_lowercase(), "seed"));
}

#[test]
fn a_rejection_is_learned_about_the_endpoint_that_made_it() {
    // A model id is not unique across endpoints: one gateway refusing a
    // parameter says nothing about another serving the same id.
    let model = "omission-endpoint-scope-test-model-v1";
    let one = "https://gateway-one.example/v1";
    let two = "https://gateway-two.example/v1";

    remember_omit(one, model, "max_tokens");
    assert!(is_omitted(one, model, "max_tokens"));
    assert!(
        !is_omitted(two, model, "max_tokens"),
        "the other gateway never rejected anything"
    );

    // Same service, different operation: one endpoint answering two ways.
    for operation in [
        "https://gateway-one.example/v1/chat/completions",
        "https://gateway-one.example/v1/responses",
        "https://gateway-one.example/v1/",
        "https://GATEWAY-ONE.example/v1?client_version=1",
    ] {
        assert!(
            is_omitted(operation, model, "max_tokens"),
            "{operation} is the same service"
        );
    }

    // A path-routed gateway is several services behind one origin.
    assert!(
        !is_omitted(
            "https://gateway-one.example/vendor-b/v1",
            model,
            "max_tokens"
        ),
        "another upstream behind the same host has not rejected anything"
    );
}

#[test]
fn an_unparseable_endpoint_is_used_whole() {
    assert_eq!(endpoint_scope("  Not A URL  "), "not a url");
    assert_eq!(endpoint_scope("https://"), "https://");
}
