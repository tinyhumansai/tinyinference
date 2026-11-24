use super::*;

#[test]
fn parses_openai_maximum_context_length() {
    let message = "This model's maximum context length is 128000 tokens. However, your \
                   messages resulted in 130512 tokens. Please reduce the length of the messages.";
    assert_eq!(parse_context_limit_from_error(message), Some(128_000));
}

#[test]
fn parses_openrouter_endpoint_limit() {
    let message = "This endpoint's maximum context length is 163840 tokens. However, you \
                   requested about 201733 tokens (195000 of text input, 6733 in the output). \
                   Please reduce the length of either one, or use the \"middle-out\" transform.";
    assert_eq!(parse_context_limit_from_error(message), Some(163_840));
}

#[test]
fn parses_deepseek_limit() {
    let message = "This model's maximum context length is 131072 tokens. However, you \
                   requested 140000 tokens (139000 in the messages, 1000 in the completion).";
    assert_eq!(parse_context_limit_from_error(message), Some(131_072));
}

#[test]
fn parses_anthropic_prompt_too_long() {
    let message = "prompt is too long: 208000 tokens > 200000 maximum";
    assert_eq!(parse_context_limit_from_error(message), Some(200_000));
}

#[test]
fn parses_vllm_maximum_model_length() {
    let message = "The decoder prompt (length 40000) is longer than the maximum model length \
                   of 32768. Make sure that `max_model_len` is no smaller than the number of \
                   text tokens.";
    assert_eq!(parse_context_limit_from_error(message), Some(32_768));
}

#[test]
fn parses_mistral_trailing_limit() {
    let message = "Prompt contains 40000 tokens and 0 draft tokens, too large for model with 32768 \
         maximum context length";
    assert_eq!(parse_context_limit_from_error(message), Some(32_768));
}

#[test]
fn parses_trailing_limit_after_a_multibyte_character() {
    // A no-break space or a CJK character right before the number used to
    // panic: the start of the digits was taken as one byte past it.
    let message = "too large for model with\u{a0}32768 maximum context length";
    assert_eq!(parse_context_limit_from_error(message), Some(32_768));
    let message = "输入过长，模型32768 maximum context length";
    assert_eq!(parse_context_limit_from_error(message), Some(32_768));
}

#[test]
fn parses_gemini_allowed_maximum() {
    let message = "The input token count (1200000) exceeds the maximum number of tokens \
                   allowed (1048576).";
    assert_eq!(parse_context_limit_from_error(message), Some(1_048_576));
}

#[test]
fn parses_llama_cpp_n_ctx() {
    let message = "The number of tokens to keep from the initial prompt is greater than the \
                   context length (n_keep: 10978 >= n_ctx: 8192).";
    assert_eq!(parse_context_limit_from_error(message), Some(8_192));
    let message = "the request exceeds the available context size (8192 tokens), try \
                   increasing it";
    assert_eq!(parse_context_limit_from_error(message), Some(8_192));
}

#[test]
fn accepts_grouped_digits() {
    let message = "maximum context length is 1,048,576 tokens";
    assert_eq!(parse_context_limit_from_error(message), Some(1_048_576));
}

#[test]
fn rejects_messages_without_a_stated_limit() {
    assert_eq!(
        parse_context_limit_from_error("exceeds the context window of this model, requested 40000"),
        None
    );
    assert_eq!(
        parse_context_limit_from_error("requested 40000 tokens, maximum context length exceeded"),
        None
    );
    assert_eq!(parse_context_limit_from_error("rate limit reached"), None);
    assert_eq!(parse_context_limit_from_error(""), None);
}

#[test]
fn rejects_implausible_limits() {
    assert_eq!(
        parse_context_limit_from_error("maximum context length is 12 tokens"),
        None
    );
}

#[test]
fn parses_dashscope_input_length_range() {
    // Alibaba's Qwen3-8B endpoint (via OpenRouter) advertises 131072 but
    // rejects anything over 98304 with this body.
    let message = "Provider returned error: {\"error\":{\"code\":\"invalid_parameter_error\",\
                   \"message\":\"Range of input length should be [1, 98304]\",\
                   \"type\":\"invalid_request_error\"}}";
    assert_eq!(parse_context_limit_from_error(message), Some(98_304));
}

#[test]
fn dashscope_range_without_an_upper_bound_is_not_a_window() {
    assert_eq!(
        parse_context_limit_from_error("Range of input length should be [1, ]"),
        None
    );
}
