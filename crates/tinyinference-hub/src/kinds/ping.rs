//! The one-token completion ping, per wire protocol.

use serde_json::json;

use crate::catalog::unreadable;
use crate::error::HubError;
use crate::ids::ModelId;
use crate::ports::{HubRequest, HubResponse};
use crate::taxonomy::Protocol;

use crate::descriptor::{ProviderDescriptor, Quirk};
use crate::endpoint::endpoint_host;

use super::context::Classifier;
use super::{DriverContext, Target};

/// The most tokens a ping asks for. Small enough to cost almost nothing, large
/// enough that reasoning models that insist on a floor still accept it.
pub(super) const PING_MAX_TOKENS: u32 = 16;

/// The prompt a ping sends.
const PING_PROMPT: &str = "ping";

/// Whether the endpoint wants `max_completion_tokens`: the catalogue row that
/// says so, or an endpoint that is OpenAI itself or an Azure OpenAI resource
/// however the operator reached it (a `custom` row pointed at api.openai.com
/// gets the same 400 on its reasoning models).
fn wants_max_completion_tokens(descriptor: &ProviderDescriptor, target: &Target<'_>) -> bool {
    // Only the OpenAI chat wire has the field; Anthropic's native `/messages`
    // requires `max_tokens` wherever it is hosted.
    descriptor.protocol != Protocol::AnthropicMessages
        && (descriptor.has_quirk(Quirk::MaxCompletionTokens)
            // The full endpoint, not `base()`: the `api-version` is in its query.
            || azure_accepts_max_completion_tokens(target.base_url)
            || endpoint_host(target.base()).is_some_and(|host| host == "api.openai.com"))
}

/// Azure OpenAI took `max_completion_tokens` from `api-version`
/// `2024-09-01-preview`; an older pinned version rejects it (a 400), so those
/// stay on `max_tokens`. No `api-version` at all is the versionless `/openai/v1`
/// surface, which is current.
fn azure_accepts_max_completion_tokens(base: &str) -> bool {
    if !crate::catalogue::is_azure_endpoint(base) {
        return false;
    }
    // Read the query text directly: the endpoint may be scheme-less, which a
    // URL parser refuses, and the version is all we want from it.
    let base = base.trim();
    let base = &base[..base.find('#').unwrap_or(base.len())];
    let version = base.split_once('?').and_then(|(_, query)| {
        url::form_urlencoded::parse(query.as_bytes())
            .find(|(name, _)| name.eq_ignore_ascii_case("api-version"))
            .map(|(_, value)| value.into_owned())
    });
    let Some(version) = version else {
        return true;
    };
    // `2024-09-01`, `2024-09-01-preview`, `2024-06`, `2023-5-15`: compare the
    // numeric date. A name (`v1`, `preview`) is a current alias.
    let mut parts = version.split('-').map(|part| part.parse::<u32>());
    match (parts.next(), parts.next()) {
        (Some(Ok(year)), Some(Ok(month))) => {
            let day = parts.next().and_then(Result::ok).unwrap_or(1);
            (year, month, day) >= (2024, 9, 1)
        }
        _ => true,
    }
}

/// Pings with the protocol the descriptor names.
///
/// OpenAI Chat (and, for the one row that has it, the Responses API, whose
/// chat-completions path also exists) posts to `{base}/chat/completions`; the
/// Anthropic native protocol posts to `{base}/messages`. A `404` on the
/// Responses row is not retried at `/responses`: chat completions is the
/// universal path, and the fallback is a turn concern.
pub(super) async fn ping_by_protocol(
    cx: &DriverContext<'_>,
    descriptor: &ProviderDescriptor,
    classify: &Classifier<'_>,
    target: &Target<'_>,
    model: &ModelId,
) -> Result<(), HubError> {
    let path = match descriptor.protocol {
        Protocol::AnthropicMessages => "/messages",
        _ => "/chat/completions",
    };
    // OpenAI's newer (reasoning) models reject `max_tokens` with a 400 and want
    // `max_completion_tokens`, which OpenAI accepts for every chat model. Other
    // OpenAI-compatible servers know only `max_tokens`, and Anthropic's native
    // API requires it, so the switch is its own descriptor quirk (not a proxy
    // such as the Responses-API flag, which says something else).
    let limit_field = if wants_max_completion_tokens(descriptor, target) {
        "max_completion_tokens"
    } else {
        "max_tokens"
    };
    let request = HubRequest::post_json(
        target.join(path),
        &json!({
            "model": model.as_str(),
            limit_field: PING_MAX_TOKENS,
            "messages": [{"role": "user", "content": PING_PROMPT}],
        }),
    );
    let request = cx.request(
        descriptor,
        target,
        request.with_body_cap(cx.policy.answer_cap),
    );
    let response = cx.call_with(classify, request).await?;
    require_answer(&response, "the completion")
}

/// A `2xx` is not yet an answer. A captive portal, an HTML landing page, or a
/// gateway that wraps an upstream error as `200 {"error": ...}` all answer 200,
/// and counting them as a pass would mark a provider that cannot serve a turn
/// as proven. The body must be a JSON object with no `error` member. The
/// failure is `unknown`, never `auth`: it says nothing about the credential.
pub(super) fn require_answer(response: &HubResponse, what: &str) -> Result<(), HubError> {
    let value: Result<serde_json::Value, _> = serde_json::from_slice(&response.body);
    match value {
        Ok(serde_json::Value::Object(map))
            if !response.truncated && map.get("error").is_none_or(serde_json::Value::is_null) =>
        {
            Ok(())
        }
        _ => Err(HubError::Provider(unreadable(format!(
            "{what} answered 2xx with a body that is not a JSON answer"
        )))),
    }
}
