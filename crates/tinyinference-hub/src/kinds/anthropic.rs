//! [`AnthropicDriver`]: Anthropic's native `/models` and `/messages` calls.

use async_trait::async_trait;
use serde_json::Value;

use crate::catalog::{Fetched, ModelEntry, parse_openai_value, too_large, unreadable};
use crate::descriptor::ProviderDescriptor;
use crate::error::HubError;
use crate::ports::HubRequest;

use super::{DriverContext, KindDriver, Target};

/// The most models Anthropic returns per page (`limit` is capped at 1000).
const PAGE_SIZE: usize = 1000;

/// The most pages one read follows, so a `has_more` that never clears cannot
/// loop.
const MAX_PAGES: usize = 10;

/// Anthropic's native API.
///
/// Its `GET /models` is paged with a default of **20** per page, so a plain
/// read (OpenCompany's) sees a fraction of the catalog. This driver asks for the
/// maximum page size and follows `has_more`/`last_id`. The native API also
/// rejects a bearer-authenticated request with no `anthropic-version` header as
/// malformed (a `400`, not a `401`), which is why the credential is presented
/// through [`AuthStyle::Anthropic`](crate::AuthStyle) and never as a bearer.
#[derive(Clone, Debug)]
pub struct AnthropicDriver {
    descriptor: ProviderDescriptor,
}

impl AnthropicDriver {
    /// A driver for the Anthropic catalogue row.
    pub fn for_descriptor(descriptor: ProviderDescriptor) -> Self {
        Self { descriptor }
    }
}

#[async_trait]
impl KindDriver for AnthropicDriver {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }

    async fn list_models(
        &self,
        cx: &DriverContext<'_>,
        target: &Target<'_>,
    ) -> Result<Fetched, HubError> {
        let mut models: Vec<ModelEntry> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut skipped = 0usize;
        let mut after: Option<String> = None;
        let started = cx.clock.now();
        for page in 0..MAX_PAGES {
            let left = cx.time_left(started)?;
            let mut path = format!("/models?limit={PAGE_SIZE}");
            if let Some(cursor) = &after {
                path.push_str("&after_id=");
                path.push_str(&urlencode(cursor));
            }
            let url = target.join(&path);
            let mut request = cx.request(
                &self.descriptor,
                target,
                HubRequest::get(url).with_body_cap(cx.policy.catalog_cap),
            );
            request.timeout = request.timeout.min(left);
            let response = cx.call(self, request).await?;
            if response.truncated {
                return Err(HubError::Provider(too_large("the model list")));
            }
            let envelope: Value = serde_json::from_slice(&response.body)
                .map_err(|_| HubError::Provider(unreadable("the model list was not JSON")))?;
            let parsed = parse_openai_value(&envelope).map_err(HubError::Provider)?;
            skipped += parsed.skipped;
            for entry in parsed.entries {
                if seen.insert(entry.id.clone()) {
                    models.push(entry);
                }
            }
            let has_more = envelope.get("has_more").and_then(Value::as_bool) == Some(true);
            let last = envelope
                .get("last_id")
                .and_then(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string);
            match (has_more, last) {
                (false, _) => break,
                // Follow the cursor while it makes progress and the page cap
                // allows.
                (true, Some(cursor))
                    if page + 1 < MAX_PAGES && after.as_deref() != Some(cursor.as_str()) =>
                {
                    after = Some(cursor);
                }
                // `has_more` with no cursor, a cursor that did not move (a proxy
                // ignoring `after_id`), or the page cap: what was read is a
                // prefix, and says so instead of asking again for the same page.
                (true, _) => {
                    return Ok(Fetched {
                        truncated: true,
                        ..Fetched::new(models)
                    });
                }
            }
        }
        // Judged over the whole read: one bad late page must not discard the
        // good pages before it, but rows with no usable entry at all is not a
        // healthy empty catalog.
        if models.is_empty() && skipped > 0 {
            return Err(HubError::Provider(unreadable(format!(
                "the model list had {skipped} rows and none was usable"
            ))));
        }
        Ok(Fetched::new(models))
    }
}

/// Percent-encodes a cursor for a query value.
fn urlencode(raw: &str) -> String {
    url::form_urlencoded::byte_serialize(raw.as_bytes()).collect()
}
