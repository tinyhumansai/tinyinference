# Client module

The client validates a request, sends it to the System One endpoint, classifies
HTTP failures, retries timeouts and connection-establishment failures, validates
the response against the original questions, and returns attempts and latency. API keys
remain private and render only as `[REDACTED]`.

Production endpoints require HTTPS. Plain HTTP is accepted only for literal
loopback IP addresses used by local tests and development services. Successful
and failed evaluations report attempts and end-to-end latency. Timeouts,
connection-establishment failures, and response body-transfer errors use the
bounded retry policy; other transport errors are terminal. Response bodies are
capped at 16 MiB, and automatic redirects are disabled.

`ClientConfig::openrouter` targets OpenRouter's compatible System One API at
`https://openrouter.ai/api/v1/systemone`.

`ClientConfig::tinyhumans_openrouter` targets the Tiny Humans OpenRouter proxy
at `https://api.tinyhumans.ai/agent-integrations/openrouter/systemone`.
`ClientConfig::with_sdk_name` sanitizes product attribution and sends
`x-sdk-name` only to this exact HTTPS endpoint. OpenRouter, TypeSafe, and
other endpoint overrides do not receive it.

`ClientConfig::self_hosted(endpoint_url, api_key)` targets an operator-declared,
Jev-compatible System One endpoint such as a self-hosted open decision model.
The endpoint is used exactly as given and must pass the same URL checks. There
is no default model, so callers set `EvaluationRequest::model`; the response
must echo that model id. An empty key is allowed for unauthenticated local
servers and then no `Authorization` header is sent.
