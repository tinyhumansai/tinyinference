# Specifications

TinyInference owns the provider boundary: normalized messages, tool-call wire
shapes, model requests and responses, usage, capability profiles, streaming,
provider transports, response caching, failure classification, embeddings,
vector storage, and retrieval.

It does not own agent execution, middleware, graph scheduling, capability
registries, persistence beyond the response-cache contract, or tool execution.
Those layers consume `ChatModel`, `EmbeddingModel`, and the normalized types.

Provider adapters must preserve provider-neutral semantics, reject invalid
configuration before network access, avoid credential exposure, classify
structured failures, and reconstruct streamed output identically to unary
output. Embedding models return one fixed-size vector per input in input order.
