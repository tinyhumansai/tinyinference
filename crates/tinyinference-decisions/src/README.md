# Source layout

| directory | responsibility |
| --- | --- |
| `client/` | HTTP execution, bounded retry, secret handling, and measurements |
| `request/` | typed Choice, Score, and Noul request payloads and validation |
| `emulation/` | Box questions emulated with Noul questions over image crops |
| `response/` | typed answers and request-relative response validation |
| `error/` | crate-wide classified failures |
| `sage/` | Levanto Sage decision requests, results, and HTTP transport |
