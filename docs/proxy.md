# Proxy: `POST /v1/chat/completions`

The OpenAI-compatible inference entry point. The developer's entire hot path.

## Wire shape

Request: standard OpenAI chat completion. Plus identity metadata (see
`personas.md` for the full table).

Response: standard OpenAI chat completion. The response's `model` field
reports the **resolved backend name**, not necessarily what the client
sent in its `model` field. Divergence is by design under the two-persona
contract.

Streaming follows OpenAI's data-only SSE format, terminated by
`data: [DONE]`.

## Request handling order

The handler runs these checks in order. Each is a 400 unless noted.

1. **Decode body.** Body too large (>10 MB) → 400.
2. **`messages` non-empty.**
3. **Stage extracted.** From `X-Orla-Stage` header, falling back to
   `metadata.orla.stage` in the body. Missing → 400.
4. **Resolve backend.** `registry.GetOrCreate(stage)` auto-creates a default
   stage record on first sighting. If `stage.Backend` is set, use it. If not,
   fall back to `req.Model`. If neither is set → 400.
5. **Apply inference policy** from the stage record. Currently just
   `reasoning_effort`.
6. **Convert messages and tools** to the internal model types.
7. **Dispatch** via `LayerExecute` → `BackendManager.ScheduleChat` →
   per-backend queue → worker → openai-go provider.
8. **Encode response** as OpenAI chat completion (or stream chunks).

## What orla does *not* do in the proxy

- No content-based routing. We never look at `messages` to pick a backend.
- No cost/quality-based routing. The platform engineer's mapper is the
  one that uses cost and quality; orla just exposes the priors.
- No fallback chain. If the resolved backend fails, return the error.
  Same-backend retries on 5xx/429 happen inside `chatWithRetry` and are
  invisible to the client.
- No model-name override. The request's `model` is a fallback only.

## Auto-create on first sighting

If a developer uses a stage id that orla has never seen, the daemon
inserts a default stage record (empty backend, empty everything) and
the request falls back to `req.Model` for that one call. The platform
engineer can later `PUT /api/v1/stages/{id}` to map it.

This means a developer can deploy new agent code without coordination
with the platform engineer. Their requests still flow; the mapper picks
them up on the next pass.

## Identity tags become completion-record dimensions

Every dispatched request results in one row in `completion_records` with:

- `completion_id` (UUID assigned by orla)
- `stage_id` (from the request)
- `workflow_run` (if set; nullable)
- `backend` (the resolved name)
- `tags_json` (the full `X-Orla-Tag-*` map)
- `prompt_tokens`, `completion_tokens`, `latency_ms`, `cost_usd`, `status`,
  `created_at`

This is the mapper's primary observation channel. See `storage.md`.

## Streaming semantics

For `stream: true`:

- orla opens an upstream stream and proxies chunks. Each chunk is rewritten
  to include the resolved backend in the `model` field.
- The worker holds its concurrency slot until the upstream stream finishes
  draining, not just until the first chunk arrives. This is load-bearing
  for the `max_concurrency` invariant.
- On client disconnect (ctx cancellation), orla drains the upstream stream
  silently and the worker releases its slot.

## Error shape

All non-200 responses use the OpenAI error envelope:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error" | "permission_denied" | "rate_limit_exceeded" | "server_error" | "api_error"
  }
}
```

Status code drives the `type` field. Clients that already handle OpenAI
errors handle these correctly.
