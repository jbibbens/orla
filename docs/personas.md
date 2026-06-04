# Personas

orla is built around two distinct personas. Every API surface, every storage
table, every line of code should fit on one side of this line.

## The agent logic writer (developer)

Writes the agent's prompts, tools, and orchestration logic. Cares about getting
good LLM responses for their workflow. Does not pick backends, does not set
inference policy, does not configure scheduling.

**What the developer says to orla**, on every LLM call:

- "This call is part of stage X." (`X-Orla-Stage` header, required)
- "It's part of workflow run Y." (`X-Orla-Workflow-Run` header, optional)
- "Here are arbitrary tags for context." (`X-Orla-Tag-*` headers, optional)
- Standard OpenAI chat completion body: messages, tools, sampling params.

**What the developer says to orla, occasionally:**

- "Completion `chatcmpl-abc` was good/bad." (`POST /v1/feedback`)

That's the entire developer surface.

## The platform engineer (or external mapper)

Owns the mapping of stages to backends and reads the data needed to do that
mapping well. In production this is an external system — often a
reinforcement-learning agent — but it can also be a human running ad-hoc
queries against the daemon.

**What the platform engineer does:**

- Registers backends with `POST /api/v1/backends`. Each backend has an
  endpoint, a model id, optional cost and quality priors, and a
  max-concurrency cap.
- Maps stages to backends with `PUT /api/v1/stages/{id}`. The stage record
  also holds inference policy (reasoning_effort) and arbitrary labels.
- Reads observation data: completion records (per-call cost, latency, tokens,
  resolved backend) and feedback records (developer ratings) via the
  `/api/v1/stages/{id}/completions` and `/feedback` endpoints.
- Reads aggregates: `/api/v1/stages/{id}/metrics` returns rolled-up cost,
  latency percentiles, rating distribution, broken down by backend.

The platform engineer's tools (RL agent, dashboards, ad-hoc scripts) live
*outside* orla. orla exposes REST; the platform engineer's mapper drinks
from it. orla does not run the RL loop itself.

## What this contract enforces

The persona split is not a guideline — it's a structural property of the
codebase. Concretely:

- The developer **cannot** pick a backend. The `model` field on chat
  completion requests is a fallback only, consulted when the stage has no
  mapping. The platform engineer's mapping always wins.
- The developer **cannot** set scheduling policy or priority. There is no
  request field for it. All scheduling is the daemon's decision.
- The platform engineer **cannot** see developer prompts or messages. The
  control-plane API exposes statistics and identity tags, not request
  payloads.
- The platform engineer **cannot** change a routing decision after a
  request has dispatched. Mapping changes apply to the next call to that
  stage.

## What orla is, what orla isn't

orla **is**:

- A dispatcher that takes tagged LLM calls and routes them to the platform
  engineer's chosen backend.
- A persistent registry of stages, backends, completion records, and
  feedback.
- The substrate the platform engineer's mapper reads and writes.

orla **is not**:

- An optimizer. The mapper is external.
- An agent runtime. Tools execute in the developer's process; orla just
  observes the tool calls in the wire format for telemetry.
- A multi-tenant fair-share scheduler (v1). Single-tenant by design;
  multi-tenant fairness can be added in a later phase when there's a real
  user asking for it.

## Identity in the request

Every chat completion request carries identity metadata that the daemon uses
for routing and observation. None of it affects the response shape.

| Header | Required | Meaning |
|---|---|---|
| `X-Orla-Stage` | yes | Stage this call belongs to. Drives routing. |
| `X-Orla-Workflow-Run` | no | Workflow run identifier. Persisted on completion records for the mapper to group calls by run. |
| `X-Orla-Tag-<Key>` | no, repeatable | Arbitrary tag, lowercased. Persisted on completion records. |

If the SDK can't easily set headers, the body's `metadata.orla` field is a
fallback for the same data:

```json
{
  "model": "...",
  "messages": [...],
  "metadata": {
    "orla": {
      "stage": "planning",
      "workflow_run": "wf-abc",
      "tags": {"tenant": "alice"}
    }
  }
}
```

Headers win over body when both are set.
