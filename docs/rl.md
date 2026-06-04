# The RL interface

What the platform engineer's reinforcement-learning mapper sees, decides,
and writes back. orla's job is to expose this loop cleanly; orla does
not run the RL itself.

## The loop

```
       ┌──────────────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      │
┌─────────────┐                          ┌──────────────┐     │
│  Mapper     │   1. Read state ───────► │              │     │
│ (RL agent,  │                          │    orla     │     │
│  external)  │   2. Update mappings ──► │              │     │
│             │                          │              │     │
│             │   3. Observe rewards ◄── │              │     │
└─────────────┘                          └──────────────┘     │
       │                                        ▲             │
       │                                        │             │
       │                                        │ requests    │
       │                                        │             │
       │                                  ┌──────────────┐    │
       │                                  │  Developer's │    │
       └──────────────────────────────────│    agent     │────┘
                                          │              │
                                          └──────────────┘
```

The mapper is a separate process (any language; Python is the natural
choice for ML work). It talks to orla only over HTTP.

## What the mapper observes

### 1. Stage and backend state

`GET /api/v1/stages` — list of all known stages and their current
mappings.

`GET /api/v1/backends` — list of registered backends, including
`cost_model`, `quality`, `max_concurrency`. These are the mapper's
arm priors.

### 2. Completion records (telemetry)

For each LLM call, orla records:

- `completion_id` — joinable with feedback
- `stage_id` — which arm-context this call was
- `backend` — which arm was pulled
- `cost_usd` — immediate negative reward signal
- `latency_ms` — immediate negative reward signal
- `prompt_tokens`, `completion_tokens` — context for cost/latency
- `workflow_run`, `tags_json` — context for batched-trajectory updates
- `status` — success or error
- `created_at`

Endpoints:
- `GET /api/v1/stages/{id}/completions?limit=&since=` — recent records
  for a stage.
- `GET /api/v1/stages/{id}/metrics?since=` — pre-aggregated stats
  (count, avg latency, p50/p95, total cost, broken down by backend).
  This is what most mappers will want to read most often.

### 3. Feedback (rewards)

The developer submits feedback per completion. Joined with completion
records, this gives the mapper the canonical reward signal:

- `rating` — 0..1, optional. The headline reward.
- `labels` — array of strings (e.g., `["hallucinated", "off-topic"]`).
  Useful for the mapper to learn structured failure modes.
- `notes` — free-form text. The mapper probably ignores this, but the
  product team reads it.

Endpoints:
- `GET /api/v1/stages/{id}/feedback?since=`

## What the mapper writes

Two write operations, both small:

### 1. Remap a stage to a new backend

```
PATCH /api/v1/stages/{id}
{
  "backend": "gpt4o-mini",
  "labels": {
    "mapper_state": "{\"last_pull\": \"...\", \"epsilon\": 0.1}"
  }
}
```

The mapping change applies immediately to the next call to that stage.
In-flight requests dispatched against the old backend run to completion;
no preemption.

The `labels` field is opaque to orla — the mapper can stash whatever
state it needs there (exploration timestamps, arm-pull counters, etc.)
to survive its own restarts.

### 2. Update backend priors

```
PATCH /api/v1/backends/{name}
{
  "cost_model": {"input_cost_per_mtoken": 0.15, "output_cost_per_mtoken": 0.60},
  "quality": 0.82
}
```

Useful when the mapper's own estimates have improved beyond the manual
ones the human platform engineer entered initially.

## A typical mapper loop, in pseudocode

```python
import orla_client as bc

client = bc.Client("http://orla:8081")

while True:
    # 1. Read state
    stages = client.list_stages()
    backends = client.list_backends()

    # 2. For each stage, decide whether to remap
    for stage in stages:
        recent = client.get_completions(stage.id, since=last_seen)
        feedback = client.get_feedback(stage.id, since=last_seen)
        joined = join_on_completion_id(recent, feedback)

        # Mapper's policy logic. E.g., epsilon-greedy over backends.
        new_backend = policy.choose(stage, backends, joined)

        if new_backend != stage.backend:
            client.patch_stage(stage.id, backend=new_backend)

    time.sleep(EVALUATION_INTERVAL)
```

The mapper is stateless from orla's perspective; orla keeps the
durable state. The mapper restarts cleanly by reading state from orla
on startup.

## Why this shape

A few decisions worth being explicit about:

- **The mapper doesn't run in-process.** orla stays a pure dispatcher.
  Keeps the persona boundary clean, keeps orla Go-only, lets the
  mapper be Python/anything.
- **orla does not implement any policy.** No epsilon-greedy, no
  Thompson sampling, no UCB. The mapper owns all decisions.
- **No streaming push to the mapper.** Pull-based via REST. Simpler,
  matches how RL training loops naturally work (periodic batch updates).
  WebSocket / Server-Sent-Event push can be added if a real mapper
  asks for it.
- **Mapper state lives in `stages.labels`**, not in a separate table.
  Keeps the schema small. The mapper's own state is its problem.
- **The mapper can read raw observation rows, not just aggregates.**
  Aggregates are convenience; the mapper can implement its own windows
  and stats if it needs to.

## What orla does *not* need to implement to support RL

These come up in conversation about "the RL story" but orla doesn't
need them for v1:

- **A/B testing framework.** The mapper handles exploration. If it
  wants to A/B test, it remaps a stage to a new backend and observes.
- **Reward shaping.** The mapper transforms `rating`, `cost`, `latency`
  into its own internal reward however it wants.
- **Model training infrastructure.** Not orla's job.
- **Online evaluation harnesses.** Not orla's job.
- **Multi-armed bandit math.** Not orla's job.

orla is the substrate. The mapper is the policy.
