# orla v1 — Build plan

A focused single-tenant MVP whose only product job is to be the substrate
the platform engineer's RL mapper reads and writes. Everything else is
deferred.

## Scope

In scope for v1:

- `POST /v1/chat/completions` — OpenAI-compatible proxy with stage routing
- `POST /v1/feedback` — developer-submitted feedback
- Stage registry (`/api/v1/stages`)
- Backend registry (`/api/v1/backends`)
- Completion records writer (async batched)
- Feedback writer (async batched)
- Read endpoints for the mapper:
  `/api/v1/stages/{id}/completions`, `/feedback`, `/metrics`
- Per-backend `max_concurrency` cap (operational necessity)
- Streaming SSE support
- `/healthz` (liveness) and `/readyz` (readiness, DB ping)
- Graceful shutdown on SIGTERM (drain HTTP, workers, BatchWriter buffers)
- Structured logging via `log/slog` with JSON output option
- Prometheus metrics
- Crude per-backend rate limiter (per-instance; single-tenant safety net)

Out of scope (deferred to later phases):

- Fair-share scheduling (single FCFS queue per backend in v1)
- Multi-tenant tenant credit / workflow-run fairness
- Workflow DAG / label propagation
- Access control policies
- KV-cache / memory manager / SGLang flush controller
- `orla agent` / TUI / one-shot CLI
- `pkg/api/` Go client
- pyorla SDK (defer; if the mapper needs a client it can hit REST directly)
- Reasoning hint header
- Interactive vs batch workflow class
- Cache flush on stage completion

## Module layout

```
cmd/orla/                 - main + cobra subcommands
  commands/                - root, serve
internal/
  storage/                 - pgxpool open, goose migrations, BatchWriter[T]
    migrations/            - .sql migrations (goose Up/Down)
    queries/               - .sql files consumed by sqlc generate
    db/                    - sqlc-generated code (committed)
  stages/                  - Stage record + Registry (wraps sqlc.Queries)
  backends/                - Backend record + Registry
  proxy/                   - HTTP handler for /v1/chat/completions
  scheduler/               - Per-backend FCFS executor + worker pool
  provider/                - openai-go integration + retry classification
  telemetry/               - Completion record + feedback writers + aggregation
  api/                     - chi router, middleware, control-plane handlers, healthz/readyz
  config/                  - envconfig loader (ORLA_ prefix)
docs/                      - personas, proxy, storage, rl, plan
```

Differences from Orla's layout, on purpose:

- `backends/` is its own package, not nested in `serving/`. Backend
  registry has nothing to do with serving HTTP.
- `proxy/` is its own package. Not nested in `serving/api/`.
- `scheduler/` is its own package, separate from the backend manager.
  Lets the scheduler interface be cleanly swappable.
- No `serving/` umbrella package. The "serving layer" abstraction in
  Orla is an artifact of an older design; from scratch, the packages
  are flat under `internal/`.
- No `agent/`, no `tui/`. Already-deleted concepts.

## Dependencies

- `github.com/openai/openai-go` — provider
- `github.com/jackc/pgx/v5` (with `pgx/v5/stdlib` adapter for goose) — Postgres
- `github.com/sqlc-dev/sqlc` — generates type-safe pgx code from SQL files
- `github.com/pressly/goose/v3` — migrations (uses `*sql.DB` via pgx/stdlib)
- `github.com/go-chi/chi/v5` — HTTP router + middleware (RequestID, Logger,
  Recoverer, Timeout, BodyLimit, RealIP)
- `github.com/sethvargo/go-envconfig` — env-var-first config with `ORLA_` prefix
- `github.com/spf13/cobra` — CLI subcommands
- `github.com/cenkalti/backoff/v4` — retry
- `github.com/google/uuid` — completion IDs
- `github.com/jonboulle/clockwork` — testable time
- `github.com/prometheus/client_golang` — metrics
- `golang.org/x/time/rate` — rate limiting
- stdlib `log/slog`, `database/sql`, `container/heap`, `net/http`,
  `encoding/json`

Test deps:
- `github.com/stretchr/testify` — assertions
- `github.com/google/go-cmp` — deep struct diffs
- `github.com/testcontainers/testcontainers-go` +
  `github.com/testcontainers/testcontainers-go/modules/postgres` — ephemeral
  Postgres for storage tests

No DI framework. Plain struct composition.

CI tooling (added day 1):

- `.golangci.yml` — `errcheck`, `govet`, `ineffassign`, `staticcheck`,
  `unused`, `errorlint`, `gosec`, `copyloopvar`, `misspell`
- `.github/workflows/ci.yml` — build, test (with race), lint, coverage upload
- `.github/dependabot.yml` — weekly gomod + github-actions updates

## Schedule

Three weeks of focused work, plus a buffer week.

### Week 1: substrate

Goal: HTTP server starts, stages and backends persist, the proxy dispatches
to a backend, and the basic shape works end-to-end against Ollama.

- Day 1: bootstrap — `cmd/orla/main.go`, `internal/config` (env-var-first
  via `ORLA_*` prefix), `internal/core/logger.go` (slog with JSON option),
  `internal/storage` with `Open` (pgx/stdlib + goose), `BatchWriter[T]`,
  first migration (empty), graceful shutdown skeleton in `main.go`. Local
  Postgres expected to be running (see `docs/storage.md` for setup).
- Day 2: stages package — `Stage`, `Registry`, schema in
  `0001_stages.sql`, REST handlers in `internal/api`.
- Day 3: backends package — same shape as stages, schema in
  `0002_backends.sql`. Plus the `Registry.GetProvider(name)` lazy
  provider construction.
- Day 4: provider package — port the openai-go integration. The
  `internal/model/openai.go` from Orla is a good starting reference
  but retype, don't paste.
- Day 5: scheduler package — single FCFS queue per backend, fixed
  worker pool sized by `backend.max_concurrency`. ~150 lines.
- Day 6-7: proxy package — `/v1/chat/completions` handler. Stage lookup,
  backend resolution, message/tool conversion, streaming. End-to-end
  test against Ollama. `/healthz` and `/readyz` endpoints wired into the
  HTTP server, with `/readyz` doing a DB ping.

### Week 2: telemetry + RL surface

Goal: the mapper can read everything it needs to make decisions.

- Day 8: completion records — schema `0003_completion_records.sql`,
  `BatchWriter[*CompletionRecord]` instance, integrate into the scheduler
  worker (records emit on dispatch completion).
- Day 9: feedback — schema `0004_feedback.sql`, `BatchWriter[*Feedback]`,
  `POST /v1/feedback` handler returning 202.
- Day 10: read endpoints — `GET /api/v1/stages/{id}/completions`,
  `/feedback`, with pagination via `since=` and `limit=`.
- Day 11: metrics aggregation — `GET /api/v1/stages/{id}/metrics`. The
  query (per-backend aggregates, p50/p95 latency, rating distribution)
  is the meat of this day.
- Day 12: Prometheus integration — `requests_total`, `queue_depth`,
  `backend_latency_seconds`, `batch_writer_drops_total`,
  `inflight_per_backend`.
- Day 13-14: rate limiter — per-backend `golang.org/x/time/rate.Limiter`
  applied on dispatch. Plus an end-to-end smoke test that exercises the
  full loop: register backend, map stage, send completion, submit
  feedback, query metrics.

### Week 3: tests + polish

- Day 15-17: tests. Storage tests, registry tests, scheduler tests,
  proxy handler tests, integration test (mock LLM server end-to-end).
  ~1,500 lines of test code.
- Day 18: documentation pass on the four design docs + a top-level README.
- Day 19: smoke test against a real backend (Ollama). Fix whatever real-world
  quirks surface.
- Day 20-21: buffer / polish / `go vet ./... && golangci-lint run` cleanup.

### Week 4: buffer

Reality always intrudes. If the first three weeks went smoothly, week 4
is for:

- Real-world testing against a second backend (vLLM or OpenAI proper).
- The first version of the mapper-side reference implementation (small
  Python script that exercises the REST API end-to-end).
- A `BSL` or `Elastic License v2` decision and license headers on every
  file.
- A draft README that pitches the product, not just the protocol.

## Decisions to lock in before day 1

These take 10 minutes each and prevent a week of revisiting:

1. **Header prefix**: `X-Orla-*`. Lowercase tag keys.
2. **Module path**: `github.com/harvard-cns/orla`. Locked.
3. **License**: none for now. Private repo, all-rights-reserved by
   copyright default. Revisit before any public release.
4. **Header for completion id in responses**: don't bother. The body's
   `id` field already carries it. Clients that need it parse the body
   (they do anyway for `model`).
5. **Storage**: Postgres 17. Driver `jackc/pgx/v5` via `pgx/v5/stdlib`.
   Default dev URL `postgres://orla:orla@localhost:5432/orla?sslmode=disable`.
   Connection via `ORLA_DATABASE_URL` env var (preferred) or
   `database_url` in YAML config.
6. **Default listen address**: `localhost:8081`. Same as Orla, no reason
   to change.
7. **Config env prefix**: `ORLA_`. Every config field bindable via env var
   (e.g., `ORLA_DATABASE_URL`, `ORLA_LISTEN_ADDRESS`, `ORLA_LOG_FORMAT`)
   so containerized deployments work without a YAML file.

## What to keep open

A few decisions worth leaving for week 4 instead of day 1:

- **Mapper SDK in Python**: yes, but not in this repo. Separate repo,
  separate release cycle. v1 of the SDK is "the OpenAI Python client
  pointed at orla plus a small `orla_client.py` for the control plane."
- **Authentication**: defer to a reverse proxy. orla v1 ships
  unauthenticated and documents "run behind nginx/cloudflare with your
  auth of choice."
- **Multi-tenant fairness**: defer to v2. Document the limitation.
- **The platform-engineer dashboard**: separate product. Out of scope.

## Source of truth for this work

This file (`docs/orla/plan.md`) is the source of truth for what v1
contains and how it's structured. Update it as decisions change.
Don't keep a mental map; keep it here.
