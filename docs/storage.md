# Storage

orla persists everything in a single Postgres database. The schema is the
contract between the daemon and the platform engineer's mapper.

## Driver and connection

- `github.com/jackc/pgx/v5` with the `pgx/v5/stdlib` adapter, so
  `database/sql` works for both application queries and `goose` migrations.
  Native pgx is available for hot paths if profiling later warrants it.
- Connection URL via `ORLA_DATABASE_URL` env var (preferred) or
  `database_url` in YAML config. Conventional Postgres URL:
  `postgres://user:pass@host:5432/orla?sslmode=...`.
- Connection pool: `*sql.DB` defaults are sufficient for v1.
  `MaxOpenConns` set to `2 × backend.max_concurrency_sum` (an upper bound
  on simultaneous dispatch goroutines plus headroom for the API and
  BatchWriter). `MaxIdleConns` set equal. `ConnMaxLifetime` 30 minutes so
  managed Postgres rotations don't surprise us.
- `sslmode=disable` only for local dev. Managed deployments use
  `sslmode=require` or `sslmode=verify-full` per the provider's guidance.

## Migrations

`github.com/pressly/goose/v3` over embedded `.sql` files under
`internal/storage/migrations/`. Files named `NNNN_description.sql` with
`-- +goose Up` and `-- +goose Down` sections. The Postgres dialect is set
via `goose.SetDialect("postgres")`. Migrations run on every
`storage.Open` so a fresh database comes up ready to use.

## Write strategy

Two write classes with different durability needs:

| Class | Examples | Write mode |
|---|---|---|
| Control plane | Stage records, backend records | Synchronous |
| Data plane | Completion records, feedback | Async batched via `BatchWriter[T]` |

Control-plane writes return only after the row is durable; the caller
needs the confirmation. Data-plane writes go to a buffered channel and
are flushed in batches of ~100 or every ~100ms, whichever comes first.
Buffer-full drops are counted in a Prometheus metric, never block the
producer. Flushes use Postgres `COPY` via `pgx.CopyFrom` for throughput;
this is the meaningful win over per-row `INSERT`.

## Schema (v1)

### `stages`

The platform engineer's action surface.

```sql
CREATE TABLE stages (
    id                TEXT PRIMARY KEY,
    backend           TEXT NOT NULL DEFAULT '',
    reasoning_effort  TEXT NOT NULL DEFAULT '',
    labels            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Auto-created on first sighting with empty fields. The platform engineer
fills in `backend` (and optionally `reasoning_effort`, `labels`) via
`PUT /api/v1/stages/{id}`.

`labels` is intentionally free-form JSONB. The mapper encodes RL-specific
state there (last action timestamp, exploration flag, arm-pull counters,
etc.) without schema migrations. JSONB lets the mapper query directly:
`SELECT id FROM stages WHERE labels @> '{"exploring":true}'`.

### `backends`

```sql
CREATE TABLE backends (
    name                    TEXT PRIMARY KEY,
    endpoint                TEXT NOT NULL,
    model_id                TEXT NOT NULL,
    api_key_env_var         TEXT NOT NULL DEFAULT '',
    max_concurrency         INTEGER NOT NULL DEFAULT 1,
    input_cost_per_mtoken   DOUBLE PRECISION,
    output_cost_per_mtoken  DOUBLE PRECISION,
    quality                 DOUBLE PRECISION,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`*_cost_per_mtoken` and `quality` are platform-engineer-supplied priors.
orla does not act on them directly; the mapper does. They're persisted
so the mapper can read them as part of its state.

`max_concurrency` is the only operational cap orla enforces directly.

### `completion_records` (data plane)

The mapper's primary observation channel.

```sql
CREATE TABLE completion_records (
    completion_id     TEXT PRIMARY KEY,
    stage_id          TEXT NOT NULL,
    workflow_run      TEXT,
    backend           TEXT NOT NULL,
    status            TEXT NOT NULL,           -- "success" | "error"
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER,
    cost_usd          DOUBLE PRECISION,
    tags              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_completion_stage_time ON completion_records(stage_id, created_at DESC);
CREATE INDEX idx_completion_workflow ON completion_records(workflow_run) WHERE workflow_run IS NOT NULL;
```

One row per `/v1/chat/completions` call. Written async via `BatchWriter`.
`tags` carries the `X-Orla-Tag-*` map verbatim.

A GIN index on `tags` is not added in v1 — most mapper queries filter on
`stage_id` first, which the b-tree already covers. Add `CREATE INDEX
idx_completion_tags ON completion_records USING gin (tags)` later if
profiling shows tag-filtered queries are hot.

### `feedback` (data plane)

```sql
CREATE TABLE feedback (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    completion_id TEXT NOT NULL,
    stage_id      TEXT NOT NULL,
    workflow_run  TEXT,
    rating        DOUBLE PRECISION,           -- 0..1, nullable
    labels        JSONB NOT NULL DEFAULT '[]'::jsonb,
    notes         TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_feedback_completion ON feedback(completion_id);
CREATE INDEX idx_feedback_stage_time ON feedback(stage_id, created_at DESC);
```

Developer-submitted. Written async; the endpoint returns 202. Joined
against `completion_records` by `completion_id` for the mapper to
attribute feedback to backends.

## How the mapper consumes this

Three access patterns the schema is optimized for:

1. **Recent observations per stage.**

   ```sql
   SELECT * FROM completion_records
   WHERE stage_id = $1 AND created_at > $2
   ORDER BY created_at DESC LIMIT $3;
   ```

   Backed by `idx_completion_stage_time`.

2. **Feedback joined to completion.**

   ```sql
   SELECT f.rating, c.backend, c.cost_usd, c.latency_ms
   FROM feedback f
   JOIN completion_records c USING (completion_id)
   WHERE c.stage_id = $1 AND f.created_at > $2;
   ```

3. **Aggregates by (stage, backend).**

   ```sql
   SELECT backend,
          COUNT(*),
          AVG(latency_ms),
          SUM(cost_usd),
          PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY latency_ms) AS p50,
          PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95
   FROM completion_records
   WHERE stage_id = $1 AND created_at > $2
   GROUP BY backend;
   ```

   This is what the `/metrics` endpoint surfaces. Native Postgres
   percentile aggregates replace the SQL gymnastics SQLite would need.

## Read-only access for the mapper

A second Postgres role is provisioned with `SELECT` only on the four
tables, intended for the RL mapper to query directly when the REST
endpoints aren't expressive enough:

```sql
CREATE ROLE orla_reader LOGIN PASSWORD '...';
GRANT CONNECT ON DATABASE orla TO orla_reader;
GRANT USAGE ON SCHEMA public TO orla_reader;
GRANT SELECT ON stages, backends, completion_records, feedback TO orla_reader;
```

Mappers that want raw SQL access use this role. The REST API stays
authoritative for the common patterns; direct SQL is the escape hatch
for heavy analytical queries that don't deserve a new endpoint.

## Deployment

- **Local dev (macOS)**:

  ```
  brew install postgresql@17
  brew services start postgresql@17
  createdb orla
  ```

  Default `ORLA_DATABASE_URL=postgres://$(whoami)@localhost:5432/orla?sslmode=disable`.
- **Local dev (Linux)**: equivalent via the system package manager, then
  `createdb orla`.
- **Production**: any Postgres 14+ deployment. Tested against managed
  Postgres (RDS, Cloud SQL, Neon). orla does not require any extensions
  beyond the default contrib set.

## Multiple orla instances

The schema is designed for HA from day one — multiple orla processes
can point at the same database:

- Synchronous control-plane writes use single-row upserts; concurrent
  writers are serialized by row locks naturally.
- BatchWriter batches are append-only inserts; conflicting completion
  IDs (vanishingly unlikely with UUIDs) become a single `ON CONFLICT DO
  NOTHING`.
- Scheduler queues live in-process per instance. A request hits
  whichever instance the load balancer routes to and is dispatched from
  that instance's worker pool. Per-backend `max_concurrency` is therefore
  enforced *per instance*, not globally. v1 documents this; a future
  phase can add a Postgres-advisory-lock-based global cap.

## Retention

v1: no automatic pruning. The mapper deletes what it doesn't need (or
runs a periodic vacuum job). orla does not opinionatedly delete
observation data because the mapper may want long history.

If `completion_records` growth becomes a concern, Postgres native
partitioning by `created_at` (monthly) is the planned mitigation. Adding
partitioning later is a non-breaking schema change.

## What's NOT in storage (v1)

- No workflow registry. The DAG / label propagation feature exists in
  Orla; not load-bearing for the RL story.
- No access control policies. Single-tenant.
- No memory-manager / KV-cache state.
- No multi-tenant fairness state (tenant credit, run-in-flight counters).
  Single-tenant scheduler is FCFS per backend.
