-- +goose Up
CREATE TABLE completion_records (
    completion_id     TEXT PRIMARY KEY,
    stage_id          TEXT NOT NULL,
    workflow_run      TEXT,
    backend           TEXT NOT NULL,
    status            TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    latency_ms        INTEGER,
    cost_usd          DOUBLE PRECISION,
    tags              JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_completion_stage_time
    ON completion_records(stage_id, created_at DESC);
CREATE INDEX idx_completion_workflow
    ON completion_records(workflow_run) WHERE workflow_run IS NOT NULL;

-- +goose Down
DROP INDEX IF EXISTS idx_completion_workflow;
DROP INDEX IF EXISTS idx_completion_stage_time;
DROP TABLE completion_records;
