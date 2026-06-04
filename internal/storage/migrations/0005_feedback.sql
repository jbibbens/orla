-- +goose Up
CREATE TABLE feedback (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    completion_id TEXT NOT NULL,
    stage_id      TEXT NOT NULL,
    workflow_run  TEXT,
    rating        DOUBLE PRECISION,
    labels        JSONB NOT NULL DEFAULT '[]'::jsonb,
    notes         TEXT NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_feedback_completion ON feedback(completion_id);
CREATE INDEX idx_feedback_stage_time ON feedback(stage_id, created_at DESC);

-- +goose Down
DROP INDEX IF EXISTS idx_feedback_stage_time;
DROP INDEX IF EXISTS idx_feedback_completion;
DROP TABLE feedback;
