-- +goose Up
-- Tool dispatches don't have token counts; they have GPU-seconds.
-- Both columns are nullable so a single completion_records table
-- holds LLM and tool dispatches without contortions.
ALTER TABLE completion_records
    ADD COLUMN gpu_seconds DOUBLE PRECISION,
    ADD COLUMN tool_kind TEXT;

-- +goose Down
ALTER TABLE completion_records
    DROP COLUMN tool_kind,
    DROP COLUMN gpu_seconds;
