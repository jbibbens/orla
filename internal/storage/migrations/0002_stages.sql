-- +goose Up
CREATE TABLE stages (
    id                TEXT PRIMARY KEY,
    backend           TEXT NOT NULL DEFAULT '',
    reasoning_effort  TEXT NOT NULL DEFAULT '',
    labels            JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- +goose Down
DROP TABLE stages;
