-- +goose Up
CREATE TABLE backends (
    name                    TEXT PRIMARY KEY,
    endpoint                TEXT NOT NULL,
    model_id                TEXT NOT NULL,
    api_key_env_var         TEXT NOT NULL DEFAULT '',
    max_concurrency         INTEGER NOT NULL DEFAULT 1 CHECK (max_concurrency >= 1),
    input_cost_per_mtoken   DOUBLE PRECISION,
    output_cost_per_mtoken  DOUBLE PRECISION,
    quality                 DOUBLE PRECISION,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- +goose Down
DROP TABLE backends;
