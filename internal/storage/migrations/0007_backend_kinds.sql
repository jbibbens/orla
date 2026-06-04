-- +goose Up
-- Backends gain a `kind` discriminator. v1 has two kinds:
--   'llm'   the existing OpenAI-compatible chat completion backends
--   'tool'  scientific computation tools (structure prediction, etc.)
-- Default 'llm' so existing rows keep working unchanged.
ALTER TABLE backends
    ADD COLUMN kind TEXT NOT NULL DEFAULT 'llm'
        CHECK (kind IN ('llm', 'tool'));

-- For tool backends, `tool_kind` distinguishes families of tools
-- (e.g., 'structure-prediction', 'docking', 'admet-prediction').
-- NULL for LLM backends.
ALTER TABLE backends
    ADD COLUMN tool_kind TEXT;

-- Tool cost is denominated in GPU-seconds, not tokens. NULL for LLM
-- backends; LLM cost continues to come from input_/output_cost_per_mtoken.
ALTER TABLE backends
    ADD COLUMN cost_per_gpu_second DOUBLE PRECISION;

-- `model_id` is meaningful only for LLM backends. Tool backends don't
-- have a model id in the OpenAI sense; their identity is (name, endpoint).
ALTER TABLE backends
    ALTER COLUMN model_id DROP NOT NULL;

-- +goose Down
ALTER TABLE backends
    ALTER COLUMN model_id SET NOT NULL;
ALTER TABLE backends
    DROP COLUMN cost_per_gpu_second,
    DROP COLUMN tool_kind,
    DROP COLUMN kind;
