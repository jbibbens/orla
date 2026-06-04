-- The SELECT column order matches the canonical backends table order.
-- Keeping that order lets sqlc reuse the db.Backend row type instead of
-- emitting a per-query row type.

-- name: GetBackend :one
SELECT name, endpoint, model_id, api_key_env_var, max_concurrency,
       input_cost_per_mtoken, output_cost_per_mtoken, quality,
       created_at, updated_at, rate_per_second,
       kind, tool_kind, cost_per_gpu_second
FROM backends
WHERE name = $1;

-- name: ListBackends :many
SELECT name, endpoint, model_id, api_key_env_var, max_concurrency,
       input_cost_per_mtoken, output_cost_per_mtoken, quality,
       created_at, updated_at, rate_per_second,
       kind, tool_kind, cost_per_gpu_second
FROM backends
ORDER BY name;

-- name: InsertBackend :one
INSERT INTO backends (
    name, endpoint, model_id, api_key_env_var, max_concurrency,
    input_cost_per_mtoken, output_cost_per_mtoken, quality, rate_per_second,
    kind, tool_kind, cost_per_gpu_second
)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
RETURNING name, endpoint, model_id, api_key_env_var, max_concurrency,
          input_cost_per_mtoken, output_cost_per_mtoken, quality,
          created_at, updated_at, rate_per_second,
          kind, tool_kind, cost_per_gpu_second;

-- name: UpdateBackend :one
UPDATE backends
SET endpoint = $2,
    model_id = $3,
    api_key_env_var = $4,
    max_concurrency = $5,
    input_cost_per_mtoken = $6,
    output_cost_per_mtoken = $7,
    quality = $8,
    rate_per_second = $9,
    kind = $10,
    tool_kind = $11,
    cost_per_gpu_second = $12,
    updated_at = now()
WHERE name = $1
RETURNING name, endpoint, model_id, api_key_env_var, max_concurrency,
          input_cost_per_mtoken, output_cost_per_mtoken, quality,
          created_at, updated_at, rate_per_second,
          kind, tool_kind, cost_per_gpu_second;

-- name: DeleteBackend :execrows
DELETE FROM backends WHERE name = $1;
