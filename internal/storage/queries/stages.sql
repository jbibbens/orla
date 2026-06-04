-- name: GetStage :one
SELECT id, backend, reasoning_effort, labels, created_at, updated_at
FROM stages
WHERE id = $1;

-- name: ListStages :many
SELECT id, backend, reasoning_effort, labels, created_at, updated_at
FROM stages
ORDER BY id;

-- name: UpsertStageDefault :one
-- Auto-create a stage with empty fields on first sighting. If the row
-- already exists, the no-op SET keeps the existing values and RETURNING
-- returns them. (A bare ON CONFLICT DO NOTHING with RETURNING returns
-- no row when the conflict fires, which loses the "fetch existing"
-- behavior we want here.)
INSERT INTO stages (id, backend, reasoning_effort, labels)
VALUES ($1, '', '', '{}'::jsonb)
ON CONFLICT (id) DO UPDATE
SET id = stages.id
RETURNING id, backend, reasoning_effort, labels, created_at, updated_at;

-- name: ReplaceStage :one
INSERT INTO stages (id, backend, reasoning_effort, labels, updated_at)
VALUES ($1, $2, $3, $4, now())
ON CONFLICT (id) DO UPDATE
SET backend = EXCLUDED.backend,
    reasoning_effort = EXCLUDED.reasoning_effort,
    labels = EXCLUDED.labels,
    updated_at = now()
RETURNING id, backend, reasoning_effort, labels, created_at, updated_at;

-- name: DeleteStage :execrows
DELETE FROM stages WHERE id = $1;
