-- name: ListStageFeedback :many
SELECT id, completion_id, stage_id, workflow_run, rating,
       labels, notes, created_at
FROM feedback
WHERE stage_id = $1
  AND created_at > COALESCE(@since::timestamptz, '-infinity'::timestamptz)
ORDER BY created_at DESC
LIMIT @limit_count;
