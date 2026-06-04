package telemetry_test

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/telemetry"
)

func TestFeedbackWriter_RoundTrip(t *testing.T) {
	pool := freshPool(t)
	w := telemetry.NewFeedbackWriter(telemetry.FeedbackWriterConfig{
		Pool:      pool,
		BatchSize: 2,
		Interval:  50 * time.Millisecond,
	})
	t.Cleanup(func() { _ = w.Close(context.Background()) })

	rating := 0.8
	fb := &telemetry.Feedback{
		CompletionID: "chatcmpl-abc",
		StageID:      "planning",
		WorkflowRun:  "wf-1",
		Rating:       &rating,
		Labels:       []string{"accurate", "concise"},
		Notes:        "looks good",
		CreatedAt:    time.Now(),
	}
	require.True(t, w.Submit(fb))

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) && w.Flushes() == 0 {
		time.Sleep(20 * time.Millisecond)
	}
	require.GreaterOrEqual(t, w.Flushes(), int64(1))

	var (
		completionID string
		stageID      string
		workflowRun  *string
		ratingOut    *float64
		labels       []byte
		notes        string
	)
	row := pool.QueryRow(context.Background(),
		`SELECT completion_id, stage_id, workflow_run, rating, labels, notes
		 FROM feedback WHERE completion_id = $1`, "chatcmpl-abc")
	require.NoError(t, row.Scan(&completionID, &stageID, &workflowRun, &ratingOut, &labels, &notes))
	assert.Equal(t, "planning", stageID)
	require.NotNil(t, workflowRun)
	assert.Equal(t, "wf-1", *workflowRun)
	require.NotNil(t, ratingOut)
	assert.InDelta(t, 0.8, *ratingOut, 1e-9)
	assert.Contains(t, string(labels), "accurate")
	assert.Equal(t, "looks good", notes)
}

func TestFeedbackWriter_NullableRatingAndWorkflow(t *testing.T) {
	pool := freshPool(t)
	w := telemetry.NewFeedbackWriter(telemetry.FeedbackWriterConfig{
		Pool:      pool,
		BatchSize: 1,
		Interval:  10 * time.Millisecond,
	})
	t.Cleanup(func() { _ = w.Close(context.Background()) })

	require.True(t, w.Submit(&telemetry.Feedback{
		CompletionID: "chatcmpl-minimal",
		StageID:      "s",
		CreatedAt:    time.Now(),
	}))

	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) && w.Flushes() == 0 {
		time.Sleep(20 * time.Millisecond)
	}

	var (
		workflowRun *string
		rating      *float64
	)
	row := pool.QueryRow(context.Background(),
		`SELECT workflow_run, rating FROM feedback WHERE completion_id = $1`,
		"chatcmpl-minimal")
	require.NoError(t, row.Scan(&workflowRun, &rating))
	assert.Nil(t, workflowRun)
	assert.Nil(t, rating)
}

func TestFeedbackWriter_CloseDrains(t *testing.T) {
	pool := freshPool(t)
	w := telemetry.NewFeedbackWriter(telemetry.FeedbackWriterConfig{
		Pool:      pool,
		BatchSize: 100,
		Interval:  time.Minute,
	})

	for i := range 3 {
		require.True(t, w.Submit(&telemetry.Feedback{
			CompletionID: "id-" + string(rune('a'+i)),
			StageID:      "s",
			CreatedAt:    time.Now(),
		}))
	}
	require.NoError(t, w.Close(context.Background()))

	var n int
	row := pool.QueryRow(context.Background(), `SELECT COUNT(*) FROM feedback`)
	require.NoError(t, row.Scan(&n))
	assert.Equal(t, 3, n)
}
