package serving

import (
	"testing"
	"time"

	"github.com/dorcha-inc/orla/internal/model"
	"github.com/stretchr/testify/assert"
)

func TestSelectNextStageKey_FCFSPrefersOldestAcrossStages(t *testing.T) {
	now := time.Now()
	stageQueues := map[string][]*scheduledRequest{
		"stage-new": {
			{
				stageName:  "stage-new",
				enqueuedAt: now.Add(-1 * time.Second),
				opts:       model.InferenceOptions{SchedulingPolicy: model.SchedulingPolicyFCFS},
			},
		},
		"stage-old": {
			{
				stageName:  "stage-old",
				enqueuedAt: now.Add(-5 * time.Second),
				opts:       model.InferenceOptions{SchedulingPolicy: model.SchedulingPolicyFCFS},
			},
		},
	}

	selected := selectNextStageKey(stageQueues, model.SchedulingPolicyFCFS)
	assert.Equal(t, "stage-old", selected)
}

func TestSelectNextStageKey_PriorityPolicyPrefersHigherPriority(t *testing.T) {
	low := 1
	high := 9
	now := time.Now()
	stageQueues := map[string][]*scheduledRequest{
		"light": {
			{
				enqueuedAt: now.Add(-1 * time.Second),
				stageName:  "light",
				opts: model.InferenceOptions{
					SchedulingPolicy: model.SchedulingPolicyPriority,
					SchedulingHints:  &model.SchedulingHints{Priority: &low},
				},
			},
		},
		"heavy": {
			{
				enqueuedAt: now.Add(-1 * time.Second),
				stageName:  "heavy",
				opts: model.InferenceOptions{
					SchedulingPolicy: model.SchedulingPolicyPriority,
					SchedulingHints:  &model.SchedulingHints{Priority: &high},
				},
			},
		},
	}

	selected := selectNextStageKey(stageQueues, model.SchedulingPolicyPriority)
	assert.Equal(t, "heavy", selected)
}

func TestSelectNextStageKey_PriorityTieBreaksByOldestHead(t *testing.T) {
	now := time.Now()
	priority := 5
	stageQueues := map[string][]*scheduledRequest{
		"older": {
			{
				enqueuedAt: now.Add(-3 * time.Second),
				stageName:  "older",
				opts: model.InferenceOptions{
					SchedulingPolicy: model.SchedulingPolicyPriority,
					SchedulingHints:  &model.SchedulingHints{Priority: &priority},
				},
			},
		},
		"newer": {
			{
				enqueuedAt: now.Add(-1 * time.Second),
				stageName:  "newer",
				opts: model.InferenceOptions{
					SchedulingPolicy: model.SchedulingPolicyPriority,
					SchedulingHints:  &model.SchedulingHints{Priority: &priority},
				},
			},
		},
	}

	selected := selectNextStageKey(stageQueues, model.SchedulingPolicyPriority)
	assert.Equal(t, "older", selected)
}

// --- selectNextRequest tests ---

func TestSelectNextRequest_FIFODefault(t *testing.T) {
	now := time.Now()
	queue := []*scheduledRequest{
		{enqueuedAt: now.Add(-2 * time.Second), opts: model.InferenceOptions{}},
		{enqueuedAt: now.Add(-1 * time.Second), opts: model.InferenceOptions{}},
	}
	assert.Equal(t, 0, selectNextRequest(queue))
}

func TestSelectNextRequest_PriorityPicksHighest(t *testing.T) {
	now := time.Now()
	low := 1
	high := 9
	queue := []*scheduledRequest{
		{
			enqueuedAt: now.Add(-2 * time.Second),
			opts: model.InferenceOptions{
				RequestSchedulingPolicy: model.RequestSchedulingPolicyPriority,
				SchedulingHints:         &model.SchedulingHints{Priority: &low},
			},
		},
		{
			enqueuedAt: now.Add(-1 * time.Second),
			opts: model.InferenceOptions{
				RequestSchedulingPolicy: model.RequestSchedulingPolicyPriority,
				SchedulingHints:         &model.SchedulingHints{Priority: &high},
			},
		},
	}
	assert.Equal(t, 1, selectNextRequest(queue))
}

func TestSelectNextRequest_PriorityTieBreaksByOldest(t *testing.T) {
	now := time.Now()
	same := 5
	queue := []*scheduledRequest{
		{
			enqueuedAt: now.Add(-1 * time.Second),
			opts: model.InferenceOptions{
				RequestSchedulingPolicy: model.RequestSchedulingPolicyPriority,
				SchedulingHints:         &model.SchedulingHints{Priority: &same},
			},
		},
		{
			enqueuedAt: now.Add(-3 * time.Second),
			opts: model.InferenceOptions{
				RequestSchedulingPolicy: model.RequestSchedulingPolicyPriority,
				SchedulingHints:         &model.SchedulingHints{Priority: &same},
			},
		},
	}
	assert.Equal(t, 1, selectNextRequest(queue))
}

func TestSelectNextRequest_SingleElement(t *testing.T) {
	queue := []*scheduledRequest{{opts: model.InferenceOptions{}}}
	assert.Equal(t, 0, selectNextRequest(queue))
}
