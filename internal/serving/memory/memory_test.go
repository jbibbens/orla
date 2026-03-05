package memory

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Tracker tests ---

func TestTracker_RegisterDeregister(t *testing.T) {
	tr := NewTracker()
	tr.RegisterWorkflow("wf1")
	assert.Contains(t, tr.ActiveWorkflowIDs(), "wf1")

	tr.DeregisterWorkflow("wf1")
	assert.NotContains(t, tr.ActiveWorkflowIDs(), "wf1")
}

func TestTracker_StageLifecycle(t *testing.T) {
	tr := NewTracker()
	tr.RegisterWorkflow("wf1")

	tr.OnStageStart(StageTransition{
		WorkflowID: "wf1", StageID: "s1", AgentName: "a1",
		Backend: "b1", Model: "m1", ContextTokens: 100,
		TransitionType: TransitionStageStart,
	})

	wf := tr.GetWorkflow("wf1")
	require.NotNil(t, wf)
	assert.Contains(t, wf.ActiveStages, "s1")
	assert.NotContains(t, wf.CompletedStages, "s1")

	tr.OnStageComplete(StageTransition{
		WorkflowID: "wf1", StageID: "s1", AgentName: "a1",
		Backend: "b1", Model: "m1", ContextTokens: 150,
		TransitionType: TransitionStageComplete,
	})

	wf = tr.GetWorkflow("wf1")
	assert.NotContains(t, wf.ActiveStages, "s1")
	assert.Contains(t, wf.CompletedStages, "s1")
	assert.Equal(t, 150, wf.CompletedStages["s1"].Tokens)
}

func TestTracker_Inflight(t *testing.T) {
	tr := NewTracker()
	tr.RecordInflight(InflightRequest{
		RequestID: "r1", WorkflowID: "wf1", StageID: "s1", Backend: "b1",
	})
	assert.Equal(t, 1, tr.InflightOnBackend("b1"))

	wfs := tr.InflightWorkflowsOnBackend("b1")
	assert.Contains(t, wfs, "wf1")

	tr.ClearInflight("b1", "r1")
	assert.Equal(t, 0, tr.InflightOnBackend("b1"))
}

func TestTracker_PreservedCache(t *testing.T) {
	tr := NewTracker()
	tr.RegisterWorkflow("wf1")
	tr.RegisterWorkflow("wf2")

	tr.OnStageStart(StageTransition{
		WorkflowID: "wf1", StageID: "s1", Backend: "b1", Model: "m1",
		TransitionType: TransitionStageStart, ContextTokens: 100,
	})
	tr.OnStageStart(StageTransition{
		WorkflowID: "wf2", StageID: "s2", Backend: "b1", Model: "m1",
		TransitionType: TransitionStageStart, ContextTokens: 200,
	})

	preserved := tr.WorkflowsWithPreservedCacheOnBackend("b1")
	assert.Len(t, preserved, 2)

	tr.MarkBackendFlushed("wf1", "b1")
	preserved = tr.WorkflowsWithPreservedCacheOnBackend("b1")
	assert.Len(t, preserved, 1)
	assert.Contains(t, preserved, "wf2")
}

// --- Policy tests ---

func TestPreserveOnSmallIncrement_SameBackendSmallDelta(t *testing.T) {
	p := NewPreserveOnSmallIncrementPolicy(256)
	action := p.Evaluate(StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID:     "wf1",
		Backend:        "b1", Model: "m1",
		PrevBackend: "b1", PrevModel: "m1",
		DeltaTokens: 50,
	}, nil)
	assert.Equal(t, CacheActionPreserve, action.Type)
}

func TestPreserveOnSmallIncrement_SameBackendLargeDelta(t *testing.T) {
	p := NewPreserveOnSmallIncrementPolicy(256)
	action := p.Evaluate(StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID:     "wf1",
		Backend:        "b1", Model: "m1",
		PrevBackend: "b1", PrevModel: "m1",
		DeltaTokens: 500,
	}, nil)
	assert.Equal(t, CacheActionNoop, action.Type)
}

func TestPreserveOnSmallIncrement_DifferentBackend(t *testing.T) {
	p := NewPreserveOnSmallIncrementPolicy(256)
	action := p.Evaluate(StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID:     "wf1",
		Backend:        "b2", Model: "m1",
		PrevBackend: "b1", PrevModel: "m1",
		DeltaTokens: 10,
	}, nil)
	assert.Equal(t, CacheActionNoop, action.Type)
}

func TestPreserveOnSmallIncrement_StageOverrideThreshold(t *testing.T) {
	p := NewPreserveOnSmallIncrementPolicy(100)
	threshold := 500
	action := p.Evaluate(StageTransition{
		TransitionType:          TransitionStageStart,
		WorkflowID:              "wf1",
		Backend:                 "b1", Model: "m1",
		PrevBackend:             "b1", PrevModel: "m1",
		DeltaTokens:             300,
		PreserveThresholdTokens: &threshold,
	}, nil)
	assert.Equal(t, CacheActionPreserve, action.Type)
}

func TestFlushAtBoundary_WorkflowComplete(t *testing.T) {
	p := NewFlushAtBoundaryPolicy()
	action := p.Evaluate(StageTransition{
		TransitionType: TransitionWorkflowComplete,
		WorkflowID:     "wf1",
		Backend:        "b1",
	}, nil)
	assert.Equal(t, CacheActionFlush, action.Type)
}

func TestFlushAtBoundary_BackendSwitch(t *testing.T) {
	p := NewFlushAtBoundaryPolicy()
	action := p.Evaluate(StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID:     "wf1",
		Backend:        "b2", Model: "m1",
		PrevBackend: "b1", PrevModel: "m1",
	}, nil)
	assert.Equal(t, CacheActionFlush, action.Type)
	assert.Equal(t, "b1", action.Backend, "should flush the OLD backend")
}

func TestFlushAtBoundary_SameBackendNoop(t *testing.T) {
	p := NewFlushAtBoundaryPolicy()
	action := p.Evaluate(StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID:     "wf1",
		Backend:        "b1", Model: "m1",
		PrevBackend: "b1", PrevModel: "m1",
	}, nil)
	assert.Equal(t, CacheActionNoop, action.Type)
}

func TestFlushUnderPressure_BelowThreshold(t *testing.T) {
	p := NewFlushUnderPressurePolicy(0.85)
	tr := NewTracker()
	tr.RegisterWorkflow("wf1")
	tr.OnStageStart(StageTransition{
		WorkflowID: "wf1", StageID: "s1", Backend: "b1", Model: "m1",
		TransitionType: TransitionStageStart, ContextTokens: 100,
	})

	actions := p.EvaluatePressure("b1", 0.50, tr)
	assert.Empty(t, actions)
}

func TestFlushUnderPressure_AboveThreshold(t *testing.T) {
	p := NewFlushUnderPressurePolicy(0.85)
	tr := NewTracker()
	tr.RegisterWorkflow("wf1")
	tr.OnStageStart(StageTransition{
		WorkflowID: "wf1", StageID: "s1", Backend: "b1", Model: "m1",
		TransitionType: TransitionStageStart, ContextTokens: 100,
	})
	tr.OnStageComplete(StageTransition{
		WorkflowID: "wf1", StageID: "s1", Backend: "b1", Model: "m1",
		TransitionType: TransitionStageComplete, ContextTokens: 100,
	})

	actions := p.EvaluatePressure("b1", 0.90, tr)
	require.Len(t, actions, 1)
	assert.Equal(t, CacheActionFlush, actions[0].Type)
	assert.Equal(t, "wf1", actions[0].WorkflowID)
}

func TestPolicyChain_FirstNonNoopWins(t *testing.T) {
	chain := NewPolicyChain(
		NewPreserveOnSmallIncrementPolicy(256),
		NewFlushAtBoundaryPolicy(),
	)
	action := chain.Evaluate(StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID:     "wf1",
		Backend:        "b1", Model: "m1",
		PrevBackend: "b1", PrevModel: "m1",
		DeltaTokens: 50,
	}, nil)
	assert.Equal(t, CacheActionPreserve, action.Type, "preserve should win over flush since same backend")
}

// --- Manager tests ---

func TestDefaultManager_StageOverridePreserve(t *testing.T) {
	mm := NewDefaultManager(DefaultManagerConfig{})
	mm.RegisterWorkflow("wf1")

	action := mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID:     "wf1", StageID: "s1",
		Backend: "b1", Model: "m1",
		CachePolicy: "preserve",
	})
	assert.Equal(t, CacheActionPreserve, action.Type)
	assert.Contains(t, action.Reason, "stage-level override")
}

func TestDefaultManager_StageOverrideFlush(t *testing.T) {
	mm := NewDefaultManager(DefaultManagerConfig{})
	mm.RegisterWorkflow("wf1")

	action := mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionStageComplete,
		WorkflowID:     "wf1", StageID: "s1",
		Backend: "b1", Model: "m1",
		CachePolicy: "flush",
	})
	assert.Equal(t, CacheActionFlush, action.Type)
	assert.Contains(t, action.Reason, "stage-level override")
}

func TestDefaultManager_PolicyChainDecides(t *testing.T) {
	mm := NewDefaultManager(DefaultManagerConfig{PreserveThreshold: 256})
	mm.RegisterWorkflow("wf1")

	mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID: "wf1", StageID: "s1",
		Backend: "b1", Model: "m1",
	})

	action := mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID:     "wf1", StageID: "s2",
		Backend: "b1", Model: "m1",
		PrevBackend: "b1", PrevModel: "m1",
		DeltaTokens: 100,
	})
	assert.Equal(t, CacheActionPreserve, action.Type)
}

func TestDefaultManager_WorkflowCompleteFlushes(t *testing.T) {
	mm := NewDefaultManager(DefaultManagerConfig{})
	mm.RegisterWorkflow("wf1")

	action := mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionWorkflowComplete,
		WorkflowID:     "wf1",
		Backend:        "b1",
	})
	assert.Equal(t, CacheActionFlush, action.Type)
}

func TestDefaultManager_MemoryPressure(t *testing.T) {
	mm := NewDefaultManager(DefaultManagerConfig{PressureThreshold: 0.80})
	mm.RegisterWorkflow("wf1")

	mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionStageStart,
		WorkflowID: "wf1", StageID: "s1", Backend: "b1", Model: "m1",
		ContextTokens: 100,
	})
	mm.OnTransition(context.Background(), StageTransition{
		TransitionType: TransitionStageComplete,
		WorkflowID: "wf1", StageID: "s1", Backend: "b1", Model: "m1",
		ContextTokens: 100,
	})

	actions := mm.OnMemoryPressure(context.Background(), "b1", 0.90)
	require.Len(t, actions, 1)
	assert.Equal(t, CacheActionFlush, actions[0].Type)
	assert.Equal(t, "wf1", actions[0].WorkflowID)
}

func TestDefaultManager_InflightTracking(t *testing.T) {
	mm := NewDefaultManager(DefaultManagerConfig{})
	mm.RegisterWorkflow("wf1")

	mm.RecordInflight(InflightRequest{
		RequestID: "r1", WorkflowID: "wf1", StageID: "s1", Backend: "b1",
	})
	assert.Equal(t, 1, mm.tracker.InflightOnBackend("b1"))

	mm.ClearInflight("b1", "r1")
	assert.Equal(t, 0, mm.tracker.InflightOnBackend("b1"))
}
