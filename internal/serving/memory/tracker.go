package memory

import (
	"sync"
	"time"
)

// StageStatus tracks the lifecycle of a single stage within the tracker.
type StageStatus string

const (
	StageStatusActive    StageStatus = "active"
	StageStatusCompleted StageStatus = "completed"
)

// StageState records per-stage metadata needed for cache decisions.
type StageState struct {
	StageID   string
	AgentName string
	Backend   string
	Model     string
	Tokens    int
	Status    StageStatus
	StartedAt time.Time
}

// BackendCacheEntry tracks a workflow's cache footprint on a specific backend.
type BackendCacheEntry struct {
	Backend    string
	Model      string
	Tokens     int
	Preserved  bool
	LastUpdate time.Time
}

// WorkflowState is the Memory Manager's view of a single active workflow.
type WorkflowState struct {
	ID              string
	ActiveStages    map[string]*StageState       // stageID -> state
	CompletedStages map[string]*StageState       // stageID -> state
	BackendUsage    map[string]*BackendCacheEntry // backend name -> cache entry
	StartedAt       time.Time
	LastActivityAt  time.Time
}

// InflightRequest describes a request currently being processed by a backend worker.
type InflightRequest struct {
	RequestID  string
	WorkflowID string
	StageID    string
	Backend    string
	Streaming  bool
	StartedAt  time.Time
}

// Tracker maintains the state of all active workflows and in-flight requests.
// It is the Memory Manager's source of truth for making cache decisions.
type Tracker struct {
	mu        sync.Mutex
	workflows map[string]*WorkflowState            // workflowID -> state
	inflight  map[string]map[string]*InflightRequest // backend -> requestID -> request
}

// NewTracker creates a new workflow state tracker.
func NewTracker() *Tracker {
	return &Tracker{
		workflows: make(map[string]*WorkflowState),
		inflight:  make(map[string]map[string]*InflightRequest),
	}
}

// RegisterWorkflow initializes tracking for a new workflow execution.
// Idempotent: calling with an already-registered workflowID is a no-op.
func (t *Tracker) RegisterWorkflow(workflowID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if _, exists := t.workflows[workflowID]; exists {
		return
	}
	now := time.Now()
	t.workflows[workflowID] = &WorkflowState{
		ID:              workflowID,
		ActiveStages:    make(map[string]*StageState),
		CompletedStages: make(map[string]*StageState),
		BackendUsage:    make(map[string]*BackendCacheEntry),
		StartedAt:       now,
		LastActivityAt:  now,
	}
}

// DeregisterWorkflow removes a workflow from tracking.
func (t *Tracker) DeregisterWorkflow(workflowID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.workflows, workflowID)
}

// GetWorkflow returns a snapshot of a workflow's state, or nil if not found.
func (t *Tracker) GetWorkflow(workflowID string) *WorkflowState {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.workflows[workflowID]
}

// ActiveWorkflowIDs returns the IDs of all active workflows.
func (t *Tracker) ActiveWorkflowIDs() []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	ids := make([]string, 0, len(t.workflows))
	for id := range t.workflows {
		ids = append(ids, id)
	}
	return ids
}

// OnStageStart records that a stage has begun executing.
func (t *Tracker) OnStageStart(signal StageTransition) {
	t.mu.Lock()
	defer t.mu.Unlock()
	wf, ok := t.workflows[signal.WorkflowID]
	if !ok {
		return
	}
	now := time.Now()
	wf.ActiveStages[signal.StageID] = &StageState{
		StageID:   signal.StageID,
		AgentName: signal.AgentName,
		Backend:   signal.Backend,
		Model:     signal.Model,
		Tokens:    signal.ContextTokens,
		Status:    StageStatusActive,
		StartedAt: now,
	}
	wf.BackendUsage[signal.Backend] = &BackendCacheEntry{
		Backend:    signal.Backend,
		Model:      signal.Model,
		Tokens:     signal.ContextTokens,
		Preserved:  true,
		LastUpdate: now,
	}
	wf.LastActivityAt = now
}

// OnStageComplete records that a stage has finished executing.
func (t *Tracker) OnStageComplete(signal StageTransition) {
	t.mu.Lock()
	defer t.mu.Unlock()
	wf, ok := t.workflows[signal.WorkflowID]
	if !ok {
		return
	}
	now := time.Now()
	stage, exists := wf.ActiveStages[signal.StageID]
	if exists {
		stage.Status = StageStatusCompleted
		stage.Tokens = signal.ContextTokens
		wf.CompletedStages[signal.StageID] = stage
		delete(wf.ActiveStages, signal.StageID)
	}
	if entry, ok := wf.BackendUsage[signal.Backend]; ok {
		entry.Tokens = signal.ContextTokens
		entry.LastUpdate = now
	}
	wf.LastActivityAt = now
}

// MarkBackendFlushed marks a workflow's cache on a backend as no longer preserved.
func (t *Tracker) MarkBackendFlushed(workflowID, backend string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	wf, ok := t.workflows[workflowID]
	if !ok {
		return
	}
	if entry, ok := wf.BackendUsage[backend]; ok {
		entry.Preserved = false
	}
}

// LastStageOnBackend returns the most recently completed stage for a workflow
// on a given backend, or nil if none.
func (t *Tracker) LastStageOnBackend(workflowID, backend string) *StageState {
	t.mu.Lock()
	defer t.mu.Unlock()
	wf, ok := t.workflows[workflowID]
	if !ok {
		return nil
	}
	var latest *StageState
	for _, s := range wf.CompletedStages {
		if s.Backend == backend {
			if latest == nil || s.StartedAt.After(latest.StartedAt) {
				latest = s
			}
		}
	}
	return latest
}

// RecordInflight marks a request as in-flight on a backend.
func (t *Tracker) RecordInflight(req InflightRequest) {
	t.mu.Lock()
	defer t.mu.Unlock()
	byBackend, ok := t.inflight[req.Backend]
	if !ok {
		byBackend = make(map[string]*InflightRequest)
		t.inflight[req.Backend] = byBackend
	}
	byBackend[req.RequestID] = &req
}

// ClearInflight removes an in-flight request.
func (t *Tracker) ClearInflight(backend, requestID string) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if byBackend, ok := t.inflight[backend]; ok {
		delete(byBackend, requestID)
	}
}

// InflightOnBackend returns the number of in-flight requests on a backend.
func (t *Tracker) InflightOnBackend(backend string) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.inflight[backend])
}

// InflightWorkflowsOnBackend returns the set of workflow IDs with in-flight
// requests on a given backend.
func (t *Tracker) InflightWorkflowsOnBackend(backend string) map[string]struct{} {
	t.mu.Lock()
	defer t.mu.Unlock()
	wfIDs := make(map[string]struct{})
	for _, req := range t.inflight[backend] {
		wfIDs[req.WorkflowID] = struct{}{}
	}
	return wfIDs
}

// WorkflowsWithPreservedCacheOnBackend returns workflow IDs that have preserved
// cache entries on the given backend, excluding any currently in-flight.
func (t *Tracker) WorkflowsWithPreservedCacheOnBackend(backend string) []string {
	t.mu.Lock()
	defer t.mu.Unlock()
	inflightWFs := make(map[string]struct{})
	for _, req := range t.inflight[backend] {
		inflightWFs[req.WorkflowID] = struct{}{}
	}
	var ids []string
	for wfID, wf := range t.workflows {
		if _, busy := inflightWFs[wfID]; busy {
			continue
		}
		if entry, ok := wf.BackendUsage[backend]; ok && entry.Preserved {
			ids = append(ids, wfID)
		}
	}
	return ids
}
