package memory

import (
	"context"
	"sync"
	"time"

	"go.uber.org/zap"
)

// CacheController is an optional interface that LLM providers can implement
// to support explicit cache management. Providers that don't support it
// simply don't implement the interface, and the Memory Manager falls back
// to soft-flush (mark as stale and let LRU reclaim).
type CacheController interface {
	FlushPrefix(ctx context.Context, sessionID string) error
	MemoryUsage(ctx context.Context) (*MemoryStats, error)
}

// MemoryStats reports a backend's memory utilization.
type MemoryStats struct {
	UsedBytes  int64
	TotalBytes int64
	Pressure   float64 // 0.0 to 1.0
}

// Manager is the Memory Manager interface. It receives stage transition
// signals from the Workflow Executor and returns cache management actions.
type Manager interface {
	OnTransition(ctx context.Context, signal StageTransition) CacheAction
	OnMemoryPressure(ctx context.Context, backend string, pressure float64) []CacheAction
	RegisterWorkflow(workflowID string)
	DeregisterWorkflow(workflowID string)
	RecordInflight(req InflightRequest)
	ClearInflight(backend, requestID string)
}

// DefaultManagerConfig configures the DefaultManager.
type DefaultManagerConfig struct {
	PreserveThreshold int
	PressureThreshold float64
	PressurePollMs    int
}

// DefaultManager is the default Memory Manager implementation. It composes
// the three paper policies (preserve on small increment, flush at boundary,
// flush under pressure) with workflow state tracking and in-flight awareness.
type DefaultManager struct {
	tracker          *Tracker
	chain            *PolicyChain
	pressurePolicy   *FlushUnderPressurePolicy
	cacheControllers map[string]CacheController
	mu               sync.RWMutex
}

// NewDefaultManager creates a new DefaultManager with the standard policy chain.
func NewDefaultManager(cfg DefaultManagerConfig) *DefaultManager {
	if cfg.PreserveThreshold <= 0 {
		cfg.PreserveThreshold = defaultPreserveThresholdTokens
	}
	if cfg.PressureThreshold <= 0 {
		cfg.PressureThreshold = 0.85
	}
	pressurePolicy := NewFlushUnderPressurePolicy(cfg.PressureThreshold)
	chain := NewPolicyChain(
		NewPreserveOnSmallIncrementPolicy(cfg.PreserveThreshold),
		NewFlushAtBoundaryPolicy(),
	)
	return &DefaultManager{
		tracker:          NewTracker(),
		chain:            chain,
		pressurePolicy:   pressurePolicy,
		cacheControllers: make(map[string]CacheController),
	}
}

// RegisterCacheController associates a CacheController with a backend name.
func (m *DefaultManager) RegisterCacheController(backend string, cc CacheController) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.cacheControllers[backend] = cc
}

func (m *DefaultManager) getCacheController(backend string) CacheController {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.cacheControllers[backend]
}

// RegisterWorkflow initializes tracking for a new workflow execution.
func (m *DefaultManager) RegisterWorkflow(workflowID string) {
	m.tracker.RegisterWorkflow(workflowID)
	zap.L().Debug("Memory manager: registered workflow", zap.String("workflow_id", workflowID))
}

// DeregisterWorkflow removes a workflow from tracking.
func (m *DefaultManager) DeregisterWorkflow(workflowID string) {
	m.tracker.DeregisterWorkflow(workflowID)
	zap.L().Debug("Memory manager: deregistered workflow", zap.String("workflow_id", workflowID))
}

// RecordInflight marks a request as in-flight.
func (m *DefaultManager) RecordInflight(req InflightRequest) {
	m.tracker.RecordInflight(req)
}

// ClearInflight removes an in-flight request.
func (m *DefaultManager) ClearInflight(backend, requestID string) {
	m.tracker.ClearInflight(backend, requestID)
}

// OnTransition processes a stage lifecycle signal and returns a cache action.
//
// Resolution order:
//  1. Stage-level explicit CachePolicy override ("preserve" or "flush")
//  2. Policy chain (preserve on small increment -> flush at boundary)
//  3. Noop (let the backend's default LRU handle it)
func (m *DefaultManager) OnTransition(ctx context.Context, signal StageTransition) CacheAction {
	// Update tracker state.
	switch signal.TransitionType {
	case TransitionStageStart:
		m.tracker.OnStageStart(signal)
	case TransitionStageComplete, TransitionAgentComplete:
		m.tracker.OnStageComplete(signal)
	case TransitionWorkflowComplete:
		m.tracker.OnStageComplete(signal)
	}

	// Stage-level explicit override takes precedence.
	if signal.CachePolicy == "preserve" {
		action := CacheAction{
			Type:       CacheActionPreserve,
			WorkflowID: signal.WorkflowID,
			Backend:    signal.Backend,
			Reason:     "stage-level override: preserve",
		}
		m.logAction(signal, action)
		return action
	}
	if signal.CachePolicy == "flush" {
		action := CacheAction{
			Type:       CacheActionFlush,
			WorkflowID: signal.WorkflowID,
			Backend:    signal.Backend,
			Reason:     "stage-level override: flush",
		}
		m.executeFlush(ctx, action)
		m.logAction(signal, action)
		return action
	}

	// Evaluate the policy chain.
	action := m.chain.Evaluate(signal, m.tracker)
	if action.Type == CacheActionFlush {
		m.executeFlush(ctx, action)
	}
	if action.Type == CacheActionPreserve || action.Type == CacheActionFlush {
		m.logAction(signal, action)
	}
	return action
}

// OnMemoryPressure evaluates whether to flush idle workflow caches under pressure.
func (m *DefaultManager) OnMemoryPressure(_ context.Context, backend string, pressure float64) []CacheAction {
	actions := m.pressurePolicy.EvaluatePressure(backend, pressure, m.tracker)
	for _, action := range actions {
		m.tracker.MarkBackendFlushed(action.WorkflowID, action.Backend)
		zap.L().Info("Memory manager: pressure flush",
			zap.String("workflow_id", action.WorkflowID),
			zap.String("backend", action.Backend),
			zap.Float64("pressure", pressure))
	}
	return actions
}

// executeFlush performs a soft flush: marks the cache as stale in the tracker
// and, if available, calls the backend's CacheController.
func (m *DefaultManager) executeFlush(ctx context.Context, action CacheAction) {
	m.tracker.MarkBackendFlushed(action.WorkflowID, action.Backend)
	cc := m.getCacheController(action.Backend)
	if cc == nil {
		return
	}
	// Only attempt a hard flush if no other workflow is in-flight on this backend.
	inflightWFs := m.tracker.InflightWorkflowsOnBackend(action.Backend)
	delete(inflightWFs, action.WorkflowID)
	if len(inflightWFs) > 0 {
		zap.L().Debug("Memory manager: skipping hard flush, other workflows in-flight",
			zap.String("backend", action.Backend),
			zap.Int("other_inflight", len(inflightWFs)))
		return
	}
	if err := cc.FlushPrefix(ctx, action.WorkflowID); err != nil {
		zap.L().Warn("Memory manager: flush failed",
			zap.String("backend", action.Backend),
			zap.Error(err))
	}
}

func (m *DefaultManager) logAction(signal StageTransition, action CacheAction) {
	zap.L().Debug("Memory manager: cache action",
		zap.String("workflow_id", signal.WorkflowID),
		zap.String("stage_id", signal.StageID),
		zap.String("transition", string(signal.TransitionType)),
		zap.String("action", string(action.Type)),
		zap.String("reason", action.Reason))
}

// StartPressureMonitor launches a loop that periodically queries backends
// for memory pressure and triggers flush actions. It blocks until ctx is cancelled.
// backendsFn is called on each tick to discover the current set of backends,
// so dynamically registered backends are picked up automatically.
func (m *DefaultManager) StartPressureMonitor(ctx context.Context, backendsFn func() []string, interval time.Duration) {
	if interval <= 0 {
		zap.L().Warn("Memory manager: pressure monitor interval is less than or equal to 0, setting to 2 seconds")
		interval = 2 * time.Second
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			m.pollPressure(ctx, backendsFn())
		}
	}
}

func (m *DefaultManager) pollPressure(ctx context.Context, backends []string) {
	for _, backend := range backends {
		cc := m.getCacheController(backend)
		if cc == nil {
			continue
		}
		stats, err := cc.MemoryUsage(ctx)
		if err != nil {
			zap.L().Debug("Memory manager: failed to query memory usage",
				zap.String("backend", backend),
				zap.Error(err))
			continue
		}
		if stats.Pressure >= m.pressurePolicy.PressureThreshold {
			m.OnMemoryPressure(ctx, backend, stats.Pressure)
		}
	}
}
