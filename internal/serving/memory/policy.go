package memory

const defaultPreserveThresholdTokens = 256

// Policy evaluates a stage transition and returns a cache action.
// Policies are composable: the first policy in a chain that returns a
// non-Noop action wins.
type Policy interface {
	Evaluate(signal StageTransition, tracker *Tracker) CacheAction
}

// PreserveOnSmallIncrementPolicy preserves KV cache when a new stage on the
// same backend/model adds only a small number of tokens to the shared context.
type PreserveOnSmallIncrementPolicy struct {
	ThresholdTokens int
}

func NewPreserveOnSmallIncrementPolicy(thresholdTokens int) *PreserveOnSmallIncrementPolicy {
	if thresholdTokens <= 0 {
		thresholdTokens = defaultPreserveThresholdTokens
	}
	return &PreserveOnSmallIncrementPolicy{ThresholdTokens: thresholdTokens}
}

func (p *PreserveOnSmallIncrementPolicy) Evaluate(signal StageTransition, _ *Tracker) CacheAction {
	if signal.TransitionType != TransitionStageStart {
		return CacheAction{Type: CacheActionNoop}
	}
	sameBackendModel := signal.PrevBackend == signal.Backend && signal.PrevModel == signal.Model
	if !sameBackendModel || signal.PrevBackend == "" {
		return CacheAction{Type: CacheActionNoop}
	}
	threshold := p.ThresholdTokens
	if signal.PreserveThresholdTokens != nil {
		threshold = *signal.PreserveThresholdTokens
	}
	if signal.DeltaTokens <= threshold {
		return CacheAction{
			Type:       CacheActionPreserve,
			WorkflowID: signal.WorkflowID,
			Backend:    signal.Backend,
			Reason:     "small increment: delta within threshold",
		}
	}
	return CacheAction{Type: CacheActionNoop}
}

// FlushAtBoundaryPolicy flushes KV cache at workflow boundaries or when
// a stage transitions to a different backend/model.
type FlushAtBoundaryPolicy struct{}

func NewFlushAtBoundaryPolicy() *FlushAtBoundaryPolicy {
	return &FlushAtBoundaryPolicy{}
}

func (p *FlushAtBoundaryPolicy) Evaluate(signal StageTransition, _ *Tracker) CacheAction {
	if signal.TransitionType == TransitionWorkflowComplete {
		return CacheAction{
			Type:       CacheActionFlush,
			WorkflowID: signal.WorkflowID,
			Backend:    signal.Backend,
			Reason:     "workflow completed",
		}
	}
	if signal.TransitionType == TransitionAgentComplete {
		switchedBackend := signal.PrevBackend != "" && signal.PrevBackend != signal.Backend
		switchedModel := signal.PrevModel != "" && signal.PrevModel != signal.Model
		if switchedBackend || switchedModel {
			return CacheAction{
				Type:       CacheActionFlush,
				WorkflowID: signal.WorkflowID,
				Backend:    signal.PrevBackend,
				Reason:     "backend/model switch at agent boundary",
			}
		}
	}
	if signal.TransitionType == TransitionStageStart && signal.PrevBackend != "" {
		switchedBackend := signal.PrevBackend != signal.Backend
		switchedModel := signal.PrevModel != signal.Model
		if switchedBackend || switchedModel {
			return CacheAction{
				Type:       CacheActionFlush,
				WorkflowID: signal.WorkflowID,
				Backend:    signal.PrevBackend,
				Reason:     "backend/model switch at stage boundary",
			}
		}
	}
	return CacheAction{Type: CacheActionNoop}
}

// FlushUnderPressurePolicy flushes cache for idle workflows when memory
// pressure is reported. It selects the oldest idle workflow with preserved
// cache on the pressured backend.
type FlushUnderPressurePolicy struct {
	PressureThreshold float64
}

func NewFlushUnderPressurePolicy(pressureThreshold float64) *FlushUnderPressurePolicy {
	if pressureThreshold <= 0 {
		pressureThreshold = 0.85
	}
	return &FlushUnderPressurePolicy{PressureThreshold: pressureThreshold}
}

// EvaluatePressure is called from the memory pressure monitor, not from the
// normal transition path. It returns flush actions for idle workflows.
func (p *FlushUnderPressurePolicy) EvaluatePressure(backend string, currentPressure float64, tracker *Tracker) []CacheAction {
	if currentPressure < p.PressureThreshold {
		return nil
	}
	candidates := tracker.WorkflowsWithPreservedCacheOnBackend(backend)
	if len(candidates) == 0 {
		return nil
	}
	// Flush the oldest idle workflow first (greedy: free memory quickly).
	var oldestID string
	var oldestTime *StageState
	for _, wfID := range candidates {
		last := tracker.LastStageOnBackend(wfID, backend)
		if last == nil {
			continue
		}
		if oldestTime == nil || last.StartedAt.Before(oldestTime.StartedAt) {
			oldestID = wfID
			oldestTime = last
		}
	}
	if oldestID == "" {
		return nil
	}
	return []CacheAction{{
		Type:       CacheActionFlush,
		WorkflowID: oldestID,
		Backend:    backend,
		Reason:     "memory pressure: flushing oldest idle workflow",
	}}
}

// PolicyChain evaluates policies in order. The first non-Noop result wins.
type PolicyChain struct {
	policies []Policy
}

func NewPolicyChain(policies ...Policy) *PolicyChain {
	return &PolicyChain{policies: policies}
}

func (c *PolicyChain) Evaluate(signal StageTransition, tracker *Tracker) CacheAction {
	for _, p := range c.policies {
		action := p.Evaluate(signal, tracker)
		if action.Type != CacheActionNoop {
			return action
		}
	}
	return CacheAction{Type: CacheActionNoop}
}
