package orla

import "context"

// CachePolicy constants for stage-level cache policy overrides.
const (
	CachePolicyPreserve = "preserve"
	CachePolicyFlush    = "flush"
	CachePolicyAuto     = "auto"
)

// CacheHints are optional per-stage parameters that influence the Memory Manager's
// cache decisions. They follow the same pattern as SchedulingHints.
type CacheHints struct {
	PreserveThresholdTokens *int  `json:"preserve_threshold_tokens,omitempty"`
	FlushOnComplete         *bool `json:"flush_on_complete,omitempty"`
}

// CacheEvent describes a stage transition for the MemoryPolicy to evaluate.
type CacheEvent struct {
	PrevStageBackend string
	PrevStageModel   string
	NextStageBackend string
	NextStageModel   string
	DeltaTokens      int
	TotalTokens      int
	TransitionType   string // "stage", "agent", "workflow_complete"
}

// MemoryPolicy determines cache actions at workflow level. Developers can
// implement custom policies or use the built-in ones shipped with Orla.
type MemoryPolicy interface {
	Decide(ctx context.Context, event CacheEvent) string // returns CachePolicyPreserve, CachePolicyFlush, or CachePolicyAuto
}

// --- Built-in policies ---

// preserveOnSmallIncrementPolicy preserves cache when the context delta
// is below a threshold and the backend/model hasn't changed.
type preserveOnSmallIncrementPolicy struct {
	thresholdTokens int
}

// NewPreserveOnSmallIncrementPolicy creates a MemoryPolicy that preserves
// KV cache when the new stage adds fewer than thresholdTokens to the context.
func NewPreserveOnSmallIncrementPolicy(thresholdTokens int) MemoryPolicy {
	if thresholdTokens <= 0 {
		thresholdTokens = 256
	}
	return &preserveOnSmallIncrementPolicy{thresholdTokens: thresholdTokens}
}

func (p *preserveOnSmallIncrementPolicy) Decide(_ context.Context, event CacheEvent) string {
	if event.PrevStageBackend != event.NextStageBackend || event.PrevStageModel != event.NextStageModel {
		return CachePolicyAuto
	}
	if event.PrevStageBackend == "" {
		return CachePolicyAuto
	}
	if event.DeltaTokens <= p.thresholdTokens {
		return CachePolicyPreserve
	}
	return CachePolicyAuto
}

// flushAtBoundaryPolicy flushes cache at workflow completion or backend/model switches.
type flushAtBoundaryPolicy struct{}

// NewFlushAtBoundaryPolicy creates a MemoryPolicy that flushes cache at
// workflow boundaries and when the backend/model changes between stages.
func NewFlushAtBoundaryPolicy() MemoryPolicy {
	return &flushAtBoundaryPolicy{}
}

func (p *flushAtBoundaryPolicy) Decide(_ context.Context, event CacheEvent) string {
	if event.TransitionType == "workflow_complete" {
		return CachePolicyFlush
	}
	if event.PrevStageBackend != "" &&
		(event.PrevStageBackend != event.NextStageBackend || event.PrevStageModel != event.NextStageModel) {
		return CachePolicyFlush
	}
	return CachePolicyAuto
}

// composedMemoryPolicy chains multiple policies; first non-auto result wins.
type composedMemoryPolicy struct {
	policies []MemoryPolicy
}

// MemoryPolicyOption configures a DefaultMemoryPolicy.
type MemoryPolicyOption func(*defaultMemoryPolicyConfig)

type defaultMemoryPolicyConfig struct {
	preserveThreshold int
}

// WithPreserveThreshold overrides the default token threshold for the
// preserve-on-small-increment policy.
func WithPreserveThreshold(tokens int) MemoryPolicyOption {
	return func(c *defaultMemoryPolicyConfig) {
		c.preserveThreshold = tokens
	}
}

// NewDefaultMemoryPolicy creates a MemoryPolicy that composes the three
// paper policies: preserve on small increment, then flush at boundary.
// The flush-under-pressure policy is handled server-side by the Memory Manager.
func NewDefaultMemoryPolicy(opts ...MemoryPolicyOption) MemoryPolicy {
	cfg := &defaultMemoryPolicyConfig{preserveThreshold: 256}
	for _, opt := range opts {
		opt(cfg)
	}
	return &composedMemoryPolicy{
		policies: []MemoryPolicy{
			NewPreserveOnSmallIncrementPolicy(cfg.preserveThreshold),
			NewFlushAtBoundaryPolicy(),
		},
	}
}

func (p *composedMemoryPolicy) Decide(ctx context.Context, event CacheEvent) string {
	for _, policy := range p.policies {
		result := policy.Decide(ctx, event)
		if result != CachePolicyAuto {
			return result
		}
	}
	return CachePolicyAuto
}
