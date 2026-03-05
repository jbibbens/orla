// Package memory implements the Memory Manager for Orla's agentic serving layer.
// It owns the KV cache lifecycle across workflow stage transitions, making
// cache preserve/flush decisions based on workflow-level signals that are
// invisible to request-level serving systems.
package memory

// TransitionType classifies a stage or workflow lifecycle event.
type TransitionType string

const (
	TransitionStageStart       TransitionType = "stage_start"
	TransitionStageComplete    TransitionType = "stage_complete"
	TransitionAgentComplete    TransitionType = "agent_complete"
	TransitionWorkflowComplete TransitionType = "workflow_complete"
)

// StageTransition is a signal from the Workflow Executor to the Memory Manager
// describing a lifecycle event in the workflow DAG.
type StageTransition struct {
	WorkflowID     string
	AgentName      string
	StageID        string
	Backend        string
	Model          string
	TransitionType TransitionType

	// PrevBackend and PrevModel describe the preceding stage on the same
	// workflow, if any. Empty when this is the first stage or the previous
	// stage used a different backend/model.
	PrevBackend string
	PrevModel   string

	// ContextTokens is the approximate total context size in tokens.
	ContextTokens int
	// DeltaTokens is the number of tokens added since the previous stage.
	DeltaTokens int

	// CachePolicy is the stage-level override ("preserve", "flush", or empty for auto).
	CachePolicy string
	// PreserveThresholdTokens is the stage-level override for the small-increment threshold.
	PreserveThresholdTokens *int
}

// CacheActionType is the action the Memory Manager instructs the backend executor to take.
type CacheActionType string

const (
	CacheActionPreserve CacheActionType = "preserve"
	CacheActionFlush    CacheActionType = "flush"
	CacheActionNoop     CacheActionType = "noop"
)

// CacheAction is the Memory Manager's response to a stage transition signal.
type CacheAction struct {
	Type       CacheActionType
	WorkflowID string
	Backend    string
	Reason     string
}
