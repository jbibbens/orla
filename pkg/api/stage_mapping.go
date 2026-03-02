package orla

import "fmt"

// StageAssignment describes the backend and inference parameters assigned to a stage.
type StageAssignment struct {
	Backend         *LLMBackend
	MaxTokens       *int
	Temperature     *float64
	TopP            *float64
	ResponseFormat  *StructuredOutputRequest
}

// StageMappingInput is the input to a StageMapping: all stages that need assignment
// and the available infrastructure (backends).
type StageMappingInput struct {
	Stages   []*Stage
	Backends []*LLMBackend
}

// StageMappingOutput is the result of stage mapping: per-stage assignments.
type StageMappingOutput struct {
	Assignments map[string]*StageAssignment // stageID -> assignment
}

// StageMapping takes a set of stages and available backends, and assigns each stage
// to a specific backend with inference parameters. This is the planning step before
// workflow execution (Section 4.1 of the Orla paper).
type StageMapping interface {
	Map(input *StageMappingInput) (*StageMappingOutput, error)
}

// ExplicitStageMapping validates that every stage already has a backend assigned.
// This is the default mapping strategy when the user sets backends directly on stages.
type ExplicitStageMapping struct{}

// Map validates all stages have backends and returns their current assignments.
func (m *ExplicitStageMapping) Map(input *StageMappingInput) (*StageMappingOutput, error) {
	if input == nil {
		return nil, fmt.Errorf("stage mapping input cannot be nil")
	}
	out := &StageMappingOutput{
		Assignments: make(map[string]*StageAssignment, len(input.Stages)),
	}
	for _, s := range input.Stages {
		if s.Backend == nil {
			return nil, fmt.Errorf("stage %q (%s) has no backend assigned", s.Name, s.ID)
		}
		out.Assignments[s.ID] = &StageAssignment{
			Backend:        s.Backend,
			MaxTokens:      s.MaxTokens,
			Temperature:    s.Temperature,
			TopP:           s.TopP,
			ResponseFormat: s.ResponseFormat,
		}
	}
	return out, nil
}

// ApplyStageMappingOutput applies mapping output to stages, overwriting their backend
// and inference parameters with the assigned values.
func ApplyStageMappingOutput(stages []*Stage, output *StageMappingOutput) error {
	if output == nil {
		return fmt.Errorf("stage mapping output cannot be nil")
	}
	for _, s := range stages {
		assignment, ok := output.Assignments[s.ID]
		if !ok {
			continue
		}
		if assignment.Backend != nil {
			s.Backend = assignment.Backend
		}
		if assignment.MaxTokens != nil {
			s.MaxTokens = assignment.MaxTokens
		}
		if assignment.Temperature != nil {
			s.Temperature = assignment.Temperature
		}
		if assignment.TopP != nil {
			s.TopP = assignment.TopP
		}
		if assignment.ResponseFormat != nil {
			s.ResponseFormat = assignment.ResponseFormat
		}
	}
	return nil
}
