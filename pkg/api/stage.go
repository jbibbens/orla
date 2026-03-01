package orla

import (
	"context"
	"fmt"
)

// AgentStage holds backend, inference options, output format, and tools for a phase.
// The agent uses the current stage when building requests.
type AgentStage struct {
	Name string
	// LLMBackend is the backend used for inference.
	LLMBackend *LLMBackend
	// MaxTokens is optional; nil means backend default.
	MaxTokens *int
	// Temperature is optional; nil means backend default.
	Temperature *float64
	// TopP is optional; nil means backend default.
	TopP *float64
	// ResponseFormat requests structured output (JSON Schema). Nil means no structured output.
	ResponseFormat *StructuredOutputRequest
	// ChatTemplateKwargs are extra kwargs passed to the chat template renderer (e.g. SGLang/vLLM).
	ChatTemplateKwargs map[string]any
	// Tools are the tools available in this stage (e.g. different stages can expose different tool sets).
	Tools map[string]*Tool
}

// NewAgentStage returns a stage with the given backend; other options are nil (backend defaults), Tools is empty.
func NewAgentStage(name string, backend *LLMBackend) *AgentStage {
	return &AgentStage{Name: name, LLMBackend: backend, Tools: make(map[string]*Tool)}
}

// SetMaxTokens sets the maximum tokens for execute calls (nil means backend default).
func (s *AgentStage) SetMaxTokens(n int) { s.MaxTokens = &n }

// SetTemperature sets the sampling temperature for execute calls (nil means backend default).
func (s *AgentStage) SetTemperature(f float64) { s.Temperature = &f }

// SetTopP sets the nucleus sampling top_p for execute calls (nil means backend default).
func (s *AgentStage) SetTopP(f float64) { s.TopP = &f }

// SetResponseFormat sets the structured output (JSON Schema) for execute calls. Use nil to disable.
func (s *AgentStage) SetResponseFormat(r *StructuredOutputRequest) { s.ResponseFormat = r }

// SetChatTemplateKwargs sets extra kwargs for the chat template renderer
func (s *AgentStage) SetChatTemplateKwargs(kwargs map[string]any) { s.ChatTemplateKwargs = kwargs }

// AddTool adds a tool to this stage. Returns an error if t is nil.
func (s *AgentStage) AddTool(t *Tool) error {
	if t == nil {
		return fmt.Errorf("tool cannot be nil")
	}
	s.Tools[t.Name] = t
	return nil
}

// OneBitStageMapper is a stage mapper that uses a one bit predictor and a prompt to do stage mapping.
type OneBitStageMapper struct {
	OneBitPredictor *OneBitPredictor
	StageOne        *AgentStage
	StageTwo        *AgentStage
	Prompt          string
}

// NewOneBitStageMapper returns a new one bit stage mapper.
func NewOneBitStageMapper(client *OrlaClient, backend *LLMBackend, stageOne *AgentStage, stageTwo *AgentStage) *OneBitStageMapper {
	return &OneBitStageMapper{OneBitPredictor: NewOneBitPredictor(client, backend), StageOne: stageOne, StageTwo: stageTwo}
}

// MapStage maps the stage based on the prompt.
func (m *OneBitStageMapper) MapStage(ctx context.Context, prompt string) (*AgentStage, error) {
	prediction, err := m.OneBitPredictor.Predict(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("failed to predict stage: %w", err)
	}

	if prediction {
		return m.StageOne, nil
	}

	return m.StageTwo, nil
}
