// Package serving implements a minimal programmatic serving layer.
package serving

import (
	"context"
	"fmt"

	"github.com/dorcha-inc/orla/internal/core"
	"github.com/dorcha-inc/orla/internal/model"
	"github.com/dorcha-inc/orla/internal/serving/memory"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"go.uber.org/zap"
)

// AgenticLayer is the serving layer that manages LLM backends and executes inference.
type AgenticLayer struct {
	llmBackendManager *LLMBackendManager
	MemoryManager     *memory.DefaultManager
}

// NewAgenticLayer creates a new serving layer.
func NewAgenticLayer() *AgenticLayer {
	mm := memory.NewDefaultManager(memory.DefaultManagerConfig{})
	return &AgenticLayer{
		llmBackendManager: NewLLMBackendManager(mm),
		MemoryManager:     mm,
	}
}

// AddLLMBackend registers an LLM backend by name.
func (l *AgenticLayer) AddLLMBackend(name string, backend *core.LLMBackend, modelID string) {
	l.llmBackendManager.AddLLMBackend(name, backend, modelID)
}

// GetModelProvider returns the model provider for a named LLM backend.
func (l *AgenticLayer) GetModelProvider(ctx context.Context, backendName string) (model.Provider, error) {
	return l.llmBackendManager.GetModelProvider(ctx, backendName)
}

// Execute runs a single non-streaming inference call against the named LLM backend.
// For streaming, use ExecuteStream instead. opts.Stream must be false.
func (l *AgenticLayer) Execute(ctx context.Context, serverName, stageName string, messages []model.Message, tools []*mcp.Tool, opts model.InferenceOptions, chatOpts ...ChatOptions) (*model.Response, error) {
	if opts.Stream {
		return nil, fmt.Errorf("Execute does not support streaming, use ExecuteStream instead")
	}

	response, _, err := l.llmBackendManager.ScheduleChat(ctx, serverName, stageName, messages, tools, opts, chatOpts...)
	if err != nil {
		return nil, fmt.Errorf("inference failed on server '%s': %w", serverName, err)
	}
	zap.L().Debug("Executed inference",
		zap.String("server", serverName),
		zap.Int("response_length", len(response.Content)))
	return response, nil
}

// ExecuteStream runs inference with streaming. It returns the response (filled as the stream
// is consumed), a channel of stream events, and an error. The caller must consume the channel
// until closed; the response content, tool_calls, and metrics are populated by the provider's
// goroutine as the stream completes. opts.Stream must be true.
func (l *AgenticLayer) ExecuteStream(ctx context.Context, serverName, stageName string, messages []model.Message, tools []*mcp.Tool, opts model.InferenceOptions, chatOpts ...ChatOptions) (*model.Response, <-chan model.StreamEvent, error) {
	if !opts.Stream {
		return nil, nil, fmt.Errorf("ExecuteStream requires opts.Stream to be true")
	}

	response, ch, err := l.llmBackendManager.ScheduleChat(ctx, serverName, stageName, messages, tools, opts, chatOpts...)
	if err != nil {
		return nil, nil, fmt.Errorf("inference failed on server '%s': %w", serverName, err)
	}
	return response, ch, nil
}

// GetLLMBackendHealth returns the health status of a named LLM backend.
func (l *AgenticLayer) GetLLMBackendHealth(ctx context.Context, serverName string) (HealthStatus, error) {
	return l.llmBackendManager.GetHealthStatus(ctx, serverName)
}

// ListLLMBackends returns all registered LLM backend names.
func (l *AgenticLayer) ListLLMBackends() []string {
	return l.llmBackendManager.ListLLMBackends()
}
