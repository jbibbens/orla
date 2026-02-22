package orla

import (
	"context"
	"fmt"
)

// Agent holds a client, a registered backend, and optional prompt/settings for execute calls.
// Use Execute for a single response, or ExecuteStream for token-by-token events.
type Agent struct {
	Client    *OrlaClient
	Backend   *LLMBackend
	Prompt    string
	MaxTokens int
}

// NewAgent returns an agent that uses the given client and backend.
func NewAgent(client *OrlaClient, backend *LLMBackend) *Agent {
	return &Agent{Client: client, Backend: backend}
}

// NewAgentWithPrompt returns an agent with the prompt set (e.g. for a one-shot run).
func NewAgentWithPrompt(client *OrlaClient, backend *LLMBackend, prompt string) *Agent {
	return &Agent{Client: client, Backend: backend, Prompt: prompt}
}

// SetPrompt sets the agent's prompt.
func (a *Agent) SetPrompt(prompt string) {
	a.Prompt = prompt
}

// SetMaxTokens sets the maximum tokens for execute calls (0 means backend default).
func (a *Agent) SetMaxTokens(n int) {
	a.MaxTokens = n
}

// req returns an ExecuteRequest from the agent's current fields.
func (a *Agent) req() *ExecuteRequest {
	r := &ExecuteRequest{Backend: a.Backend.Name, Prompt: a.Prompt}
	if a.MaxTokens > 0 {
		r.MaxTokens = a.MaxTokens
	}
	return r
}

// Execute runs a single non-streaming inference and returns the full response.
func (a *Agent) Execute(ctx context.Context) (*InferenceResponse, error) {
	return a.Client.Execute(ctx, a.req())
}

// ExecuteStream runs inference with streaming and returns a channel of events (content, thinking, tool_call, done).
func (a *Agent) ExecuteStream(ctx context.Context) (<-chan StreamEvent, error) {
	return a.Client.ExecuteStream(ctx, a.req())
}

// StreamHandler is an optional callback invoked for each stream event (e.g. to print tokens).
// ConsumeStream always accumulates and returns the full InferenceResponse; the handler is for side effects only.
type StreamHandler func(event StreamEvent) error

// ConsumeStream reads the stream until "done", accumulates content/thinking/metrics, and returns the result.
// If streamHandler is non-nil, it is called for each event before processing (e.g. to print content as it arrives).
func (a *Agent) ConsumeStream(ctx context.Context, stream <-chan StreamEvent, streamHandler StreamHandler) (*InferenceResponse, error) {
	response := &InferenceResponse{
		Content:     "",
		Thinking:    "",
		ToolCalls:   []any{},
		ToolResults: []any{},
		Metrics:     &InferenceResponseMetrics{},
	}
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case event, ok := <-stream:
			if !ok {
				return nil, fmt.Errorf("stream closed without a final response")
			}
			if streamHandler != nil {
				if err := streamHandler(event); err != nil {
					return nil, fmt.Errorf("stream handler: %w", err)
				}
			}
			switch event.Type {
			case "content":
				response.Content += event.Content
			case "thinking":
				response.Thinking += event.Thinking
			case "tool_call":
				return nil, fmt.Errorf("tool calls not supported for now")
			case "done":
				if event.Response != nil && event.Response.Metrics != nil {
					response.Metrics.TTFTMs = event.Response.Metrics.TTFTMs
					response.Metrics.TPOTMs = event.Response.Metrics.TPOTMs
				}
				return response, nil
			default:
				// Ignore unknown event types for forward compatibility
			}
		}
	}
}
