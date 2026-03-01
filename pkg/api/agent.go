package orla

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// Agent represents a single agent profile. It uses the current Stage for backend, inference options, and tools.
// Use it for execute calls and pass the prompt per call to Execute or ExecuteStream.
// Configure the stage via a.Stage.SetMaxTokens, a.Stage.AddTool, etc., or build an AgentStage and SetStage.
// Note that this is safe for concurrent use i.e. multiple threads can use the same Agent instance to execute calls.
type Agent struct {
	Client *OrlaClient
	// Stage is the current backend, inference options, and tools; used for all execute calls.
	Stage *AgentStage
}

// NewAgent returns an agent that uses the given client and backend (wrapped in a default stage).
func NewAgent(client *OrlaClient) *Agent {
	return &Agent{Client: client}
}

// SetStage sets the current stage. Use this to switch stages.
func (a *Agent) SetStage(s *AgentStage) { a.Stage = s }

// req builds a request with a prompt and the current stage's inference options.
func (a *Agent) req(prompt string) (*ExecuteRequest, error) {
	if a.Stage == nil {
		return nil, fmt.Errorf("stage is nil")
	}

	s := a.Stage
	r := &ExecuteRequest{Backend: s.LLMBackend.Name, Prompt: prompt}
	r.MaxTokens = s.MaxTokens
	r.Temperature = s.Temperature
	r.TopP = s.TopP
	r.ResponseFormat = s.ResponseFormat
	r.ChatTemplateKwargs = s.ChatTemplateKwargs
	return r, nil
}

// reqWithMessages builds a request with existing messages and tools, for agent loops.
func (a *Agent) reqWithMessages(messages []Message) (*ExecuteRequest, error) {
	if a.Stage == nil {
		return nil, fmt.Errorf("stage is nil")
	}

	s := a.Stage
	r := &ExecuteRequest{Backend: s.LLMBackend.Name, Messages: messages}
	r.MaxTokens = s.MaxTokens
	r.Temperature = s.Temperature
	r.TopP = s.TopP
	r.ResponseFormat = s.ResponseFormat
	r.ChatTemplateKwargs = s.ChatTemplateKwargs

	if len(a.Stage.Tools) > 0 {
		r.Tools = a.toolsToMCP()
	}

	return r, nil
}

func (a *Agent) toolsToMCP() []*mcp.Tool {
	tools := a.Stage.Tools
	out := make([]*mcp.Tool, 0, len(tools))
	for _, t := range tools {
		out = append(out, t.toMCP())
	}
	return out
}

// Execute runs a single non-streaming inference with the given prompt.
func (a *Agent) Execute(ctx context.Context, prompt string) (*InferenceResponse, error) {
	req, err := a.req(prompt)
	if err != nil {
		return nil, err
	}
	return a.Client.Execute(ctx, req)
}

// ExecuteStream runs inference with streaming; returns a channel of events (content, thinking, tool_call, done).
func (a *Agent) ExecuteStream(ctx context.Context, prompt string) (<-chan StreamEvent, error) {
	req, err := a.req(prompt)
	if err != nil {
		return nil, err
	}
	return a.Client.ExecuteStream(ctx, req)
}

// ExecuteWithMessages runs a single non-streaming inference with the given message list and any tools attached to the agent.
// Use this for agent loops: append assistant and tool result messages, then call again until the model returns no tool calls.
func (a *Agent) ExecuteWithMessages(ctx context.Context, messages []Message) (*InferenceResponse, error) {
	req, err := a.reqWithMessages(messages)
	if err != nil {
		return nil, err
	}
	return a.Client.Execute(ctx, req)
}

// ExecuteStreamWithMessages runs streaming inference with the given message list and any tools attached to the agent.
func (a *Agent) ExecuteStreamWithMessages(ctx context.Context, messages []Message) (<-chan StreamEvent, error) {
	req, err := a.reqWithMessages(messages)
	if err != nil {
		return nil, err
	}
	return a.Client.ExecuteStream(ctx, req)
}

// StreamHandler is an optional callback invoked for each stream event (e.g. to print tokens).
// ConsumeStream always accumulates and returns the full InferenceResponse; the handler is for side effects only.
type StreamHandler func(event StreamEvent) error

// ConsumeStream reads the stream until "done", accumulates content/thinking/metrics, and returns the result.
// If streamHandler is non-nil, it is called for each event before processing (e.g. to print content as it arrives).
func (a *Agent) ConsumeStream(ctx context.Context, stream <-chan StreamEvent, handler StreamHandler) (*InferenceResponse, error) {
	response := &InferenceResponse{Metrics: &InferenceResponseMetrics{}}
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case event, ok := <-stream:
			if !ok {
				if ctx.Err() != nil {
					return nil, ctx.Err()
				}
				return nil, fmt.Errorf("stream closed without done")
			}
			if handler != nil {
				if err := handler(event); err != nil {
					return nil, err
				}
			}
			switch event.Type {
			case "content":
				response.Content += event.Content
			case "thinking":
				response.Thinking += event.Thinking
			case "tool_call":
				// Streaming tool_call deltas are for display; final ToolCalls come in "done"
			case "done":
				if event.Response != nil {
					response.Content = event.Response.Content
					response.Thinking = event.Response.Thinking
					response.ToolCalls = event.Response.ToolCalls
					if event.Response.Metrics != nil {
						response.Metrics = event.Response.Metrics
					}
				}
				return response, nil
			}
		}
	}
}

func (a *Agent) RunToolCall(ctx context.Context, toolCall *ToolCall) (*ToolResult, error) {
	if toolCall == nil {
		return nil, fmt.Errorf("tool call cannot be nil")
	}

	tool, ok := a.Stage.Tools[toolCall.Name]
	if !ok {
		return nil, fmt.Errorf("unknown tool %q", toolCall.Name)
	}

	toolResult, err := tool.Run(ctx, toolCall.InputArguments)

	if err != nil {
		return nil, fmt.Errorf("failed to run tool call: %w", err)
	}

	if toolResult == nil {
		return nil, fmt.Errorf("tool result is nil")
	}

	toolResult.ID = toolCall.ID
	toolResult.Name = toolCall.Name

	return toolResult, nil
}

// RunToolCallsInResponseAndGetToolResults parses the response's tool calls, runs each tool by name, and returns results.
func (a *Agent) RunToolCallsInResponseAndGetToolResults(ctx context.Context, response *InferenceResponse) ([]*ToolResult, error) {
	toolResults := make([]*ToolResult, 0, len(response.ToolCalls))

	for _, call := range response.ToolCalls {
		toolCall, err := NewToolCallFromRawToolCall(call)
		if err != nil {
			return nil, fmt.Errorf("failed to parse tool call: %w", err)
		}

		toolResult, err := a.RunToolCall(ctx, toolCall)
		if err != nil {
			return nil, fmt.Errorf("failed to run tool call: %w", err)
		}
		toolResults = append(toolResults, toolResult)
	}
	return toolResults, nil
}

// RunToolCallsInResponse runs the tool calls in the response and returns the tool result messages.
func (a *Agent) RunToolCallsInResponse(ctx context.Context, response *InferenceResponse) ([]*Message, error) {
	toolResults, err := a.RunToolCallsInResponseAndGetToolResults(ctx, response)
	if err != nil {
		return nil, fmt.Errorf("failed to run tool calls: %w", err)
	}

	toolMessages := make([]*Message, 0, len(toolResults))
	for _, toolResult := range toolResults {
		toolMessage, err := toolResult.ToMessage()
		if err != nil {
			return nil, fmt.Errorf("failed to convert tool result to message: %w", err)
		}
		toolMessages = append(toolMessages, toolMessage)
	}

	return toolMessages, nil
}
