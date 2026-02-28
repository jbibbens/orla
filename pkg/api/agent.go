package orla

import (
	"context"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// Agent represents a single agent profile including the backend and inference options.
// Use it for execute calls and pass the prompt per call to Execute or ExecuteStream.
// Add tools with AddTool so the model can call them when using ExecuteWithMessages
// or ExecuteStreamWithMessages. Note that this is safe for concurrent use i.e.
// multiple threads can use the same Agent instance to execute calls.
type Agent struct {
	Client  *OrlaClient
	Backend *LLMBackend
	// MaxTokens is optional; nil means backend default.
	MaxTokens *int
	// Temperature is optional; nil means backend default.
	Temperature *float64
	// TopP is optional; nil means backend default.
	TopP *float64
	// Tools are the tools attached to this agent.
	Tools []*Tool
}

// NewAgent returns an agent that uses the given client and backend.
func NewAgent(client *OrlaClient, backend *LLMBackend) *Agent {
	tools := make([]*Tool, 0)
	return &Agent{Client: client, Backend: backend, Tools: tools}
}

// SetMaxTokens sets the maximum tokens for execute calls (nil means backend default).
func (a *Agent) SetMaxTokens(n int) { a.MaxTokens = &n }

// SetTemperature sets the sampling temperature for execute calls (nil means backend default).
func (a *Agent) SetTemperature(f float64) { a.Temperature = &f }

// SetTopP sets the nucleus sampling top_p for execute calls (nil means backend default).
func (a *Agent) SetTopP(f float64) { a.TopP = &f }

// AddTool adds a tool to this agent. The tool spec is sent to the model via the
// configured LLM backend. Run is invoked locally when the model returns a tool call.
func (a *Agent) AddTool(t *Tool) error {
	if t == nil {
		return fmt.Errorf("tool cannot be nil")
	}

	a.Tools = append(a.Tools, t)
	return nil
}

// req builds a request with a prompt and optional inference options.
func (a *Agent) req(prompt string) *ExecuteRequest {
	r := &ExecuteRequest{Backend: a.Backend.Name, Prompt: prompt}
	r.MaxTokens = a.MaxTokens
	r.Temperature = a.Temperature
	r.TopP = a.TopP
	return r
}

// reqWithMessages builds a request with existing messages and tools, for agent loops.
func (a *Agent) reqWithMessages(messages []Message) *ExecuteRequest {
	r := &ExecuteRequest{Backend: a.Backend.Name, Messages: messages}
	r.MaxTokens = a.MaxTokens
	r.Temperature = a.Temperature
	r.TopP = a.TopP

	if len(a.Tools) > 0 {
		r.Tools = a.toolsToMCP()
	}

	return r
}

func (a *Agent) toolsToMCP() []*mcp.Tool {
	out := make([]*mcp.Tool, len(a.Tools))
	for i, t := range a.Tools {
		out[i] = t.toMCP()
	}
	return out
}

// Execute runs a single non-streaming inference with the given prompt.
func (a *Agent) Execute(ctx context.Context, prompt string) (*InferenceResponse, error) {
	return a.Client.Execute(ctx, a.req(prompt))
}

// ExecuteStream runs inference with streaming; returns a channel of events (content, thinking, tool_call, done).
func (a *Agent) ExecuteStream(ctx context.Context, prompt string) (<-chan StreamEvent, error) {
	return a.Client.ExecuteStream(ctx, a.req(prompt))
}

// ExecuteWithMessages runs a single non-streaming inference with the given message list and any tools attached to the agent.
// Use this for agent loops: append assistant and tool result messages, then call again until the model returns no tool calls.
func (a *Agent) ExecuteWithMessages(ctx context.Context, messages []Message) (*InferenceResponse, error) {
	return a.Client.Execute(ctx, a.reqWithMessages(messages))
}

// ExecuteStreamWithMessages runs streaming inference with the given message list and any tools attached to the agent.
func (a *Agent) ExecuteStreamWithMessages(ctx context.Context, messages []Message) (<-chan StreamEvent, error) {
	return a.Client.ExecuteStream(ctx, a.reqWithMessages(messages))
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

// RunToolCalls parses the response's tool calls, runs each tool by name, and returns results.
// The caller can append each result with result.ToMessage() to messages and call ExecuteWithMessages again.
func (a *Agent) RunToolCalls(ctx context.Context, response *InferenceResponse) ([]ToolResult, error) {
	calls, err := parseToolCalls(response.ToolCalls)
	if err != nil {
		return nil, err
	}
	if len(calls) == 0 {
		return nil, nil
	}
	byName := make(map[string]*Tool)
	for _, t := range a.Tools {
		byName[t.Name] = t
	}
	results := make([]ToolResult, 0, len(calls))
	for _, call := range calls {
		tool, ok := byName[call.Name]
		if !ok {
			return nil, fmt.Errorf("unknown tool %q", call.Name)
		}
		args := call.InputArguments
		if args == nil {
			args = ToolSchema{}
		}
		out, err := tool.Run(ctx, args)
		if err != nil {
			results = append(results, ToolResult{
				ID: call.ID, Name: call.Name,
				Error: err.Error(), IsError: true,
			})
			continue
		}
		if out == nil {
			out = ToolSchema{}
		}
		results = append(results, ToolResult{ID: call.ID, Name: call.Name, OutputValues: out})
	}
	return results, nil
}
