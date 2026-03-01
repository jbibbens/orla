// Package model provides model integration for Orla Agent Mode (RFC 4).
package model

import (
	"context"
	"encoding/json"
	"io"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type MessageRole string

const (
	MessageRoleUser      MessageRole = "user"
	MessageRoleAssistant MessageRole = "assistant"
	MessageRoleSystem    MessageRole = "system"
	MessageRoleTool      MessageRole = "tool"
)

func (r MessageRole) String() string {
	return string(r)
}

// Message represents a chat message in a conversation.
// For role "tool" i.e tool calls, set both ToolCallID and ToolName when building messages.
// The OpenAI API uses ToolCallID and the Ollama API uses ToolName. The providers ignore the field they do not need.
type Message struct {
	// Role of the message
	Role MessageRole `json:"role"`
	// Content of the message
	Content string `json:"content"`
	// ToolName is used by the Ollama API
	ToolName string `json:"tool_name,omitempty"`
	// ToolCallID is used by the OpenAI API and vLLM
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// ToolCallWithID represents a tool invocation request from the model.
// It embeds mcp.CallToolParams for MCP compatibility, and adds an ID
// for tracking in the agent loop so we can match results back to calls.
type ToolCallWithID struct {
	ID                string `json:"id"` // Unique identifier for this tool call
	McpCallToolParams mcp.CallToolParams
}

type ResponseMetrics struct {
	// TTFTMs is time to first token in milliseconds. Only set when task was executed with streaming.
	TTFTMs int64 `json:"ttft_ms,omitempty"`
	// TPOTMs is time per output token in milliseconds. Only set when task was executed with streaming.
	TPOTMs int64 `json:"tpot_ms,omitempty"`
}

// Response represents a model response
type Response struct {
	Content   string           `json:"content"`    // Text content from the model
	Thinking  string           `json:"thinking"`   // Thinking trace from the model (if supported)
	ToolCalls []ToolCallWithID `json:"tool_calls"` // Tool calls requested by the model
	Metrics   *ResponseMetrics `json:"metrics"`    // Response metrics
}

// StructuredOutputOptions requests the model to return content conforming to a JSON Schema.
type StructuredOutputOptions struct {
	Name   string          `json:"name"`             // Required by OpenAI for json_schema response_format
	Strict bool            `json:"strict,omitempty"` // If true, response is guaranteed to conform to schema (default true when used)
	Schema json.RawMessage `json:"schema"`           // JSON Schema object. The schema must be valid when set
}

// InferenceOptions holds per-request inference settings (agent profile knobs).
// JSON tags match the execute API so the server can embed this in ExecuteRequest.
type InferenceOptions struct {
	// Stream is whether to stream the response. A nil value means no streaming.
	Stream bool `json:"stream,omitempty"`
	// MaxTokens is the maximum number of tokens to generate. A nil value means use the backend default.
	MaxTokens *int `json:"max_tokens,omitempty"`
	// Temperature is the temperature parameter for sampling. A nil value means use the backend default.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopP is the nucleus sampling top_p parameter. A nil value means use the backend default.
	TopP *float64 `json:"top_p,omitempty"`
	// ResponseFormat is the structured output options. A nil value means no structured output.
	ResponseFormat *StructuredOutputOptions `json:"response_format,omitempty"`
	// ChatTemplateKwargs are extra kwargs passed to the chat template renderer
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

// Provider is the interface that all model providers must implement
type Provider interface {
	// Name returns the provider name (e.g., "ollama", "openai", "anthropic")
	Name() string

	// Chat sends a chat request to the model with the given inference options and returns the response.
	Chat(ctx context.Context, messages []Message, tools []*mcp.Tool, opts InferenceOptions) (*Response, <-chan StreamEvent, error)

	// EnsureReady ensures the model provider is ready (e.g., starts Ollama if needed)
	// Returns an error if the provider cannot be made ready
	EnsureReady(ctx context.Context) error
}

// StreamWriter is an interface for writing streaming responses
type StreamWriter interface {
	io.Writer
	Flush() error
}

// StreamEvent represents a single event in the streaming response
type StreamEvent interface {
	// Type returns the type of stream event
	Type() StreamEventType
}

// StreamEventType represents the type of stream event
type StreamEventType string

const (
	StreamEventTypeContent  StreamEventType = "content"  // Text content chunk
	StreamEventTypeToolCall StreamEventType = "toolcall" // Tool call notification
	StreamEventTypeThinking StreamEventType = "thinking" // Thinking trace chunk
)

// ContentEvent represents a content chunk in the stream
type ContentEvent struct {
	Content string
}

func (e *ContentEvent) Type() StreamEventType {
	return StreamEventTypeContent
}

// ToolCallEvent represents a tool call notification in the stream
type ToolCallEvent struct {
	Name      string
	Arguments map[string]any
}

func (e *ToolCallEvent) Type() StreamEventType {
	return StreamEventTypeToolCall
}

// ThinkingEvent represents a thinking trace chunk in the stream
type ThinkingEvent struct {
	Content string
}

func (e *ThinkingEvent) Type() StreamEventType {
	return StreamEventTypeThinking
}
