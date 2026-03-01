// Package orla provides a public Go client library for Orla server
package orla

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// OrlaClient is the public API client for the Orla daemon
type OrlaClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewOrlaClient creates a new daemon API client.
func NewOrlaClient(baseURL string) *OrlaClient {
	return &OrlaClient{
		baseURL:    baseURL,
		httpClient: &http.Client{},
	}
}

// Health checks the health of the daemon
func (c *OrlaClient) Health(ctx context.Context) error {
	url := fmt.Sprintf("%s/api/v1/health", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to check health: %w", err)
	}
	defer LogDeferredError(resp.Body.Close)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	return nil
}

// RegisterBackend registers an LLM backend with the daemon. Call this before using the backend in Execute.
// Pass the same req (or the *LLMBackend) to NewAgent after a successful registration.
func (c *OrlaClient) RegisterBackend(ctx context.Context, req *RegisterBackendRequest) error {
	url := fmt.Sprintf("%s/api/v1/backends", c.baseURL)
	body, err := json.Marshal(req)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("register backend request failed: %w", err)
	}
	defer LogDeferredError(httpResp.Body.Close)
	var resp RegisterBackendResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&resp); err != nil {
		return fmt.Errorf("failed to decode response: %w", err)
	}
	if !resp.Success {
		return fmt.Errorf("register backend failed: %s", resp.Error)
	}
	return nil
}

// StructuredOutputRequest requests the model to return content conforming to a JSON Schema.
type StructuredOutputRequest struct {
	// Name is the name of the structured output. Required for json_schema response_format.
	Name string `json:"name"`
	// Strict is whether the response is guaranteed to conform to the schema.
	Strict bool `json:"strict,omitempty"`
	// Schema is the JSON Schema object (e.g. map[string]any or struct). The schema must be valid when set.
	Schema any `json:"schema"`
}

// ExecuteRequest represents a request to execute inference on a named backend.
type ExecuteRequest struct {
	Backend string `json:"backend"`
	// Prompt is the prompt to execute.
	Prompt string `json:"prompt,omitempty"`
	// Messages are the messages to execute.
	Messages []Message `json:"messages,omitempty"`
	// Tools are the tools attached to this request.
	Tools []*mcp.Tool `json:"tools,omitempty"`
	// MaxTokens is the maximum number of tokens to generate. A nil value means use the backend default.
	MaxTokens *int `json:"max_tokens,omitempty"`
	// Stream is whether to stream the response. A nil value means no streaming.
	Stream bool `json:"stream,omitempty"`
	// Temperature is the temperature parameter for sampling. A nil value means use the backend default.
	Temperature *float64 `json:"temperature,omitempty"`
	// TopP is the nucleus sampling top_p parameter. A nil value means use the backend default.
	TopP *float64 `json:"top_p,omitempty"`
	// ResponseFormat is the structured output options. A nil value means no structured output.
	ResponseFormat *StructuredOutputRequest `json:"response_format,omitempty"`
}

// ExecuteResponse represents the response from an execute call.
type ExecuteResponse struct {
	// Success is whether the execute call was successful.
	Success bool `json:"success"`
	// Response is the response from the execute call.
	Response *InferenceResponse `json:"response,omitempty"`
	// Error is the error from the execute call, if any.
	Error string `json:"error,omitempty"`
}

// Message represents a chat message.
type Message struct {
	Role       string `json:"role"`
	Content    string `json:"content"`
	ToolCallID string `json:"tool_call_id,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
}

// RawToolCall is a raw tool call from the inference response.
type RawToolCall []byte

// UnmarshalJSON stores the raw JSON bytes of the value (object or otherwise) so that tool_calls from the API response decode correctly.
func (r *RawToolCall) UnmarshalJSON(data []byte) error {
	*r = append((*r)[0:0], data...)
	return nil
}

// InferenceResponse represents the response from inference.
type InferenceResponse struct {
	Content   string                    `json:"content"`
	Thinking  string                    `json:"thinking,omitempty"`
	ToolCalls []RawToolCall             `json:"tool_calls,omitempty"`
	Metrics   *InferenceResponseMetrics `json:"metrics,omitempty"`
}

// InferenceResponseMetrics holds timing metrics from streaming execution.
type InferenceResponseMetrics struct {
	TTFTMs int64 `json:"ttft_ms,omitempty"`
	TPOTMs int64 `json:"tpot_ms,omitempty"`
}

// StreamEvent is a single event from ExecuteStream. Exactly one of Content, Thinking, ToolCall, or Response is set, depending on Type.
type StreamEvent struct {
	Type     string             // "content", "thinking", "tool_call", or "done"
	Content  string             // content delta (Type == "content")
	Thinking string             // thinking delta (Type == "thinking")
	ToolCall *ToolCallDelta     // tool call (Type == "tool_call")
	Response *InferenceResponse // final response (Type == "done")
}

// ToolCallDelta is a streaming tool call notification.
type ToolCallDelta struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ExecuteStream runs inference with streaming. The returned channel receives content, thinking, and tool_call deltas, then a final "done" event with the full Response. Caller must consume the channel until closed.
func (c *OrlaClient) ExecuteStream(ctx context.Context, req *ExecuteRequest) (<-chan StreamEvent, error) {
	streamReq := *req
	streamReq.Stream = true

	url := fmt.Sprintf("%s/api/v1/execute", c.baseURL)
	body, err := json.Marshal(&streamReq)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}

	if httpResp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(httpResp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}

		err = httpResp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("failed to close response body: %w", err)
		}

		return nil, fmt.Errorf("execute failed with status %d: %s", httpResp.StatusCode, string(bodyBytes))
	}

	ch := make(chan StreamEvent)
	go readSSEStream(httpResp.Body, ch)
	return ch, nil
}

// readSSEStream parses SSE from r and sends StreamEvents to ch, then closes ch.
func readSSEStream(r io.ReadCloser, ch chan StreamEvent) {
	defer LogDeferredError(r.Close)
	defer close(ch)

	scanner := bufio.NewScanner(r)
	var eventType, data string
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if eventType != "" && data != "" {
				if ev := parseSSEEvent(eventType, data); ev != nil {
					ch <- *ev
				}
			}
			eventType = ""
			data = ""
			continue
		}
		if strings.HasPrefix(line, "event: ") {
			eventType = strings.TrimPrefix(line, "event: ")
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			data = strings.TrimPrefix(line, "data: ")
			continue
		}
	}
}

func parseSSEEvent(eventType, data string) *StreamEvent {
	ev := &StreamEvent{Type: eventType}
	switch eventType {
	case "content":
		var v struct {
			Content string `json:"content"`
		}
		if json.Unmarshal([]byte(data), &v) == nil {
			ev.Content = v.Content
			return ev
		}
	case "thinking":
		var v struct {
			Thinking string `json:"thinking"`
		}
		if json.Unmarshal([]byte(data), &v) == nil {
			ev.Thinking = v.Thinking
			return ev
		}
	case "tool_call":
		var v struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if json.Unmarshal([]byte(data), &v) == nil {
			ev.ToolCall = &ToolCallDelta{Name: v.Name, Arguments: v.Arguments}
			return ev
		}
	case "done":
		var v ExecuteResponse
		if json.Unmarshal([]byte(data), &v) == nil && v.Success && v.Response != nil {
			ev.Response = v.Response
			return ev
		}
	}
	return nil
}

// Execute runs inference on the named backend via the daemon.
func (c *OrlaClient) Execute(ctx context.Context, req *ExecuteRequest) (*InferenceResponse, error) {
	url := fmt.Sprintf("%s/api/v1/execute", c.baseURL)

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("failed to create execute request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	httpResp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("execute request failed: %w", err)
	}
	defer LogDeferredError(httpResp.Body.Close)

	if httpResp.StatusCode != http.StatusOK {
		bodyBytes, err := io.ReadAll(httpResp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %w", err)
		}
		return nil, fmt.Errorf("execute failed with status %d: %s", httpResp.StatusCode, string(bodyBytes))
	}

	var execResp ExecuteResponse
	if err := json.NewDecoder(httpResp.Body).Decode(&execResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if !execResp.Success {
		return nil, fmt.Errorf("execution failed: %s", execResp.Error)
	}

	return execResp.Response, nil
}
