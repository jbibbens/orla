// Package orla provides a public Go client library for Orla server.
// Tool support uses the Model Context Protocol (MCP) types from github.com/modelcontextprotocol/go-sdk/mcp.

package orla

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// ToolSchema is a JSON-serializable object (e.g. for tool input/output).
type ToolSchema map[string]any

// ToolRunner runs a tool: input from the model, output back to the model.
type ToolRunner func(ctx context.Context, input ToolSchema) (output ToolSchema, err error)

// Tool defines a single tool: name, description, schemas, and runner.
type Tool struct {
	Name         string
	Description  string
	InputSchema  ToolSchema
	OutputSchema ToolSchema
	Run          ToolRunner
}

// NewTool returns a Tool. run must be non-nil.
func NewTool(name, description string, inputSchema, outputSchema ToolSchema, run ToolRunner) (*Tool, error) {
	if run == nil {
		return nil, fmt.Errorf("tool runner cannot be nil")
	}
	return &Tool{
		Name:         name,
		Description:  description,
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
		Run:          run,
	}, nil
}

// toMCP returns the MCP tool spec for the execute request.
func (t *Tool) toMCP() *mcp.Tool {
	return &mcp.Tool{
		Name:         t.Name,
		Description:  t.Description,
		InputSchema:  t.InputSchema,
		OutputSchema: t.OutputSchema,
	}
}

// ToolCall is one tool invocation from the agent.
type ToolCall struct {
	ID             string     `json:"id"`
	Name           string     `json:"name"`
	InputArguments ToolSchema `json:"input_arguments"`
}

// ToolResult is the result of running one tool call.
type ToolResult struct {
	ID           string     `json:"id"`
	Name         string     `json:"name"`
	OutputValues ToolSchema `json:"output_values"`
	Error        string     `json:"error,omitempty"`
	IsError      bool       `json:"is_error,omitempty"`
}

// toolCallWithID is a tool call with an ID.
// NOTE: this is the same as orla/internal/model/types.go:toolCallWithID.
// If updating this, update the other one as well.
type toolCallWithID struct {
	ID                string `json:"id"`
	McpCallToolParams mcp.CallToolParams
}

func toolCallWithIDFromJSON(data []byte) (*toolCallWithID, error) {
	var tc toolCallWithID
	if err := json.Unmarshal(data, &tc); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ToolCallWithID: %w", err)
	}
	return &tc, nil
}

func (tc *toolCallWithID) toToolCall() (*ToolCall, error) {
	args, ok := tc.McpCallToolParams.Arguments.(ToolSchema)
	if !ok {
		return nil, fmt.Errorf("failed to convert arguments to ToolSchema: %v", tc.McpCallToolParams.Arguments)
	}
	return &ToolCall{
		ID:             tc.ID,
		Name:           tc.McpCallToolParams.Name,
		InputArguments: args,
	}, nil
}

// toolResultWithID is a tool result with an ID.
// NOTE: this is the same as orla/internal/model/types.go:toolResultWithID.
// If updating this, update the other one as well.
type toolResultWithID struct {
	ID                string `json:"id"`
	McpCallToolResult mcp.CallToolResult
}

func toolResultWithIDFromJSON(data []byte) (*toolResultWithID, error) {
	var tr toolResultWithID
	if err := json.Unmarshal(data, &tr); err != nil {
		return nil, fmt.Errorf("failed to unmarshal ToolResultWithID: %w", err)
	}
	return &tr, nil
}

// toToolResult converts MCP tool result to ToolResult. We only use StructuredContent
// (Tool always has OutputSchema in this API); unstructured Content is ignored.
func (tr *toolResultWithID) toToolResult(name string) (*ToolResult, error) {
	result := &ToolResult{ID: tr.ID, Name: name, IsError: tr.McpCallToolResult.IsError}

	// We are checking the error first, because if there is an error then structured content
	// is often nil and the error message is in the Content.
	if result.IsError {
		var errParts []string
		for _, c := range tr.McpCallToolResult.Content {
			t, ok := c.(*mcp.TextContent)
			if !ok {
				continue
			}

			if t.Text == "" {
				continue
			}

			errParts = append(errParts, t.Text)
		}

		result.Error = strings.Join(errParts, "\n")

		// If there is structured content, we will use it.
		if tr.McpCallToolResult.StructuredContent != nil {
			outputValues, ok := tr.McpCallToolResult.StructuredContent.(ToolSchema)
			if ok {
				result.OutputValues = outputValues
			}
		}

		return result, nil
	}

	// Now there is no error, so we enforce the use of StructuredContent.
	if tr.McpCallToolResult.StructuredContent == nil {
		return nil, fmt.Errorf("StructuredContent is nil for tool %s", name)
	}

	outputValues, ok := tr.McpCallToolResult.StructuredContent.(ToolSchema)
	if !ok {
		return nil, fmt.Errorf("could not convert StructuredContent to ToolSchema for tool %s: %T", name, tr.McpCallToolResult.StructuredContent)
	}

	result.OutputValues = outputValues
	return result, nil
}

// parseToolCalls turns InferenceResponse.ToolCalls ([][]byte) into []*ToolCall.
// Note: orla actually returns type ToolCallWithID in InferenceResponse.ToolCalls
// (see orla/internal/model/types.go:ToolCallWithID) but we convert it to the
// client-friendly ToolCall here.
func parseToolCalls(raw [][]byte) ([]*ToolCall, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	out := make([]*ToolCall, 0, len(raw))

	for i, v := range raw {
		tc, err := toolCallWithIDFromJSON(v)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal tool_call[%d]: %w", i, err)
		}

		toolCall, err := tc.toToolCall()
		if err != nil {
			return nil, fmt.Errorf("failed to convert tool_call[%d] to ToolCall: %w", i, err)
		}

		out = append(out, toolCall)
	}
	return out, nil
}

// ToMessage returns a tool-result message to append to the conversation.
func (r ToolResult) ToMessage() Message {
	content := ""
	if r.IsError && r.Error != "" {
		content = r.Error
	} else if len(r.OutputValues) > 0 {
		b, _ := json.Marshal(r.OutputValues)
		content = string(b)
	}
	return Message{
		Role:       "tool",
		Content:    content,
		ToolCallID: r.ID,
		ToolName:   r.Name,
	}
}
