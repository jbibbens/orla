// Package agent implements the agent loop and in-process tool execution (no MCP subprocess).
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/dorcha-inc/orla/internal/core"
	"github.com/dorcha-inc/orla/internal/tools"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"go.uber.org/zap"
)

// InProcessToolClient implements MCPClient using a programmatic tool list and the core executor.
// No subprocess or MCP server; tools are executed in-process.
type InProcessToolClient struct {
	registry *tools.ToolsRegistry
	executor *core.OrlaToolExecutor
}

// NewInProcessToolClient creates a client that lists and executes tools from the given registry.
func NewInProcessToolClient(registry *tools.ToolsRegistry, executor *core.OrlaToolExecutor) *InProcessToolClient {
	if registry == nil {
		registry = &tools.ToolsRegistry{Tools: make(map[string]*core.ToolManifest)}
	}
	return &InProcessToolClient{registry: registry, executor: executor}
}

// Ensure InProcessToolClient implements MCPClient
var _ MCPClient = (*InProcessToolClient)(nil)

// ListTools returns MCP tools from the configured tool list.
func (c *InProcessToolClient) ListTools(ctx context.Context) ([]*mcp.Tool, error) {
	list := c.registry.ListTools()
	out := make([]*mcp.Tool, 0, len(list))
	for _, t := range list {
		mcpTool := &mcp.Tool{
			Name:        t.Name,
			Description: t.Description,
		}
		if t.MCP != nil && t.MCP.InputSchema != nil {
			mcpTool.InputSchema = t.MCP.InputSchema
		}
		if t.MCP != nil && t.MCP.OutputSchema != nil {
			mcpTool.OutputSchema = t.MCP.OutputSchema
		}
		out = append(out, mcpTool)
	}
	return out, nil
}

// CallTool executes a tool by name using the core executor and returns an MCP CallToolResult.
func (c *InProcessToolClient) CallTool(ctx context.Context, params *mcp.CallToolParams) (*mcp.CallToolResult, error) {
	tool, err := c.registry.GetTool(params.Name)
	if err != nil {
		return &mcp.CallToolResult{
			IsError: true,
			Content: []mcp.Content{
				&mcp.TextContent{Text: fmt.Sprintf("tool not found: %s", params.Name)},
			},
		}, nil
	}

	// Only simple (on-demand) mode is supported
	if tool.Runtime != nil && tool.Runtime.Mode != "" && tool.Runtime.Mode != core.RuntimeModeSimple {
		return &mcp.CallToolResult{
			IsError: true,
			Content: []mcp.Content{
				&mcp.TextContent{Text: "only simple runtime mode is supported"},
			},
		}, nil
	}

	input, _ := params.Arguments.(map[string]any)
	if input == nil {
		input = make(map[string]any)
	}
	var args []string
	stdin := ""
	for k, v := range input {
		if k == "stdin" {
			if s, ok := v.(string); ok {
				stdin = s
			}
			continue
		}
		argName := strings.ReplaceAll(k, "_", "-")
		args = append(args, fmt.Sprintf("--%s", argName))
		args = append(args, fmt.Sprintf("%v", v))
	}

	start := time.Now()
	result, execErr := c.executor.Execute(ctx, tool, args, stdin)
	duration := time.Since(start).Seconds()
	core.LogToolExecution(tool.Name, duration, execErr)

	if execErr != nil {
		errorMsg := fmt.Sprintf("Tool execution failed: %v", execErr)
		if result != nil && result.Error != nil && strings.Contains(result.Error.Error(), "timed out") {
			errorMsg = fmt.Sprintf("Tool %q timed out", tool.Name)
		}
		return &mcp.CallToolResult{
			IsError: true,
			Content: []mcp.Content{
				&mcp.TextContent{Text: errorMsg},
			},
		}, nil
	}

	var outputSchema map[string]any
	if tool.MCP != nil && tool.MCP.OutputSchema != nil {
		outputSchema = tool.MCP.OutputSchema
	}
	res, _ := buildToolResponseFromResult(tool.Name, result.Stdout, result.Stderr, result.ExitCode, result.Error, outputSchema)
	return res, nil
}

// buildToolResponseFromResult builds an MCP CallToolResult from executor output.
func buildToolResponseFromResult(
	toolName string,
	stdout string,
	stderr string,
	exitCode int,
	execErr error,
	outputSchema map[string]any,
) (*mcp.CallToolResult, map[string]any) {
	content := []mcp.Content{
		&mcp.TextContent{Text: stdout},
	}
	if stderr != "" {
		content = append(content, &mcp.TextContent{Text: fmt.Sprintf("stderr: %s", stderr)})
	}
	if exitCode != 0 {
		content = append(content, &mcp.TextContent{Text: fmt.Sprintf("exit_code: %d", exitCode)})
	}
	isError := execErr != nil || exitCode != 0
	callToolResult := &mcp.CallToolResult{IsError: isError, Content: content}
	outputMap := make(map[string]any)
	if execErr != nil {
		outputMap["error"] = execErr.Error()
	}
	if outputSchema == nil {
		outputMap["stdout"] = stdout
		outputMap["stderr"] = stderr
		outputMap["exit_code"] = exitCode
		return callToolResult, outputMap
	}
	var parsedOutput any
	if err := json.Unmarshal([]byte(stdout), &parsedOutput); err != nil {
		zap.L().Error("Failed to parse tool output as JSON", zap.String("tool", toolName), zap.String("stdout", stdout), zap.Error(err))
		return &mcp.CallToolResult{
			IsError: true,
			Content: []mcp.Content{
				&mcp.TextContent{Text: fmt.Sprintf("Tool output is not valid JSON: %v", err)},
			},
		}, nil
	}
	parsedMap, ok := parsedOutput.(map[string]any)
	if !ok {
		zap.L().Error("Tool output is not a map", zap.String("tool", toolName), zap.String("stdout", stdout))
		return &mcp.CallToolResult{
			IsError: true,
			Content: []mcp.Content{
				&mcp.TextContent{Text: "Tool output is not a JSON object"},
			},
		}, nil
	}
	callToolResult.Content = nil
	return callToolResult, parsedMap
}
