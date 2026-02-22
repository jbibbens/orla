package orla

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// AgentExecutor provides high-level agent execution with tool support.
// The daemon handles inference; the client handles the agent loop with tools.
type AgentExecutor struct {
	client *Client
}

// NewAgentExecutor creates a new agent executor.
func NewAgentExecutor(daemonURL string) *AgentExecutor {
	return &AgentExecutor{
		client: NewClient(daemonURL),
	}
}

// AgentExecuteRequest represents a request to execute an agent.
type AgentExecuteRequest struct {
	Backend   string      `json:"backend"`
	Prompt    string      `json:"prompt"`
	Messages  []Message   `json:"messages,omitempty"`
	Tools     []*mcp.Tool `json:"tools,omitempty"`
	MaxTokens int         `json:"max_tokens,omitempty"`
	Stream    bool        `json:"stream,omitempty"`
}

// Execute runs a single inference call against the named backend.
func (e *AgentExecutor) Execute(ctx context.Context, req *AgentExecuteRequest) (*TaskResponse, error) {
	return e.client.Execute(ctx, &ExecuteRequest{
		Backend:   req.Backend,
		Prompt:    req.Prompt,
		Messages:  req.Messages,
		Tools:     req.Tools,
		MaxTokens: req.MaxTokens,
		Stream:    req.Stream,
	})
}

// ExecuteWithTools executes an agent with tool support, handling the full agent loop.
// The daemon handles inference; the client handles tool execution via MCP.
func (e *AgentExecutor) ExecuteWithTools(
	ctx context.Context,
	backend string,
	prompt string,
	mcpSession *mcp.ClientSession,
	maxIterations int,
	onIteration func(iteration int, response *TaskResponse) error,
) (*TaskResponse, error) {
	if maxIterations <= 0 {
		maxIterations = 10
	}

	listResult, err := mcpSession.ListTools(ctx, &mcp.ListToolsParams{})
	if err != nil {
		return nil, fmt.Errorf("failed to list tools: %w", err)
	}

	validTools := make([]*mcp.Tool, 0, len(listResult.Tools))
	for _, tool := range listResult.Tools {
		if tool != nil && tool.Name != "" {
			validTools = append(validTools, tool)
		}
	}

	var conversation []Message

	for iteration := 0; iteration < maxIterations; iteration++ {
		req := &AgentExecuteRequest{
			Backend:  backend,
			Prompt:   prompt,
			Messages: conversation,
			Tools:    validTools,
		}

		response, err := e.Execute(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("inference failed on iteration %d: %w", iteration+1, err)
		}

		if onIteration != nil {
			if err := onIteration(iteration+1, response); err != nil {
				return nil, fmt.Errorf("iteration callback error: %w", err)
			}
		}

		if response.Content != "" {
			conversation = append(conversation, Message{
				Role:    "assistant",
				Content: response.Content,
			})
		}

		if len(response.ToolCalls) == 0 {
			return response, nil
		}

		for _, toolCall := range response.ToolCalls {
			toolCallMap, ok := toolCall.(map[string]any)
			if !ok {
				continue
			}

			toolName, ok := toolCallMap["name"].(string)
			if !ok || toolName == "" {
				continue
			}

			arguments, ok := toolCallMap["arguments"].(map[string]any)
			if !ok || arguments == nil {
				arguments = make(map[string]any)
			}

			params := &mcp.CallToolParams{
				Name:      toolName,
				Arguments: arguments,
			}
			result, err := mcpSession.CallTool(ctx, params)
			if err != nil {
				return nil, fmt.Errorf("tool call failed for %s: %w", toolName, err)
			}

			var content string
			if result != nil && len(result.Content) > 0 {
				for _, c := range result.Content {
					if textContent, ok := c.(*mcp.TextContent); ok {
						content += textContent.Text
					} else if imageContent, ok := c.(*mcp.ImageContent); ok {
						if len(imageContent.Data) > 0 {
							content += fmt.Sprintf("[Image: %d bytes]", len(imageContent.Data))
						}
					} else {
						if jsonBytes, marshalErr := json.Marshal(c); marshalErr == nil {
							content += string(jsonBytes)
						}
					}
				}
			}

			conversation = append(conversation, Message{
				Role:    "tool",
				Content: content,
			})
		}

		prompt = ""
	}

	return nil, fmt.Errorf("max iterations (%d) reached", maxIterations)
}
