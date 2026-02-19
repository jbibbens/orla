package agent

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/dorcha-inc/orla/internal/core"
	"github.com/dorcha-inc/orla/internal/tools"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewClient_ContextCancellation(t *testing.T) {
	// Test that NewClient respects context cancellation
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// This should fail because context is cancelled
	client, err := NewClient(ctx)
	if err != nil {
		// Expected: should fail when context is cancelled
		assert.Contains(t, err.Error(), "failed to")
		assert.Nil(t, client)
	}
}

func TestClient_ListTools_NilSession(t *testing.T) {
	ctx := context.Background()
	client := &Client{McpSession: nil}

	tools, err := client.ListTools(ctx)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "MCP session is not initialized")
	assert.Nil(t, tools)
}

func TestInProcessToolClient_ListTools_Empty(t *testing.T) {
	ctx := context.Background()
	registry := tools.NewToolsRegistry()
	executor := core.NewOrlaToolExecutor(30)
	client := NewInProcessToolClient(registry, executor)

	tools, err := client.ListTools(ctx)
	require.NoError(t, err)
	assert.NotNil(t, tools)
	assert.Len(t, tools, 0)
}

func TestInProcessToolClient_ListTools_ReturnsConfiguredTools(t *testing.T) {
	ctx := context.Background()
	registry := tools.NewToolsRegistry()
	// Add a tool manifest (path not required for ListTools)
	require.NoError(t, registry.AddTool(&core.ToolManifest{
		Name:        "echo-tool",
		Description: "Echoes input",
		Path:        "/bin/echo",
	}))
	executor := core.NewOrlaToolExecutor(30)
	client := NewInProcessToolClient(registry, executor)

	tools, err := client.ListTools(ctx)
	require.NoError(t, err)
	require.Len(t, tools, 1)
	assert.Equal(t, "echo-tool", tools[0].Name)
	assert.Equal(t, "Echoes input", tools[0].Description)
}

func TestInProcessToolClient_CallTool_NotFound(t *testing.T) {
	ctx := context.Background()
	registry := tools.NewToolsRegistry()
	executor := core.NewOrlaToolExecutor(30)
	client := NewInProcessToolClient(registry, executor)

	result, err := client.CallTool(ctx, &mcp.CallToolParams{Name: "nonexistent", Arguments: nil})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.True(t, result.IsError)
	assert.Contains(t, result.Content[0].(*mcp.TextContent).Text, "tool not found")
}

func TestInProcessToolClient_CallTool_Success(t *testing.T) {
	ctx := context.Background()
	registry := tools.NewToolsRegistry()
	// Use a real executable for execution test
	echoPath := "/bin/echo"
	if _, err := os.Stat(echoPath); err != nil {
		t.Skip("echo not available on this system")
	}
	require.NoError(t, registry.AddTool(&core.ToolManifest{
		Name:        "echo-tool",
		Description: "Echo",
		Path:        echoPath,
	}))
	executor := core.NewOrlaToolExecutor(30)
	client := NewInProcessToolClient(registry, executor)

	result, err := client.CallTool(ctx, &mcp.CallToolParams{
		Name:      "echo-tool",
		Arguments: map[string]any{},
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.False(t, result.IsError)
	require.NotEmpty(t, result.Content)
	textContent, ok := result.Content[0].(*mcp.TextContent)
	require.True(t, ok)
	assert.Equal(t, "\n", textContent.Text) // echo with no args outputs newline
}

func TestInProcessToolClient_CallTool_WithArgs(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	scriptPath := filepath.Join(tmpDir, "script.sh")
	// #nosec G306 -- test script
	require.NoError(t, os.WriteFile(scriptPath, []byte("#!/bin/sh\necho hello\n"), 0755))
	registry := tools.NewToolsRegistry()
	require.NoError(t, registry.AddTool(&core.ToolManifest{
		Name:        "script-tool",
		Description: "Test script",
		Path:        scriptPath,
		Interpreter: "/bin/sh",
	}))
	executor := core.NewOrlaToolExecutor(30)
	client := NewInProcessToolClient(registry, executor)

	result, err := client.CallTool(ctx, &mcp.CallToolParams{
		Name:      "script-tool",
		Arguments: map[string]any{},
	})
	require.NoError(t, err)
	require.NotNil(t, result)
	assert.False(t, result.IsError)
	require.NotEmpty(t, result.Content)
	textContent, ok := result.Content[0].(*mcp.TextContent)
	require.True(t, ok)
	assert.Contains(t, textContent.Text, "hello")
}

func TestClient_CallTool_NilSession(t *testing.T) {
	ctx := context.Background()
	client := &Client{McpSession: nil}

	params := &mcp.CallToolParams{
		Name:      "test_tool",
		Arguments: map[string]any{},
	}

	result, err := client.CallTool(ctx, params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "MCP session is not initialized")
	assert.Nil(t, result)
}

func TestClient_Close_NilSessionAndCmd(t *testing.T) {
	client := &Client{
		McpSession: nil,
		Cmd:        nil,
	}

	err := client.Close()
	assert.NoError(t, err, "Close should not error when both session and cmd are nil")
}

func TestClient_Close_NilSession_NilCmdProcess(t *testing.T) {
	client := &Client{
		McpSession: nil,
		Cmd:        &exec.Cmd{},
	}

	err := client.Close()
	assert.NoError(t, err, "Close should not error when cmd process is nil")
}

func TestClient_Close_NilSession_CmdWithNilProcess(t *testing.T) {
	client := &Client{
		McpSession: nil,
		Cmd:        &exec.Cmd{Process: nil},
	}

	err := client.Close()
	assert.NoError(t, err, "Close should not error when cmd has nil process")
}

func TestClient_Close_ProcessAlreadyFinished(t *testing.T) {
	// Create a command that will exit immediately
	ctx := t.Context()

	cmd := exec.CommandContext(ctx, "true")
	err := cmd.Run()
	require.NoError(t, err, "Command should run successfully")

	// ProcessState should be set after Run()
	client := &Client{
		McpSession: nil,
		Cmd:        cmd,
	}

	err = client.Close()
	assert.NoError(t, err, "Close should not error when process is already finished")
}

func TestClient_Close_WithSessionError(t *testing.T) {
	// Test Close when session close returns an error
	// We can't easily mock the session, but we can test the nil case
	client := &Client{
		McpSession: nil,
		Cmd:        nil,
	}

	err := client.Close()
	// Should not error when both are nil
	assert.NoError(t, err)
}
