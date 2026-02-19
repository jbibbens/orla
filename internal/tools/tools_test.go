package tools

import (
	"testing"

	"github.com/dorcha-inc/orla/internal/core"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNewToolNotFoundError tests the creation of a ToolNotFoundError
func TestNewToolNotFoundError(t *testing.T) {
	err := NewToolNotFoundError("test-tool")
	require.NotNil(t, err)
	assert.Equal(t, "test-tool", err.Name)
}

// TestToolNotFoundError_Error tests the error message formatting
func TestToolNotFoundError_Error(t *testing.T) {
	err := NewToolNotFoundError("test-tool")
	require.NotNil(t, err)
	assert.Equal(t, "tool not found: test-tool", err.Error())
}

// TestNewToolsRegistry tests the creation of a new tools registry
func TestNewToolsRegistry(t *testing.T) {
	registry := NewToolsRegistry()
	require.NotNil(t, registry)
	assert.NotNil(t, registry.Tools)
	assert.Equal(t, 0, len(registry.Tools))
}

// TestToolsRegistry_AddTool tests adding a tool to the registry
func TestToolsRegistry_AddTool(t *testing.T) {
	registry := NewToolsRegistry()

	tool := &core.ToolManifest{
		Name:        "test-tool",
		Description: "A test tool",
		Path:        "/path/to/tool",
		Interpreter: "/bin/sh",
	}

	err := registry.AddTool(tool)
	require.NoError(t, err)

	// Verify the tool was added
	retrievedTool, err := registry.GetTool("test-tool")
	require.NoError(t, err)
	assert.Equal(t, tool, retrievedTool)
}

// TestToolsRegistry_AddTool_Duplicate tests that adding a tool with the same name returns an error
func TestToolsRegistry_AddTool_Duplicate(t *testing.T) {
	registry := NewToolsRegistry()

	tool1 := &core.ToolManifest{
		Name:        "test-tool",
		Description: "First tool",
		Path:        "/path/to/tool1",
		Interpreter: "",
	}

	tool2 := &core.ToolManifest{
		Name:        "test-tool",
		Description: "Second tool",
		Path:        "/path/to/tool2",
		Interpreter: "/bin/sh",
	}

	err := registry.AddTool(tool1)
	require.NoError(t, err)

	// Adding a duplicate tool should return an error
	err = registry.AddTool(tool2)
	assert.Error(t, err)
	var duplicateErr *DuplicateToolNameError
	assert.ErrorAs(t, err, &duplicateErr)
	assert.Equal(t, "test-tool", duplicateErr.Name)

	// Verify the first tool is still in the registry
	retrievedTool, err := registry.GetTool("test-tool")
	require.NoError(t, err)
	assert.Equal(t, tool1, retrievedTool)
	assert.Equal(t, "First tool", retrievedTool.Description)
}

// TestToolsRegistry_GetTool tests retrieving a tool from the registry
func TestToolsRegistry_GetTool(t *testing.T) {
	registry := NewToolsRegistry()

	tool := &core.ToolManifest{
		Name:        "test-tool",
		Description: "A test tool",
		Path:        "/path/to/tool",
		Interpreter: "",
	}

	err := registry.AddTool(tool)
	require.NoError(t, err)

	// Test successful retrieval
	retrievedTool, err := registry.GetTool("test-tool")
	require.NoError(t, err)
	assert.Equal(t, tool, retrievedTool)

	// Test retrieval of non-existent tool
	_, err = registry.GetTool("nonexistent")
	assert.Error(t, err)
	var toolNotFoundErr *ToolNotFoundError
	assert.ErrorAs(t, err, &toolNotFoundErr)
	assert.Equal(t, "nonexistent", toolNotFoundErr.Name)
}

// TestToolsRegistry_ListTools tests listing all tools in the registry
func TestToolsRegistry_ListTools(t *testing.T) {
	registry := NewToolsRegistry()

	// Initially empty
	tools := registry.ListTools()
	assert.Equal(t, 0, len(tools))

	// Add some tools
	tool1 := &core.ToolManifest{
		Name:        "tool1",
		Description: "First tool",
		Path:        "/path/to/tool1",
		Interpreter: "",
	}

	tool2 := &core.ToolManifest{
		Name:        "tool2",
		Description: "Second tool",
		Path:        "/path/to/tool2",
		Interpreter: "/bin/sh",
	}

	tool3 := &core.ToolManifest{
		Name:        "tool3",
		Description: "Third tool",
		Path:        "/path/to/tool3",
		Interpreter: "/usr/bin/python3",
	}

	err := registry.AddTool(tool1)
	require.NoError(t, err)
	err = registry.AddTool(tool2)
	require.NoError(t, err)
	err = registry.AddTool(tool3)
	require.NoError(t, err)

	// List tools
	tools = registry.ListTools()
	assert.Equal(t, 3, len(tools))

	// Verify all tools are present (order may vary)
	toolMap := make(map[string]*core.ToolManifest)
	for _, tool := range tools {
		toolMap[tool.Name] = tool
	}

	assert.Contains(t, toolMap, "tool1")
	assert.Contains(t, toolMap, "tool2")
	assert.Contains(t, toolMap, "tool3")
	assert.Equal(t, tool1, toolMap["tool1"])
	assert.Equal(t, tool2, toolMap["tool2"])
	assert.Equal(t, tool3, toolMap["tool3"])
}

// TestDuplicateToolNameError_Error tests the error message formatting
func TestDuplicateToolNameError_Error(t *testing.T) {
	err := NewDuplicateToolNameError("test-tool")
	require.NotNil(t, err)
	assert.Equal(t, "test-tool", err.Name)
	assert.Equal(t, "tool with name test-tool already exists", err.Error())
}
