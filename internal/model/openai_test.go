package model

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/sashabaranov/go-openai"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/core"
)

func TestNormalizeSchemaToMap_MapPassthrough(t *testing.T) {
	t.Parallel()

	tool := &mcp.Tool{
		Name:        "hello",
		Description: "desc",
		InputSchema: map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}},
	}

	got, err := normalizeSchemaToMap(tool)
	require.NoError(t, err)
	require.Equal(t, "object", got["type"])
}

func TestNormalizeSchemaToMap_RawMessage(t *testing.T) {
	t.Parallel()

	raw := json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`)
	tool := &mcp.Tool{Name: "raw", InputSchema: raw}

	got, err := normalizeSchemaToMap(tool)
	require.NoError(t, err)
	require.Equal(t, "object", got["type"])
}

func TestNormalizeSchemaToMap_JSONBytes(t *testing.T) {
	t.Parallel()

	b := []byte(`{"type":"object"}`)
	tool := &mcp.Tool{Name: "bytes", InputSchema: b}

	got, err := normalizeSchemaToMap(tool)
	require.NoError(t, err)
	require.Equal(t, "object", got["type"])
}

func TestNormalizeSchemaToMap_MarshalableStruct(t *testing.T) {
	t.Parallel()

	type schema struct {
		Type string `json:"type"`
	}
	tool := &mcp.Tool{Name: "struct", InputSchema: schema{Type: "object"}}

	got, err := normalizeSchemaToMap(tool)
	require.NoError(t, err)
	require.Equal(t, "object", got["type"])
}

func TestNormalizeSchemaToMap_NilSchema(t *testing.T) {
	t.Parallel()

	tool := &mcp.Tool{Name: "nil", InputSchema: nil}
	_, err := normalizeSchemaToMap(tool)
	require.Error(t, err)
}

func TestConvertToolsToOpenAIFormat_UsesNormalizedSchema(t *testing.T) {
	t.Parallel()

	tools := []*mcp.Tool{
		{
			Name:        "t1",
			Description: "d1",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"x":{"type":"string"}}}`),
		},
	}

	got, err := convertToolsToOpenAIFormat(tools)
	require.NoError(t, err)
	require.Len(t, got, 1)
	require.Equal(t, openai.ToolTypeFunction, got[0].Type)
	require.NotNil(t, got[0].Function)
	require.Equal(t, "t1", got[0].Function.Name)

	// Parameters should be a map[string]any after normalization.
	_, ok := got[0].Function.Parameters.(map[string]any)
	require.True(t, ok, "expected parameters to be map[string]any, got %T", got[0].Function.Parameters)
}

func TestConvertMessageToOpenAI_ToolRoleSetsToolCallID(t *testing.T) {
	t.Parallel()

	msg := Message{
		Role:       MessageRoleTool,
		Content:    "result",
		ToolName:   "some_tool",
		ToolCallID: "call_123",
	}

	got := convertMessageToOpenAI(msg)
	require.Equal(t, openai.ChatMessageRoleTool, got.Role)
	require.Equal(t, "call_123", got.ToolCallID)
}

func TestConvertOpenAIToolCalls_ParsesArgumentsAndUsesCallID(t *testing.T) {
	t.Parallel()

	calls := []openai.ToolCall{
		{
			ID:   "abc",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "do",
				Arguments: `{"x":"y"}`,
			},
		},
	}

	got, err := convertOpenAIToolCalls(calls)
	require.NoError(t, err)
	require.Len(t, got, 1)
	require.Equal(t, "abc", got[0].ID)
	require.Equal(t, "do", got[0].McpCallToolParams.Name)

	args, ok := got[0].McpCallToolParams.Arguments.(map[string]any)
	require.True(t, ok, "expected arguments to be map[string]any, got %T", got[0].McpCallToolParams.Arguments)
	require.Equal(t, "y", args["x"])
}

func TestConvertOpenAIToolCalls_InvalidJSONReturnsError(t *testing.T) {
	t.Parallel()

	calls := []openai.ToolCall{
		{
			ID:   "good",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "good_tool",
				Arguments: `{"ok":true}`,
			},
		},
		{
			ID:   "bad",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "bad_tool",
				Arguments: `{"oops":`, // invalid JSON
			},
		},
	}

	got, err := convertOpenAIToolCalls(calls)
	require.Error(t, err)
	require.Nil(t, got)
}

func TestConvertOpenAIToolCalls_EmptyIDReturnsError(t *testing.T) {
	t.Parallel()

	calls := []openai.ToolCall{
		{
			ID:   "",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "tool",
				Arguments: `{}`,
			},
		},
	}

	got, err := convertOpenAIToolCalls(calls)
	require.Error(t, err)
	require.Nil(t, got)
	require.Contains(t, err.Error(), "empty id")
}

func TestGetOpenAICompatibleEndpoint_Validation(t *testing.T) {
	t.Parallel()

	_, _, err := getOpenAICompatibleEndpoint(nil)
	require.Error(t, err)

	_, _, err = getOpenAICompatibleEndpoint(nil)
	require.Error(t, err)

	_, _, err = getOpenAICompatibleEndpoint(
		&core.LLMBackend{
			Endpoint: "http://example",
			Type:     "",
		},
	)
	require.Error(t, err)

	_, _, err = getOpenAICompatibleEndpoint(
		&core.LLMBackend{
			Endpoint: "http://example",
			Type:     core.LLMInferenceAPITypeOpenAI,
		},
	)
	require.NoError(t, err)

	_, _, err = getOpenAICompatibleEndpoint(
		&core.LLMBackend{
			Endpoint: "http://example",
			Type:     "unsupported_backend",
		},
	)
	require.Error(t, err)
}

func TestNewOpenAIProvider_RequiresAPIKeyEnvVarValue(t *testing.T) {
	t.Parallel()

	// API key env var is not set => should error.
	llmBackend := &core.LLMBackend{
		Endpoint:     "http://example",
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	_, err := NewOpenAIProvider("model", llmBackend)
	require.Error(t, err)
}

func TestOpenAIProvider_Chat_NonStreaming_BasicAndToolCalls(t *testing.T) {
	srv := NewMockLLMServer().
		ReturnContent("hello").
		ReturnToolCallWithID("call_abc", "do", `{"x":"y"}`).
		Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "k")
	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("m", llmBackend)
	require.NoError(t, err)
	require.NoError(t, p.EnsureReady(context.Background()))

	resp, ch, err := p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, nil, InferenceOptions{Stream: false})
	require.NoError(t, err)
	require.Nil(t, ch)
	require.NotNil(t, resp)
	require.Equal(t, "hello", resp.Content)
	require.Len(t, resp.ToolCalls, 1)
	require.Equal(t, "call_abc", resp.ToolCalls[0].ID)
	require.Equal(t, "do", resp.ToolCalls[0].McpCallToolParams.Name)
}

func TestOpenAIProvider_Chat_Streaming_Content(t *testing.T) {
	srv := NewMockLLMServer().
		ReturnStreamChunks([]string{"he", "llo"}).
		Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "k")
	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("m", llmBackend)
	require.NoError(t, err)

	resp, ch, err := p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, nil, InferenceOptions{Stream: true})
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, ch)

	// Drain channel to let goroutine finish without leaks.
	for range ch {
	}
	// Response content is built incrementally by streaming handler; should include all chunks.
	require.Equal(t, "hello", resp.Content)
}

func TestOpenAIProvider_Chat_WithMaxTokens(t *testing.T) {
	maxTokens := 100
	srv := NewMockLLMServer().ReturnContent("Short response").Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "test-key")
	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("test-model", llmBackend)
	require.NoError(t, err)
	require.NoError(t, p.EnsureReady(context.Background()))

	resp, ch, err := p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, nil, InferenceOptions{Stream: false, MaxTokens: core.Ptr(maxTokens)})
	require.NoError(t, err)
	require.Nil(t, ch)
	require.NotNil(t, resp)
	require.Equal(t, "Short response", resp.Content)

	var req openai.ChatCompletionRequest
	require.NoError(t, json.Unmarshal(srv.LastRequestBody(), &req))
	require.Equal(t, maxTokens, req.MaxTokens)
}

func TestOpenAIProvider_Chat_WithoutMaxTokens(t *testing.T) {
	srv := NewMockLLMServer().ReturnContent("Response").Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "test-key")
	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("test-model", llmBackend)
	require.NoError(t, err)
	require.NoError(t, p.EnsureReady(context.Background()))

	resp, ch, err := p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, nil, InferenceOptions{Stream: false})
	require.NoError(t, err)
	require.Nil(t, ch)
	require.NotNil(t, resp)
	require.Equal(t, "Response", resp.Content)

	var req openai.ChatCompletionRequest
	require.NoError(t, json.Unmarshal(srv.LastRequestBody(), &req))
	require.Equal(t, 0, req.MaxTokens)
}

func TestOpenAIProvider_Chat_WithMaxTokensZero(t *testing.T) {
	maxTokens := 0
	srv := NewMockLLMServer().ReturnContent("").Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "test-key")
	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("test-model", llmBackend)
	require.NoError(t, err)
	require.NoError(t, p.EnsureReady(context.Background()))

	resp, ch, err := p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, nil, InferenceOptions{Stream: false, MaxTokens: &maxTokens})
	require.NoError(t, err)
	require.Nil(t, ch)
	require.NotNil(t, resp)

	var req openai.ChatCompletionRequest
	require.NoError(t, json.Unmarshal(srv.LastRequestBody(), &req))
	require.Equal(t, 0, req.MaxTokens)
}

func TestOpenAIProvider_Chat_Streaming_WithToolCalls(t *testing.T) {
	srv := NewMockLLMServer().
		ReturnStreamWithToolCalls("hi", mockLLMToolCall{ID: "call_1", Name: "tool", Args: `{"x":"y"}`}).
		Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "k")

	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("m", llmBackend)
	require.NoError(t, err)

	resp, ch, err := p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, nil, InferenceOptions{Stream: true})
	require.NoError(t, err)
	require.NotNil(t, resp)
	require.NotNil(t, ch)

	// Drain channel
	for range ch {
	}

	require.Equal(t, "hi", resp.Content)
	require.Len(t, resp.ToolCalls, 1)
	require.Equal(t, "call_1", resp.ToolCalls[0].ID)
	require.Equal(t, "tool", resp.ToolCalls[0].McpCallToolParams.Name)
}

func TestConvertMessageToOpenAI_ToolRoleMissingToolCallID(t *testing.T) {
	t.Parallel()

	msg := Message{
		Role:     MessageRoleTool,
		Content:  "result",
		ToolName: "some_tool",
		// ToolCallID is empty
	}

	got := convertMessageToOpenAI(msg)
	require.Equal(t, openai.ChatMessageRoleTool, got.Role)
	require.Empty(t, got.ToolCallID)
}

func TestConvertMessageToOpenAI_NonToolRole(t *testing.T) {
	t.Parallel()

	msg := Message{
		Role:    MessageRoleUser,
		Content: "hello",
	}

	got := convertMessageToOpenAI(msg)
	require.Equal(t, openai.ChatMessageRoleUser, got.Role)
	require.Equal(t, "hello", got.Content)
}

func TestOpenAIProvider_Chat_NonStreaming_NoChoices(t *testing.T) {
	srv := NewMockLLMServer().ReturnNoChoices().Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "k")

	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("m", llmBackend)
	require.NoError(t, err)

	_, _, err = p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, nil, InferenceOptions{Stream: false})
	require.Error(t, err)
	require.Contains(t, err.Error(), "no choices")
}

func TestOpenAIProvider_Chat_ToolConversionError(t *testing.T) {
	t.Setenv("ORLA_TEST_OPENAI_KEY", "k")

	llmBackend := &core.LLMBackend{
		Endpoint:     "http://example",
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("m", llmBackend)
	require.NoError(t, err)

	// Tool with nil InputSchema should cause conversion error
	tools := []*mcp.Tool{
		{
			Name:        "bad_tool",
			Description: "desc",
			InputSchema: nil,
		},
	}

	_, _, err = p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, tools, InferenceOptions{Stream: false})
	require.Error(t, err)
	require.Contains(t, err.Error(), "normalize tool input schema")
}

func TestGetOpenAICompatibleEndpoint_MissingEndpoint(t *testing.T) {
	t.Parallel()

	_, _, err := getOpenAICompatibleEndpoint(
		&core.LLMBackend{
			Endpoint: "",
			Type:     core.LLMInferenceAPITypeOpenAI,
		})
	require.Error(t, err)
	require.Contains(t, err.Error(), "endpoint is required")
}

func TestNormalizeSchemaToMap_UnmarshalError(t *testing.T) {
	t.Parallel()

	tool := &mcp.Tool{
		Name:        "bad",
		InputSchema: []byte(`{invalid json}`),
	}

	_, err := normalizeSchemaToMap(tool)
	require.Error(t, err)
	require.Contains(t, err.Error(), "unmarshal")
}

func TestNormalizeSchemaToMap_MarshalError(t *testing.T) {
	t.Parallel()

	// Channel cannot be marshaled to JSON
	tool := &mcp.Tool{
		Name:        "bad",
		InputSchema: make(chan int),
	}

	_, err := normalizeSchemaToMap(tool)
	require.Error(t, err)
	require.Contains(t, err.Error(), "marshal")
}

func TestConvertOpenAIToolCalls_ErrorPropagates(t *testing.T) {
	t.Parallel()

	calls := []openai.ToolCall{
		{
			ID:   "bad",
			Type: openai.ToolTypeFunction,
			Function: openai.FunctionCall{
				Name:      "tool",
				Arguments: `{invalid json}`,
			},
		},
	}

	_, err := convertOpenAIToolCalls(calls)
	require.Error(t, err)
	require.Contains(t, err.Error(), "unmarshal")
}

func TestOpenAIProvider_Chat_WithResponseFormat_NonStreaming(t *testing.T) {
	minimalSchema := json.RawMessage(`{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}`)
	srv := NewMockLLMServer().ReturnContent(`{"answer":"hello"}`).Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "k")
	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("m", llmBackend)
	require.NoError(t, err)

	opts := InferenceOptions{
		Stream: false,
		ResponseFormat: &StructuredOutputOptions{
			Name:   "test-schema",
			Strict: true,
			Schema: minimalSchema,
		},
	}
	resp, ch, err := p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "Say hello in JSON"}}, nil, opts)
	require.NoError(t, err)
	require.Nil(t, ch)
	require.NotNil(t, resp)
	require.Equal(t, `{"answer":"hello"}`, resp.Content)

	var req openai.ChatCompletionRequest
	require.NoError(t, json.Unmarshal(srv.LastRequestBody(), &req))
	require.NotNil(t, req.ResponseFormat, "request should include response_format")
	require.Equal(t, openai.ChatCompletionResponseFormatTypeJSONSchema, req.ResponseFormat.Type)
	require.NotNil(t, req.ResponseFormat.JSONSchema)
	require.Equal(t, "test-schema", req.ResponseFormat.JSONSchema.Name)
	require.True(t, req.ResponseFormat.JSONSchema.Strict)
	require.NotNil(t, req.ResponseFormat.JSONSchema.Schema)

	var parsed map[string]any
	require.NoError(t, json.Unmarshal([]byte(resp.Content), &parsed))
	require.Equal(t, "hello", parsed["answer"])
}

func TestOpenAIProvider_Chat_WithoutResponseFormat_RequestOmitsResponseFormat(t *testing.T) {
	srv := NewMockLLMServer().ReturnContent("plain text").Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "k")
	llmBackend := &core.LLMBackend{
		Endpoint:     srv.URL(),
		Type:         core.LLMInferenceAPITypeOpenAI,
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}

	p, err := NewOpenAIProvider("m", llmBackend)
	require.NoError(t, err)

	resp, ch, err := p.Chat(context.Background(), []Message{{Role: MessageRoleUser, Content: "hi"}}, nil, InferenceOptions{Stream: false})
	require.NoError(t, err)
	require.Nil(t, ch)
	require.NotNil(t, resp)
	require.Equal(t, "plain text", resp.Content)

	var req openai.ChatCompletionRequest
	require.NoError(t, json.Unmarshal(srv.LastRequestBody(), &req))
	require.Nil(t, req.ResponseFormat, "request should not include response_format when opts.ResponseFormat is nil")
}
