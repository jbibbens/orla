package serving

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/sashabaranov/go-openai"

	"github.com/dorcha-inc/orla/internal/core"
	"github.com/dorcha-inc/orla/internal/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLayer_NewLayer(t *testing.T) {
	layer := NewAgenticLayer()
	require.NotNil(t, layer)
	assert.Empty(t, layer.ListLLMBackends())
}

func TestLayer_AddServer(t *testing.T) {
	layer := NewAgenticLayer()
	layer.AddLLMBackend("test-server", &core.LLMBackend{
		Type:     core.LLMInferenceAPITypeOpenAI,
		Endpoint: "http://localhost:11434/v1",
	}, "openai:test-model")
	assert.Contains(t, layer.ListLLMBackends(), "test-server")
}

func TestLayer_Execute_WithMaxTokens(t *testing.T) {
	srv := model.NewMockLLMServer().ReturnContent("test response").Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "test-key")
	layer := NewAgenticLayer()
	layer.AddLLMBackend("test-server", &core.LLMBackend{
		Type:         core.LLMInferenceAPITypeOpenAI,
		Endpoint:     srv.URL() + "/v1",
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}, "openai:test-model")

	response, err := layer.Execute(context.Background(), "test-server", "test", []model.Message{
		{Role: model.MessageRoleUser, Content: "test prompt"},
	}, nil, model.InferenceOptions{MaxTokens: core.IntPtr(42)})
	require.NoError(t, err)
	assert.Equal(t, "test response", response.Content)

	var req openai.ChatCompletionRequest
	require.NoError(t, json.Unmarshal(srv.LastRequestBody(), &req))
	assert.Equal(t, 42, req.MaxTokens)
}

func TestLayer_Execute_WithoutMaxTokens(t *testing.T) {
	srv := model.NewMockLLMServer().ReturnContent("test response").Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "test-key")
	layer := NewAgenticLayer()
	layer.AddLLMBackend("test-server", &core.LLMBackend{
		Type:         core.LLMInferenceAPITypeOpenAI,
		Endpoint:     srv.URL() + "/v1",
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}, "openai:test-model")

	response, err := layer.Execute(context.Background(), "test-server", "test", []model.Message{
		{Role: model.MessageRoleUser, Content: "test prompt"},
	}, nil, model.InferenceOptions{})
	require.NoError(t, err)
	assert.Equal(t, "test response", response.Content)

	var req openai.ChatCompletionRequest
	require.NoError(t, json.Unmarshal(srv.LastRequestBody(), &req))
	assert.Equal(t, 0, req.MaxTokens)
}

func TestLayer_Execute_ServerNotFound(t *testing.T) {
	layer := NewAgenticLayer()
	_, err := layer.Execute(context.Background(), "nonexistent", "", nil, nil, model.InferenceOptions{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "not found")
}

func TestLayer_Execute_RejectsStream(t *testing.T) {
	srv := model.NewMockLLMServer().ReturnContent("ignored").Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "test-key")
	layer := NewAgenticLayer()
	layer.AddLLMBackend("test-server", &core.LLMBackend{
		Type:         core.LLMInferenceAPITypeOpenAI,
		Endpoint:     srv.URL() + "/v1",
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}, "openai:test-model")

	_, err := layer.Execute(context.Background(), "test-server", "test", []model.Message{
		{Role: model.MessageRoleUser, Content: "test"},
	}, nil, model.InferenceOptions{Stream: true})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "ExecuteStream")
}

func TestLayer_ExecuteStream(t *testing.T) {
	srv := model.NewMockLLMServer().ReturnStreamChunks([]string{"test ", "response"}).Start()
	t.Cleanup(srv.Close)

	t.Setenv("ORLA_TEST_OPENAI_KEY", "test-key")
	layer := NewAgenticLayer()
	layer.AddLLMBackend("test-server", &core.LLMBackend{
		Type:         core.LLMInferenceAPITypeOpenAI,
		Endpoint:     srv.URL() + "/v1",
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}, "openai:test-model")

	response, ch, err := layer.ExecuteStream(context.Background(), "test-server", "test", []model.Message{
		{Role: model.MessageRoleUser, Content: "test"},
	}, nil, model.InferenceOptions{Stream: true, MaxTokens: core.IntPtr(10)})
	require.NoError(t, err)
	require.NotNil(t, ch)
	for range ch {
	}
	assert.Equal(t, "test response", response.Content)
}
