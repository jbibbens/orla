package orla

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewStage(t *testing.T) {
	backend := &LLMBackend{Name: "b", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "m"}
	s := NewStage("my_stage", backend)
	require.NotNil(t, s)
	assert.NotEmpty(t, s.ID)
	assert.Equal(t, "my_stage", s.Name)
	assert.Equal(t, backend, s.Backend)
	assert.Equal(t, ExecutionModeSingleShot, s.ExecutionMode)
	assert.Nil(t, s.MaxTokens)
	assert.Nil(t, s.Temperature)
	assert.Nil(t, s.TopP)
	assert.Nil(t, s.ResponseFormat)
	assert.NotNil(t, s.Tools)
	assert.Empty(t, s.Tools)
}

func TestStage_Setters(t *testing.T) {
	s := NewStage("s", &LLMBackend{Name: "b", Endpoint: "http://x", Type: "openai", ModelID: "m"})

	s.SetMaxTokens(100)
	require.NotNil(t, s.MaxTokens)
	assert.Equal(t, 100, *s.MaxTokens)

	s.SetTemperature(0.5)
	require.NotNil(t, s.Temperature)
	assert.Equal(t, 0.5, *s.Temperature)

	s.SetTopP(0.9)
	require.NotNil(t, s.TopP)
	assert.Equal(t, 0.9, *s.TopP)

	rf := &StructuredOutputRequest{Name: "schema", Schema: map[string]any{"type": "object"}}
	s.SetResponseFormat(rf)
	assert.Equal(t, rf, s.ResponseFormat)

	s.SetExecutionMode(ExecutionModeAgentLoop)
	assert.Equal(t, ExecutionModeAgentLoop, s.ExecutionMode)

	s.SetMaxTurns(50)
	assert.Equal(t, 50, s.MaxTurns)

	s.SetSchedulingPolicy(SchedulingPolicyPriority)
	assert.Equal(t, SchedulingPolicyPriority, s.StageSchedulingPolicy)

	s.SetRequestSchedulingPolicy("custom")
	assert.Equal(t, "custom", s.RequestSchedulingPolicy)
}

func TestStage_AddTool_success(t *testing.T) {
	s := NewStage("s", &LLMBackend{Name: "b", Endpoint: "http://x", Type: "openai", ModelID: "m"})
	tool, err := NewTool("t1", "desc", nil, nil, func(ctx context.Context, in ToolSchema) (*ToolResult, error) {
		return &ToolResult{OutputValues: in}, nil
	})
	require.NoError(t, err)

	err = s.AddTool(tool)
	require.NoError(t, err)
	assert.Len(t, s.Tools, 1)
	assert.Equal(t, tool, s.Tools["t1"])
}

func TestStage_AddTool_nilReturnsError(t *testing.T) {
	s := NewStage("s", &LLMBackend{Name: "b", Endpoint: "http://x", Type: "openai", ModelID: "m"})
	err := s.AddTool(nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cannot be nil")
	assert.Empty(t, s.Tools)
}

func TestStage_ID_isUnique(t *testing.T) {
	backend := &LLMBackend{Name: "b", Endpoint: "http://x", Type: "openai", ModelID: "m"}
	s1 := NewStage("a", backend)
	s2 := NewStage("a", backend)
	assert.NotEqual(t, s1.ID, s2.ID)
}

func TestOneBitStageMapper_MapStage_returnsStageOneWhenTrue(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: `{"prediction":true}`},
		})
	}))
	defer server.Close()

	client := NewOrlaClient(server.URL)
	backend := &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"}
	stageOne := NewStage("one", backend)
	stageTwo := NewStage("two", backend)
	mapper := NewOneBitStageMapper(client, backend, stageOne, stageTwo)

	got, err := mapper.MapStage(context.Background(), "prompt")
	require.NoError(t, err)
	assert.Same(t, stageOne, got)
}

func TestOneBitStageMapper_MapStage_returnsStageTwoWhenFalse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: `{"prediction":false}`},
		})
	}))
	defer server.Close()

	client := NewOrlaClient(server.URL)
	backend := &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"}
	stageOne := NewStage("one", backend)
	stageTwo := NewStage("two", backend)
	mapper := NewOneBitStageMapper(client, backend, stageOne, stageTwo)

	got, err := mapper.MapStage(context.Background(), "prompt")
	require.NoError(t, err)
	assert.Same(t, stageTwo, got)
}

func TestOneBitStageMapper_MapStage_predictError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	client := NewOrlaClient(server.URL)
	backend := &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"}
	stageOne := NewStage("one", backend)
	stageTwo := NewStage("two", backend)
	mapper := NewOneBitStageMapper(client, backend, stageOne, stageTwo)

	got, err := mapper.MapStage(context.Background(), "prompt")
	require.Error(t, err)
	assert.Nil(t, got)
	assert.Contains(t, err.Error(), "failed to predict stage")
}

func TestOneBitStageMapper_ImplementsStageMapper(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: `{"prediction":true}`},
		})
	}))
	defer server.Close()

	client := NewOrlaClient(server.URL)
	backend := &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"}
	stageOne := NewStage("one", backend)
	stageTwo := NewStage("two", backend)
	var mapper StageMapper = NewOneBitStageMapper(client, backend, stageOne, stageTwo)

	got, err := mapper.MapStage(context.Background(), "prompt")
	require.NoError(t, err)
	assert.Same(t, stageOne, got)
}

func TestThresholdStageMapper_RoutesByPromptLengthDefault(t *testing.T) {
	backend := &LLMBackend{Name: "b", Endpoint: "http://x", Type: "openai", ModelID: "m"}
	low := NewStage("low", backend)
	high := NewStage("high", backend)

	mapper := NewThresholdStageMapper(10, low, high, nil)
	gotShort, err := mapper.MapStage(context.Background(), "short")
	require.NoError(t, err)
	assert.Same(t, low, gotShort)

	gotLong, err := mapper.MapStage(context.Background(), "this is definitely long")
	require.NoError(t, err)
	assert.Same(t, high, gotLong)
}
