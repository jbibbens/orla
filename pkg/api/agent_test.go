package orla

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStage_buildRequest(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "test", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "model"}
	s := NewStage("test", backend)
	s.Client = client

	r, err := s.buildRequest("hello")
	require.NoError(t, err)
	require.NotNil(t, r)
	assert.Equal(t, "test", r.Backend)
	assert.Equal(t, "hello", r.Prompt)
	assert.Equal(t, s.ID, r.StageID)
	assert.Zero(t, r.MaxTokens)

	s.SetMaxTokens(100)
	r, err = s.buildRequest("world")
	require.NoError(t, err)
	require.NotNil(t, r.MaxTokens)
	assert.Equal(t, 100, *r.MaxTokens)
	assert.Equal(t, "world", r.Prompt)

	s.SetResponseFormat(&StructuredOutputRequest{
		Name:   "out",
		Strict: true,
		Schema: map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}},
	})
	r, err = s.buildRequest("structured")
	require.NoError(t, err)
	require.NotNil(t, r.ResponseFormat)
	assert.Equal(t, "out", r.ResponseFormat.Name)
	assert.True(t, r.ResponseFormat.Strict)
}

func TestStage_buildRequest_nilBackendReturnsError(t *testing.T) {
	s := &Stage{Name: "s", Tools: make(map[string]*Tool)}
	r, err := s.buildRequest("hello")
	require.Error(t, err)
	assert.Nil(t, r)
	assert.Contains(t, err.Error(), "backend is nil")
}

func TestStage_buildRequestWithMessages_includesResponseFormat(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "b", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "m"}
	s := NewStage("b", backend)
	s.Client = client
	s.SetResponseFormat(&StructuredOutputRequest{Name: "schema", Strict: true, Schema: map[string]any{"type": "object"}})

	r, err := s.buildRequestWithMessages([]Message{{Role: "user", Content: "hi"}})
	require.NoError(t, err)
	require.NotNil(t, r.ResponseFormat)
	assert.Equal(t, "schema", r.ResponseFormat.Name)
}

func TestStage_buildRequest_setsStageID(t *testing.T) {
	backend := &LLMBackend{Name: "b", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "m"}
	s := NewStage("stage_a", backend)

	r, err := s.buildRequest("hello")
	require.NoError(t, err)
	assert.Equal(t, s.ID, r.StageID)
	assert.NotEmpty(t, r.StageID)
}

func TestAgent_AddStage_and_AddDependency(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "b", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "m"}
	a := NewAgent(client)

	s1 := NewStage("s1", backend)
	s2 := NewStage("s2", backend)

	require.NoError(t, a.AddStage(s1))
	require.NoError(t, a.AddStage(s2))
	require.NoError(t, a.AddDependency(s2.ID, s1.ID))

	assert.Len(t, a.Stages(), 2)
	assert.Same(t, client, s1.Client)
	assert.Same(t, client, s2.Client)
}

func TestAgent_AddStage_duplicateReturnsError(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "b", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "m"}
	a := NewAgent(client)
	s := NewStage("s", backend)
	require.NoError(t, a.AddStage(s))
	err := a.AddStage(s)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already exists")
}
