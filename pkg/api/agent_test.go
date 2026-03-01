package orla

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgent_req(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "test", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "model"}
	a := NewAgent(client)
	a.SetStage(NewAgentStage("test", backend))

	r, err := a.req("hello")
	require.NoError(t, err)
	require.NotNil(t, r)
	assert.Equal(t, "test", r.Backend)
	assert.Equal(t, "hello", r.Prompt)
	assert.Zero(t, r.MaxTokens)

	a.Stage.SetMaxTokens(100)
	r, err = a.req("world")
	require.NoError(t, err)
	require.NotNil(t, r.MaxTokens)
	assert.Equal(t, 100, *r.MaxTokens)
	assert.Equal(t, "world", r.Prompt)

	// ResponseFormat is passed through when set on stage
	a.Stage.SetResponseFormat(&StructuredOutputRequest{
		Name:   "out",
		Strict: true,
		Schema: map[string]any{"type": "object", "properties": map[string]any{"x": map[string]any{"type": "string"}}},
	})
	r, err = a.req("structured")
	require.NoError(t, err)
	require.NotNil(t, r.ResponseFormat)
	assert.Equal(t, "out", r.ResponseFormat.Name)
	assert.True(t, r.ResponseFormat.Strict)
}

func TestAgent_req_nilStageReturnsError(t *testing.T) {
	a := NewAgent(NewOrlaClient("http://localhost:8081"))
	r, err := a.req("hello")
	require.Error(t, err)
	assert.Nil(t, r)
	assert.Contains(t, err.Error(), "stage is nil")
}

func TestAgent_reqWithMessages_includesResponseFormat(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "b", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "m"}
	a := NewAgent(client)
	a.SetStage(NewAgentStage("b", backend))
	a.Stage.SetResponseFormat(&StructuredOutputRequest{Name: "schema", Strict: true, Schema: map[string]any{"type": "object"}})

	r, err := a.reqWithMessages([]Message{{Role: "user", Content: "hi"}})
	require.NoError(t, err)
	require.NotNil(t, r.ResponseFormat)
	assert.Equal(t, "schema", r.ResponseFormat.Name)
}
