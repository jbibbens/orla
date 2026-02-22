package orla

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAgent_req(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "test", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "model"}
	a := NewAgentWithPrompt(client, backend, "hello")
	r := a.req()
	require.NotNil(t, r)
	assert.Equal(t, "test", r.Backend)
	assert.Equal(t, "hello", r.Prompt)
	assert.Zero(t, r.MaxTokens)

	a.SetMaxTokens(100)
	r = a.req()
	assert.Equal(t, 100, r.MaxTokens)
}
