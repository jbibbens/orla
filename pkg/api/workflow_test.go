package orla

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestWorkflow_AgentDAG(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ExecuteRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		content := req.Prompt
		if len(req.Messages) > 0 {
			content = req.Messages[len(req.Messages)-1].Content
		}
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: content},
		})
	}))
	defer server.Close()

	client := NewOrlaClient(server.URL)
	backend := &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "openai:test"}

	agent1 := NewAgent(client)
	agent1.Name = "agent1"
	s1 := NewStage("classify", backend)
	s1.Prompt = "classify task"
	require.NoError(t, agent1.AddStage(s1))

	agent2 := NewAgent(client)
	agent2.Name = "agent2"
	s2 := NewStage("generate", backend)
	s2.Prompt = "generate code"
	require.NoError(t, agent2.AddStage(s2))

	wf := NewWorkflow()
	require.NoError(t, wf.AddAgent(agent1))
	require.NoError(t, wf.AddAgent(agent2))
	require.NoError(t, wf.AddDependency("agent2", "agent1"))

	results, err := wf.Execute(context.Background())
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.NotNil(t, results["agent1"])
	assert.NotNil(t, results["agent2"])
}

func TestWorkflow_AgentCycle(t *testing.T) {
	client := NewOrlaClient("http://example.com")
	backend := &LLMBackend{Name: "b", Endpoint: "http://x", Type: "openai", ModelID: "openai:test"}

	a1 := NewAgent(client)
	a1.Name = "a1"
	s1 := NewStage("s", backend)
	s1.Prompt = "p"
	require.NoError(t, a1.AddStage(s1))

	a2 := NewAgent(client)
	a2.Name = "a2"
	s2 := NewStage("s", backend)
	s2.Prompt = "p"
	require.NoError(t, a2.AddStage(s2))

	wf := NewWorkflow()
	require.NoError(t, wf.AddAgent(a1))
	require.NoError(t, wf.AddAgent(a2))
	require.NoError(t, wf.AddDependency("a1", "a2"))
	require.NoError(t, wf.AddDependency("a2", "a1"))

	_, err := wf.Execute(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "cycle")
}
