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

func TestAgent_ExecuteDAG_SingleShotLinear(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ExecuteRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: req.Prompt},
		})
	}))
	defer server.Close()

	client := NewOrlaClient(server.URL)
	backend := &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "openai:test"}

	agent := NewAgent(client)
	agent.Name = "test"

	s1 := NewStage("step1", backend)
	s1.Prompt = "first"
	require.NoError(t, agent.AddStage(s1))

	s2 := NewStage("step2", backend)
	s2.PromptBuilder = func(results map[string]*StageResult) (string, error) {
		return results[s1.ID].Response.Content + "+second", nil
	}
	require.NoError(t, agent.AddStage(s2))
	require.NoError(t, agent.AddDependency(s2.ID, s1.ID))

	results, err := agent.ExecuteDAG(context.Background())
	require.NoError(t, err)
	require.Len(t, results, 2)
	assert.Equal(t, "first", results[s1.ID].Response.Content)
	assert.Equal(t, "first+second", results[s2.ID].Response.Content)
}

func TestAgent_ExecuteDAG_FanOut(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ExecuteRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: req.Prompt},
		})
	}))
	defer server.Close()

	client := NewOrlaClient(server.URL)
	backend := &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "openai:test"}

	agent := NewAgent(client)
	agent.Name = "fanout"

	root := NewStage("root", backend)
	root.Prompt = "root"
	require.NoError(t, agent.AddStage(root))

	branchA := NewStage("branchA", backend)
	branchA.Prompt = "A"
	require.NoError(t, agent.AddStage(branchA))
	require.NoError(t, agent.AddDependency(branchA.ID, root.ID))

	branchB := NewStage("branchB", backend)
	branchB.Prompt = "B"
	require.NoError(t, agent.AddStage(branchB))
	require.NoError(t, agent.AddDependency(branchB.ID, root.ID))

	results, err := agent.ExecuteDAG(context.Background())
	require.NoError(t, err)
	require.Len(t, results, 3)
	assert.Equal(t, "root", results[root.ID].Response.Content)
	assert.Equal(t, "A", results[branchA.ID].Response.Content)
	assert.Equal(t, "B", results[branchB.ID].Response.Content)
}

func TestAgent_ExecuteDAG_NoStagesReturnsError(t *testing.T) {
	agent := NewAgent(NewOrlaClient("http://x"))
	agent.Name = "empty"
	_, err := agent.ExecuteDAG(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no stages")
}

func TestAgent_ExecuteDAG_AgentLoopMode(t *testing.T) {
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: "done"},
		})
	}))
	defer server.Close()

	client := NewOrlaClient(server.URL)
	backend := &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "openai:test"}

	agent := NewAgent(client)
	agent.Name = "agent-loop"

	s := NewStage("loop", backend)
	s.Prompt = "do something"
	s.ExecutionMode = ExecutionModeAgentLoop
	s.MaxTurns = 3
	require.NoError(t, agent.AddStage(s))

	results, err := agent.ExecuteDAG(context.Background())
	require.NoError(t, err)
	require.NotNil(t, results[s.ID])
	assert.Equal(t, "done", results[s.ID].Response.Content)
	assert.True(t, len(results[s.ID].Messages) > 0)
	assert.Equal(t, 1, callCount)
}
