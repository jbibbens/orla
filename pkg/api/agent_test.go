package orla

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewAgentExecutor(t *testing.T) {
	executor := NewAgentExecutor("http://localhost:8081")
	assert.NotNil(t, executor)
	assert.NotNil(t, executor.client)
	assert.Equal(t, "http://localhost:8081", executor.client.baseURL)
}

func TestAgentExecutor_Execute_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/execute", r.URL.Path)
		assert.Equal(t, "POST", r.Method)

		var req ExecuteRequest
		require.NoError(t, decodeJSON(r, &req))
		assert.Equal(t, "my-server", req.Backend)

		response := ExecuteResponse{
			Success: true,
			Response: &TaskResponse{
				Content:  "test response",
				Thinking: "test thinking",
			},
		}
		encodeJSON(w, response)
	}))
	defer server.Close()

	executor := NewAgentExecutor(server.URL)
	ctx := context.Background()
	req := &AgentExecuteRequest{
		Backend:   "my-server",
		Prompt:    "test prompt",
		MaxTokens: 100,
	}

	taskResp, err := executor.Execute(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, taskResp)
	assert.Equal(t, "test response", taskResp.Content)
	assert.Equal(t, "test thinking", taskResp.Thinking)
}

func TestAgentExecutor_Execute_WithMessages(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req ExecuteRequest
		_ = decodeJSON(r, &req) //nolint:errcheck
		assert.Len(t, req.Messages, 2)

		response := ExecuteResponse{
			Success: true,
			Response: &TaskResponse{
				Content: "test response",
			},
		}
		encodeJSON(w, response)
	}))
	defer server.Close()

	executor := NewAgentExecutor(server.URL)
	ctx := context.Background()
	req := &AgentExecuteRequest{
		Backend: "my-server",
		Prompt: "test prompt",
		Messages: []Message{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi"},
		},
	}

	taskResp, err := executor.Execute(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, taskResp)
}

func TestAgentExecutor_Execute_WithTools(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := ExecuteResponse{
			Success: true,
			Response: &TaskResponse{
				Content: "test response",
			},
		}
		encodeJSON(w, response)
	}))
	defer server.Close()

	executor := NewAgentExecutor(server.URL)
	ctx := context.Background()
	req := &AgentExecuteRequest{
		Backend: "my-server",
		Prompt: "test prompt",
		Tools: []*mcp.Tool{
			{
				Name:        "test_tool",
				Description: "A test tool",
			},
		},
	}

	taskResp, err := executor.Execute(ctx, req)
	require.NoError(t, err)
	assert.NotNil(t, taskResp)
}

func TestAgentExecutor_Execute_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := ExecuteResponse{
			Success: false,
			Error:   "execution failed",
		}
		encodeJSON(w, response)
	}))
	defer server.Close()

	executor := NewAgentExecutor(server.URL)
	ctx := context.Background()
	req := &AgentExecuteRequest{
		Backend: "my-server",
		Prompt: "test prompt",
	}

	_, err := executor.Execute(ctx, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "execution failed")
}

func TestAgentExecutor_Execute_NonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("bad request")) //nolint:errcheck
	}))
	defer server.Close()

	executor := NewAgentExecutor(server.URL)
	ctx := context.Background()
	req := &AgentExecuteRequest{
		Backend: "my-server",
		Prompt: "test prompt",
	}

	_, err := executor.Execute(ctx, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "status 400")
}
