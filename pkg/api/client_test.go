package orla

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func encodeJSON(w http.ResponseWriter, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(v) //nolint:errcheck
}

func decodeJSON(r *http.Request, v interface{}) error {
	return json.NewDecoder(r.Body).Decode(v)
}

func TestNewClient(t *testing.T) {
	client := NewClient("http://localhost:8081")
	assert.NotNil(t, client)
	assert.Equal(t, "http://localhost:8081", client.baseURL)
	assert.NotNil(t, client.httpClient)
	assert.Equal(t, time.Duration(0), client.httpClient.Timeout)
}

func TestClient_Health_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/health", r.URL.Path)
		assert.Equal(t, "GET", r.Method)
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	ctx := context.Background()
	err := client.Health(ctx)
	assert.NoError(t, err)
}

func TestClient_Health_NonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	ctx := context.Background()
	err := client.Health(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "status 500")
}

func TestClient_Health_RequestError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	server.Close()

	client := NewClient(server.URL)
	ctx := context.Background()
	err := client.Health(ctx)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to check health")
}

func TestClient_Execute_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/execute", r.URL.Path)
		assert.Equal(t, "POST", r.Method)

		var req ExecuteRequest
		require.NoError(t, decodeJSON(r, &req))
		assert.Equal(t, "my-server", req.Backend)
		assert.Equal(t, "test prompt", req.Prompt)

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

	client := NewClient(server.URL)
	ctx := context.Background()
	resp, err := client.Execute(ctx, &ExecuteRequest{
		Backend: "my-server",
		Prompt: "test prompt",
	})
	require.NoError(t, err)
	assert.Equal(t, "test response", resp.Content)
	assert.Equal(t, "test thinking", resp.Thinking)
}

func TestClient_Execute_ErrorResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		response := ExecuteResponse{
			Success: false,
			Error:   "execution failed",
		}
		encodeJSON(w, response)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	ctx := context.Background()
	_, err := client.Execute(ctx, &ExecuteRequest{
		Backend: "my-server",
		Prompt: "test prompt",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "execution failed")
}

func TestClient_Execute_NonOKStatus(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("bad request")) //nolint:errcheck
	}))
	defer server.Close()

	client := NewClient(server.URL)
	ctx := context.Background()
	_, err := client.Execute(ctx, &ExecuteRequest{
		Backend: "my-server",
		Prompt: "test prompt",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "status 400")
}

func TestClient_Execute_RequestError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	server.Close()

	client := NewClient(server.URL)
	ctx := context.Background()
	_, err := client.Execute(ctx, &ExecuteRequest{
		Backend: "my-server",
		Prompt: "test prompt",
	})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "execute request failed")
}
