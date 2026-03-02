package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/dorcha-inc/orla/internal/model"
	"github.com/dorcha-inc/orla/internal/serving"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServer_HandleHealth(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0")

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/v1/health", nil)
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var result map[string]string
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	assert.Equal(t, "healthy", result["status"])
}

func TestServer_HandleExecute_NoBackend(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0")

	reqBody := ExecuteRequest{
		Prompt: "test prompt",
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader(body))
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestServer_HandleExecute_InvalidJSON(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0")

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader([]byte("invalid json")))
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestServer_HandleExecute_BackendNotFound(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0")

	reqBody := ExecuteRequest{
		Backend: "nonexistent",
		Prompt:  "test prompt",
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader(body))
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)
	var result ExecuteResponse
	err = json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	assert.False(t, result.Success)
	assert.Contains(t, result.Error, "not found")
}

func TestServer_HandleExecute_InvalidSchedulingPolicy(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0")

	reqBody := ExecuteRequest{
		Backend: "nonexistent",
		Prompt:  "test prompt",
		InferenceOptions: model.InferenceOptions{
			SchedulingPolicy: "not_supported",
		},
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader(body))
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestServer_HandleExecute_InvalidRequestSchedulingPolicy(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0")

	reqBody := ExecuteRequest{
		Backend: "nonexistent",
		Prompt:  "test prompt",
		InferenceOptions: model.InferenceOptions{
			RequestSchedulingPolicy: "not_supported",
		},
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader(body))
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestServer_HandleRegisterBackend(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0")

	reqBody := RegisterBackendRequest{
		Name:     "vllm",
		Endpoint: "http://localhost:8000/v1",
		Type:     "openai",
		ModelID:  "openai:Qwen/Qwen3-4B",
	}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/v1/backends", bytes.NewReader(body))
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var result RegisterBackendResponse
	err = json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	assert.True(t, result.Success)

	// List backends
	resp2 := httptest.NewRecorder()
	req2 := httptest.NewRequest("GET", "/api/v1/backends", nil)
	server.mux.ServeHTTP(resp2, req2)
	require.Equal(t, http.StatusOK, resp2.Code)
	var listResult ListBackendsResponse
	err = json.Unmarshal(resp2.Body.Bytes(), &listResult)
	require.NoError(t, err)
	assert.Equal(t, []string{"vllm"}, listResult.Backends)
}

func TestServer_HandleRegisterBackend_Validation(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0")

	tests := []struct {
		name string
		req  RegisterBackendRequest
		want int
	}{
		{"missing name", RegisterBackendRequest{Endpoint: "http://x", Type: "openai", ModelID: "openai:m"}, http.StatusBadRequest},
		{"missing endpoint", RegisterBackendRequest{Name: "v", Type: "openai", ModelID: "openai:m"}, http.StatusBadRequest},
		{"missing type", RegisterBackendRequest{Name: "v", Endpoint: "http://x", ModelID: "openai:m"}, http.StatusBadRequest},
		{"missing model_id", RegisterBackendRequest{Name: "v", Endpoint: "http://x", Type: "openai"}, http.StatusBadRequest},
		{"invalid type", RegisterBackendRequest{Name: "v", Endpoint: "http://x", Type: "invalid", ModelID: "openai:m"}, http.StatusBadRequest},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body, err := json.Marshal(tt.req)
			require.NoError(t, err)
			resp := httptest.NewRecorder()
			req := httptest.NewRequest("POST", "/api/v1/backends", bytes.NewReader(body))
			server.mux.ServeHTTP(resp, req)
			assert.Equal(t, tt.want, resp.Code)
		})
	}
}
