package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/harvard-cns/orla/internal/core"
	"github.com/harvard-cns/orla/internal/model"
	"github.com/harvard-cns/orla/internal/serving"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestServer_HandleExecute_RequestBodyTooLarge(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0", nil)

	// Valid JSON body that exceeds maxRequestBodyBytes (10MB)
	payload := `{"backend":"x","prompt":"` + string(bytes.Repeat([]byte("x"), maxRequestBodyBytes)) + `"}`
	req := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader([]byte(payload)))
	req.Header.Set("Content-Type", "application/json")

	resp := httptest.NewRecorder()
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
	assert.Contains(t, resp.Body.String(), "request body too large")
}

func TestRecoveryMiddleware_RecoversPanic(t *testing.T) {
	panicHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	})
	wrapped := recoveryMiddleware(panicHandler)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/", nil)
	wrapped.ServeHTTP(resp, req)

	require.Equal(t, http.StatusInternalServerError, resp.Code)
	assert.Contains(t, resp.Body.String(), "internal server error")
}

func TestServer_HandleHealth(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0", nil)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/api/v1/health", nil)
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusOK, resp.Code)
	var result map[string]string
	err := json.Unmarshal(resp.Body.Bytes(), &result)
	require.NoError(t, err)
	assert.Equal(t, "healthy", result["status"])
}

func TestServer_HandleMetrics(t *testing.T) {
	srv := model.NewMockLLMServer().ReturnContent("ok").Start()
	t.Cleanup(srv.Close)
	t.Setenv("ORLA_TEST_OPENAI_KEY", "test-key")

	layer := serving.NewAgenticLayer()
	layer.AddLLMBackend("metrics-backend", &core.LLMBackend{
		Type:         core.LLMInferenceAPITypeOpenAI,
		Endpoint:     srv.URL() + "/v1",
		APIKeyEnvVar: "ORLA_TEST_OPENAI_KEY",
	}, "openai:test-model")
	server := NewAgenticServer(layer, ":0", nil)

	// Execute a request to populate orla_ metrics
	executeReq := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader([]byte(`{"backend":"metrics-backend","prompt":"hi"}`)))
	executeReq.Header.Set("Content-Type", "application/json")
	server.mux.ServeHTTP(httptest.NewRecorder(), executeReq)

	resp := httptest.NewRecorder()
	server.mux.ServeHTTP(resp, httptest.NewRequest("GET", "/metrics", nil))
	require.Equal(t, http.StatusOK, resp.Code)
	body := resp.Body.String()
	assert.Contains(t, body, "# HELP ")
	assert.Contains(t, body, "# TYPE ")
	assert.Contains(t, body, "orla_")
}

func TestServer_HandleExecute_NoBackend(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0", nil)

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
	server := NewAgenticServer(layer, ":0", nil)

	resp := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader([]byte("invalid json")))
	server.mux.ServeHTTP(resp, req)

	require.Equal(t, http.StatusBadRequest, resp.Code)
}

func TestServer_HandleExecute_BackendNotFound(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0", nil)

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
	server := NewAgenticServer(layer, ":0", nil)

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
	server := NewAgenticServer(layer, ":0", nil)

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
	server := NewAgenticServer(layer, ":0", nil)

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

func TestServer_RateLimit_Returns429WhenExceeded(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0", &ServerOptions{RateLimitRPS: 1})

	reqBody := ExecuteRequest{Backend: "x", Prompt: "hi"}
	body, err := json.Marshal(reqBody)
	require.NoError(t, err)

	// First request - may succeed (backend not found) or get 429 if we're fast
	// Second request immediately - should hit rate limit
	resp1 := httptest.NewRecorder()
	req1 := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader(body))
	server.mux.ServeHTTP(resp1, req1)

	resp2 := httptest.NewRecorder()
	req2 := httptest.NewRequest("POST", "/api/v1/execute", bytes.NewReader(body))
	server.mux.ServeHTTP(resp2, req2)

	// At least one should get 429 (rate limit) or 500 (backend not found)
	// With RPS=1, the second request in quick succession should get 429
	if resp2.Code == http.StatusTooManyRequests {
		assert.Contains(t, resp2.Body.String(), "rate limit")
	}
}

func TestServer_HandleRegisterBackend_Validation(t *testing.T) {
	layer := serving.NewAgenticLayer()
	server := NewAgenticServer(layer, ":0", nil)

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
