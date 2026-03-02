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

func TestNewOneBitPredictor(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "b", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "m"}
	p := NewOneBitPredictor(client, backend)
	require.NotNil(t, p)
	require.NotNil(t, p.Stage)
	require.NotNil(t, p.Stage.ResponseFormat)
	assert.Equal(t, oneBitPredictorName, p.Stage.ResponseFormat.Name)
	assert.Equal(t, oneBitPredictorSchema, p.Stage.ResponseFormat.Schema)
}

func TestOneBitPredictor_Predict_True(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/api/v1/execute", r.URL.Path)
		encodeExecuteResponse(w, ExecuteResponse{
			Success: true,
			Response: &InferenceResponse{
				Content: `{"prediction":true}`,
			},
		})
	}))
	defer server.Close()

	p := NewOneBitPredictor(NewOrlaClient(server.URL), &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"})
	got, err := p.Predict(context.Background(), "prompt")
	require.NoError(t, err)
	assert.True(t, got)
}

func TestOneBitPredictor_Predict_False(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: `{"prediction":false}`},
		})
	}))
	defer server.Close()

	p := NewOneBitPredictor(NewOrlaClient(server.URL), &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"})
	got, err := p.Predict(context.Background(), "prompt")
	require.NoError(t, err)
	assert.False(t, got)
}

func TestOneBitPredictor_Predict_InvalidJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encodeExecuteResponse(w, ExecuteResponse{
			Success:  true,
			Response: &InferenceResponse{Content: `not json`},
		})
	}))
	defer server.Close()

	p := NewOneBitPredictor(NewOrlaClient(server.URL), &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"})
	_, err := p.Predict(context.Background(), "prompt")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "unmarshal")
	assert.Contains(t, err.Error(), "response:")
}

func TestOneBitPredictor_Predict_ExecuteError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	p := NewOneBitPredictor(NewOrlaClient(server.URL), &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"})
	_, err := p.Predict(context.Background(), "prompt")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to execute")
}

func TestOneBitPredictor_Predict_ExecuteSuccessFalse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		encodeExecuteResponse(w, ExecuteResponse{
			Success: false,
			Error:   "model error",
		})
	}))
	defer server.Close()

	p := NewOneBitPredictor(NewOrlaClient(server.URL), &LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"})
	_, err := p.Predict(context.Background(), "prompt")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to execute")
}

func encodeExecuteResponse(w http.ResponseWriter, resp ExecuteResponse) {
	w.Header().Set("Content-Type", "application/json")
	err := json.NewEncoder(w).Encode(resp)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
