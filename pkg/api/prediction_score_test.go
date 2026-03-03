package orla

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewScorePredictor(t *testing.T) {
	client := NewOrlaClient("http://localhost:8081")
	backend := &LLMBackend{Name: "b", Endpoint: "http://vllm:8000/v1", Type: "openai", ModelID: "m"}
	p := NewScorePredictor(client, backend)
	require.NotNil(t, p)
	require.NotNil(t, p.Stage)
	require.NotNil(t, p.Stage.ResponseFormat)
	assert.Equal(t, scorePredictorName, p.Stage.ResponseFormat.Name)
}

func TestScorePredictor_Predict(t *testing.T) {
	tests := []struct {
		name       string
		response   string
		wantScore  int
		wantErr    bool
		errContain string
	}{
		{"low score", `{"score":1}`, 1, false, ""},
		{"mid score", `{"score":3}`, 3, false, ""},
		{"high score", `{"score":5}`, 5, false, ""},
		{"clamps below 1", `{"score":0}`, 1, false, ""},
		{"clamps above 5", `{"score":9}`, 5, false, ""},
		{"invalid json", `not json`, 0, true, "unmarshal"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				encodeExecuteResponse(w, ExecuteResponse{
					Success:  true,
					Response: &InferenceResponse{Content: tt.response},
				})
			}))
			defer server.Close()

			p := NewScorePredictor(
				NewOrlaClient(server.URL),
				&LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"},
			)
			got, err := p.Predict(context.Background(), "prompt")
			if tt.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.errContain)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantScore, got)
		})
	}
}

func TestScorePredictor_Predict_ExecuteError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	p := NewScorePredictor(
		NewOrlaClient(server.URL),
		&LLMBackend{Name: "b", Endpoint: server.URL, Type: "openai", ModelID: "m"},
	)
	_, err := p.Predict(context.Background(), "prompt")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "failed to execute")
}
