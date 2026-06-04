package provider_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/provider"
)

func TestParseModelID(t *testing.T) {
	tests := []struct {
		in              string
		wantProvider    string
		wantModel       string
	}{
		{"openai:gpt-4o", "openai", "gpt-4o"},
		{"sglang:Qwen/Qwen3-4B", "sglang", "Qwen/Qwen3-4B"},
		{"ollama:llama3:8b", "ollama", "llama3:8b"}, // first colon wins
		{"plain-model", "", "plain-model"},
		{"", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			p, m := provider.ParseModelID(tt.in)
			assert.Equal(t, tt.wantProvider, p)
			assert.Equal(t, tt.wantModel, m)
		})
	}
}

// openaiTestServer returns an httptest.Server that responds to
// /chat/completions with the supplied handler.
func openaiTestServer(t *testing.T, handler http.HandlerFunc) (string, func()) {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/chat/completions", handler)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv.URL, srv.Close
}

func TestOpenAIProvider_Chat_ReturnsResponse(t *testing.T) {
	url, _ := openaiTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		// Verify the provider overwrote Model with the configured value.
		var req map[string]any
		require.NoError(t, json.Unmarshal(body, &req))
		assert.Equal(t, "configured-model", req["model"])

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-abc",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "configured-model",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "ok"},
			}},
			"usage": map[string]any{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
		})
	})

	p := provider.NewOpenAI(&backends.Backend{
		Name: "gpt4o", Endpoint: url, ModelID: new("openai:configured-model"),
		MaxConcurrency: 1,
	})

	resp, err := p.Chat(context.Background(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage("hello"),
		},
		Model: "client-requested-model", // should be overwritten
	})
	require.NoError(t, err)
	assert.Equal(t, "chatcmpl-abc", resp.ID)
	require.Len(t, resp.Choices, 1)
	assert.Equal(t, "ok", resp.Choices[0].Message.Content)
}

func TestOpenAIProvider_Chat_RetriesOn5xx(t *testing.T) {
	var calls atomic.Int32
	url, _ := openaiTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n < 3 {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = w.Write([]byte(`{"error":{"message":"transient","type":"server_error"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-ok",
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   "m",
			"choices": []map[string]any{{
				"index":         0,
				"finish_reason": "stop",
				"message":       map[string]any{"role": "assistant", "content": "ok"},
			}},
		})
	})

	p := provider.NewOpenAI(
		&backends.Backend{Name: "b", Endpoint: url, ModelID: new("openai:m"), MaxConcurrency: 1},
		provider.WithRetry(3, 1*time.Millisecond, 5*time.Millisecond),
	)

	resp, err := p.Chat(context.Background(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	require.NoError(t, err)
	assert.Equal(t, "chatcmpl-ok", resp.ID)
	assert.Equal(t, int32(3), calls.Load(), "retry until success")
}

func TestOpenAIProvider_Chat_DoesNotRetryOn4xx(t *testing.T) {
	var calls atomic.Int32
	url, _ := openaiTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		_, _ = fmt.Fprintln(w, `{"error":{"message":"bad","type":"invalid_request_error"}}`)
	})

	p := provider.NewOpenAI(
		&backends.Backend{Name: "b", Endpoint: url, ModelID: new("openai:m"), MaxConcurrency: 1},
		provider.WithRetry(3, 1*time.Millisecond, 5*time.Millisecond),
	)

	_, err := p.Chat(context.Background(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	require.Error(t, err)
	assert.Equal(t, int32(1), calls.Load(), "4xx must fail fast, no retry")
}

func TestOpenAIProvider_Chat_RetriesOn429(t *testing.T) {
	var calls atomic.Int32
	url, _ := openaiTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n < 2 {
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = fmt.Fprintln(w, `{"error":{"message":"slow down","type":"rate_limit_exceeded"}}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      "chatcmpl-ok", "object": "chat.completion",
			"created": time.Now().Unix(), "model": "m",
			"choices": []map[string]any{{"index": 0, "finish_reason": "stop",
				"message": map[string]any{"role": "assistant", "content": "ok"}}},
		})
	})

	p := provider.NewOpenAI(
		&backends.Backend{Name: "b", Endpoint: url, ModelID: new("openai:m"), MaxConcurrency: 1},
		provider.WithRetry(3, 1*time.Millisecond, 5*time.Millisecond),
	)
	_, err := p.Chat(context.Background(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	require.NoError(t, err)
	assert.Equal(t, int32(2), calls.Load(), "retried 429 once")
}

func TestOpenAIProvider_Chat_RespectsContext(t *testing.T) {
	url, _ := openaiTestServer(t, func(w http.ResponseWriter, r *http.Request) {
		// Slow handler; should be canceled by the client ctx.
		select {
		case <-r.Context().Done():
		case <-time.After(2 * time.Second):
		}
		w.WriteHeader(http.StatusInternalServerError)
	})

	p := provider.NewOpenAI(
		&backends.Backend{Name: "b", Endpoint: url, ModelID: new("openai:m"), MaxConcurrency: 1},
		provider.WithRetry(3, 1*time.Millisecond, 5*time.Millisecond),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	_, err := p.Chat(ctx, openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	require.Error(t, err)
}

func TestMockProvider_RecordsCalls(t *testing.T) {
	m := provider.NewMockProvider().WithName("test").WithModelID("model-x")
	resp, err := m.Chat(context.Background(), openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{openai.UserMessage("hi")},
	})
	require.NoError(t, err)
	assert.NotNil(t, resp)
	assert.Equal(t, "test", m.Name())
	assert.Equal(t, "model-x", m.ModelID())
	assert.Equal(t, 1, m.CallCount())
}

func TestMockProvider_WithError(t *testing.T) {
	m := provider.NewMockProvider().WithError(provider.ErrMockUnconfigured)
	_, err := m.Chat(context.Background(), openai.ChatCompletionNewParams{})
	assert.ErrorIs(t, err, provider.ErrMockUnconfigured)
}
