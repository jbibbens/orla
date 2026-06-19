package api

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/provider"
	"github.com/harvard-cns/orla/internal/scheduler"
	"github.com/harvard-cns/orla/internal/stages"
)

// proxyEnv wires up a fake stage registry and a real scheduler with a
// mock provider, exactly the dependencies the handler needs.
type proxyEnv struct {
	srv    *Server
	stages *stages.FakeRegistry
	sched  *scheduler.Scheduler
	mock   *provider.MockProvider
}

func newProxyEnv(t *testing.T) *proxyEnv {
	t.Helper()
	mock := provider.NewMockProvider().WithName("gpt4o").
		WithResponse(&openai.ChatCompletion{
			ID:    "chatcmpl-stub",
			Model: "openai:gpt-4o",
			Choices: []openai.ChatCompletionChoice{{
				Index:        0,
				FinishReason: "stop",
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: "hello",
				},
			}},
		})

	sched := scheduler.New(
		func(b *backends.Backend) provider.Backend { return mock },
		slog.New(slog.NewTextHandler(io.Discard, nil)),
	)
	sched.Register(&backends.Backend{
		Name: "gpt4o", Endpoint: "x", ModelID: new("openai:gpt-4o"), MaxConcurrency: 2,
	})
	t.Cleanup(func() { _ = sched.Shutdown(context.Background()) })

	stageReg := stages.NewFakeRegistry()

	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterProxyRoutes(srv.Router(), ProxyDeps{
		Stages:    stageReg,
		Scheduler: sched,
	})

	return &proxyEnv{srv: srv, stages: stageReg, sched: sched, mock: mock}
}

func bodyForChat(messages ...string) []byte {
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	var ms []msg
	for _, m := range messages {
		ms = append(ms, msg{Role: "user", Content: m})
	}
	b, _ := json.Marshal(map[string]any{
		"model":    "gpt-4o",
		"messages": ms,
	})
	return b
}

func TestProxy_ComputesLLMCostUSD(t *testing.T) {
	mock := provider.NewMockProvider().WithName("gpt4o").
		WithResponse(&openai.ChatCompletion{
			ID:    "chatcmpl-cost",
			Model: "openai:gpt-4o",
			Choices: []openai.ChatCompletionChoice{{
				Message: openai.ChatCompletionMessage{Role: "assistant", Content: "hi"},
			}},
			Usage: openai.CompletionUsage{PromptTokens: 2_000_000, CompletionTokens: 1_000_000},
		})

	sched := scheduler.New(
		func(b *backends.Backend) provider.Backend { return mock },
		slog.New(slog.NewTextHandler(io.Discard, nil)),
	)
	t.Cleanup(func() { _ = sched.Shutdown(context.Background()) })
	in, out := 0.5, 1.5
	sched.Register(&backends.Backend{
		Name: "gpt4o", Endpoint: "x", ModelID: new("openai:gpt-4o"),
		MaxConcurrency:      2,
		InputCostPerMtoken:  &in,
		OutputCostPerMtoken: &out,
	})

	stageReg := stages.NewFakeRegistry()
	_, err := stageReg.Replace(context.Background(), &stages.Stage{
		ID: "planning", Backend: "gpt4o",
	})
	require.NoError(t, err)

	sink := &recordingSink{}
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterProxyRoutes(srv.Router(), ProxyDeps{
		Stages: stageReg, Scheduler: sched, CompletionSink: sink,
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(bodyForChat("hi")))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())

	require.Len(t, sink.got, 1)
	require.NotNil(t, sink.got[0].CostUSD)
	// 2M prompt × $0.50/Mt + 1M completion × $1.50/Mt = 1.0 + 1.5 = $2.50.
	assert.InDelta(t, 2.5, *sink.got[0].CostUSD, 1e-9)
}

func TestProxy_RequiresStageHeader(t *testing.T) {
	env := newProxyEnv(t)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(bodyForChat("hi")))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestProxy_StageInHeader(t *testing.T) {
	env := newProxyEnv(t)
	// Pre-configure stage with a backend mapping.
	_, err := env.stages.Replace(context.Background(), &stages.Stage{
		ID: "planning", Backend: "gpt4o",
	})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(bodyForChat("hi")))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	var resp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &resp))
	assert.Equal(t, "chatcmpl-stub", resp.ID)
	assert.Equal(t, "gpt4o", resp.Model,
		"response model field reports resolved backend name")
}

func TestProxy_StageInBodyMetadata(t *testing.T) {
	env := newProxyEnv(t)
	_, err := env.stages.Replace(context.Background(), &stages.Stage{
		ID: "planning", Backend: "gpt4o",
	})
	require.NoError(t, err)

	body, _ := json.Marshal(map[string]any{
		"model": "gpt-4o",
		"messages": []map[string]string{
			{"role": "user", "content": "hi"},
		},
		"metadata": map[string]string{
			"orla.stage": "planning",
		},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(body))
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
}

func TestProxy_AutoCreateStageOnFirstSighting(t *testing.T) {
	env := newProxyEnv(t)
	// "planning" has no backend, request body's model field is the fallback.
	body, _ := json.Marshal(map[string]any{
		"model":    "gpt4o", // <-- treated as backend name fallback
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())

	// Stage should now exist with empty backend (auto-created).
	got, err := env.stages.Get(context.Background(), "planning")
	require.NoError(t, err)
	assert.Equal(t, "", got.Backend, "auto-create leaves backend empty")
}

func TestProxy_RejectsRequestWithoutBackendOrModel(t *testing.T) {
	env := newProxyEnv(t)
	body, _ := json.Marshal(map[string]any{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		// no model, no stage backend
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestProxy_UnknownBackendReturns502(t *testing.T) {
	env := newProxyEnv(t)
	_, err := env.stages.Replace(context.Background(), &stages.Stage{
		ID: "planning", Backend: "does-not-exist",
	})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(bodyForChat("hi")))
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadGateway, rr.Code)
}

func TestProxy_ProviderErrorRendersUpstreamShape(t *testing.T) {
	env := newProxyEnv(t)
	_, err := env.stages.Replace(context.Background(), &stages.Stage{
		ID: "planning", Backend: "gpt4o",
	})
	require.NoError(t, err)

	env.mock.WithError(errors.New("simulated upstream failure"))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(bodyForChat("hi")))
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)

	assert.Equal(t, http.StatusBadGateway, rr.Code)
	var env2 errorEnvelope
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &env2))
	assert.Contains(t, env2.Error.Message, "simulated upstream failure")
}

func TestProxy_ResponseModelOverwrittenWithBackendName(t *testing.T) {
	env := newProxyEnv(t)
	_, err := env.stages.Replace(context.Background(), &stages.Stage{
		ID: "planning", Backend: "gpt4o",
	})
	require.NoError(t, err)

	env.mock.WithResponse(&openai.ChatCompletion{
		ID:    "chatcmpl-x",
		Model: "upstream-disagrees-with-us", // proxy must overwrite
	})

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewReader(bodyForChat("hi")))
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	var resp openai.ChatCompletion
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &resp))
	assert.Equal(t, "gpt4o", resp.Model)
}

func TestProxy_RejectsEmptyMessages(t *testing.T) {
	env := newProxyEnv(t)
	body, _ := json.Marshal(map[string]any{
		"model":    "gpt-4o",
		"messages": []any{},
	})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestExtractRequestContext_TagsLowercased(t *testing.T) {
	r := httptest.NewRequest(http.MethodPost, "/", nil)
	r.Header.Set(HeaderStage, "planning")
	r.Header.Set("X-Orla-Tag-Tenant", "alice")
	r.Header.Set("X-Orla-Tag-PROJECT", "core")

	rc := extractRequestContext(r, nil)
	assert.Equal(t, "planning", rc.Stage)
	assert.Equal(t, "alice", rc.Tags["tenant"])
	assert.Equal(t, "core", rc.Tags["project"])
}

// TestProxy_StreamingClientDisconnect verifies that when the client
// closes the connection mid-stream, the worker still releases its slot
// so subsequent requests aren't stuck behind a phantom in-flight call.
func TestProxy_StreamingClientDisconnect(t *testing.T) {
	upstreamGate := make(chan struct{})
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		_, _ = io.WriteString(w, `data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"u","choices":[{"index":0,"delta":{"content":"a"}}]}`+"\n\n")
		flusher.Flush()
		// Hang waiting for more chunks until upstreamGate or ctx fires.
		select {
		case <-upstreamGate:
		case <-r.Context().Done():
		}
	}))
	t.Cleanup(func() { close(upstreamGate); upstream.Close() })

	stageReg := stages.NewFakeRegistry()
	_, err := stageReg.Replace(context.Background(), &stages.Stage{
		ID: "planning", Backend: "real",
	})
	require.NoError(t, err)

	sched := scheduler.New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	sched.Register(&backends.Backend{
		Name: "real", Endpoint: upstream.URL,
		ModelID: new("openai:upstream"), MaxConcurrency: 1,
	})
	t.Cleanup(func() { _ = sched.Shutdown(context.Background()) })

	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterProxyRoutes(srv.Router(), ProxyDeps{Stages: stageReg, Scheduler: sched})

	ts := httptest.NewServer(srv.Router())
	t.Cleanup(ts.Close)

	body, _ := json.Marshal(map[string]any{
		"model": "upstream", "stream": true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	ctx, cancel := context.WithCancel(context.Background())
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		ts.URL+"/v1/chat/completions", bytes.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "planning")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)

	// Read one chunk to confirm the stream opened, then disconnect.
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		if strings.HasPrefix(scanner.Text(), "data: ") {
			break
		}
	}
	cancel()
	_ = resp.Body.Close()

	// The scheduler must release its slot. Capacity is 1, so the second
	// dispatch can't proceed if the slot is still held. We don't make a
	// real second request (it'd just hang), instead, verify Stats()
	// drops to 0 in-flight after disconnect.
	require.Eventually(t, func() bool {
		stats := sched.Stats()
		if len(stats) == 0 {
			return false
		}
		return stats[0].InFlight == 0
	}, 3*time.Second, 50*time.Millisecond,
		"scheduler must release the slot on client disconnect")
}

// Streaming smoke test: the proxy must produce SSE frames followed by
// data: [DONE]. We can't easily exercise the openai-go stream type
// with the mock (its ChatStream panics intentionally), so we point the
// scheduler at a real provider talking to an httptest SSE server.
func TestProxy_StreamingSmoke(t *testing.T) {
	// Upstream SSE server emits two chunks then [DONE].
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		chunks := []string{
			`{"id":"chunk-1","object":"chat.completion.chunk","created":1,"model":"upstream","choices":[{"index":0,"delta":{"content":"hello"}}]}`,
			`{"id":"chunk-2","object":"chat.completion.chunk","created":1,"model":"upstream","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}`,
		}
		for _, c := range chunks {
			_, _ = io.WriteString(w, "data: "+c+"\n\n")
			flusher.Flush()
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	t.Cleanup(upstream.Close)

	stageReg := stages.NewFakeRegistry()
	_, err := stageReg.Replace(context.Background(), &stages.Stage{
		ID: "planning", Backend: "real",
	})
	require.NoError(t, err)

	sched := scheduler.New(nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
	sched.Register(&backends.Backend{
		Name: "real", Endpoint: upstream.URL,
		ModelID: new("openai:upstream"), MaxConcurrency: 1,
	})
	t.Cleanup(func() { _ = sched.Shutdown(context.Background()) })

	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
		WriteTimeout:  0,
	})
	RegisterProxyRoutes(srv.Router(), ProxyDeps{Stages: stageReg, Scheduler: sched})

	// Use httptest.Server so the response can stream (httptest.ResponseRecorder
	// doesn't implement http.Flusher).
	ts := httptest.NewServer(srv.Router())
	t.Cleanup(ts.Close)

	body, _ := json.Marshal(map[string]any{
		"model":    "upstream",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	req, err := http.NewRequest(http.MethodPost, ts.URL+"/v1/chat/completions", bytes.NewReader(body))
	require.NoError(t, err)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "planning")

	resp, err := http.DefaultClient.Do(req)
	require.NoError(t, err)
	defer func() { _ = resp.Body.Close() }()
	require.Equal(t, http.StatusOK, resp.StatusCode)
	require.Equal(t, "text/event-stream", resp.Header.Get("Content-Type"))

	var (
		chunks  []string
		sawDone bool
	)
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 8192), 1<<20)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		chunks = append(chunks, payload)
	}
	require.NoError(t, scanner.Err())
	require.Len(t, chunks, 2, "expected 2 data chunks")
	assert.True(t, sawDone, "expected [DONE] terminator")

	// Verify Model was rewritten in each chunk.
	for _, c := range chunks {
		var chunk openai.ChatCompletionChunk
		require.NoError(t, json.Unmarshal([]byte(c), &chunk))
		assert.Equal(t, "real", chunk.Model, "chunk.model rewritten to backend name")
		assert.NotContains(t, c, `"role":""`,
			"must forward the upstream chunk verbatim, not emit zero-value fields strict clients reject")
	}
}
