package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/backends"
)

func newBackendTestServer(t *testing.T) (*Server, *backends.FakeRegistry) {
	t.Helper()
	reg := backends.NewFakeRegistry()
	srv := NewServer(ServerConfig{
		ListenAddress: "127.0.0.1:0",
		Logger:        slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	RegisterBackendRoutes(srv.Router(), BackendDeps{Registry: reg})
	return srv, reg
}

func TestBackendHandlers_CreateReturns201(t *testing.T) {
	srv, reg := newBackendTestServer(t)

	body := mustJSON(t, map[string]any{
		"name":            "gpt4o",
		"endpoint":        "https://api.openai.com/v1",
		"model_id":        "openai:gpt-4o",
		"api_key_env_var": "OPENAI_API_KEY",
		"max_concurrency": 8,
		"quality":         0.85,
	})
	req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusCreated, rr.Code, rr.Body.String())
	var got backends.Backend
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, "gpt4o", got.Name)
	assert.Equal(t, int32(8), got.MaxConcurrency)
	require.NotNil(t, got.Quality)
	assert.InDelta(t, 0.85, *got.Quality, 1e-9)

	stored, err := reg.Get(context.Background(), "gpt4o")
	require.NoError(t, err)
	require.NotNil(t, stored.ModelID)
	assert.Equal(t, "openai:gpt-4o", *stored.ModelID)
}

func TestBackendHandlers_CreateDuplicateReturns409(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	body := mustJSON(t, map[string]any{
		"name": "gpt4o", "endpoint": "x", "model_id": "openai:gpt-4o", "max_concurrency": 1,
	})
	for i := range 2 {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
		rr := httptest.NewRecorder()
		srv.Router().ServeHTTP(rr, req)
		if i == 0 {
			require.Equal(t, http.StatusCreated, rr.Code)
		} else {
			assert.Equal(t, http.StatusConflict, rr.Code)
		}
	}
}

func TestBackendHandlers_CreateRejectsMissingFields(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	tests := []struct {
		name string
		body map[string]any
	}{
		{"no name", map[string]any{"endpoint": "x", "model_id": "y", "max_concurrency": 1}},
		{"no endpoint", map[string]any{"name": "x", "model_id": "y", "max_concurrency": 1}},
		{"no model_id", map[string]any{"name": "x", "endpoint": "y", "max_concurrency": 1}},
		{"zero concurrency", map[string]any{"name": "x", "endpoint": "y", "model_id": "z", "max_concurrency": 0}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := mustJSON(t, tt.body)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/backends", bytes.NewReader(body))
			rr := httptest.NewRecorder()
			srv.Router().ServeHTTP(rr, req)
			assert.Equal(t, http.StatusBadRequest, rr.Code)
		})
	}
}

func TestBackendHandlers_GetMissing404(t *testing.T) {
	srv, _ := newBackendTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/api/v1/backends/missing", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNotFound, rr.Code)
}

func TestBackendHandlers_PatchOnlyChangesProvidedFields(t *testing.T) {
	srv, reg := newBackendTestServer(t)

	q := 0.85
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "gpt4o", Endpoint: "x", ModelID: new("openai:gpt-4o"),
		MaxConcurrency: 4, Quality: &q,
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"max_concurrency": 16})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/gpt4o", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)

	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())
	var got backends.Backend
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &got))
	assert.Equal(t, int32(16), got.MaxConcurrency)
	require.NotNil(t, got.Quality)
	assert.InDelta(t, 0.85, *got.Quality, 1e-9, "untouched")
}

func TestBackendHandlers_PatchZeroConcurrencyReturns400(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "x", Endpoint: "y", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	body := mustJSON(t, map[string]any{"max_concurrency": 0})
	req := httptest.NewRequest(http.MethodPatch, "/api/v1/backends/x", bytes.NewReader(body))
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusBadRequest, rr.Code)
}

func TestBackendHandlers_DeleteReturns204(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	_, err := reg.Insert(context.Background(), &backends.Backend{
		Name: "x", Endpoint: "y", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
	})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodDelete, "/api/v1/backends/x", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	assert.Equal(t, http.StatusNoContent, rr.Code)
}

func TestBackendHandlers_ListOrderedByName(t *testing.T) {
	srv, reg := newBackendTestServer(t)
	for _, n := range []string{"zeta", "alpha", "mu"} {
		_, err := reg.Insert(context.Background(), &backends.Backend{
			Name: n, Endpoint: "y", ModelID: new("openai:gpt-4o"), MaxConcurrency: 1,
		})
		require.NoError(t, err)
	}

	req := httptest.NewRequest(http.MethodGet, "/api/v1/backends", nil)
	rr := httptest.NewRecorder()
	srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code)

	var body struct {
		Backends []backends.Backend `json:"backends"`
	}
	require.NoError(t, json.Unmarshal(rr.Body.Bytes(), &body))
	require.Len(t, body.Backends, 3)
	assert.Equal(t, []string{"alpha", "mu", "zeta"},
		[]string{body.Backends[0].Name, body.Backends[1].Name, body.Backends[2].Name})
}
