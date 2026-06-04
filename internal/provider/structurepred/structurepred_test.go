package structurepred_test

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/provider"
	"github.com/harvard-cns/orla/internal/provider/structurepred"
)

// wrapPayload encodes an inner Request into the generic ToolRequest
// envelope the way the proxy will.
func wrapPayload(t *testing.T, inner structurepred.Request) provider.ToolRequest {
	t.Helper()
	b, err := json.Marshal(inner)
	require.NoError(t, err)
	return provider.ToolRequest{Kind: structurepred.ToolKind, Payload: b}
}

// unwrapPayload decodes a Response from a ToolResponse envelope, the
// way a downstream caller would.
func unwrapPayload(t *testing.T, resp *provider.ToolResponse) structurepred.Response {
	t.Helper()
	var out structurepred.Response
	require.NoError(t, json.Unmarshal(resp.Payload, &out))
	return out
}

func TestClient_Invoke_RoundTrip(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "/v1/tools/structure-prediction", r.URL.Path)
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))

		// Decode the inner Request out of the envelope.
		var env provider.ToolRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&env))
		assert.Equal(t, structurepred.ToolKind, env.Kind)

		var inner structurepred.Request
		require.NoError(t, json.Unmarshal(env.Payload, &inner))
		assert.Equal(t, []string{"MKTV"}, inner.Sequences)
		assert.Equal(t, []string{"CCO"}, inner.LigandSMILES)

		// Return a Response wrapped in the ToolResponse envelope.
		respPayload, _ := json.Marshal(structurepred.Response{
			StructureCIF:    "data_test\n#",
			PLDDTPerResidue: []float64{82.5, 90.1, 75.0},
		})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(provider.ToolResponse{
			Payload:    respPayload,
			GPUSeconds: 12.3,
			Metadata:   map[string]any{"backend_version": "boltz-2.0.1"},
		})
	}))
	t.Cleanup(srv.Close)

	c := structurepred.New(&backends.Backend{
		Name:     "boltz",
		Endpoint: srv.URL,
	})

	resp, err := c.Invoke(context.Background(), wrapPayload(t, structurepred.Request{
		Sequences:    []string{"MKTV"},
		LigandSMILES: []string{"CCO"},
	}))
	require.NoError(t, err)
	assert.InDelta(t, 12.3, resp.GPUSeconds, 1e-9)
	assert.Equal(t, "boltz-2.0.1", resp.Metadata["backend_version"])

	out := unwrapPayload(t, resp)
	assert.Equal(t, "data_test\n#", out.StructureCIF)
	assert.Equal(t, []float64{82.5, 90.1, 75.0}, out.PLDDTPerResidue)
}

func TestClient_Invoke_SendsBearerToken(t *testing.T) {
	t.Setenv("ORLA_TOOL_TOKEN_TEST", "supersecret")

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "Bearer supersecret", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"payload": {"structure_cif": ""}}`))
	}))
	t.Cleanup(srv.Close)

	c := structurepred.New(&backends.Backend{
		Name:         "boltz",
		Endpoint:     srv.URL,
		APIKeyEnvVar: "ORLA_TOOL_TOKEN_TEST",
	})

	_, err := c.Invoke(context.Background(), wrapPayload(t, structurepred.Request{
		Sequences: []string{"MKTV"},
	}))
	require.NoError(t, err)
}

func TestClient_Invoke_OmitsAuthHeaderWhenNoEnvVar(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Empty(t, r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"payload": {"structure_cif": ""}}`))
	}))
	t.Cleanup(srv.Close)

	c := structurepred.New(&backends.Backend{Name: "boltz", Endpoint: srv.URL})

	_, err := c.Invoke(context.Background(), wrapPayload(t, structurepred.Request{
		Sequences: []string{"MKTV"},
	}))
	require.NoError(t, err)
}

func TestClient_Invoke_NonOKStatusReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"bad sequence"}}`)
	}))
	t.Cleanup(srv.Close)

	c := structurepred.New(&backends.Backend{Name: "boltz", Endpoint: srv.URL})
	_, err := c.Invoke(context.Background(), wrapPayload(t, structurepred.Request{
		Sequences: []string{"X"},
	}))
	require.Error(t, err)
	assert.Contains(t, err.Error(), "status=400")
	assert.Contains(t, err.Error(), "bad sequence")
}

func TestClient_Invoke_RejectsWrongKindInEnvelope(t *testing.T) {
	c := structurepred.New(&backends.Backend{Name: "boltz", Endpoint: "http://unused"})
	_, err := c.Invoke(context.Background(), provider.ToolRequest{
		Kind:    "docking",
		Payload: []byte(`{"sequences":["X"]}`),
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "wrong kind")
}

func TestClient_Invoke_RespectsContext(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Block until the request is cancelled — used to verify the ctx
		// propagates through the http client.
		<-r.Context().Done()
		w.WriteHeader(http.StatusInternalServerError)
	}))
	t.Cleanup(srv.Close)

	c := structurepred.New(&backends.Backend{Name: "boltz", Endpoint: srv.URL})

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // immediately
	_, err := c.Invoke(ctx, wrapPayload(t, structurepred.Request{Sequences: []string{"X"}}))
	require.Error(t, err)
}

func TestClient_NameAndToolKind(t *testing.T) {
	c := structurepred.New(&backends.Backend{Name: "boltz-2", Endpoint: "http://unused"})
	assert.Equal(t, "boltz-2", c.Name())
	assert.Equal(t, structurepred.ToolKind, c.ToolKind())
}
