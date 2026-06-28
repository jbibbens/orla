package api

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/harvard-cns/orla/internal/stages"
)

const jsonSchemaBody = `{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],` +
	`"response_format":{"type":"json_schema","json_schema":{"name":"Selection","strict":true,` +
	`"schema":{"type":"object","properties":{"passages":{"type":"array","items":{"type":"integer"}}},` +
	`"required":["passages"],"additionalProperties":false}}}}`

func TestProxy_ForwardsJSONSchemaResponseFormat(t *testing.T) {
	env := newProxyEnv(t)
	_, err := env.stages.Replace(context.Background(), &stages.Stage{ID: "planning", Backend: "gpt4o"})
	require.NoError(t, err)

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(jsonSchemaBody)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(HeaderStage, "planning")
	rr := httptest.NewRecorder()
	env.srv.Router().ServeHTTP(rr, req)
	require.Equal(t, http.StatusOK, rr.Code, rr.Body.String())

	calls := env.mock.Calls()
	require.Len(t, calls, 1)
	out, err := json.Marshal(calls[0].ResponseFormat)
	require.NoError(t, err)
	var rf map[string]any
	require.NoError(t, json.Unmarshal(out, &rf))
	js, ok := rf["json_schema"].(map[string]any)
	require.True(t, ok, "json_schema must reach the provider, got %s", out)
	assert.Equal(t, "Selection", js["name"])
	assert.Equal(t, true, js["strict"])
	assert.NotNil(t, js["schema"])
}

func TestReattachJSONSchema_NoOpCases(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{name: "no response_format", body: `{"model":"m","messages":[]}`},
		{name: "json_object format", body: `{"response_format":{"type":"json_object"}}`},
		{name: "text format", body: `{"response_format":{"type":"text"}}`},
		{name: "empty body", body: `{}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var params openai.ChatCompletionNewParams
			assert.NoError(t, reattachJSONSchema(&params, []byte(tt.body)))
		})
	}
}

func TestReattachJSONSchema_RejectsMalformedFormat(t *testing.T) {
	body := []byte(`{"response_format":"not-an-object"}`)
	var params openai.ChatCompletionNewParams
	assert.Error(t, reattachJSONSchema(&params, body))
}
