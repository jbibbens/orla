// Package provider integrates the backends orla dispatches to.
//
// Two backend kinds are supported:
//
//   - LLM backends speak OpenAI-compatible chat completions. The
//     openAIProvider in openai.go is the canonical implementation.
//
//   - Tool backends speak a kind-specific JSON RPC over HTTP for
//     scientific computation (structure prediction, docking, etc.).
//     Subpackages under provider/ implement ToolProvider; the first
//     is provider/structurepred.
//
// The Backend interface is the common identity both share. Scheduler
// machinery (concurrency caps, rate limits, telemetry) is kind-agnostic
// and operates on Backend; the proxy layer is kind-aware and routes
// per-kind to LLMProvider.Chat or ToolProvider.Invoke.
package provider

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// Backend is the kind-agnostic identity shared by every backend the
// scheduler knows about. Concrete providers (LLMProvider, ToolProvider)
// embed this contract.
type Backend interface {
	// Name returns the backend's registered name (matches the
	// `name` column in the backends table). Stable across the
	// process lifetime.
	Name() string
}

// LLMProvider is implemented by OpenAI-compatible chat backends.
//
// This is what existing orla code used to call "Provider". The
// rename clears the way for tool providers to coexist; behavior is
// otherwise unchanged.
type LLMProvider interface {
	Backend

	// ModelID returns the resolved model identifier (without the
	// provider prefix). The proxy overwrites the request's Model
	// field with this before dispatch, since the developer's value
	// is advisory only.
	ModelID() string

	// Chat sends a non-streaming chat completion.
	Chat(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)

	// ChatStream opens a streaming chat completion. The caller must
	// Close() the stream when done; concurrency slots are held by
	// the scheduler until that happens.
	ChatStream(ctx context.Context, params openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk]
}

// ToolProvider is implemented by scientific-computation backends
// (structure prediction, docking, ADMET property prediction, ...).
// Each concrete implementation handles one ToolKind.
type ToolProvider interface {
	Backend

	// ToolKind returns the kind of tool (e.g., "structure-prediction").
	// The proxy uses this to validate that the request's kind matches
	// the resolved backend's kind.
	ToolKind() string

	// Invoke dispatches a tool request. Payload schema is kind-specific
	// and opaque to the scheduler; the proxy decodes per kind.
	Invoke(ctx context.Context, req ToolRequest) (*ToolResponse, error)
}

// ToolRequest is the wire-shape envelope the proxy passes to
// ToolProvider.Invoke. Payload is kind-specific JSON; the concrete
// provider decodes it according to its ToolKind.
type ToolRequest struct {
	Kind    string          `json:"kind"`
	Payload json.RawMessage `json:"payload"`
}

// ToolResponse is the wire-shape envelope returned from
// ToolProvider.Invoke. Payload is kind-specific. GPUSeconds is the
// measured GPU compute time for cost accounting (the wrapper service
// reports it). Metadata is opaque diagnostic data.
type ToolResponse struct {
	Payload    json.RawMessage `json:"payload"`
	GPUSeconds float64         `json:"gpu_seconds,omitempty"`
	Metadata   map[string]any  `json:"metadata,omitempty"`
}

// ParseModelID splits a backend model id of the form "provider:model"
// into its (provider, model) parts. If no colon is present, the input
// is treated as a model name with an empty provider.
func ParseModelID(s string) (provider, model string) {
	before, after, ok := strings.Cut(s, ":")
	if !ok {
		return "", s
	}
	return before, after
}
