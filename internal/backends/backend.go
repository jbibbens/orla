// Package backends owns the LLM-or-tool backend record (endpoint,
// model id / tool kind, concurrency cap, cost/quality priors) and the
// registry that persists it. Backends are explicitly registered by
// the platform engineer; the proxy refuses to dispatch to an unknown
// backend.
package backends

import "time"

// Kind discriminates between the two backend flavors:
//
//   - KindLLM:  OpenAI-compatible chat completion backend.
//     Cost is token-denominated; ModelID is required.
//
//   - KindTool: scientific computation tool (structure prediction,
//     docking, etc.). Cost is GPU-second-denominated; ToolKind is
//     required; ModelID is unused.
type Kind string

const (
	KindLLM  Kind = "llm"
	KindTool Kind = "tool"
)

// Backend is the persistent record for a single backend.
//
// Cost and quality are platform-engineer-supplied priors. orla does
// not act on them; the mapper does. They are persisted so the mapper
// can read them as part of its state.
type Backend struct {
	Name           string   `json:"name"`
	Endpoint       string   `json:"endpoint"`
	APIKeyEnvVar   string   `json:"api_key_env_var"`
	MaxConcurrency int32    `json:"max_concurrency"`
	Quality        *float64 `json:"quality,omitempty"`

	// RatePerSecond is the requests/second cap applied per orla
	// instance. Nil or 0 means no limit. Bursting is allowed up to the
	// rounded-up rate so a steady-state limit of 5 rps doesn't reject
	// a sudden 5-request batch.
	RatePerSecond *float64 `json:"rate_per_second,omitempty"`

	// Kind discriminates LLM vs tool. Defaults to KindLLM at registration
	// time if unset.
	Kind Kind `json:"kind"`

	// ModelID is the provider-prefixed model identifier
	// (e.g., "openai:gpt-4o"). Required for KindLLM, unused for KindTool.
	ModelID *string `json:"model_id,omitempty"`

	// LLM cost model: per-million-token rates. NULL for KindTool.
	InputCostPerMtoken  *float64 `json:"input_cost_per_mtoken,omitempty"`
	OutputCostPerMtoken *float64 `json:"output_cost_per_mtoken,omitempty"`

	// ToolKind identifies the family of tool (e.g., "structure-prediction",
	// "docking"). Required for KindTool, unused for KindLLM.
	ToolKind *string `json:"tool_kind,omitempty"`

	// Tool cost model: per-GPU-second rate. NULL for KindLLM.
	CostPerGPUSecond *float64 `json:"cost_per_gpu_second,omitempty"`

	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// PatchRequest describes a partial update. nil pointers leave the
// corresponding field unchanged. Name, Kind, ModelID, and ToolKind
// cannot be patched — to change them, delete and re-create the
// backend.
type PatchRequest struct {
	Endpoint            *string  `json:"endpoint,omitempty"`
	APIKeyEnvVar        *string  `json:"api_key_env_var,omitempty"`
	MaxConcurrency      *int32   `json:"max_concurrency,omitempty"`
	InputCostPerMtoken  *float64 `json:"input_cost_per_mtoken,omitempty"`
	OutputCostPerMtoken *float64 `json:"output_cost_per_mtoken,omitempty"`
	Quality             *float64 `json:"quality,omitempty"`
	RatePerSecond       *float64 `json:"rate_per_second,omitempty"`
	CostPerGPUSecond    *float64 `json:"cost_per_gpu_second,omitempty"`
}
