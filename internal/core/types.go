// Package core implements the core functionality for orla that is shared across all components.
package core

// LLMInferenceAPIType represents the type of LLM inference API
type LLMInferenceAPIType string

const (
	// LLMInferenceAPITypeOpenAI represents any inference server that has an
	// OpenAI-compatible API (including Ollama via /v1/chat/completions).
	LLMInferenceAPITypeOpenAI LLMInferenceAPIType = "openai"
	// LLMInferenceAPITypeSGLang represents SGLang, which provides an OpenAI-compatible
	// API for inference and a separate /flush_cache endpoint for cache control.
	LLMInferenceAPITypeSGLang LLMInferenceAPIType = "sglang"
)

// LLMBackend represents an LLM inference server. This allows configuring
// remote Ollama servers, OpenAI-compatible APIs, and other LLM inference servers.
type LLMBackend struct {
	// Endpoint is the URL of the LLM inference server
	Endpoint string `yaml:"endpoint,omitempty" mapstructure:"endpoint"`
	// Type is the type of the LLM inference API
	Type LLMInferenceAPIType `yaml:"type,omitempty" mapstructure:"type"`
	// APIKeyEnvVar is the *ENVIRONMENT VARIABLE*  storing the API key for the LLM inference API
	// Orla *does not* allow you to store the API key in the config file. You must use an environment variable.
	// This is to prevent the API key from being accidentally logged or leaked.
	APIKeyEnvVar string `yaml:"api_key_env_var,omitempty" mapstructure:"api_key_env_var"`
	// MaxConcurrency is the maximum number of concurrent inference requests dispatched
	// to this backend. Backends like vLLM and SGLang support continuous batching and can
	// process multiple requests simultaneously. A value of 0 or 1 means serial dispatch.
	MaxConcurrency int `yaml:"max_concurrency,omitempty" mapstructure:"max_concurrency"`
	// QueueCapacity is the maximum number of requests that may be queued for this backend.
	// When the queue is full, ScheduleChat returns an error. A value of 0 means use the default (4096).
	QueueCapacity int `yaml:"queue_capacity,omitempty" mapstructure:"queue_capacity"`
}
