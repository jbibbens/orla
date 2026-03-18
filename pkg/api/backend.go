package orla

import "github.com/docker/docker/pkg/namesgenerator"

// randomBackendName generates a random name for the backend that is human readable.
// I liked the way Docker does it with namesgenerator so I used it here.
func randomBackendName() string {
	return namesgenerator.GetRandomName(0)
}

const (
	backendTypeOpenAI = "openai"
	backendTypeSGLang = "sglang"
)

func modelIDForBackendType(backendType string, modelID string) string {
	return backendType + ":" + modelID
}

// RegisterBackendRequest is the request body for registering an LLM backend.
type RegisterBackendRequest struct {
	Name           string `json:"name"`                        // backend name (used as Backend in execute requests)
	Endpoint       string `json:"endpoint"`                    // e.g. "http://localhost:8000/v1"
	Type           string `json:"type"`                        // "openai" or "sglang"
	ModelID        string `json:"model_id"`                    // e.g. "openai:Qwen/Qwen3-4B-Instruct-2507"
	APIKeyEnvVar   string `json:"api_key_env_var,omitempty"`   // optional env var for API key (openai-type)
	MaxConcurrency int    `json:"max_concurrency,omitempty"`   // max concurrent requests dispatched to this backend (default 1)
	QueueCapacity  int    `json:"queue_capacity,omitempty"`   // max queued requests for this backend; 0 = default (4096)
}

// RegisterBackendResponse is the response from register backend.
type RegisterBackendResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

type LLMBackend = RegisterBackendRequest

// SetMaxConcurrency sets the maximum number of concurrent inference requests
// dispatched to this backend. Backends that support continuous batching (e.g.
// vLLM, SGLang) can process multiple requests simultaneously for better throughput.
// A value of 0 or 1 means serial dispatch (default).
func (r *RegisterBackendRequest) SetMaxConcurrency(n int) {
	r.MaxConcurrency = n
}

// SetQueueCapacity sets the maximum number of requests that may be queued for
// this backend. When full, new requests get an error (backpressure). Zero
// means use the server default (4096).
func (r *RegisterBackendRequest) SetQueueCapacity(n int) {
	r.QueueCapacity = n
}

func NewVLLMBackend(modelID string, endpoint string) *LLMBackend {
	return &LLMBackend{
		Name:     randomBackendName(),
		Endpoint: endpoint,
		Type:     backendTypeOpenAI,
		ModelID:  modelIDForBackendType(backendTypeOpenAI, modelID),
	}
}

func NewSGLangBackend(modelID string, endpoint string) *LLMBackend {
	return &LLMBackend{
		Name:     randomBackendName(),
		Endpoint: endpoint,
		Type:     backendTypeSGLang,
		ModelID:  modelIDForBackendType(backendTypeOpenAI, modelID), // OpenAI-compatible API
	}
}

// NewOllamaBackend creates a backend that talks to Ollama's OpenAI-compatible
// API (/v1/chat/completions). The endpoint should be the base Ollama URL
// (e.g. "http://ollama:11434"); "/v1" is appended automatically.
func NewOllamaBackend(modelID string, endpoint string) *LLMBackend {
	return &LLMBackend{
		Name:     randomBackendName(),
		Endpoint: endpoint + "/v1",
		Type:     backendTypeOpenAI,
		ModelID:  modelIDForBackendType(backendTypeOpenAI, modelID),
	}
}
