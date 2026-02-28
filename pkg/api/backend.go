package orla

import "github.com/docker/docker/pkg/namesgenerator"

// randomBackendName generates a random name for the backend that is human readable.
// I liked the way Docker does it with namesgenerator so I used it here.
func randomBackendName() string {
	return namesgenerator.GetRandomName(0)
}

const (
	backendTypeOpenAI = "openai"
	backendTypeOllama = "ollama"
)

func modelIDForBackendType(backendType string, modelID string) string {
	return backendType + ":" + modelID
}

// RegisterBackendRequest is the request body for registering an LLM backend.
type RegisterBackendRequest struct {
	Name         string `json:"name"`                      // backend name (used as Backend in execute requests)
	Endpoint     string `json:"endpoint"`                  // e.g. "http://localhost:8000/v1"
	Type         string `json:"type"`                      // "openai", "ollama", or "sglang"
	ModelID      string `json:"model_id"`                  // e.g. "openai:Qwen/Qwen3-4B-Instruct-2507"
	APIKeyEnvVar string `json:"api_key_env_var,omitempty"` // optional env var for API key (openai-type)
}

// RegisterBackendResponse is the response from register backend.
type RegisterBackendResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

type LLMBackend = RegisterBackendRequest

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
		Type:     backendTypeOpenAI,
		ModelID:  modelIDForBackendType(backendTypeOpenAI, modelID),
	}
}

func NewOllamaBackend(modelID string, endpoint string) *LLMBackend {
	return &LLMBackend{
		Name:     randomBackendName(),
		Endpoint: endpoint,
		Type:     backendTypeOllama,
		ModelID:  modelIDForBackendType(backendTypeOllama, modelID),
	}
}
