package provider

import (
	"context"
	"errors"
	"sync"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
)

// MockProvider is a Provider for tests. Construct with NewMockProvider
// and configure via the With* helpers.
type MockProvider struct {
	mu sync.Mutex

	name     string
	modelID  string
	response *openai.ChatCompletion
	err      error
	chatFunc func(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)

	calls []openai.ChatCompletionNewParams
}

// NewMockProvider returns a mock with sensible defaults: name=mock,
// model=mock-model, an empty success response.
func NewMockProvider() *MockProvider {
	return &MockProvider{
		name:    "mock",
		modelID: "mock-model",
		response: &openai.ChatCompletion{
			ID: "chatcmpl-mock",
		},
	}
}

// Compile-time interface check.
var _ LLMProvider = (*MockProvider)(nil)

// WithName overrides the provider name.
func (m *MockProvider) WithName(n string) *MockProvider {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.name = n
	return m
}

// WithModelID overrides the provider model id.
func (m *MockProvider) WithModelID(id string) *MockProvider {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.modelID = id
	return m
}

// WithResponse sets the canned response.
func (m *MockProvider) WithResponse(resp *openai.ChatCompletion) *MockProvider {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.response = resp
	m.err = nil
	m.chatFunc = nil
	return m
}

// WithError makes Chat return err on every call.
func (m *MockProvider) WithError(err error) *MockProvider {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.err = err
	m.response = nil
	m.chatFunc = nil
	return m
}

// WithChatFunc installs a custom handler used in place of the canned
// response/error.
func (m *MockProvider) WithChatFunc(fn func(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error)) *MockProvider {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.chatFunc = fn
	return m
}

func (m *MockProvider) Name() string    { return m.name }
func (m *MockProvider) ModelID() string { return m.modelID }

func (m *MockProvider) Chat(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	m.mu.Lock()
	m.calls = append(m.calls, params)
	chatFunc := m.chatFunc
	resp := m.response
	err := m.err
	m.mu.Unlock()

	if chatFunc != nil {
		return chatFunc(ctx, params)
	}
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// ChatStream is intentionally unimplemented for v1. Streaming tests
// use a real httptest server.
func (m *MockProvider) ChatStream(_ context.Context, _ openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk] {
	// ssestream.Stream has unexported fields; we can't construct an
	// erroring stream from outside the package. Tests needing streams
	// should use httptest + the real OpenAI provider.
	panic("MockProvider.ChatStream: not implemented; use the real provider against an httptest server")
}

// Calls returns the params of every Chat invocation in order. Safe for
// concurrent use.
func (m *MockProvider) Calls() []openai.ChatCompletionNewParams {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]openai.ChatCompletionNewParams, len(m.calls))
	copy(out, m.calls)
	return out
}

// CallCount returns the number of Chat calls so far.
func (m *MockProvider) CallCount() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.calls)
}

// ErrMockUnconfigured is what tests should return from custom chatFunc
// when they want to simulate an unconfigured backend without having to
// invent a transport-level error.
var ErrMockUnconfigured = errors.New("mock provider: unconfigured")
