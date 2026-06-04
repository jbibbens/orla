package provider

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/ssestream"

	"github.com/harvard-cns/orla/internal/backends"
)

// openAIProvider implements Provider against an OpenAI-compatible
// HTTP endpoint. Used for both literal OpenAI (provider="openai") and
// SGLang/vLLM/Ollama (other prefixes) — they all share the wire format.
type openAIProvider struct {
	name    string
	modelID string
	client  openai.Client

	retryAttempts uint64
	retryInitial  time.Duration
	retryMax      time.Duration
}

// Compile-time interface check.
var _ LLMProvider = (*openAIProvider)(nil)

// Option configures an openAIProvider.
type Option func(*openAIProvider)

// WithRetry tunes the retry policy. Defaults: 3 attempts, 250ms initial,
// 5s max. Zero values keep the default.
func WithRetry(attempts uint64, initial, max time.Duration) Option {
	return func(p *openAIProvider) {
		if attempts > 0 {
			p.retryAttempts = attempts
		}
		if initial > 0 {
			p.retryInitial = initial
		}
		if max > 0 {
			p.retryMax = max
		}
	}
}

// NewOpenAI constructs a Provider for the given backend. The API key
// is resolved via os.Getenv(backend.APIKeyEnvVar); an unset value is
// permitted (some local backends like Ollama don't require auth).
func NewOpenAI(b *backends.Backend, opts ...Option) LLMProvider {
	// LLM backends always have a model_id (enforced at registration).
	// Defensive nil-check: if a caller built a Backend by hand without
	// setting it, treat ParseModelID as having no prefix and no model.
	rawModelID := ""
	if b.ModelID != nil {
		rawModelID = *b.ModelID
	}
	_, modelID := ParseModelID(rawModelID)

	requestOpts := []option.RequestOption{
		option.WithBaseURL(b.Endpoint),
	}
	if b.APIKeyEnvVar != "" {
		if key := os.Getenv(b.APIKeyEnvVar); key != "" {
			requestOpts = append(requestOpts, option.WithAPIKey(key))
		}
	}

	p := &openAIProvider{
		name:          b.Name,
		modelID:       modelID,
		client:        openai.NewClient(requestOpts...),
		retryAttempts: 3,
		retryInitial:  250 * time.Millisecond,
		retryMax:      5 * time.Second,
	}
	for _, o := range opts {
		o(p)
	}
	return p
}

func (p *openAIProvider) Name() string    { return p.name }
func (p *openAIProvider) ModelID() string { return p.modelID }

func (p *openAIProvider) Chat(ctx context.Context, params openai.ChatCompletionNewParams) (*openai.ChatCompletion, error) {
	params.Model = p.modelID

	var out *openai.ChatCompletion
	err := p.retry(ctx, func() error {
		resp, err := p.client.Chat.Completions.New(ctx, params)
		if err != nil {
			return err
		}
		out = resp
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("provider %s: chat: %w", p.name, err)
	}
	return out, nil
}

// ChatStream opens a streaming chat completion. Retries do not apply
// to streaming because resumption mid-stream is not safe (partial
// content has already been delivered).
func (p *openAIProvider) ChatStream(ctx context.Context, params openai.ChatCompletionNewParams) *ssestream.Stream[openai.ChatCompletionChunk] {
	params.Model = p.modelID
	return p.client.Chat.Completions.NewStreaming(ctx, params)
}

// retry runs fn with exponential backoff. Retries only fire on errors
// classified as transient (see isTransient); other errors fail fast.
func (p *openAIProvider) retry(ctx context.Context, fn func() error) error {
	policy := backoff.NewExponentialBackOff()
	policy.InitialInterval = p.retryInitial
	policy.MaxInterval = p.retryMax
	policy.MaxElapsedTime = 0 // bounded by MaxRetries below
	policy.Reset()

	wrapped := backoff.WithMaxRetries(policy, p.retryAttempts-1)
	wrappedCtx := backoff.WithContext(wrapped, ctx)

	return backoff.Retry(func() error {
		if err := fn(); err != nil {
			if !isTransient(err) {
				return backoff.Permanent(err)
			}
			return err
		}
		return nil
	}, wrappedCtx)
}

// isTransient reports whether err is retriable. Network errors and
// 5xx / 429 responses are; everything else is permanent.
func isTransient(err error) bool {
	if err == nil {
		return false
	}
	if apiErr, ok := errors.AsType[*openai.Error](err); ok {
		switch {
		case apiErr.StatusCode == http.StatusTooManyRequests:
			return true
		case apiErr.StatusCode >= 500 && apiErr.StatusCode < 600:
			return true
		default:
			return false
		}
	}
	// Network errors, EOFs, context errors at the transport layer are
	// classified transient. context.Canceled / DeadlineExceeded reaching
	// here without an openai.Error wrap means the request never landed;
	// the retry loop's ctx will short-circuit the next attempt anyway.
	return true
}
