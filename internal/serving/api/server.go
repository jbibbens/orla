// Package api provides the HTTP API for the serving layer daemon.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/harvard-cns/orla/internal/core"
	"github.com/harvard-cns/orla/internal/model"
	"github.com/harvard-cns/orla/internal/serving"
	_ "github.com/harvard-cns/orla/internal/serving/metrics" // register Prometheus metrics
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.uber.org/zap"
	"golang.org/x/time/rate"
)

// SSE event types for execute stream
const (
	sseEventContent  = "content"
	sseEventThinking = "thinking"
	sseEventDone     = "done"

	// maxRequestBodyBytes limits JSON request body size to prevent memory exhaustion (10MB)
	maxRequestBodyBytes = 10 << 20
	// readTimeout is the time we wait for the http client to send the request body
	readTimeout = 30 * time.Second
	// readHeaderTimeout is the time we wait for the http client to send the request header
	readHeaderTimeout = 10 * time.Second

	// writeTimeout is the time we wait for from the request being processed to the response being sent back to the client.
	// This can be long for inference and streaming.
	writeTimeout = 30 * time.Minute
	// idleTimeout is the time we wait for the http client to send the next request.
	// If the client doesn't send a request within this time, the connection is closed.
	idleTimeout = 120 * time.Second
	// maxHeaderBytes is the maximum size of the HTTP header. This is 1MB by default.
	// This is used to prevent HTTP request smuggling attacks.
	maxHeaderBytes = 1 << 20
)

// AgenticServer is the HTTP API server for the daemon
type AgenticServer struct {
	layer      *serving.AgenticLayer
	httpServer *http.Server
	mux        *http.ServeMux
}

// ServerOptions configures the API server.
type ServerOptions struct {
	// RateLimitRPS limits requests per second for execute and backends. 0 = disabled.
	RateLimitRPS int
}

// NewAgenticServer creates a new daemon API server.
func NewAgenticServer(layer *serving.AgenticLayer, listenAddress string, opts *ServerOptions) *AgenticServer {
	if opts == nil {
		opts = &ServerOptions{}
	}
	mux := http.NewServeMux()
	server := &AgenticServer{
		layer: layer,
		mux:   mux,
		httpServer: &http.Server{
			Addr:              listenAddress,
			Handler:           recoveryMiddleware(mux),
			ReadTimeout:       readTimeout,
			ReadHeaderTimeout: readHeaderTimeout,
			WriteTimeout:      writeTimeout, // long-running inference and streaming
			IdleTimeout:       idleTimeout,
			MaxHeaderBytes:    maxHeaderBytes, // 1MB
		},
	}

	server.registerRoutes(opts.RateLimitRPS)
	return server
}

// recoveryMiddleware catches panics in handlers, logs the stack, and returns 500.
func recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				zap.L().Error("handler panic", zap.Any("panic", err), zap.Stack("stack"))
				http.Error(w, "internal server error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func (s *AgenticServer) registerRoutes(rateLimitRPS int) {
	s.mux.HandleFunc("GET /api/v1/health", s.handleHealth)
	s.mux.Handle("GET /metrics", promhttp.Handler())

	var executeHandler, backendsHandler http.Handler
	executeHandler = http.HandlerFunc(s.handleExecute)
	backendsHandler = http.HandlerFunc(s.handleRegisterBackend)
	if rateLimitRPS > 0 {
		limiter := rate.NewLimiter(rate.Limit(rateLimitRPS), rateLimitRPS) // burst = RPS for predictable limiting
		executeHandler = rateLimitMiddleware(limiter, executeHandler)
		backendsHandler = rateLimitMiddleware(limiter, backendsHandler)
	}
	s.mux.Handle("POST /api/v1/execute", executeHandler)
	s.mux.Handle("POST /api/v1/backends", backendsHandler)

	s.mux.HandleFunc("GET /api/v1/backends", s.handleListBackends)
	s.mux.HandleFunc("POST /api/v1/workflow/complete", s.handleWorkflowComplete)
}

// rateLimitMiddleware returns 429 when the limiter rejects the request.
func rateLimitMiddleware(limiter *rate.Limiter, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !limiter.Allow() {
			http.Error(w, "rate limit exceeded", http.StatusTooManyRequests)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// Start starts the HTTP server
func (s *AgenticServer) Start() error {
	zap.L().Info("Starting daemon API server",
		zap.String("address", s.httpServer.Addr))
	return s.httpServer.ListenAndServe()
}

// Shutdown gracefully shuts down the HTTP server
func (s *AgenticServer) Shutdown(ctx context.Context) error {
	return s.httpServer.Shutdown(ctx)
}

func (s *AgenticServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	core.WriteJSONResponse(w, map[string]string{
		"status": "healthy",
	})
}

// ExecuteRequest is the request body for the execute endpoint.
// Inference options (stream, max_tokens, temperature, top_p) are embedded so the JSON body stays flat.
type ExecuteRequest struct {
	Backend  string          `json:"backend"`
	StageID  string          `json:"stage_id,omitempty"`
	Prompt   string          `json:"prompt,omitempty"`
	Messages []model.Message `json:"messages,omitempty"`
	Tools    []*mcp.Tool     `json:"tools,omitempty"`
	model.InferenceOptions

	WorkflowID  string `json:"workflow_id,omitempty"`
	CachePolicy string `json:"cache_policy,omitempty"`
}

// ExecuteResponse is the response body for the execute endpoint.
type ExecuteResponse struct {
	Success  bool            `json:"success"`
	Response *model.Response `json:"response,omitempty"`
	Error    string          `json:"error,omitempty"`
}

func (s *AgenticServer) handleExecute(w http.ResponseWriter, r *http.Request) {
	body := http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	var req ExecuteRequest
	if err := json.NewDecoder(body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Backend == "" {
		http.Error(w, "backend is required", http.StatusBadRequest)
		return
	}

	messages := req.Messages
	if req.Prompt != "" {
		messages = append(messages, model.Message{
			Role:    model.MessageRoleUser,
			Content: req.Prompt,
		})
	}

	ctx := r.Context()
	opts := req.InferenceOptions
	switch opts.GetSchedulingPolicy() {
	case model.SchedulingPolicyFCFS, model.SchedulingPolicyPriority:
		// supported
	default:
		http.Error(w, fmt.Sprintf("unsupported scheduling policy %q", opts.SchedulingPolicy), http.StatusBadRequest)
		return
	}
	switch opts.RequestSchedulingPolicy {
	case "", model.RequestSchedulingPolicyFIFO, model.RequestSchedulingPolicyPriority:
		// supported
	default:
		http.Error(w, fmt.Sprintf("unsupported request scheduling policy %q", opts.RequestSchedulingPolicy), http.StatusBadRequest)
		return
	}

	stageID := req.StageID
	chatOpts := serving.ChatOptions{
		WorkflowID:  req.WorkflowID,
		CachePolicy: req.CachePolicy,
	}

	if req.Stream {
		s.handleExecuteStream(w, ctx, req.Backend, stageID, messages, req.Tools, opts, chatOpts)
		return
	}

	response, err := s.layer.Execute(ctx, req.Backend, stageID, messages, req.Tools, opts, chatOpts)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		core.WriteJSONResponse(w, ExecuteResponse{
			Success: false,
			Error:   err.Error(),
		})
		return
	}

	zap.L().Debug("Executed inference via API",
		zap.String("backend", req.Backend),
		zap.Int("response_length", len(response.Content)))

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	core.WriteJSONResponse(w, ExecuteResponse{
		Success:  true,
		Response: response,
	})
}

func (s *AgenticServer) handleExecuteStream(w http.ResponseWriter, ctx context.Context, backend, stage string, messages []model.Message, tools []*mcp.Tool, opts model.InferenceOptions, chatOpts serving.ChatOptions) {
	response, eventCh, err := s.layer.ExecuteStream(ctx, backend, stage, messages, tools, opts, chatOpts)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		core.WriteJSONResponse(w, ExecuteResponse{Success: false, Error: err.Error()})
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher, ok := w.(http.Flusher)
	if !ok {
		zap.L().Error("flusher not supported", zap.String("backend", backend))
		http.Error(w, "flusher not supported", http.StatusInternalServerError)
		return
	}

	if eventCh != nil {
		for ev := range eventCh {
			switch e := ev.(type) {
			case *model.ContentEvent:
				core.WriteSSEResponse(w, flusher, sseEventContent, map[string]string{"content": e.Content})
			case *model.ThinkingEvent:
				core.WriteSSEResponse(w, flusher, sseEventThinking, map[string]string{"thinking": e.Content})
			case *model.ToolCallEvent:
				core.WriteSSEResponse(w, flusher, "tool_call", map[string]any{"name": e.Name, "arguments": e.Arguments})
			}
		}
	}

	core.WriteSSEResponse(w, flusher, sseEventDone, ExecuteResponse{
		Success:  true,
		Response: response,
	})
}

// RegisterBackendRequest is the request body for registering an LLM backend.
type RegisterBackendRequest struct {
	Name           string `json:"name"`                      // backend name (used as "backend" in execute requests)
	Endpoint       string `json:"endpoint"`                  // e.g. "http://vllm:8000/v1", "http://localhost:11434"
	Type           string `json:"type"`                      // "openai" or "sglang"
	ModelID        string `json:"model_id"`                  // full model identifier e.g. "openai:Qwen/Qwen3-4B-Instruct-2507", "openai:llama3"
	APIKeyEnvVar   string `json:"api_key_env_var,omitempty"` // optional env var name for API key (for openai-type backends)
	MaxConcurrency int    `json:"max_concurrency,omitempty"` // max concurrent requests to this backend (default 1)
	QueueCapacity  int    `json:"queue_capacity,omitempty"`  // max queued requests; 0 = default (4096)
}

// RegisterBackendResponse is the response body for register backend.
type RegisterBackendResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

func (s *AgenticServer) handleRegisterBackend(w http.ResponseWriter, r *http.Request) {
	body := http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	var req RegisterBackendRequest
	if err := json.NewDecoder(body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Name == "" {
		http.Error(w, "name is required", http.StatusBadRequest)
		return
	}
	if req.Endpoint == "" {
		http.Error(w, "endpoint is required", http.StatusBadRequest)
		return
	}
	if req.Type == "" {
		http.Error(w, "type is required", http.StatusBadRequest)
		return
	}
	if req.ModelID == "" {
		http.Error(w, "model_id is required", http.StatusBadRequest)
		return
	}

	backendType := core.LLMInferenceAPIType(req.Type)
	switch backendType {
	case core.LLMInferenceAPITypeOpenAI, core.LLMInferenceAPITypeSGLang:
		// supported
	default:
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		core.WriteJSONResponse(w, RegisterBackendResponse{
			Success: false,
			Error:   fmt.Sprintf("type must be one of: openai, sglang; got %q", req.Type),
		})
		return
	}

	backend := &core.LLMBackend{
		Endpoint:       req.Endpoint,
		Type:           backendType,
		APIKeyEnvVar:   req.APIKeyEnvVar,
		MaxConcurrency: req.MaxConcurrency,
		QueueCapacity:  req.QueueCapacity,
	}
	s.layer.AddLLMBackend(req.Name, backend, req.ModelID)

	zap.L().Info("Registered LLM backend",
		zap.String("name", req.Name),
		zap.String("endpoint", req.Endpoint),
		zap.String("model_id", req.ModelID),
		zap.Int("max_concurrency", req.MaxConcurrency),
		zap.Int("queue_capacity", req.QueueCapacity))

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	core.WriteJSONResponse(w, RegisterBackendResponse{Success: true})
}

// ListBackendsResponse is the response body for list backends.
type ListBackendsResponse struct {
	Backends []string `json:"backends"`
}

func (s *AgenticServer) handleListBackends(w http.ResponseWriter, r *http.Request) {
	names := s.layer.ListLLMBackends()
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	core.WriteJSONResponse(w, ListBackendsResponse{Backends: names})
}

// WorkflowCompleteRequest is the request body for the workflow/complete endpoint.
type WorkflowCompleteRequest struct {
	WorkflowID string   `json:"workflow_id"`
	Backends   []string `json:"backends"`
}

// WorkflowCompleteResponse is the response body for the workflow/complete endpoint.
type WorkflowCompleteResponse struct {
	Success bool   `json:"success"`
	Error   string `json:"error,omitempty"`
}

func (s *AgenticServer) handleWorkflowComplete(w http.ResponseWriter, r *http.Request) {
	body := http.MaxBytesReader(w, r.Body, maxRequestBodyBytes)
	var req WorkflowCompleteRequest
	if err := json.NewDecoder(body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}
	if req.WorkflowID == "" {
		http.Error(w, "workflow_id is required", http.StatusBadRequest)
		return
	}

	s.layer.NotifyWorkflowComplete(r.Context(), req.WorkflowID, req.Backends)

	zap.L().Debug("Workflow complete notification",
		zap.String("workflow_id", req.WorkflowID),
		zap.Strings("backends", req.Backends))

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	core.WriteJSONResponse(w, WorkflowCompleteResponse{Success: true})
}
