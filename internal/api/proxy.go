package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"

	"github.com/harvard-cns/orla/internal/scheduler"
	"github.com/harvard-cns/orla/internal/stages"
	"github.com/harvard-cns/orla/internal/telemetry"
)

// Header names. Lowercased on lookup; chi/net.http normalize incoming
// headers to canonical case so we can compare via Get().
const (
	HeaderStage       = "X-Orla-Stage"
	HeaderWorkflowRun = "X-Orla-Workflow-Run"
	HeaderTagPrefix   = "X-Orla-Tag-"
)

// metadata key fallbacks for SDKs that can't easily set headers.
const (
	metaStage       = "orla.stage"
	metaWorkflowRun = "orla.workflow_run"
)

// CompletionSink receives one record per dispatched chat completion.
// Implementations are typically wrapping telemetry.CompletionWriter;
// nil is acceptable for tests that don't care about records.
type CompletionSink interface {
	Submit(rec *telemetry.CompletionRecord) bool
}

// ProxyMetrics is the subset of metrics.Metrics consumed by the proxy
// hot path. Nil is allowed for tests that don't care about metrics.
type ProxyMetrics interface {
	IncRequest(stage, backend, status string)
	ObserveBackendLatency(backend string, seconds float64)
}

// ProxyDeps bundles the dependencies of the proxy handler.
type ProxyDeps struct {
	Stages         stages.Registry
	Scheduler      *scheduler.Scheduler
	CompletionSink CompletionSink
	Metrics        ProxyMetrics
}

// RegisterProxyRoutes mounts POST /v1/chat/completions.
func RegisterProxyRoutes(r chi.Router, deps ProxyDeps) {
	h := &proxyHandler{deps: deps}
	r.Post("/v1/chat/completions", h.chatCompletions)
}

type proxyHandler struct {
	deps ProxyDeps
}

// requestContext aggregates the identity metadata we extract from
// headers + body fallbacks. Stages and tags are persisted later by the
// completion-records writer; the proxy only consumes Stage and (in a
// future phase) ReasoningEffort.
type requestContext struct {
	Stage       string
	WorkflowRun string
	Tags        map[string]string
}

func (h *proxyHandler) chatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("read body: %w", err))
		return
	}

	// Peek for the stream flag — openai.ChatCompletionNewParams doesn't
	// carry it; client-side it's controlled by which method (New vs
	// NewStreaming) is called.
	var peek struct {
		Stream bool `json:"stream"`
	}
	if len(body) > 0 {
		_ = json.Unmarshal(body, &peek)
	}

	var params openai.ChatCompletionNewParams
	if err := json.Unmarshal(body, &params); err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("decode chat completion params: %w", err))
		return
	}
	if len(params.Messages) == 0 {
		writeErrorMsg(w, http.StatusBadRequest, "messages is required and must not be empty")
		return
	}

	rc := extractRequestContext(r, params.Metadata)
	if rc.Stage == "" {
		writeErrorMsg(w, http.StatusBadRequest, "stage is required (set X-Orla-Stage or metadata.orla.stage)")
		return
	}

	stage, err := h.deps.Stages.GetOrCreate(r.Context(), rc.Stage)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	backendName := stage.Backend
	if backendName == "" {
		// Fall back to the client-supplied model field if the stage
		// has no mapping yet.
		backendName = string(params.Model)
	}
	if backendName == "" {
		writeErrorMsg(w, http.StatusBadRequest,
			fmt.Sprintf("stage %q has no backend mapping and request did not specify model", rc.Stage))
		return
	}

	// Apply stage-level inference policy.
	if stage.ReasoningEffort != "" {
		params.ReasoningEffort = shared.ReasoningEffort(stage.ReasoningEffort)
	}

	if peek.Stream {
		h.serveStreaming(w, r, rc, backendName, params)
		return
	}
	h.serveNonStreaming(w, r, rc, backendName, params)
}

func (h *proxyHandler) serveNonStreaming(w http.ResponseWriter, r *http.Request, rc *requestContext, backendName string, params openai.ChatCompletionNewParams) {
	p, release, err := h.deps.Scheduler.AcquireLLM(r.Context(), backendName)
	if err != nil {
		statusForSchedulerErr(w, err, backendName)
		return
	}
	defer release()

	start := time.Now()
	resp, err := p.Chat(r.Context(), params)
	latencyMs := int(time.Since(start) / time.Millisecond)

	if err != nil {
		h.recordCompletion(&completionInputs{
			completionID: uuid.NewString(),
			rc:           rc,
			backend:      backendName,
			status:       "error",
			latencyMs:    &latencyMs,
		})
		h.emitMetrics(rc.Stage, backendName, "error", latencyMs)
		writeUpstreamError(w, err)
		return
	}

	// The body's "model" field is the canonical source of truth for
	// which backend ran the call. Overwrite with the resolved backend
	// name so the developer can correlate against /api/v1/backends.
	resp.Model = backendName

	completionID := resp.ID
	if completionID == "" {
		completionID = uuid.NewString()
		resp.ID = completionID
	}
	prompt := int(resp.Usage.PromptTokens)
	completion := int(resp.Usage.CompletionTokens)
	h.recordCompletion(&completionInputs{
		completionID:     completionID,
		rc:               rc,
		backend:          backendName,
		status:           "success",
		promptTokens:     &prompt,
		completionTokens: &completion,
		latencyMs:        &latencyMs,
	})
	h.emitMetrics(rc.Stage, backendName, "success", latencyMs)

	writeJSON(w, http.StatusOK, resp)
}

func (h *proxyHandler) serveStreaming(w http.ResponseWriter, r *http.Request, rc *requestContext, backendName string, params openai.ChatCompletionNewParams) {
	p, release, err := h.deps.Scheduler.AcquireLLM(r.Context(), backendName)
	if err != nil {
		statusForSchedulerErr(w, err, backendName)
		return
	}
	defer release()

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeErrorMsg(w, http.StatusInternalServerError, "streaming not supported by this server")
		return
	}

	start := time.Now()
	var completionID string
	var promptTokens, completionTokens int

	stream := p.ChatStream(r.Context(), params)
	defer func() { _ = stream.Close() }()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	for stream.Next() {
		chunk := stream.Current()
		if completionID == "" && chunk.ID != "" {
			completionID = chunk.ID
		}
		// Some upstreams (notably OpenAI) only attach Usage to the
		// final chunk. We capture whichever value lands last.
		if chunk.Usage.PromptTokens > 0 {
			promptTokens = int(chunk.Usage.PromptTokens)
		}
		if chunk.Usage.CompletionTokens > 0 {
			completionTokens = int(chunk.Usage.CompletionTokens)
		}
		chunk.Model = backendName
		data, err := json.Marshal(chunk)
		if err != nil {
			return
		}
		if _, werr := fmt.Fprintf(w, "data: %s\n\n", data); werr != nil {
			return
		}
		flusher.Flush()
	}

	latencyMs := int(time.Since(start) / time.Millisecond)

	if err := stream.Err(); err != nil && !errors.Is(err, io.EOF) {
		errEnv := errorEnvelope{Error: errorBody{Message: err.Error(), Type: "server_error"}}
		if data, mErr := json.Marshal(errEnv); mErr == nil {
			_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
		if completionID == "" {
			completionID = uuid.NewString()
		}
		h.recordCompletion(&completionInputs{
			completionID: completionID,
			rc:           rc,
			backend:      backendName,
			status:       "error",
			latencyMs:    &latencyMs,
		})
		h.emitMetrics(rc.Stage, backendName, "error", latencyMs)
		return
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()

	if completionID == "" {
		completionID = uuid.NewString()
	}
	pt, ct := &promptTokens, &completionTokens
	if promptTokens == 0 {
		pt = nil
	}
	if completionTokens == 0 {
		ct = nil
	}
	h.recordCompletion(&completionInputs{
		completionID:     completionID,
		rc:               rc,
		backend:          backendName,
		status:           "success",
		promptTokens:     pt,
		completionTokens: ct,
		latencyMs:        &latencyMs,
	})
	h.emitMetrics(rc.Stage, backendName, "success", latencyMs)
}

type completionInputs struct {
	completionID     string
	rc               *requestContext
	backend          string
	status           string
	promptTokens     *int
	completionTokens *int
	latencyMs        *int
	costUSD          *float64
}

// emitMetrics is a no-op when ProxyMetrics is nil.
func (h *proxyHandler) emitMetrics(stage, backend, status string, latencyMs int) {
	if h.deps.Metrics == nil {
		return
	}
	h.deps.Metrics.IncRequest(stage, backend, status)
	h.deps.Metrics.ObserveBackendLatency(backend, float64(latencyMs)/1000.0)
}

func (h *proxyHandler) recordCompletion(in *completionInputs) {
	if h.deps.CompletionSink == nil {
		return
	}
	rec := &telemetry.CompletionRecord{
		CompletionID:     in.completionID,
		StageID:          in.rc.Stage,
		WorkflowRun:      in.rc.WorkflowRun,
		Backend:          in.backend,
		Status:           in.status,
		PromptTokens:     in.promptTokens,
		CompletionTokens: in.completionTokens,
		LatencyMs:        in.latencyMs,
		CostUSD:          in.costUSD,
		Tags:             in.rc.Tags,
		CreatedAt:        time.Now(),
	}
	_ = h.deps.CompletionSink.Submit(rec)
}

// extractRequestContext parses X-Orla-* headers and metadata fallbacks.
// Header values win over body metadata when both are set.
func extractRequestContext(r *http.Request, metadata shared.Metadata) *requestContext {
	rc := &requestContext{Tags: make(map[string]string)}

	rc.Stage = r.Header.Get(HeaderStage)
	if rc.Stage == "" {
		rc.Stage = metadata[metaStage]
	}

	rc.WorkflowRun = r.Header.Get(HeaderWorkflowRun)
	if rc.WorkflowRun == "" {
		rc.WorkflowRun = metadata[metaWorkflowRun]
	}

	for name, values := range r.Header {
		if !strings.HasPrefix(name, HeaderTagPrefix) {
			continue
		}
		if len(values) == 0 {
			continue
		}
		key := strings.ToLower(strings.TrimPrefix(name, HeaderTagPrefix))
		if key == "" {
			continue
		}
		rc.Tags[key] = values[0]
	}
	return rc
}

func statusForSchedulerErr(w http.ResponseWriter, err error, backendName string) {
	if errors.Is(err, scheduler.ErrUnknownBackend) {
		writeError(w, http.StatusBadGateway,
			fmt.Errorf("backend %q is not registered", backendName))
		return
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		writeError(w, http.StatusRequestTimeout, err)
		return
	}
	writeError(w, http.StatusInternalServerError, err)
}

func writeUpstreamError(w http.ResponseWriter, err error) {
	if apiErr, ok := errors.AsType[*openai.Error](err); ok {
		// Mirror the upstream status when sensible (4xx). 5xx surfaces
		// as a 502 to make "orla failed" distinguishable from "client
		// asked for something silly".
		if apiErr.StatusCode >= 400 && apiErr.StatusCode < 500 {
			writeError(w, apiErr.StatusCode, err)
			return
		}
	}
	writeError(w, http.StatusBadGateway, err)
}
