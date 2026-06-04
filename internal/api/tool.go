// Tool HTTP route: POST /v1/tools/{kind}
//
// The wire shape mirrors the OpenAI-compatible chat path's role of
// "developer hands orla a request, orla routes by stage, records
// completion + cost, returns the response". For tool dispatches the
// request and response are kind-specific JSON payloads; orla is
// agnostic to their shapes — it just routes, accounts, and feeds
// the mapper.

package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/google/uuid"

	"github.com/harvard-cns/orla/internal/backends"
	"github.com/harvard-cns/orla/internal/provider"
	"github.com/harvard-cns/orla/internal/scheduler"
	"github.com/harvard-cns/orla/internal/stages"
	"github.com/harvard-cns/orla/internal/telemetry"
)

// ToolDeps bundles the tool handler's dependencies. Parallels ProxyDeps.
type ToolDeps struct {
	Stages         stages.Registry
	Scheduler      *scheduler.Scheduler
	Backends       backends.Registry
	CompletionSink CompletionSink
	Metrics        ProxyMetrics
}

// RegisterToolRoutes mounts POST /v1/tools/{kind}.
func RegisterToolRoutes(r chi.Router, deps ToolDeps) {
	h := &toolHandler{deps: deps}
	r.Post("/v1/tools/{kind}", h.invoke)
}

type toolHandler struct {
	deps ToolDeps
}

// invoke is the POST /v1/tools/{kind} handler. Flow:
//
//   1. Parse the URL kind + extract request context (stage, tags).
//   2. Resolve the stage to a backend via the stages registry.
//   3. Look up the backend record (need its Kind + cost rate).
//   4. Verify backend.Kind == "tool" and backend.ToolKind == URL kind.
//   5. Acquire a worker slot via the scheduler (concurrency cap + rate limit).
//   6. Decode the body into a provider.ToolRequest and dispatch.
//   7. Record completion + emit metrics + return the response.
func (h *toolHandler) invoke(w http.ResponseWriter, r *http.Request) {
	kind := chi.URLParam(r, "kind")
	if kind == "" {
		writeErrorMsg(w, http.StatusBadRequest, "tool kind is required in URL path")
		return
	}

	rc := extractRequestContext(r, nil)
	if rc.Stage == "" {
		writeErrorMsg(w, http.StatusBadRequest,
			"stage is required (set X-Orla-Stage)")
		return
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadRequest, fmt.Errorf("read body: %w", err))
		return
	}

	stage, err := h.deps.Stages.GetOrCreate(r.Context(), rc.Stage)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err)
		return
	}
	backendName := stage.Backend
	if backendName == "" {
		writeErrorMsg(w, http.StatusBadRequest,
			fmt.Sprintf("stage %q has no backend mapping", rc.Stage))
		return
	}

	// Backend record lookup — we need Kind, ToolKind, and the cost rate.
	bk, err := h.deps.Backends.Get(r.Context(), backendName)
	if err != nil {
		if errors.Is(err, backends.ErrNotFound) {
			writeError(w, http.StatusBadGateway,
				fmt.Errorf("backend %q is not registered", backendName))
			return
		}
		writeError(w, http.StatusInternalServerError, err)
		return
	}

	if bk.Kind != backends.KindTool {
		writeError(w, http.StatusBadRequest,
			fmt.Errorf("backend %q is kind=%q; the /v1/tools route requires kind=tool",
				bk.Name, bk.Kind))
		return
	}
	if bk.ToolKind == nil || *bk.ToolKind != kind {
		got := ""
		if bk.ToolKind != nil {
			got = *bk.ToolKind
		}
		writeError(w, http.StatusBadRequest,
			fmt.Errorf("backend %q has tool_kind=%q; request asked for %q",
				bk.Name, got, kind))
		return
	}

	tp, release, err := h.deps.Scheduler.AcquireTool(r.Context(), backendName)
	if err != nil {
		statusForSchedulerErr(w, err, backendName)
		return
	}
	defer release()

	req := provider.ToolRequest{
		Kind:    kind,
		Payload: body,
	}

	completionID := uuid.NewString()

	start := time.Now()
	resp, err := tp.Invoke(r.Context(), req)
	latencyMs := int(time.Since(start) / time.Millisecond)

	if err != nil {
		h.recordToolCompletion(&toolCompletionInputs{
			completionID: completionID,
			rc:           rc,
			backend:      backendName,
			toolKind:     kind,
			status:       "error",
			latencyMs:    &latencyMs,
		})
		h.emitMetrics(rc.Stage, backendName, "error", latencyMs)
		writeError(w, http.StatusBadGateway, err)
		return
	}

	// Compute cost: gpu_seconds × $/gpu-second.
	var costUSD *float64
	if bk.CostPerGPUSecond != nil && resp.GPUSeconds > 0 {
		c := resp.GPUSeconds * *bk.CostPerGPUSecond
		costUSD = &c
	}
	gpuSeconds := resp.GPUSeconds

	h.recordToolCompletion(&toolCompletionInputs{
		completionID: completionID,
		rc:           rc,
		backend:      backendName,
		toolKind:     kind,
		status:       "success",
		latencyMs:    &latencyMs,
		costUSD:      costUSD,
		gpuSeconds:   &gpuSeconds,
	})
	h.emitMetrics(rc.Stage, backendName, "success", latencyMs)

	// Return the response envelope verbatim so downstream callers see
	// the same shape the underlying ToolProvider returned.
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Orla-Completion-Id", completionID)
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(resp)
}

type toolCompletionInputs struct {
	completionID string
	rc           *requestContext
	backend      string
	toolKind     string
	status       string
	latencyMs    *int
	costUSD      *float64
	gpuSeconds   *float64
}

func (h *toolHandler) recordToolCompletion(in *toolCompletionInputs) {
	if h.deps.CompletionSink == nil {
		return
	}
	_ = h.deps.CompletionSink.Submit(&telemetry.CompletionRecord{
		CompletionID: in.completionID,
		StageID:      in.rc.Stage,
		WorkflowRun:  in.rc.WorkflowRun,
		Backend:      in.backend,
		Status:       in.status,
		LatencyMs:    in.latencyMs,
		CostUSD:      in.costUSD,
		GPUSeconds:   in.gpuSeconds,
		ToolKind:     in.toolKind,
		Tags:         in.rc.Tags,
		CreatedAt:    time.Now(),
	})
}

func (h *toolHandler) emitMetrics(stage, backend, status string, latencyMs int) {
	if h.deps.Metrics == nil {
		return
	}
	h.deps.Metrics.IncRequest(stage, backend, status)
	h.deps.Metrics.ObserveBackendLatency(backend, float64(latencyMs)/1000.0)
}
