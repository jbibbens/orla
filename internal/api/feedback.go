package api

import (
	"net/http"
	"time"

	"github.com/go-chi/chi/v5"

	"github.com/harvard-cns/orla/internal/telemetry"
)

// FeedbackSink is the subset of telemetry.FeedbackWriter used by the
// handler. Tests pass a fake.
type FeedbackSink interface {
	Submit(fb *telemetry.Feedback) bool
}

// FeedbackMetrics is the subset of metrics.Metrics consumed by the
// feedback handler. Nil is allowed for tests.
type FeedbackMetrics interface {
	IncFeedback(stage string)
}

// FeedbackDeps bundles handler dependencies.
type FeedbackDeps struct {
	Sink    FeedbackSink
	Metrics FeedbackMetrics
}

// RegisterFeedbackRoutes mounts POST /v1/feedback.
func RegisterFeedbackRoutes(r chi.Router, deps FeedbackDeps) {
	h := &feedbackHandler{deps: deps}
	r.Post("/v1/feedback", h.submit)
}

type feedbackHandler struct {
	deps FeedbackDeps
}

// feedbackRequest is the wire shape. completion_id and stage_id are
// required; the developer's SDK can pull stage_id from the original
// chat completion call. Doing it on the developer side avoids a sync
// DB lookup against an async-batched table.
type feedbackRequest struct {
	CompletionID string   `json:"completion_id"`
	StageID      string   `json:"stage_id"`
	WorkflowRun  string   `json:"workflow_run,omitempty"`
	Rating       *float64 `json:"rating,omitempty"`
	Labels       []string `json:"labels,omitempty"`
	Notes        string   `json:"notes,omitempty"`
}

func (h *feedbackHandler) submit(w http.ResponseWriter, r *http.Request) {
	var req feedbackRequest
	if !decodeJSON(w, r, &req) {
		return
	}
	if req.CompletionID == "" {
		writeErrorMsg(w, http.StatusBadRequest, "completion_id is required")
		return
	}
	if req.StageID == "" {
		writeErrorMsg(w, http.StatusBadRequest, "stage_id is required")
		return
	}
	if req.Rating != nil && (*req.Rating < 0 || *req.Rating > 1) {
		writeErrorMsg(w, http.StatusBadRequest, "rating must be between 0 and 1")
		return
	}

	if h.deps.Sink != nil {
		_ = h.deps.Sink.Submit(&telemetry.Feedback{
			CompletionID: req.CompletionID,
			StageID:      req.StageID,
			WorkflowRun:  req.WorkflowRun,
			Rating:       req.Rating,
			Labels:       req.Labels,
			Notes:        req.Notes,
			CreatedAt:    time.Now(),
		})
	}
	if h.deps.Metrics != nil {
		h.deps.Metrics.IncFeedback(req.StageID)
	}

	// 202 Accepted: write is async-batched.
	writeJSON(w, http.StatusAccepted, map[string]string{"status": "accepted"})
}
