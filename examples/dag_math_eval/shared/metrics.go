package shared

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"
)

// StageMetrics is the timing and token usage for one stage within a workflow.
type StageMetrics struct {
	StepID           int   `json:"step_id"`
	StartTimeMs      int64 `json:"start_time_ms"`
	EndTimeMs        int64 `json:"end_time_ms"`
	DurationMs       int64 `json:"duration_ms"`
	PromptTokens     int   `json:"prompt_tokens,omitempty"`
	CompletionTokens int   `json:"completion_tokens,omitempty"`
	QueueWaitMs      int64 `json:"queue_wait_ms,omitempty"`
	BackendLatencyMs int64 `json:"backend_latency_ms,omitempty"`
	TTFTMs           int64 `json:"ttft_ms,omitempty"`
	TPOTMs           int64 `json:"tpot_ms,omitempty"`
}

// WorkflowMetrics is the timing and token usage for one DAG-Math workflow.
type WorkflowMetrics struct {
	ProblemID             int            `json:"problem_id"`
	NumStages             int            `json:"num_stages"`
	Difficulty            float64        `json:"difficulty"`
	StartTimeMs           int64          `json:"start_time_ms"`
	EndTimeMs             int64          `json:"end_time_ms"`
	DurationMs            int64          `json:"duration_ms"`
	TotalPromptTokens     int            `json:"total_prompt_tokens"`
	TotalCompletionTokens int            `json:"total_completion_tokens"`
	Stages                []StageMetrics `json:"stages"`
}

// RunMetrics is the full run output.
type RunMetrics struct {
	ExperimentName        string            `json:"experiment_name"`
	StartTimeMs           int64             `json:"start_time_ms"`
	EndTimeMs             int64             `json:"end_time_ms"`
	TotalDurationMs       int64             `json:"total_duration_ms"`
	WorkflowsCount        int               `json:"workflows_count"`
	TotalPromptTokens     int               `json:"total_prompt_tokens"`
	TotalCompletionTokens int               `json:"total_completion_tokens"`
	Workflows             []WorkflowMetrics `json:"workflows"`
}

// RunMetricsRecorder records timings for a run. Thread-safe for concurrent AddWorkflow calls.
type RunMetricsRecorder struct {
	ExperimentName string
	TotalWorkflows int
	startTime      time.Time
	endTime        time.Time
	workflows      []WorkflowMetrics
	mu             sync.Mutex
}

func NewRunMetricsRecorder(experimentName string) *RunMetricsRecorder {
	return &RunMetricsRecorder{ExperimentName: experimentName}
}

func (r *RunMetricsRecorder) BeginRun() {
	r.startTime = time.Now()
}

func (r *RunMetricsRecorder) EndRun() {
	r.endTime = time.Now()
	log.Printf("[metrics] run: duration=%dms", r.endTime.Sub(r.startTime).Milliseconds())
}

// AddWorkflow appends workflow metrics, logs progress, and flushes the metrics file. Thread-safe.
func (r *RunMetricsRecorder) AddWorkflow(wf WorkflowMetrics) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.workflows = append(r.workflows, wf)
	done := len(r.workflows)
	log.Printf("[%d/%d] problem_%d: stages=%d duration=%dms prompt_tokens=%d completion_tokens=%d",
		done, r.TotalWorkflows,
		wf.ProblemID, wf.NumStages, wf.DurationMs,
		wf.TotalPromptTokens, wf.TotalCompletionTokens)
	if err := r.writeLocked(""); err != nil {
		log.Printf("warning: flush metrics: %v", err)
	}
}

// Write writes the collected metrics to path. Uses MetricsPath if path is empty. Thread-safe.
func (r *RunMetricsRecorder) Write(path string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.writeLocked(path)
}

func (r *RunMetricsRecorder) writeLocked(path string) error {
	if path == "" {
		path = os.Getenv("METRICS_PATH")
		if path == "" {
			path = MetricsPath
		}
	}
	endMs := r.endTime.UnixMilli()
	if r.endTime.IsZero() {
		endMs = time.Now().UnixMilli()
	}
	startMs := r.startTime.UnixMilli()

	var totalPrompt, totalCompletion int
	for _, wf := range r.workflows {
		totalPrompt += wf.TotalPromptTokens
		totalCompletion += wf.TotalCompletionTokens
	}

	m := RunMetrics{
		ExperimentName:        r.ExperimentName,
		StartTimeMs:           startMs,
		EndTimeMs:             endMs,
		TotalDurationMs:       endMs - startMs,
		WorkflowsCount:        len(r.workflows),
		TotalPromptTokens:     totalPrompt,
		TotalCompletionTokens: totalCompletion,
		Workflows:             r.workflows,
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal metrics: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil { //nolint:gosec // G703 - path from caller, example code
		return fmt.Errorf("write metrics: %w", err)
	}
	return nil
}
