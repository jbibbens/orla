// Package shared provides run metrics (end-to-end, per-instance, per-step) for SWE-bench Lite experiments.
// Use RunMetricsRecorder in baseline and two_stage_mapping; call BeginRun, BeginInstance/EndInstance,
// BeginStep/EndStep, then EndRun and Write to a separate output file.

package shared

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"
)

// StepMetrics is the timing and token usage for one ReAct step (one inference + command execution).
type StepMetrics struct {
	StepIndex        int   `json:"step_index"`
	StartTime        int64 `json:"start_time_ms"`
	EndTime          int64 `json:"end_time_ms"`
	DurationMs       int64 `json:"duration_ms"`
	PromptTokens     int   `json:"prompt_tokens,omitempty"`
	CompletionTokens int   `json:"completion_tokens,omitempty"`
}

// InstanceMetrics is the timing and token usage for one SWE-bench instance (all ReAct steps).
// MappedStage is set by experiments that do stage mapping (e.g. "light", "heavy"); empty for baseline.
type InstanceMetrics struct {
	InstanceID       string        `json:"instance_id"`
	MappedStage      string        `json:"mapped_stage,omitempty"`
	StartTime        int64         `json:"start_time_ms"`
	EndTime          int64         `json:"end_time_ms"`
	DurationMs       int64         `json:"duration_ms"`
	Steps            []StepMetrics `json:"steps"`
	StepsCount       int           `json:"steps_count"`
	TotalPromptTokens     int      `json:"total_prompt_tokens"`
	TotalCompletionTokens int      `json:"total_completion_tokens"`
}

// RunMetrics is the full run: end-to-end time, token usage, and per-instance (and per-step) metrics.
type RunMetrics struct {
	ExperimentName        string            `json:"experiment_name"`
	StartTime             int64             `json:"start_time_ms"`
	EndTime               int64             `json:"end_time_ms"`
	TotalDurationMs       int64             `json:"total_duration_ms"`
	Instances             []InstanceMetrics `json:"instances"`
	InstancesCount        int               `json:"instances_count"`
	TotalPromptTokens     int               `json:"total_prompt_tokens"`
	TotalCompletionTokens int               `json:"total_completion_tokens"`
}

// RunMetricsRecorder records timings for a run. Call BeginRun once, then for each instance
// BeginInstance/EndInstance, and for each step BeginStep/EndStep. Call EndRun and Write at the end.
type RunMetricsRecorder struct {
	ExperimentName string
	startTime      time.Time
	endTime        time.Time
	instances      []InstanceMetrics
	current        *InstanceMetrics
	stepStart      time.Time
}

// NewRunMetricsRecorder returns a new recorder. Name is used in the output (e.g. "baseline", "two_stage_mapping").
func NewRunMetricsRecorder(experimentName string) *RunMetricsRecorder {
	return &RunMetricsRecorder{ExperimentName: experimentName}
}

// BeginRun records the run start time. Call once at the start of the experiment.
func (r *RunMetricsRecorder) BeginRun() {
	r.startTime = time.Now()
}

// EndRun records the run end time. Call once at the end before Write.
func (r *RunMetricsRecorder) EndRun() {
	r.endTime = time.Now()
	log.Printf("[metrics] run: duration=%dms", r.endTime.Sub(r.startTime).Milliseconds())
}

// BeginInstance starts timing an instance. Call after PrepareWorkdir for that instance.
func (r *RunMetricsRecorder) BeginInstance(instanceID string) {
	if r.current != nil {
		// Flush previous if not ended
		r.instances = append(r.instances, *r.current)
	}
	now := time.Now()
	r.current = &InstanceMetrics{
		InstanceID: instanceID,
		StartTime:  now.UnixMilli(),
		Steps:      nil,
	}
}

// SetMappedStage records the stage chosen for the current instance (e.g. "light", "heavy").
// Only used by experiments that do stage mapping; call after MapStage and before EndInstance.
func (r *RunMetricsRecorder) SetMappedStage(stage string) {
	if r.current != nil {
		r.current.MappedStage = stage
	}
}

// EndInstance ends timing for the current instance. Call after the ReAct loop and before the next instance.
func (r *RunMetricsRecorder) EndInstance() {
	if r.current == nil {
		return
	}
	now := time.Now()
	r.current.EndTime = now.UnixMilli()
	r.current.DurationMs = r.current.EndTime - r.current.StartTime
	r.current.StepsCount = len(r.current.Steps)

	for _, s := range r.current.Steps {
		r.current.TotalPromptTokens += s.PromptTokens
		r.current.TotalCompletionTokens += s.CompletionTokens
	}

	if r.current.MappedStage != "" {
		log.Printf("[metrics] instance %s: stage=%s, duration=%dms, steps=%d, prompt_tokens=%d, completion_tokens=%d",
			r.current.InstanceID, r.current.MappedStage, r.current.DurationMs, r.current.StepsCount,
			r.current.TotalPromptTokens, r.current.TotalCompletionTokens)
	} else {
		log.Printf("[metrics] instance %s: duration=%dms, steps=%d, prompt_tokens=%d, completion_tokens=%d",
			r.current.InstanceID, r.current.DurationMs, r.current.StepsCount,
			r.current.TotalPromptTokens, r.current.TotalCompletionTokens)
	}

	r.instances = append(r.instances, *r.current)
	r.current = nil
}

// BeginStep starts timing a ReAct step. Call before ExecuteWithMessages for that step.
func (r *RunMetricsRecorder) BeginStep(stepIndex int) {
	if r.current == nil {
		return
	}
	r.stepStart = time.Now()
	r.current.Steps = append(r.current.Steps, StepMetrics{
		StepIndex: stepIndex,
		StartTime: r.stepStart.UnixMilli(),
	})
}

// RecordStepTokens records the token usage for the current step. Call after ExecuteWithMessages.
func (r *RunMetricsRecorder) RecordStepTokens(promptTokens, completionTokens int) {
	if r.current == nil || len(r.current.Steps) == 0 {
		return
	}
	last := &r.current.Steps[len(r.current.Steps)-1]
	last.PromptTokens = promptTokens
	last.CompletionTokens = completionTokens
}

// EndStep ends timing for the current step. Call after command execution.
func (r *RunMetricsRecorder) EndStep(stepIndex int) {
	if r.current == nil || len(r.current.Steps) == 0 {
		return
	}
	now := time.Now()
	last := &r.current.Steps[len(r.current.Steps)-1]
	last.EndTime = now.UnixMilli()
	last.DurationMs = last.EndTime - last.StartTime

	log.Printf("[metrics] step %d: duration=%dms, prompt_tokens=%d, completion_tokens=%d",
		stepIndex, last.DurationMs, last.PromptTokens, last.CompletionTokens)
}

// Write writes the collected metrics to path. Uses MetricsPath if path is empty. Call after EndRun.
func (r *RunMetricsRecorder) Write(path string) error {
	if path == "" {
		path = os.Getenv("METRICS_PATH")
		if path == "" {
			path = MetricsPath
		}
	}
	if r.current != nil {
		r.instances = append(r.instances, *r.current)
		r.current = nil
	}
	endMs := r.endTime.UnixMilli()
	if r.endTime.IsZero() {
		endMs = time.Now().UnixMilli()
	}
	startMs := r.startTime.UnixMilli()

	var totalPrompt, totalCompletion int
	for _, inst := range r.instances {
		totalPrompt += inst.TotalPromptTokens
		totalCompletion += inst.TotalCompletionTokens
	}

	m := RunMetrics{
		ExperimentName:        r.ExperimentName,
		StartTime:             startMs,
		EndTime:               endMs,
		TotalDurationMs:       endMs - startMs,
		Instances:             r.instances,
		InstancesCount:        len(r.instances),
		TotalPromptTokens:     totalPrompt,
		TotalCompletionTokens: totalCompletion,
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal metrics: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write metrics: %w", err)
	}
	return nil
}
