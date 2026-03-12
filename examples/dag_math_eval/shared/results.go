package shared

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
)

// StepResult is the prompt and model response for one DAG-Math step.
type StepResult struct {
	StepID   int    `json:"step_id"`
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
}

// WorkflowResult is the full results for one DAG-Math workflow.
type WorkflowResult struct {
	ProblemID int          `json:"problem_id"`
	Steps     []StepResult `json:"steps"`
}

// RunResults is the full run output (prompts + responses).
type RunResults struct {
	ExperimentName string           `json:"experiment_name"`
	Workflows      []WorkflowResult `json:"workflows"`
}

// RunResultsRecorder records workflow results and flushes to disk. Thread-safe.
type RunResultsRecorder struct {
	ExperimentName string
	workflows      []WorkflowResult
	mu             sync.Mutex
}

func NewRunResultsRecorder(experimentName string) *RunResultsRecorder {
	return &RunResultsRecorder{ExperimentName: experimentName}
}

// AddWorkflow appends workflow results and flushes to disk. Thread-safe.
func (r *RunResultsRecorder) AddWorkflow(wf WorkflowResult) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.workflows = append(r.workflows, wf)
	if err := r.writeLocked(""); err != nil {
		log.Printf("warning: flush results: %v", err)
	}
}

// Write writes the collected results to path. Uses OutputPath if path is empty. Thread-safe.
func (r *RunResultsRecorder) Write(path string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.writeLocked(path)
}

func (r *RunResultsRecorder) writeLocked(path string) error {
	if path == "" {
		path = os.Getenv("OUTPUT_PATH")
		if path == "" {
			path = OutputPath
		}
	}
	data, err := json.MarshalIndent(RunResults{
		ExperimentName: r.ExperimentName,
		Workflows:      r.workflows,
	}, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal results: %w", err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("write results: %w", err)
	}
	return nil
}
