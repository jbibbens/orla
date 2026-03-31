// Package eval runs the DAG-Math memory evaluation experiment.
// Two modes (set via EXPERIMENT_MODE):
//   - flush_per_request:  every stage gets CachePolicy="flush", so KV cache is evicted after each LLM call
//   - flush_per_workflow: default Orla policy (preserve across stages, flush at workflow boundary)
package eval

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	orla "github.com/harvard-cns/orla/pkg/api"

	"github.com/harvard-cns/orla/examples/dag_math_eval/shared"
)

const (
	ModeFlushPerRequest  = "flush_per_request"
	ModeFlushPerWorkflow = "flush_per_workflow"

	defaultModelID = "Qwen/Qwen3-8B" // override with SGLANG_MODEL
)

func Run(ctx context.Context, dataset *shared.DAGMathDataset, mode string) error {
	client := orla.NewOrlaClient(shared.OrlaURL)
	if err := client.Health(ctx); err != nil {
		return fmt.Errorf("orla health check: %w", err)
	}

	sglangURL := os.Getenv("SGLANG_URL")
	if sglangURL == "" {
		sglangURL = shared.SGLangURL
	}
	modelID := os.Getenv("SGLANG_MODEL")
	if modelID == "" {
		modelID = defaultModelID
	}

	log.Printf("Using model %s (override with SGLANG_MODEL)", modelID) //nolint:gosec // G706 - modelID is from env or hardcoded default
	backend := orla.NewSGLangBackend(modelID, sglangURL)
	backend.SetMaxConcurrency(1)
	if err := client.RegisterBackend(ctx, backend); err != nil {
		return fmt.Errorf("register backend: %w", err)
	}

	limit := shared.MaxInstances
	if limit > len(dataset.Problems) {
		limit = len(dataset.Problems)
	}
	problems := dataset.Problems[:limit]

	metrics := shared.NewRunMetricsRecorder("dag_math_" + mode)
	metrics.TotalWorkflows = len(problems)
	metrics.BeginRun()
	results := shared.NewRunResultsRecorder("dag_math_" + mode)
	defer func() {
		metrics.EndRun()
		if err := metrics.Write(""); err != nil {
			log.Printf("warning: write metrics: %v", err)
		} else {
			log.Printf("Metrics written to %s", shared.MetricsPath)
		}
		if err := results.Write(""); err != nil {
			log.Printf("warning: write results: %v", err)
		} else {
			log.Printf("Results written to %s", shared.OutputPath)
		}
	}()

	log.Printf("Running %d workflows sequentially (mode=%s)", len(problems), mode)

	for i, problem := range problems {
		log.Printf("[%d/%d] Starting problem %d (%d steps, difficulty=%.0f)",
			i+1, len(problems), problem.ProblemID, len(problem.Steps), problem.Difficulty)

		wfMetrics, wfResults, err := runWorkflow(ctx, client, backend, problem, mode)
		if err != nil {
			log.Printf("  problem %d failed: %v", problem.ProblemID, err)
			continue
		}
		metrics.AddWorkflow(*wfMetrics)
		for _, sm := range wfMetrics.Stages {
			log.Printf("  workflow %d, step %d: duration=%dms prompt_tokens=%d completion_tokens=%d",
				problem.ProblemID, sm.StepID, sm.DurationMs, sm.PromptTokens, sm.CompletionTokens)
		}
		if wfResults != nil {
			results.AddWorkflow(*wfResults)
		}
		// Progress log every 10 workflows or on last
		if (i+1)%10 == 0 || i+1 == len(problems) {
			log.Printf("Progress: %d/%d workflows complete", i+1, len(problems))
		}
	}

	log.Printf("Done. Metrics written to %s, results to %s", shared.MetricsPath, shared.OutputPath)
	return nil
}

func runWorkflow(ctx context.Context, client *orla.OrlaClient, backend *orla.LLMBackend, problem shared.DAGMathProblem, mode string) (*shared.WorkflowMetrics, *shared.WorkflowResult, error) {
	if len(problem.Steps) == 0 {
		return nil, nil, fmt.Errorf("problem %d has no steps", problem.ProblemID)
	}

	wf := orla.NewWorkflow(client)

	stageIDByStepID := make(map[int]string, len(problem.Steps))
	stepByStageID := make(map[string]shared.DAGMathStep, len(problem.Steps))

	for _, step := range problem.Steps {
		stage := orla.NewStage(fmt.Sprintf("step_%d", step.StepID), backend)
		stage.SetMaxTokens(shared.MaxOutputTokens)
		stage.SetTemperature(0)
		stage.SetChatTemplateKwargs(map[string]any{"enable_thinking": false})
		stage.SetStream(true)

		if mode == ModeFlushPerRequest {
			stage.SetCachePolicy(orla.CachePolicyFlush)
		}

		capturedStep := step
		stage.SetMessagesBuilder(func(depResults map[string]*orla.StageResult) ([]orla.Message, error) {
			depContents := make(map[int]string, len(capturedStep.DirectDependentSteps))
			for _, depStepID := range capturedStep.DirectDependentSteps {
				depStageID, ok := stageIDByStepID[depStepID]
				if !ok {
					continue
				}
				if result, ok := depResults[depStageID]; ok && result.Response != nil {
					depContents[depStepID] = result.Response.Content
				}
			}

			prompt := shared.BuildStagePrompt(problem, capturedStep, depContents)
			return []orla.Message{
				{Role: "system", Content: shared.SystemPrompt},
				{Role: "user", Content: prompt},
			}, nil
		})

		if err := wf.AddStage(stage); err != nil {
			return nil, nil, fmt.Errorf("add stage step_%d: %w", step.StepID, err)
		}

		stageIDByStepID[step.StepID] = stage.ID
		stepByStageID[stage.ID] = step
	}

	for _, step := range problem.Steps {
		if len(step.DirectDependentSteps) == 0 {
			continue
		}
		stageID := stageIDByStepID[step.StepID]
		for _, depStepID := range step.DirectDependentSteps {
			depStageID, ok := stageIDByStepID[depStepID]
			if !ok {
				log.Printf("  warning: step %d depends on unknown step %d", step.StepID, depStepID)
				continue
			}
			if err := wf.AddDependency(stageID, depStageID); err != nil {
				return nil, nil, fmt.Errorf("add dependency step_%d -> step_%d: %w", step.StepID, depStepID, err)
			}
		}
	}

	wfStart := time.Now()
	results, err := wf.Execute(ctx)
	wfDuration := time.Since(wfStart)
	if err != nil {
		return nil, nil, fmt.Errorf("execute workflow: %w", err)
	}

	wfMetrics := &shared.WorkflowMetrics{
		ProblemID:   problem.ProblemID,
		NumStages:   len(problem.Steps),
		Difficulty:  problem.Difficulty,
		StartTimeMs: wfStart.UnixMilli(),
		EndTimeMs:   wfStart.Add(wfDuration).UnixMilli(),
		DurationMs:  wfDuration.Milliseconds(),
	}

	for stageID, result := range results {
		step, ok := stepByStageID[stageID]
		if !ok {
			continue
		}
		sm := shared.StageMetrics{
			StepID:     step.StepID,
			DurationMs: wfDuration.Milliseconds(), // will be overridden below if we have per-stage metrics
		}
		if result.Response != nil && result.Response.Metrics != nil {
			m := result.Response.Metrics
			sm.PromptTokens = m.PromptTokens
			sm.CompletionTokens = m.CompletionTokens
			sm.QueueWaitMs = m.QueueWaitMs
			if m.BackendLatencyMs != nil {
				sm.BackendLatencyMs = *m.BackendLatencyMs
				sm.DurationMs = *m.BackendLatencyMs
			}
			if m.TTFTMs != nil {
				sm.TTFTMs = *m.TTFTMs
			}
			if m.TPOTMs != nil {
				sm.TPOTMs = *m.TPOTMs
			}
			if sm.DurationMs == 0 {
				sm.DurationMs = m.QueueWaitMs + sm.TTFTMs
			}
		}
		wfMetrics.TotalPromptTokens += sm.PromptTokens
		wfMetrics.TotalCompletionTokens += sm.CompletionTokens
		wfMetrics.Stages = append(wfMetrics.Stages, sm)
	}

	// Build workflow results (prompt + response per step) for output.
	wfResults := &shared.WorkflowResult{ProblemID: problem.ProblemID}
	for _, step := range problem.Steps {
		stageID := stageIDByStepID[step.StepID]
		result := results[stageID]
		depContents := make(map[int]string)
		for _, depStepID := range step.DirectDependentSteps {
			depStageID := stageIDByStepID[depStepID]
			if r, ok := results[depStageID]; ok && r != nil && r.Response != nil {
				depContents[depStepID] = r.Response.Content
			}
		}
		prompt := shared.BuildStagePrompt(problem, step, depContents)
		response := ""
		if result != nil && result.Response != nil {
			response = result.Response.Content
		}
		wfResults.Steps = append(wfResults.Steps, shared.StepResult{
			StepID:   step.StepID,
			Prompt:   prompt,
			Response: response,
		})
	}

	return wfMetrics, wfResults, nil
}
