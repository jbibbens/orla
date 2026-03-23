// Package memorymanagerdemo demonstrates the Memory Manager with a multi-stage workflow.
//
// Uses SGLang by default so the Memory Manager can perform a hard flush via
// POST /flush_cache when the workflow completes. The workflow uses
// NewFlushAtBoundaryPolicy() to explicitly test flush-at-boundary behavior.
//
// Run with: go run ./examples/memory_manager_demo/cmd/memory_manager_demo
//
// Prerequisites: Orla + SGLang (workflow-demo stack):
//
//	docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
package memorymanagerdemo

import (
	"context"
	"fmt"
	"log"
	"os"

	orla "github.com/harvard-cns/orla/pkg/api"
)

const (
	defaultOrlaURL   = "http://localhost:8081"
	defaultVLLMURL   = "http://vllm:8000/v1"
	defaultSGLangURL = "http://sglang:30000/v1"
	defaultModel     = "Qwen/Qwen3-4B-Instruct-2507"
	sampleTicket     = "Hi, I was charged twice for my monthly subscription. I only signed up once. Can you please refund the duplicate charge? My account email is customer@example.com."
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// Run executes the memory manager demo.
func Run(ctx context.Context) error {
	orlaURL := envOr("ORLA_URL", defaultOrlaURL)
	model := envOr("SGLANG_MODEL", defaultModel)

	client := orla.NewOrlaClient(orlaURL)
	if err := client.Health(ctx); err != nil {
		return fmt.Errorf("orla health check: %w", err)
	}

	var backend *orla.LLMBackend
	if vllmURL := os.Getenv("VLLM_URL"); vllmURL != "" {
		backend = orla.NewVLLMBackend(envOr("VLLM_MODEL", defaultModel), vllmURL)
	} else {
		sglangURL := envOr("SGLANG_URL", defaultSGLangURL)
		backend = orla.NewSGLangBackend(model, sglangURL)
	}
	backend.SetMaxConcurrency(1)

	if err := client.RegisterBackend(ctx, backend); err != nil {
		return fmt.Errorf("register backend: %w", err)
	}

	// --- Workflow with flush-at-boundary policy ---
	wf := orla.NewWorkflow(client)
	wf.SetMemoryPolicy(orla.NewFlushAtBoundaryPolicy())

	classify := orla.NewStage("classify", backend)
	classify.SetMaxTokens(128)
	classify.Prompt = fmt.Sprintf("Classify this support ticket in one sentence (category + key issue):\n\n%s", sampleTicket)

	prioritize := orla.NewStage("prioritize", backend)
	prioritize.SetMaxTokens(128)
	prioritize.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		cr, ok := results[classify.ID]
		if !ok || cr.Response == nil {
			return "", fmt.Errorf("missing classify result")
		}
		return fmt.Sprintf("Given this classification, assign severity (low/medium/high) and a one-sentence priority reason:\n\n%s", cr.Response.Content), nil
	})

	draft := orla.NewStage("draft", backend)
	draft.SetMaxTokens(256)
	draft.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		var triageOutput string
		if cr, ok := results[classify.ID]; ok && cr.Response != nil {
			triageOutput += cr.Response.Content + "\n"
		}
		if pr, ok := results[prioritize.ID]; ok && pr.Response != nil {
			triageOutput += pr.Response.Content + "\n"
		}
		return fmt.Sprintf("Draft a brief, professional customer response based on this triage analysis:\n\n%s\n\nOriginal ticket: %s", triageOutput, sampleTicket), nil
	})

	if err := wf.AddStage(classify); err != nil {
		return err
	}
	if err := wf.AddStage(prioritize); err != nil {
		return err
	}
	if err := wf.AddStage(draft); err != nil {
		return err
	}
	if err := wf.AddDependency(prioritize.ID, classify.ID); err != nil {
		return err
	}
	if err := wf.AddDependency(draft.ID, classify.ID); err != nil {
		return err
	}
	if err := wf.AddDependency(draft.ID, prioritize.ID); err != nil {
		return err
	}

	log.Println("Executing workflow (classify -> prioritize -> draft) with FlushAtBoundaryPolicy...")
	log.Println("On completion, SGLang cache will be flushed via POST /flush_cache")
	results, err := wf.Execute(ctx)
	if err != nil {
		return fmt.Errorf("workflow execution: %w", err)
	}

	log.Println("=== Results ===")
	if draftResult, ok := results[draft.ID]; ok && draftResult.Response != nil {
		log.Printf("Draft response:\n%s\n", draftResult.Response.Content)
	}

	return nil
}
