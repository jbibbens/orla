// Package memorymanagerdemo demonstrates the Memory Manager with a two-agent workflow.
//
// Uses SGLang by default so the Memory Manager can perform a hard flush via
// POST /flush_cache when the workflow completes. The workflow uses
// NewFlushAtBoundaryPolicy() to explicitly test flush-at-boundary behavior.
//
// Run with: go run ./examples/memory_manager_demo/cmd/memory_manager_demo
//
// Prerequisites: Orla + SGLang (workflow-demo stack):
//   docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
package memorymanagerdemo

import (
	"context"
	"fmt"
	"log"
	"os"

	orla "github.com/dorcha-inc/orla/pkg/api"
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
	backend.SetMaxConcurrency(1) // safe for SGLang global flush

	if err := client.RegisterBackend(ctx, backend); err != nil {
		return fmt.Errorf("register backend: %w", err)
	}

	// --- Triage agent ---
	triage := orla.NewAgent(client)
	triage.Name = "triage"

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

	if err := triage.AddStage(classify); err != nil {
		return err
	}
	if err := triage.AddStage(prioritize); err != nil {
		return err
	}
	if err := triage.AddDependency(prioritize.ID, classify.ID); err != nil {
		return err
	}

	// --- Resolver agent ---
	resolver := orla.NewAgent(client)
	resolver.Name = "resolver"

	draft := orla.NewStage("draft", backend)
	draft.SetMaxTokens(256)
	draft.Prompt = "Draft a brief, professional customer response based on the triage output above. Address the issue and suggest next steps."

	if err := resolver.AddStage(draft); err != nil {
		return err
	}

	// --- Workflow with flush-at-boundary policy ---
	// FlushAtBoundaryPolicy flushes at workflow completion (and on backend switch).
	// With SGLang, this triggers a hard flush via POST /flush_cache.
	wf := orla.NewWorkflow()
	wf.SetMemoryPolicy(orla.NewFlushAtBoundaryPolicy())
	if err := wf.AddAgent(triage); err != nil {
		return err
	}
	if err := wf.AddAgent(resolver); err != nil {
		return err
	}
	if err := wf.AddDependency("resolver", "triage"); err != nil {
		return err
	}

	wf.SetContextPassingFn(func(upstreamResults map[string]*orla.AgentResult, downstream *orla.Agent) error {
		triageResult, ok := upstreamResults["triage"]
		if !ok {
			return nil
		}
		var triageOutput string
		for _, sr := range triageResult.StageResults {
			if sr.Response != nil {
				triageOutput += sr.Response.Content + "\n"
			}
		}
		for _, s := range downstream.Stages() {
			if s.Name == "draft" {
				s.Prompt = fmt.Sprintf("Draft a brief, professional customer response based on this triage analysis:\n\n%s\n\nOriginal ticket: %s", triageOutput, sampleTicket)
			}
		}
		return nil
	})

	log.Println("Executing workflow (triage -> resolver) with FlushAtBoundaryPolicy...")
	log.Println("On completion, SGLang cache will be flushed via POST /flush_cache")
	results, err := wf.Execute(ctx)
	if err != nil {
		return fmt.Errorf("workflow execution: %w", err)
	}

	log.Println("=== Results ===")
	if r, ok := results["resolver"]; ok {
		if draftResult, ok := r.StageResults[draft.ID]; ok && draftResult.Response != nil {
			log.Printf("Draft response:\n%s\n", draftResult.Response.Content)
		}
	}

	return nil
}
