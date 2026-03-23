// Package concurrentstagesdemo demonstrates backend concurrency with parallel stages.
//
// Two independent stages (summarize and extract_entities) run concurrently on the
// same backend when SetMaxConcurrency(4) is set. Without it, they would queue
// one after the other.
//
// Run with: go run ./examples/concurrent_stages_demo/cmd/concurrent_stages_demo
//
// Prerequisites: Orla + vLLM running (e.g. docker compose -f deploy/docker-compose.vllm.yaml up -d)
package concurrentstagesdemo

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
	defaultModel     = "Qwen/Qwen3-4B-Instruct-2507"
	sampleReport     = "The quarterly earnings report shows revenue up 12% year-over-year. Key drivers include growth in the cloud division and strong performance in the APAC region. The company announced a new partnership with TechCorp and plans to expand into three new markets by Q4."
)

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// Run executes the concurrent stages demo.
func Run(ctx context.Context) error {
	orlaURL := envOr("ORLA_URL", defaultOrlaURL)
	vllmURL := envOr("VLLM_URL", defaultVLLMURL)
	model := envOr("VLLM_MODEL", defaultModel)

	client := orla.NewOrlaClient(orlaURL)
	if err := client.Health(ctx); err != nil {
		return fmt.Errorf("orla health check: %w", err)
	}

	backend := orla.NewVLLMBackend(model, vllmURL)
	backend.SetMaxConcurrency(4)

	if err := client.RegisterBackend(ctx, backend); err != nil {
		return fmt.Errorf("register backend: %w", err)
	}
	log.Println("Backend registered with max concurrency 4")

	wf := orla.NewWorkflow(client)

	stageA := orla.NewStage("summarize", backend)
	stageA.SetMaxTokens(256)
	stageA.Prompt = fmt.Sprintf("Summarize the key findings of this report in 2-3 sentences:\n\n%s", sampleReport)

	stageB := orla.NewStage("extract_entities", backend)
	stageB.SetMaxTokens(256)
	stageB.Prompt = fmt.Sprintf("Extract all named entities (companies, regions, metrics) from this report. List them one per line:\n\n%s", sampleReport)

	if err := wf.AddStage(stageA); err != nil {
		return err
	}
	if err := wf.AddStage(stageB); err != nil {
		return err
	}

	log.Println("Executing parallel stages (summarize + extract_entities)...")
	results, err := wf.Execute(ctx)
	if err != nil {
		return fmt.Errorf("execute workflow: %w", err)
	}

	log.Println("=== Results ===")
	if r, ok := results[stageA.ID]; ok && r.Response != nil {
		log.Printf("Summary:\n%s\n", r.Response.Content)
	}
	if r, ok := results[stageB.ID]; ok && r.Response != nil {
		log.Printf("Entities:\n%s\n", r.Response.Content)
	}

	return nil
}
