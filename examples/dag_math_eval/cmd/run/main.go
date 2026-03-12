// Command dag_math_eval runs the DAG-Math memory management evaluation.
// Set EXPERIMENT_MODE to flush_per_request or flush_per_workflow (default: flush_per_workflow).
package main

import (
	"context"
	"log"
	"os"

	"github.com/dorcha-inc/orla/examples/dag_math_eval/eval"
	"github.com/dorcha-inc/orla/examples/dag_math_eval/shared"
)

func main() {
	mode := os.Getenv("EXPERIMENT_MODE")
	if mode == "" {
		mode = eval.ModeFlushPerWorkflow
	}
	log.Printf("Starting DAG-Math memory evaluation (mode=%s)", mode)

	dataset, err := shared.LoadDataset()
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}
	log.Printf("Loaded %d problems", len(dataset.Problems))

	ctx := context.Background()
	if err := eval.Run(ctx, dataset, mode); err != nil {
		log.Fatalf("experiment failed: %v", err)
	}
}
