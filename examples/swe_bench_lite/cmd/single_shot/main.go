// Command single_shot runs the SWE-bench Lite single-shot experiment.
// Set EXPERIMENT_MODE to baseline, stage_mapping, or sjf (default: baseline).
package main

import (
	"context"
	"log"
	"os"

	singleshot "github.com/harvard-cns/orla/examples/swe_bench_lite/single_shot"

	"github.com/harvard-cns/orla/examples/swe_bench_lite/shared"
)

func main() {
	mode := os.Getenv("EXPERIMENT_MODE")
	if mode == "" {
		mode = "baseline"
	}
	log.Printf("Starting single-shot SWE-bench experiment (mode=%s)", mode) //nolint:gosec // G706 - mode is from a controlled enum

	dataset, err := shared.LoadDataset()
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}
	log.Printf("Loaded %d instances", len(dataset.Instances))

	ctx := context.Background()
	if err := singleshot.Run(ctx, dataset, mode); err != nil {
		log.Fatalf("experiment failed: %v", err)
	}
}
