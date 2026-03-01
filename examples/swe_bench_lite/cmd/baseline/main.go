// Command baseline runs the SWE-bench Lite baseline experiment.
package main

import (
	"context"
	"log"
	"os"

	"github.com/dorcha-inc/orla/examples/swe_bench_lite/baseline"
	"github.com/dorcha-inc/orla/examples/swe_bench_lite/shared"
)

func main() {
	ctx := context.Background()
	dataset, err := shared.LoadDataset()
	if err != nil {
		log.Fatal(err)
	}
	if err := baseline.Run(ctx, dataset); err != nil {
		log.Fatal(err)
	}
	os.Exit(0)
}
