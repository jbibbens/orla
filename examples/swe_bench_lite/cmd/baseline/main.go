// Command baseline runs the SWE-bench Lite baseline experiment.
package main

import (
	"context"
	"log"
	"os"

	"github.com/dorcha-inc/orla/examples/swe_bench_lite/baseline"
)

func main() {
	ctx := context.Background()
	cfg := baseline.DefaultConfig()
	if err := baseline.Run(ctx, cfg); err != nil {
		log.Fatal(err)
	}
	os.Exit(0)
}
