// Command concurrent_stages_demo runs the concurrent stages tutorial example.
//
// Prerequisites: Start Orla and vLLM (e.g. from repo root):
//
//	docker compose -f deploy/docker-compose.vllm.yaml up -d
//
// Then run (from repo root):
//
//	go run ./examples/concurrent_stages_demo/cmd/concurrent_stages_demo
//
// When Orla and vLLM run in Docker Compose, the default VLLM_URL (http://vllm:8000/v1)
// works because the Orla server resolves "vllm" from within the compose network.
// If you run Orla/vLLM on the host, set VLLM_URL=http://localhost:8000/v1.
package main

import (
	"context"
	"log"
	"os"

	concurrentstagesdemo "github.com/harvard-cns/orla/examples/concurrent_stages_demo"
)

func main() {
	log.Println("================================================")
	log.Println("Concurrent Stages Demo")
	log.Println("================================================")

	ctx := context.Background()
	if err := concurrentstagesdemo.Run(ctx); err != nil {
		log.Fatal(err)
	}
	os.Exit(0)
}
