// Command memory_manager_demo runs the Memory Manager tutorial example.
//
// Uses SGLang by default to test hard cache flush via POST /flush_cache.
// Prerequisites (from repo root):
//
//	docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
//	go run ./examples/memory_manager_demo/cmd/memory_manager_demo
//
// For vLLM (soft flush only), set VLLM_URL=http://vllm:8000/v1.
package main

import (
	"context"
	"log"
	"os"

	memorymanagerdemo "github.com/harvard-cns/orla/examples/memory_manager_demo"
)

func main() {
	log.Println("================================================")
	log.Println("Memory Manager Demo")
	log.Println("================================================")

	ctx := context.Background()
	if err := memorymanagerdemo.Run(ctx); err != nil {
		log.Fatal(err)
	}
	os.Exit(0)
}
