// Command part2 runs Demo Part 2: Stage Mapping + Priority + Cache Preserves.
//
// Prerequisites:
//
//	docker compose -f deploy/docker-compose.workflow-demo.yaml up -d
//	go run ./examples/demo_video/cmd/part2
package main

import (
	"context"
	"log"
	"os"

	workflowdemo "github.com/harvard-cns/orla/examples/workflow_demo"
	"github.com/harvard-cns/orla/examples/demo_video/part2"
)

func main() {
	log.Println("================================================")
	log.Println("5-Min Demo Part 2")
	log.Println("================================================")

	ticket := workflowdemo.SampleTicket
	if path := os.Getenv("TICKET_PATH"); path != "" {
		data, err := os.ReadFile(path) // #nosec G304 -- path from TICKET_PATH env, user-controlled for demo
		if err != nil {
			log.Fatalf("read ticket: %v", err)
		}
		ticket = string(data)
	}

	ctx := context.Background()
	if err := part2.Run(ctx, ticket); err != nil {
		log.Fatal(err)
	}
	os.Exit(0)
}
