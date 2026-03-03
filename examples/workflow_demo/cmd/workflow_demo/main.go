// Command workflow_demo runs the customer support workflow demo.
package main

import (
	"context"
	"log"
	"os"

	workflowdemo "github.com/dorcha-inc/orla/examples/workflow_demo"
)

func main() {
	log.Println("================================================")
	log.Println("Running customer support workflow demo")
	log.Println("================================================")

	ticket := workflowdemo.SampleTicket
	if path := os.Getenv("TICKET_PATH"); path != "" {
		data, err := os.ReadFile(path)
		if err != nil {
			log.Fatalf("read ticket: %v", err)
		}
		ticket = string(data)
	}

	ctx := context.Background()
	if err := workflowdemo.Run(ctx, ticket); err != nil {
		log.Fatal(err)
	}
	os.Exit(0)
}
