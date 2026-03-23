// Package part1 demonstrates the 5-min demo Part 1:
//   - Customer support workflow exactly as in workflow_demo
//   - Stage mapping only (classify→light, policy/reply/route→heavy)
//   - No scheduling or memory policy customization
package part1

import (
	"context"
	"log"

	workflowdemo "github.com/harvard-cns/orla/examples/workflow_demo"
)

// Run executes Part 1 of the demo.
func Run(ctx context.Context, ticket string) error {
	log.Println("Demo Part 1: Customer support workflow - stage mapping only")
	return workflowdemo.RunWithOptions(ctx, ticket, workflowdemo.Options{})
}
