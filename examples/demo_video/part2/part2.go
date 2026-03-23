// Package part2 demonstrates the 5-min demo Part 2:
//   - Stage Mapping (classify→light, policy/reply/route→heavy)
//   - Scheduling: Priority
//   - Cache Management: Preserves same model (default policy)
package part2

import (
	"context"
	"log"

	workflowdemo "github.com/harvard-cns/orla/examples/workflow_demo"
	orla "github.com/harvard-cns/orla/pkg/api"
)

// Run executes Part 2 of the demo.
func Run(ctx context.Context, ticket string) error {
	log.Println("Demo Part 2: Stage Mapping + Priority + Cache Preserves")
	opts := workflowdemo.Options{
		SchedulingPolicy: orla.SchedulingPolicyPriority,
		// MemoryPolicy: nil = default (preserve on same-backend small increment)
	}
	return workflowdemo.RunWithOptions(ctx, ticket, opts)
}
