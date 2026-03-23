// Package part3 demonstrates the 5-min demo Part 3:
//   - Stage Mapping (classify→light, policy/reply/route→heavy)
//   - Scheduling: Priority
//   - Second small workflow at end — Orla flushes cache at workflow boundary
package part3

import (
	"context"
	"log"

	workflowdemo "github.com/harvard-cns/orla/examples/workflow_demo"
	orla "github.com/harvard-cns/orla/pkg/api"
)

// Run executes Part 3 of the demo.
func Run(ctx context.Context, ticket string) error {
	log.Println("Demo Part 3: Stage Mapping + Priority + Flush at Boundary (two workflows)")
	opts := workflowdemo.Options{
		SchedulingPolicy:   orla.SchedulingPolicyPriority,
		MemoryPolicy:      orla.NewFlushAtBoundaryPolicy(),
		RunSecondWorkflow: true,
	}
	return workflowdemo.RunWithOptions(ctx, ticket, opts)
}
