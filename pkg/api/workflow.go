package orla

import (
	"context"
	"fmt"
	"maps"
	"sync"

	"github.com/docker/docker/pkg/namesgenerator"
)

// AgentResult wraps the output of an agent's DAG execution.
type AgentResult struct {
	StageResults map[string]*StageResult
}

// ContextPassingFn customizes how one agent's results feed into the next agent's stages.
// It is called before a dependent agent starts, allowing mutation of that agent's stage prompts.
type ContextPassingFn func(upstreamResults map[string]*AgentResult, downstream *Agent) error

// Workflow composes multiple Agents with inter-agent dependencies.
// Each agent is an independent DAG of stages. Agents with no inter-dependencies
// run concurrently; the workflow-level DAG controls ordering between agents.
type Workflow struct {
	agents       map[string]*Agent
	dependencies map[string][]string // agentName -> depends on []agentName
	contextFn    ContextPassingFn
	memoryPolicy MemoryPolicy
}

// NewWorkflow creates an empty workflow.
func NewWorkflow() *Workflow {
	return &Workflow{
		agents:       make(map[string]*Agent),
		dependencies: make(map[string][]string),
	}
}

// SetContextPassingFn sets the function called before each dependent agent starts.
func (w *Workflow) SetContextPassingFn(fn ContextPassingFn) {
	w.contextFn = fn
}

// SetMemoryPolicy sets the workflow-level MemoryPolicy used by the Memory Manager
// to decide cache actions at stage transitions. If not set, the default policy
// (preserve on small increment + flush at boundary) is used.
func (w *Workflow) SetMemoryPolicy(policy MemoryPolicy) {
	w.memoryPolicy = policy
}

// MemoryPolicyOrDefault returns the configured MemoryPolicy or the default.
func (w *Workflow) MemoryPolicyOrDefault() MemoryPolicy {
	if w.memoryPolicy != nil {
		return w.memoryPolicy
	}
	return NewDefaultMemoryPolicy()
}

// AddAgent adds an agent to the workflow.
func (w *Workflow) AddAgent(agent *Agent) error {
	if agent == nil {
		return fmt.Errorf("agent cannot be nil")
	}
	if agent.Name == "" {
		return fmt.Errorf("agent name is required")
	}
	if _, exists := w.agents[agent.Name]; exists {
		return fmt.Errorf("agent %q already exists", agent.Name)
	}
	w.agents[agent.Name] = agent
	return nil
}

// AddDependency declares that agentName depends on dependsOnAgentName.
func (w *Workflow) AddDependency(agentName, dependsOnAgentName string) error {
	if _, ok := w.agents[agentName]; !ok {
		return fmt.Errorf("agent %q not found", agentName)
	}
	if _, ok := w.agents[dependsOnAgentName]; !ok {
		return fmt.Errorf("dependency agent %q not found", dependsOnAgentName)
	}
	w.dependencies[agentName] = append(w.dependencies[agentName], dependsOnAgentName)
	return nil
}

// Agents returns all agents keyed by name.
func (w *Workflow) Agents() map[string]*Agent {
	out := make(map[string]*Agent, len(w.agents))
	maps.Copy(out, w.agents)
	return out
}

// Execute runs the workflow DAG: agents execute their internal stage DAGs,
// respecting inter-agent dependencies. Independent agents run concurrently.
func (w *Workflow) Execute(ctx context.Context) (map[string]*AgentResult, error) {
	if len(w.agents) == 0 {
		return map[string]*AgentResult{}, nil
	}

	workflowID := namesgenerator.GetRandomName(0)
	for _, agent := range w.agents {
		agent.workflowID = workflowID
	}

	// Validate dependencies
	for name, deps := range w.dependencies {
		for _, depName := range deps {
			if _, ok := w.agents[depName]; !ok {
				return nil, fmt.Errorf("agent %q depends on unknown agent %q", name, depName)
			}
		}
	}

	dependents := make(map[string][]string, len(w.agents))
	remainingDeps := make(map[string]int, len(w.agents))
	for name := range w.agents {
		for _, depName := range w.dependencies[name] {
			dependents[depName] = append(dependents[depName], name)
		}
		remainingDeps[name] = len(w.dependencies[name])
	}

	results := make(map[string]*AgentResult, len(w.agents))
	var resultsMu sync.RWMutex
	var remainingMu sync.Mutex

	type agentOutcome struct {
		name      string
		err       error
		unblocked []string
	}

	outcomeCh := make(chan agentOutcome, len(w.agents))
	readyCh := make(chan string, len(w.agents))

	startAgent := func(agentName string) {
		go func() {
			agent := w.agents[agentName]

			if w.contextFn != nil {
				resultsMu.RLock()
				snapshot := make(map[string]*AgentResult, len(results))
				maps.Copy(snapshot, results)
				resultsMu.RUnlock()

				if err := w.contextFn(snapshot, agent); err != nil {
					outcomeCh <- agentOutcome{name: agentName, err: fmt.Errorf("agent %q context passing: %w", agentName, err)}
					return
				}
			}

			stageResults, err := agent.ExecuteDAG(ctx)
			if err != nil {
				outcomeCh <- agentOutcome{name: agentName, err: fmt.Errorf("agent %q: %w", agentName, err)}
				return
			}

			resultsMu.Lock()
			results[agentName] = &AgentResult{StageResults: stageResults}
			resultsMu.Unlock()

			remainingMu.Lock()
			var unblocked []string
			for _, dep := range dependents[agentName] {
				remainingDeps[dep]--
				if remainingDeps[dep] == 0 {
					unblocked = append(unblocked, dep)
				}
			}
			remainingMu.Unlock()

			outcomeCh <- agentOutcome{name: agentName, unblocked: unblocked}
		}()
	}

	remainingMu.Lock()
	for name, deps := range remainingDeps {
		if deps == 0 {
			readyCh <- name
		}
	}
	remainingMu.Unlock()

	dispatched := 0
	completed := 0
	for {
		for {
			select {
			case agentName := <-readyCh:
				startAgent(agentName)
				dispatched++
			default:
				goto waitOutcome
			}
		}

	waitOutcome:
		if completed == len(w.agents) {
			break
		}
		if dispatched == completed {
			select {
			case agentName := <-readyCh:
				startAgent(agentName)
				dispatched++
				continue
			default:
				return nil, fmt.Errorf("workflow has a cycle; completed %d/%d agents", completed, len(w.agents))
			}
		}

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case outcome := <-outcomeCh:
			if outcome.err != nil {
				return nil, outcome.err
			}
			completed++
			for _, next := range outcome.unblocked {
				readyCh <- next
			}
		}
	}

	if dispatched != len(w.agents) {
		return nil, fmt.Errorf("workflow has a cycle; dispatched %d/%d agents", dispatched, len(w.agents))
	}
	return results, nil
}
