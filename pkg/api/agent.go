package orla

import (
	"context"
	"fmt"
	"maps"
	"sync"
)

// Agent owns a DAG of Stages. Use AddStage and AddDependency to build
// the DAG, then ExecuteDAG to run it.
type Agent struct {
	Client *OrlaClient
	Name   string

	stages       map[string]*Stage
	dependencies map[string][]string // stageID -> depends on []stageID
}

// NewAgent returns an agent bound to the given client.
func NewAgent(client *OrlaClient) *Agent {
	return &Agent{
		Client:       client,
		stages:       make(map[string]*Stage),
		dependencies: make(map[string][]string),
	}
}

// AddStage registers a stage in the agent's DAG. Sets stage.Client automatically.
func (a *Agent) AddStage(s *Stage) error {
	if s == nil {
		return fmt.Errorf("stage cannot be nil")
	}
	if s.ID == "" {
		return fmt.Errorf("stage id is required")
	}
	if _, exists := a.stages[s.ID]; exists {
		return fmt.Errorf("stage %q already exists", s.ID)
	}
	s.Client = a.Client
	a.stages[s.ID] = s
	return nil
}

// AddDependency declares that stageID depends on dependsOnStageID (must finish before stageID starts).
func (a *Agent) AddDependency(stageID, dependsOnStageID string) error {
	if _, ok := a.stages[stageID]; !ok {
		return fmt.Errorf("stage %q not found", stageID)
	}
	if _, ok := a.stages[dependsOnStageID]; !ok {
		return fmt.Errorf("dependency stage %q not found", dependsOnStageID)
	}
	a.dependencies[stageID] = append(a.dependencies[stageID], dependsOnStageID)
	return nil
}

// Stages returns all DAG stages keyed by ID.
func (a *Agent) Stages() map[string]*Stage {
	out := make(map[string]*Stage, len(a.stages))
	maps.Copy(out, a.stages)
	return out
}

// ExecuteDAG runs the agent's stage DAG with dependency-aware scheduling.
// Independent stages execute concurrently; context is passed between stages via PromptBuilder/MessagesBuilder.
// Returns results keyed by stage ID.
func (a *Agent) ExecuteDAG(ctx context.Context) (map[string]*StageResult, error) {
	if len(a.stages) == 0 {
		return nil, fmt.Errorf("agent %q has no stages", a.Name)
	}

	for id, deps := range a.dependencies {
		for _, depID := range deps {
			if _, ok := a.stages[depID]; !ok {
				return nil, fmt.Errorf("stage %q depends on unknown stage %q", id, depID)
			}
		}
	}

	dependents := make(map[string][]string, len(a.stages))
	remainingDeps := make(map[string]int, len(a.stages))
	for id := range a.stages {
		for _, depID := range a.dependencies[id] {
			dependents[depID] = append(dependents[depID], id)
		}
		remainingDeps[id] = len(a.dependencies[id])
	}

	results := make(map[string]*StageResult, len(a.stages))
	var resultsMu sync.RWMutex
	var remainingMu sync.Mutex

	type nodeOutcome struct {
		id        string
		err       error
		unblocked []string
	}

	outcomeCh := make(chan nodeOutcome, len(a.stages))
	readyCh := make(chan string, len(a.stages))

	startStage := func(stageID string) {
		go func() {
			stage := a.stages[stageID]

			resultsMu.RLock()
			depSnapshot := make(map[string]*StageResult, len(results))
			maps.Copy(depSnapshot, results)
			resultsMu.RUnlock()

			result, err := a.executeStageInDAG(ctx, stage, depSnapshot)
			if err != nil {
				outcomeCh <- nodeOutcome{id: stageID, err: fmt.Errorf("stage %q: %w", stageID, err)}
				return
			}

			resultsMu.Lock()
			results[stageID] = result
			resultsMu.Unlock()

			remainingMu.Lock()
			var unblocked []string
			for _, dep := range dependents[stageID] {
				remainingDeps[dep]--
				if remainingDeps[dep] == 0 {
					unblocked = append(unblocked, dep)
				}
			}
			remainingMu.Unlock()

			outcomeCh <- nodeOutcome{id: stageID, unblocked: unblocked}
		}()
	}

	remainingMu.Lock()
	for id, deps := range remainingDeps {
		if deps == 0 {
			readyCh <- id
		}
	}
	remainingMu.Unlock()

	dispatched := 0
	completed := 0
	for {
		for {
			select {
			case stageID := <-readyCh:
				startStage(stageID)
				dispatched++
			default:
				goto waitOutcome
			}
		}

	waitOutcome:
		if completed == len(a.stages) {
			break
		}
		if dispatched == completed {
			select {
			case stageID := <-readyCh:
				startStage(stageID)
				dispatched++
				continue
			default:
				return nil, fmt.Errorf("agent %q: stage DAG has a cycle; completed %d/%d stages", a.Name, completed, len(a.stages))
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

	if dispatched != len(a.stages) {
		return nil, fmt.Errorf("agent %q: stage DAG has a cycle; dispatched %d/%d stages", a.Name, dispatched, len(a.stages))
	}
	return results, nil
}

const defaultMaxAgentLoopTurns = 100

func (a *Agent) executeStageInDAG(ctx context.Context, stage *Stage, depResults map[string]*StageResult) (*StageResult, error) {
	switch stage.ExecutionMode {
	case ExecutionModeAgentLoop:
		return a.executeAgentLoopStage(ctx, stage, depResults)
	default:
		return a.executeSingleShotStage(ctx, stage, depResults)
	}
}

func (a *Agent) executeSingleShotStage(ctx context.Context, stage *Stage, depResults map[string]*StageResult) (*StageResult, error) {
	if stage.MessagesBuilder != nil {
		msgs, err := stage.MessagesBuilder(depResults)
		if err != nil {
			return nil, fmt.Errorf("messages builder: %w", err)
		}
		resp, err := stage.ExecuteWithMessages(ctx, msgs)
		if err != nil {
			return nil, err
		}
		return &StageResult{Response: resp, Messages: msgs}, nil
	}

	prompt := stage.Prompt
	if stage.PromptBuilder != nil {
		built, err := stage.PromptBuilder(depResults)
		if err != nil {
			return nil, fmt.Errorf("prompt builder: %w", err)
		}
		prompt = built
	}
	if prompt == "" {
		return nil, fmt.Errorf("prompt is empty")
	}
	resp, err := stage.Execute(ctx, prompt)
	if err != nil {
		return nil, err
	}
	return &StageResult{Response: resp}, nil
}

func (a *Agent) executeAgentLoopStage(ctx context.Context, stage *Stage, depResults map[string]*StageResult) (*StageResult, error) {
	var messages []Message

	if stage.MessagesBuilder != nil {
		msgs, err := stage.MessagesBuilder(depResults)
		if err != nil {
			return nil, fmt.Errorf("messages builder: %w", err)
		}
		messages = msgs
	} else {
		prompt := stage.Prompt
		if stage.PromptBuilder != nil {
			built, err := stage.PromptBuilder(depResults)
			if err != nil {
				return nil, fmt.Errorf("prompt builder: %w", err)
			}
			prompt = built
		}
		if prompt == "" {
			return nil, fmt.Errorf("prompt is empty")
		}
		messages = []Message{{Role: "user", Content: prompt}}
	}

	maxTurns := stage.MaxTurns
	if maxTurns <= 0 {
		maxTurns = defaultMaxAgentLoopTurns
	}

	var lastResp *InferenceResponse
	for turn := range maxTurns {
		_ = turn
		resp, err := stage.ExecuteWithMessages(ctx, messages)
		if err != nil {
			return nil, fmt.Errorf("turn %d: %w", turn+1, err)
		}
		lastResp = resp

		messages = append(messages, Message{Role: "assistant", Content: resp.Content})

		if len(resp.ToolCalls) == 0 {
			break
		}

		toolMsgs, err := stage.RunToolCallsInResponse(ctx, resp)
		if err != nil {
			return nil, fmt.Errorf("turn %d tool calls: %w", turn+1, err)
		}
		for _, msg := range toolMsgs {
			messages = append(messages, *msg)
		}
	}

	return &StageResult{Response: lastResp, Messages: messages}, nil
}
