package serving

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/dorcha-inc/orla/internal/model"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

const defaultBackendQueueCapacity = 1024

type scheduledRequest struct {
	ctx       context.Context
	backend   string
	stageName string
	messages  []model.Message
	tools     []*mcp.Tool
	opts      model.InferenceOptions

	enqueuedAt time.Time

	resultCh chan scheduledResult
}

type scheduledResult struct {
	response *model.Response
	streamCh <-chan model.StreamEvent
	err      error
}

// backendExecutor serializes request execution for one backend and applies a scheduling policy.
type backendExecutor struct {
	backendName string
	manager     *LLMBackendManager

	mu          sync.Mutex
	cond        *sync.Cond
	stageQueues map[string][]*scheduledRequest
	queueLen    int
	capacity    int
	policy      model.SchedulingPolicy
	policySet   bool
	closed      bool
}

func newBackendExecutor(backendName string, manager *LLMBackendManager) *backendExecutor {
	exec := &backendExecutor{
		backendName: backendName,
		manager:     manager,
		capacity:    defaultBackendQueueCapacity,
		stageQueues: make(map[string][]*scheduledRequest),
	}
	exec.cond = sync.NewCond(&exec.mu)
	go exec.worker()
	return exec
}

func (e *backendExecutor) enqueue(req *scheduledRequest) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.closed {
		return errors.New("backend executor is closed")
	}
	if e.queueLen >= e.capacity {
		return fmt.Errorf("backend queue is full for backend %q", e.backendName)
	}
	if !e.policySet {
		e.policy = req.opts.GetSchedulingPolicy()
		e.policySet = true
	}
	// Keep scheduling policy consistent per backend executor.
	req.opts.SchedulingPolicy = e.policy

	stage := req.stageName
	if stage == "" {
		stage = "default"
	}
	e.stageQueues[stage] = append(e.stageQueues[stage], req)
	e.queueLen++
	e.cond.Signal()
	return nil
}

func (e *backendExecutor) close() {
	e.mu.Lock()
	e.closed = true
	e.cond.Broadcast()
	e.mu.Unlock()
}

func (e *backendExecutor) worker() {
	for {
		req, schedulerDecisionMs, ok := e.dequeue()
		if !ok {
			return
		}
		if req.ctx.Err() != nil {
			req.resultCh <- scheduledResult{err: req.ctx.Err()}
			continue
		}

		provider, err := e.manager.GetModelProvider(req.ctx, req.backend)
		if err != nil {
			req.resultCh <- scheduledResult{err: fmt.Errorf("failed to get provider for server '%s': %w", req.backend, err)}
			continue
		}

		dispatchStart := time.Now()
		response, streamCh, err := provider.Chat(req.ctx, req.messages, req.tools, req.opts)
		dispatchMs := time.Since(dispatchStart).Milliseconds()
		queueWaitMs := dispatchStart.Sub(req.enqueuedAt).Milliseconds()
		if err != nil {
			req.resultCh <- scheduledResult{err: fmt.Errorf("inference failed on server '%s': %w", req.backend, err)}
			continue
		}

		if response.Metrics == nil {
			response.Metrics = &model.ResponseMetrics{}
		}
		response.Metrics.QueueWaitMs = queueWaitMs
		response.Metrics.SchedulerDecisionMs = schedulerDecisionMs
		response.Metrics.DispatchMs = dispatchMs
		// Non-streaming backends block in provider.Chat; for streaming this remains a lightweight setup time.
		if !req.opts.Stream {
			response.Metrics.BackendLatencyMs = dispatchMs
		}

		req.resultCh <- scheduledResult{
			response: response,
			streamCh: streamCh,
			err:      nil,
		}
	}
}

func (e *backendExecutor) dequeue() (*scheduledRequest, int64, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for e.queueLen == 0 && !e.closed {
		e.cond.Wait()
	}
	if e.closed {
		return nil, 0, false
	}

	decisionStart := time.Now()
	stage := selectNextStageKey(e.stageQueues, e.policy)
	stageQueue := e.stageQueues[stage]

	idx := selectNextRequest(stageQueue)
	req := stageQueue[idx]
	stageQueue = append(stageQueue[:idx], stageQueue[idx+1:]...)
	if len(stageQueue) == 0 {
		delete(e.stageQueues, stage)
	} else {
		e.stageQueues[stage] = stageQueue
	}
	e.queueLen--
	decisionMs := time.Since(decisionStart).Milliseconds()
	return req, decisionMs, true
}

// selectNextRequest returns the index of the next request to dequeue from a stage queue.
// If the head request's RequestSchedulingPolicy is "priority", picks the highest-priority
// request in the queue (tie-breaking by oldest enqueue time). Otherwise FIFO (index 0).
func selectNextRequest(queue []*scheduledRequest) int {
	if len(queue) <= 1 {
		return 0
	}
	head := queue[0]
	if head.opts.RequestSchedulingPolicy != model.RequestSchedulingPolicyPriority {
		return 0
	}
	bestIdx := 0
	bestPriority := head.opts.SchedulingHints.GetPriority()
	for i, req := range queue[1:] {
		p := req.opts.SchedulingHints.GetPriority()
		if p > bestPriority || (p == bestPriority && req.enqueuedAt.Before(queue[bestIdx].enqueuedAt)) {
			bestIdx = i + 1
			bestPriority = p
		}
	}
	return bestIdx
}

func selectNextStageKey(stageQueues map[string][]*scheduledRequest, policy model.SchedulingPolicy) string {
	// Stage-level scheduling only: requests are always FIFO within each stage queue.
	// Between stages, apply the backend policy:
	// 1) priority     -> highest priority head request (tie: older request first), or
	// 2) fcfs/default    -> oldest head request across stages.

	var bestStage string
	var bestReq *scheduledRequest
	for stage, q := range stageQueues {
		if len(q) == 0 {
			continue
		}
		head := q[0]
		if bestReq == nil {
			bestStage = stage
			bestReq = head
			continue
		}
		if policy == model.SchedulingPolicyPriority {
			headPriority := head.opts.SchedulingHints.GetPriority()
			bestPriority := bestReq.opts.SchedulingHints.GetPriority()
			if headPriority > bestPriority || (headPriority == bestPriority && head.enqueuedAt.Before(bestReq.enqueuedAt)) {
				bestStage = stage
				bestReq = head
			}
			continue
		}
		if head.enqueuedAt.Before(bestReq.enqueuedAt) {
			bestStage = stage
			bestReq = head
		}
	}
	return bestStage
}
