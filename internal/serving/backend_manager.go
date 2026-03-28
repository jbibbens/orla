package serving

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/harvard-cns/orla/internal/core"
	"github.com/harvard-cns/orla/internal/model"
	"github.com/harvard-cns/orla/internal/serving/memory"
	"github.com/modelcontextprotocol/go-sdk/mcp"
	"go.uber.org/zap"
)

type backendEntry struct {
	backend        *core.LLMBackend
	modelID        string
	maxConcurrency int
	queueCapacity  int
}

// LLMBackendManager manages a pool of LLM backend configurations and their providers
type LLMBackendManager struct {
	backends      map[string]*backendEntry
	providers     map[string]model.Provider
	executors     map[string]*backendExecutor
	memoryManager *memory.DefaultManager
	mu            sync.RWMutex
}

// NewLLMBackendManager creates a new LLM backend manager.
func NewLLMBackendManager(mm *memory.DefaultManager) *LLMBackendManager {
	return &LLMBackendManager{
		backends:      make(map[string]*backendEntry),
		providers:     make(map[string]model.Provider),
		executors:     make(map[string]*backendExecutor),
		memoryManager: mm,
	}
}

// AddLLMBackend registers an LLM backend by name.
func (m *LLMBackendManager) AddLLMBackend(name string, backend *core.LLMBackend, modelID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.backends[name] = &backendEntry{
		backend:        backend,
		modelID:        modelID,
		maxConcurrency: backend.MaxConcurrency,
		queueCapacity:  backend.QueueCapacity,
	}
	delete(m.providers, name)
	if exec, ok := m.executors[name]; ok {
		exec.close()
		delete(m.executors, name)
	}

	if m.memoryManager != nil && backend.Type == core.LLMInferenceAPITypeSGLang {
		baseURL := strings.TrimSuffix(strings.TrimRight(backend.Endpoint, "/"), "/v1")
		cc := memory.NewSGLangCacheController(baseURL)
		m.memoryManager.RegisterCacheController(name, cc)
		zap.L().Debug("Registered SGLang cache controller for backend", zap.String("backend", name))
	}
}

// GetModelID returns the modelID string for a registered backend, or "" if not found.
func (m *LLMBackendManager) GetModelID(backendName string) string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if entry, ok := m.backends[backendName]; ok {
		return entry.modelID
	}
	return ""
}

// GetCostModel returns the CostModel for a registered backend, or nil if not found or unset.
func (m *LLMBackendManager) GetCostModel(backendName string) *core.CostModel {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if entry, ok := m.backends[backendName]; ok {
		return entry.backend.CostModel
	}
	return nil
}

// GetModelProvider returns a cached provider for an LLM backend, creating it if necessary
func (m *LLMBackendManager) GetModelProvider(ctx context.Context, backendName string) (model.Provider, error) {
	m.mu.RLock()
	if provider, exists := m.providers[backendName]; exists {
		m.mu.RUnlock()
		return provider, nil
	}
	m.mu.RUnlock()

	m.mu.Lock()
	defer m.mu.Unlock()

	// Double-check after acquiring write lock
	if provider, exists := m.providers[backendName]; exists {
		return provider, nil
	}

	entry, exists := m.backends[backendName]
	if !exists {
		return nil, fmt.Errorf("llm_backend '%s' not found", backendName)
	}

	provider, err := model.NewProviderFromBackend(entry.backend, entry.modelID)
	if err != nil {
		return nil, fmt.Errorf("failed to create provider for llm_backend '%s': %w", backendName, err)
	}

	m.providers[backendName] = provider

	zap.L().Debug("Created and cached a model provider for LLM backend",
		zap.String("backend_name", backendName),
		zap.String("model", entry.modelID))

	return provider, nil
}

func (m *LLMBackendManager) getOrCreateExecutorLocked(backendName string) (*backendExecutor, error) {
	entry, exists := m.backends[backendName]
	if !exists {
		return nil, fmt.Errorf("llm_backend '%s' not found", backendName)
	}
	if exec, ok := m.executors[backendName]; ok {
		return exec, nil
	}
	exec := newBackendExecutor(backendName, m, entry.maxConcurrency, entry.queueCapacity, m.memoryManager)
	m.executors[backendName] = exec
	return exec, nil
}

// ChatOptions carries optional metadata for a scheduled chat request.
type ChatOptions struct {
	WorkflowID  string
	CachePolicy string
}

// ScheduleChat queues a request for execution under the backend's scheduling policy.
// stageName identifies the stage queue inside the backend. Empty uses "default".
func (m *LLMBackendManager) ScheduleChat(ctx context.Context, backendName, stageName string, messages []model.Message, tools []*mcp.Tool, opts model.InferenceOptions, chatOpts ...ChatOptions) (*model.Response, <-chan model.StreamEvent, error) {
	m.mu.Lock()
	exec, err := m.getOrCreateExecutorLocked(backendName)
	m.mu.Unlock()
	if err != nil {
		return nil, nil, err
	}

	req := &scheduledRequest{
		ctx:        ctx,
		backend:    backendName,
		stageName:  stageName,
		messages:   messages,
		tools:      tools,
		opts:       opts,
		enqueuedAt: time.Now(),
		resultCh:   make(chan scheduledResult, 1),
	}
	if len(chatOpts) > 0 {
		req.workflowID = chatOpts[0].WorkflowID
		req.cachePolicy = chatOpts[0].CachePolicy
	}
	if err := exec.enqueue(req); err != nil {
		return nil, nil, err
	}

	select {
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	case result := <-req.resultCh:
		return result.response, result.streamCh, result.err
	}
}

// HealthStatus represents the health status of an LLM backend
type HealthStatus string

const (
	HealthStatusHealthy     HealthStatus = "healthy"
	HealthStatusDegraded    HealthStatus = "degraded"
	HealthStatusUnavailable HealthStatus = "unavailable"
)

const (
	healthCheckTimeout           = 5 * time.Second
	healthCheckDegradedThreshold = 2 * time.Second
)

// GetHealthStatus returns the health status of an LLM backend
func (m *LLMBackendManager) GetHealthStatus(ctx context.Context, backendName string) (HealthStatus, error) {
	m.mu.RLock()
	_, exists := m.backends[backendName]
	m.mu.RUnlock()
	if !exists {
		return HealthStatusUnavailable, fmt.Errorf("llm_backend '%s' not found", backendName)
	}

	provider, err := m.GetModelProvider(ctx, backendName)
	if err != nil {
		return HealthStatusUnavailable, fmt.Errorf("failed to get provider: %w", err)
	}

	healthCtx, cancel := context.WithTimeout(ctx, healthCheckTimeout)
	defer cancel()

	start := time.Now()
	err = provider.EnsureReady(healthCtx)
	duration := time.Since(start)

	if healthCtx.Err() == context.DeadlineExceeded {
		return HealthStatusUnavailable, fmt.Errorf("health check timed out after %v", healthCheckTimeout)
	}

	if err != nil {
		return HealthStatusUnavailable, fmt.Errorf("health check failed: %w", err)
	}

	if duration > healthCheckDegradedThreshold {
		return HealthStatusDegraded, nil
	}

	return HealthStatusHealthy, nil
}

// SelectBackendByAccuracy returns the cheapest registered backend whose Quality >= accuracy
// and that has a CostModel set. Ties are broken by ascending output cost, then input cost.
func (m *LLMBackendManager) SelectBackendByAccuracy(accuracy float64) (string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	type candidate struct {
		name string
		cm   *core.CostModel
	}
	var candidates []candidate
	for name, entry := range m.backends {
		b := entry.backend
		if b.CostModel == nil || b.Quality < accuracy {
			continue
		}
		candidates = append(candidates, candidate{name: name, cm: b.CostModel})
	}
	if len(candidates) == 0 {
		return "", fmt.Errorf("no backend with quality >= %v and a cost model; registered backends: %s",
			accuracy, m.describeBackendsLocked())
	}
	sort.Slice(candidates, func(i, j int) bool {
		if candidates[i].cm.OutputCostPerMToken != candidates[j].cm.OutputCostPerMToken {
			return candidates[i].cm.OutputCostPerMToken < candidates[j].cm.OutputCostPerMToken
		}
		return candidates[i].cm.InputCostPerMToken < candidates[j].cm.InputCostPerMToken
	})
	return candidates[0].name, nil
}

// describeBackendsLocked returns a human-readable summary of registered backends.
// Caller must hold at least m.mu.RLock().
func (m *LLMBackendManager) describeBackendsLocked() string {
	if len(m.backends) == 0 {
		return "(none)"
	}
	parts := make([]string, 0, len(m.backends))
	for name, entry := range m.backends {
		q := entry.backend.Quality
		hasCost := entry.backend.CostModel != nil
		parts = append(parts, fmt.Sprintf("%s(quality=%.2f, has_cost=%v)", name, q, hasCost))
	}
	sort.Strings(parts)
	return strings.Join(parts, ", ")
}

// ListLLMBackends returns a list of all LLM backend names
func (m *LLMBackendManager) ListLLMBackends() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	backendNames := make([]string, 0, len(m.backends))
	for backendName := range m.backends {
		backendNames = append(backendNames, backendName)
	}
	return backendNames
}
