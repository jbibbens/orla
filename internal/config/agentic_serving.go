// Package config provides configuration management for Orla, including
// Agentic Serving Layer configuration (RFC 5).
package config

import (
	"fmt"

	"github.com/dorcha-inc/orla/internal/core"
)

const (
	WorkflowGraphStartNodeID = "__start__"
	WorkflowGraphEndNodeID   = "__end__"
)

// AgenticServingMode represents the operational mode of the Agentic Serving Layer
type AgenticServingMode string

const (
	// AgenticServingModeEmbedded runs the serving layer in-process with orla agent
	AgenticServingModeEmbedded AgenticServingMode = "embedded"
	// AgenticServingModeDaemon runs the serving layer as a separate daemon process
	AgenticServingModeDaemon AgenticServingMode = "daemon"
)

// CachePolicyType represents the type of KV cache policy
type CachePolicyType string

const (
	// CachePolicyPreserve always preserves cache (unless flush_after_final modifier is set)
	CachePolicyPreserve CachePolicyType = "preserve"
	// CachePolicyFlushUnderPressure flushes cache when memory pressure exceeds threshold
	CachePolicyFlushUnderPressure CachePolicyType = "flush_under_pressure"
	// CachePolicyAggressiveFlush always flushes cache after each request
	CachePolicyAggressiveFlush CachePolicyType = "aggressive_flush"
	// CachePolicyPreserveWithinWorkflow preserves cache within a workflow but flushes when transitioning to a different workflow
	CachePolicyPreserveWithinWorkflow CachePolicyType = "preserve_within_workflow"
)

// AgenticServingConfig represents the Agentic Serving Layer configuration (RFC 5)
type AgenticServingConfig struct {
	// Mode is the operational mode (embedded or daemon)
	Mode AgenticServingMode `yaml:"mode,omitempty" mapstructure:"mode"`
	// Daemon configuration (only when mode=daemon)
	Daemon *DaemonConfig `yaml:"daemon,omitempty" mapstructure:"daemon"`
	// LLMServers is a list of LLM server configurations
	LLMServers []*LLMServerConfig `yaml:"llm_servers,omitempty" mapstructure:"llm_servers"`
	// AgentProfiles is a list of agent profile configurations
	AgentProfiles []*AgentProfile `yaml:"agent_profiles,omitempty" mapstructure:"agent_profiles"`
	// Workflows is a list of workflow configurations
	Workflows []*Workflow `yaml:"workflows,omitempty" mapstructure:"workflows"`
}

// DaemonConfig represents daemon-specific configuration
type DaemonConfig struct {
	// ListenAddress is the address the daemon listens on (e.g., "localhost:8081")
	ListenAddress string `yaml:"listen_address,omitempty" mapstructure:"listen_address"`
}

// LLMServerConfig represents an LLM server configuration
type LLMServerConfig struct {
	// Name is a unique identifier for this LLM server configuration
	Name string `yaml:"name" mapstructure:"name"`
	// Backend is the LLM backend configuration
	Backend *core.LLMBackend `yaml:"backend" mapstructure:"backend"`
	// Model is the model identifier (e.g., "ollama:llama3:8b", "openai:gpt-4")
	Model string `yaml:"model" mapstructure:"model"`
	// Context is the context sharing configuration
	Context *ContextConfig `yaml:"context,omitempty" mapstructure:"context"`
	// Cache is the KV cache policy configuration
	Cache *CacheConfig `yaml:"cache,omitempty" mapstructure:"cache"`
}

// ContextConfig represents context sharing configuration
type ContextConfig struct {
	// Shared indicates whether multiple agents share conversation context
	Shared bool `yaml:"shared,omitempty" mapstructure:"shared"`
	// SyncInterval is how often to synchronize context across agents (in tokens)
	// Only used when shared=true
	SyncInterval int `yaml:"sync_interval,omitempty" mapstructure:"sync_interval"`
}

// CacheConfig represents KV cache policy configuration
type CacheConfig struct {
	// Policy is the cache policy type
	Policy CachePolicyType `yaml:"policy" mapstructure:"policy"`
	// FlushUnderPressure indicates whether to flush under memory pressure
	FlushUnderPressure bool `yaml:"flush_under_pressure,omitempty" mapstructure:"flush_under_pressure"`
	// MemoryPressureThreshold is the memory pressure threshold (0.0-1.0) for flush_under_pressure policy
	MemoryPressureThreshold float64 `yaml:"memory_pressure_threshold,omitempty" mapstructure:"memory_pressure_threshold"`
	// FlushAfterFinal indicates whether to flush cache after final iteration
	FlushAfterFinal bool `yaml:"flush_after_final,omitempty" mapstructure:"flush_after_final"`
}

// AgentProfile represents an agent profile configuration
type AgentProfile struct {
	// Name is a unique identifier for this agent profile
	Name string `yaml:"name" mapstructure:"name"`
	// LLMServer is the name of the LLM server configuration to use
	LLMServer string `yaml:"llm_server" mapstructure:"llm_server"`
	// Inference is the inference parameter configuration
	Inference *InferenceConfig `yaml:"inference,omitempty" mapstructure:"inference"`
	// Tools is the tool access configuration
	Tools *ToolsConfig `yaml:"tools,omitempty" mapstructure:"tools"`
}

// InferenceConfig represents inference parameter configuration
type InferenceConfig struct {
	// Temperature is the sampling temperature (0.0-2.0)
	Temperature float64 `yaml:"temperature,omitempty" mapstructure:"temperature"`
	// TopP is the nucleus sampling parameter (0.0-1.0)
	TopP float64 `yaml:"top_p,omitempty" mapstructure:"top_p"`
	// MaxTokens is the maximum number of tokens to generate (nil = no limit)
	MaxTokens *int `yaml:"max_tokens,omitempty" mapstructure:"max_tokens"`
}

// ToolsConfig represents tool access configuration
type ToolsConfig struct {
	// Allowed is a list of allowed tool names (empty = all tools available)
	Allowed []string `yaml:"allowed,omitempty" mapstructure:"allowed"`
}

// Workflow represents a workflow configuration.
// Exactly one of tasks (linear list) or graph (nodes and edges) must be set; both cannot be set at once.
type Workflow struct {
	// Name is a unique identifier for this workflow
	Name string `yaml:"name" mapstructure:"name"`
	// Tasks is a linear sequence of workflow tasks (use when graph is not set)
	Tasks []*WorkflowTask `yaml:"tasks,omitempty" mapstructure:"tasks"`
	// Graph defines the workflow as nodes and edges (use when tasks is not set)
	Graph *WorkflowGraph `yaml:"graph,omitempty" mapstructure:"graph"`
}

// WorkflowGraph defines a workflow as a graph of nodes and edges.
// Reserved node ids: __start__ (entry), __end__ (exit). Execution follows a single path;
// for now only linear chains are supported (one path from start to end).
type WorkflowGraph struct {
	// Nodes are the workflow steps\ i.e the agent invocations, keyed by id in the slice order or by ID field
	Nodes []*WorkflowNode `yaml:"nodes" mapstructure:"nodes"`
	// Edges define the flow: from -> to (node ids, or __start__ / __end__)
	Edges []*WorkflowEdge `yaml:"edges" mapstructure:"edges"`
}

// WorkflowNode is a single node in a workflow graph (one agent step).
type WorkflowNode struct {
	// ID is the unique node id (required). Use __start__ and __end__ only as edge endpoints, not as node ids.
	ID string `yaml:"id" mapstructure:"id"`
	// AgentProfile is the name of the agent profile to use
	AgentProfile string `yaml:"agent_profile" mapstructure:"agent_profile"`
	// LLMServer is an optional override for the LLM server
	LLMServer string `yaml:"llm_server,omitempty" mapstructure:"llm_server"`
	// Prompt is the prompt or template for this task
	Prompt string `yaml:"prompt,omitempty" mapstructure:"prompt"`
	// UseContext indicates whether to use previous task outputs as context
	UseContext bool `yaml:"use_context,omitempty" mapstructure:"use_context"`
}

// WorkflowEdge connects two nodes (from -> to). Use __start__ as from for entry, __end__ as to for exit.
type WorkflowEdge struct {
	From string `yaml:"from" mapstructure:"from"`
	To   string `yaml:"to" mapstructure:"to"`
}

// WorkflowTask represents a single task in a workflow
type WorkflowTask struct {
	// AgentProfile is the name of the agent profile to use for this task
	AgentProfile string `yaml:"agent_profile" mapstructure:"agent_profile"`
	// LLMServer is an optional override for the LLM server configuration
	// If not set, uses the LLM server from the agent profile
	LLMServer string `yaml:"llm_server,omitempty" mapstructure:"llm_server"`
	// Prompt is the prompt or prompt template for this task
	// If empty, uses the workflow's initial prompt
	Prompt string `yaml:"prompt,omitempty" mapstructure:"prompt"`
	// UseContext indicates whether to use previous task outputs as context
	// When true, the task receives the accumulated context from previous tasks
	UseContext bool `yaml:"use_context,omitempty" mapstructure:"use_context"`
}

// validateAgenticServingConfig validates the agentic serving configuration
func validateAgenticServingConfig(cfg *AgenticServingConfig) error {
	// Validate mode
	if cfg.Mode != "" && cfg.Mode != AgenticServingModeEmbedded && cfg.Mode != AgenticServingModeDaemon {
		return fmt.Errorf("mode must be 'embedded' or 'daemon', got '%s'", cfg.Mode)
	}

	// Validate daemon config (only when mode=daemon)
	if cfg.Mode == AgenticServingModeDaemon && cfg.Daemon == nil {
		return fmt.Errorf("daemon configuration is required when mode=daemon")
	}

	// Validate LLM server name uniqueness
	llmServerNames := make(map[string]bool)
	for _, server := range cfg.LLMServers {
		if server.Name == "" {
			return fmt.Errorf("llm_server name cannot be empty")
		}
		if llmServerNames[server.Name] {
			return fmt.Errorf("duplicate llm_server name: %s", server.Name)
		}
		llmServerNames[server.Name] = true

		// Validate LLM server configuration
		if err := validateLLMServerConfig(server); err != nil {
			return fmt.Errorf("llm_server '%s': %w", server.Name, err)
		}
	}

	// Validate agent profile name uniqueness and references
	agentProfileNames := make(map[string]bool)
	for _, profile := range cfg.AgentProfiles {
		if profile.Name == "" {
			return fmt.Errorf("agent_profile name cannot be empty")
		}
		if agentProfileNames[profile.Name] {
			return fmt.Errorf("duplicate agent_profile name: %s", profile.Name)
		}
		agentProfileNames[profile.Name] = true

		// Validate agent profile references
		if profile.LLMServer == "" {
			return fmt.Errorf("agent_profile '%s': llm_server cannot be empty", profile.Name)
		}
		if !llmServerNames[profile.LLMServer] {
			return fmt.Errorf("agent_profile '%s': references unknown llm_server '%s'", profile.Name, profile.LLMServer)
		}

		// Validate inference config if present
		if profile.Inference != nil {
			if err := validateInferenceConfig(profile.Inference); err != nil {
				return fmt.Errorf("agent_profile '%s': %w", profile.Name, err)
			}
		}
	}

	// Validate workflow name uniqueness and references
	workflowNames := make(map[string]bool)
	for _, workflow := range cfg.Workflows {
		if workflow.Name == "" {
			return fmt.Errorf("workflow name cannot be empty")
		}
		if workflowNames[workflow.Name] {
			return fmt.Errorf("duplicate workflow name: %s", workflow.Name)
		}
		workflowNames[workflow.Name] = true

		// Validate workflow: exactly one of tasks or graph
		hasTasks := len(workflow.Tasks) > 0
		hasGraph := workflow.Graph != nil && len(workflow.Graph.Nodes) > 0
		if !hasTasks && !hasGraph {
			return fmt.Errorf("workflow '%s': must have either 'tasks' or 'graph' with at least one node", workflow.Name)
		}
		if hasTasks && hasGraph {
			return fmt.Errorf("workflow '%s': cannot have both 'tasks' and 'graph'; set exactly one", workflow.Name)
		}
		if hasGraph {
			if err := validateWorkflowGraph(workflow.Name, workflow.Graph, agentProfileNames, llmServerNames); err != nil {
				return fmt.Errorf("error validating workflow '%s' graph: %w", workflow.Name, err)
			}
			continue
		}

		// If we have reached this point, we have tasks and no graph
		for i, task := range workflow.Tasks {
			if task.AgentProfile == "" {
				return fmt.Errorf("workflow '%s' task %d: agent_profile cannot be empty", workflow.Name, i+1)
			}
			if !agentProfileNames[task.AgentProfile] {
				return fmt.Errorf("workflow '%s' task %d: references unknown agent_profile '%s'", workflow.Name, i+1, task.AgentProfile)
			}
			// Validate LLM server override if present
			if task.LLMServer != "" && !llmServerNames[task.LLMServer] {
				return fmt.Errorf("workflow '%s' task %d: references unknown llm_server '%s'", workflow.Name, i+1, task.LLMServer)
			}
		}
	}

	return nil
}

// validateLLMServerConfig validates an LLM server configuration
func validateLLMServerConfig(server *LLMServerConfig) error {
	if server.Backend == nil {
		return fmt.Errorf("backend cannot be nil")
	}
	if server.Backend.Type != core.LLMInferenceAPITypeOllama && server.Backend.Type != core.LLMInferenceAPITypeOpenAI && server.Backend.Type != core.LLMInferenceAPITypeSGLang {
		return fmt.Errorf("backend type must be 'ollama', 'openai', or 'sglang', got '%s'", server.Backend.Type)
	}
	if server.Backend.Endpoint == "" {
		return fmt.Errorf("backend endpoint cannot be empty")
	}
	if server.Model == "" {
		return fmt.Errorf("model cannot be empty")
	}

	// Validate context config if present
	if server.Context != nil {
		if server.Context.Shared && server.Context.SyncInterval < 0 {
			return fmt.Errorf("sync_interval must be non-negative when shared=true")
		}
	}

	// Validate cache config if present
	if server.Cache != nil {
		if err := validateCacheConfig(server.Cache); err != nil {
			return err
		}
	}

	return nil
}

// validateCacheConfig validates a cache configuration
func validateCacheConfig(cache *CacheConfig) error {
	validPolicies := map[CachePolicyType]struct{}{
		CachePolicyPreserve:               {},
		CachePolicyFlushUnderPressure:     {},
		CachePolicyAggressiveFlush:        {},
		CachePolicyPreserveWithinWorkflow: {},
	}

	if _, ok := validPolicies[cache.Policy]; !ok {
		return fmt.Errorf("cache policy must be one of: preserve, flush_under_pressure, aggressive_flush, preserve_within_workflow, got '%s'", cache.Policy)
	}

	// Validate policy-specific parameters
	switch cache.Policy {
	case CachePolicyFlushUnderPressure:
		if cache.MemoryPressureThreshold < 0 || cache.MemoryPressureThreshold > 1 {
			return fmt.Errorf("memory_pressure_threshold must be between 0.0 and 1.0, got %f", cache.MemoryPressureThreshold)
		}
	}

	return nil
}

// validateInferenceConfig validates an inference configuration
func validateInferenceConfig(inference *InferenceConfig) error {
	if inference.Temperature < 0 || inference.Temperature > 2 {
		return fmt.Errorf("temperature must be between 0.0 and 2.0, got %f", inference.Temperature)
	}
	if inference.TopP < 0 || inference.TopP > 1 {
		return fmt.Errorf("top_p must be between 0.0 and 1.0, got %f", inference.TopP)
	}
	if inference.MaxTokens != nil && *inference.MaxTokens < 1 {
		return fmt.Errorf("max_tokens must be at least 1, got %d", *inference.MaxTokens)
	}
	return nil
}

// validateWorkflowGraph validates a workflow graph (node ids, agent refs, edges, linear path).
func validateWorkflowGraph(workflowName string, g *WorkflowGraph, agentProfileNames, llmServerNames map[string]bool) error {
	nodeIds := make(map[string]bool)
	for i, n := range g.Nodes {
		if n.ID == "" {
			return fmt.Errorf("workflow '%s' graph node %d: id cannot be empty", workflowName, i+1)
		}
		if n.ID == WorkflowGraphStartNodeID || n.ID == WorkflowGraphEndNodeID {
			return fmt.Errorf("workflow '%s' graph node %d: id cannot be reserved %s or %s", workflowName, i+1, WorkflowGraphStartNodeID, WorkflowGraphEndNodeID)
		}
		if nodeIds[n.ID] {
			return fmt.Errorf("workflow '%s' graph: duplicate node id '%s'", workflowName, n.ID)
		}
		nodeIds[n.ID] = true
		if n.AgentProfile == "" {
			return fmt.Errorf("workflow '%s' graph node '%s': agent_profile cannot be empty", workflowName, n.ID)
		}
		if !agentProfileNames[n.AgentProfile] {
			return fmt.Errorf("workflow '%s' graph node '%s': references unknown agent_profile '%s'", workflowName, n.ID, n.AgentProfile)
		}
		if n.LLMServer != "" && !llmServerNames[n.LLMServer] {
			return fmt.Errorf("workflow '%s' graph node '%s': references unknown llm_server '%s'", workflowName, n.ID, n.LLMServer)
		}
	}
	if len(g.Edges) == 0 {
		return fmt.Errorf("workflow '%s' graph: must have at least one edge", workflowName)
	}
	outEdges := make(map[string][]string)
	for _, e := range g.Edges {
		fromOk := e.From == WorkflowGraphStartNodeID || nodeIds[e.From]
		toOk := e.To == WorkflowGraphEndNodeID || nodeIds[e.To]
		if !fromOk || !toOk {
			return fmt.Errorf("workflow '%s' graph edge %s -> %s: from and to must be %s/%s or defined node ids", workflowName, e.From, e.To, WorkflowGraphStartNodeID, WorkflowGraphEndNodeID)
		}
		outEdges[e.From] = append(outEdges[e.From], e.To)
	}
	// Require explicit __start__ and __end__: no implicit entry or exit
	if nexts := outEdges[WorkflowGraphStartNodeID]; len(nexts) != 1 {
		return fmt.Errorf("workflow '%s' graph: must have exactly one edge from %s to the first node", workflowName, WorkflowGraphStartNodeID)
	}
	start := outEdges[WorkflowGraphStartNodeID][0]

	visited := make(map[string]bool)
	cur := start
	for cur != "" && cur != WorkflowGraphEndNodeID && !visited[cur] {
		visited[cur] = true
		nexts := outEdges[cur]
		if len(nexts) > 1 {
			return fmt.Errorf("workflow '%s' graph: node '%s' has multiple outgoing edges (only linear chains supported)", workflowName, cur)
		}
		if len(nexts) == 0 {
			return fmt.Errorf("workflow '%s' graph: must have an edge from the last node to %s (node '%s' has no outgoing edge)", workflowName, WorkflowGraphEndNodeID, cur)
		}
		cur = nexts[0]
	}
	if cur != WorkflowGraphEndNodeID {
		return fmt.Errorf("workflow '%s' graph: path must end at %s", workflowName, WorkflowGraphEndNodeID)
	}
	for id := range nodeIds {
		if !visited[id] {
			return fmt.Errorf("workflow '%s' graph: node '%s' is not reachable from %s", workflowName, id, WorkflowGraphStartNodeID)
		}
	}
	return nil
}

// CompileWorkflowTasks returns the task list for a workflow.
// If the workflow has a graph, it is compiled to a linear task list (execution order).
// Otherwise the workflow's tasks slice is returned.
func CompileWorkflowTasks(w *Workflow) ([]*WorkflowTask, error) {
	if w.Graph != nil && len(w.Graph.Nodes) > 0 {
		return compileGraphToTasks(w.Graph)
	}
	return w.Tasks, nil
}

// compileGraphToTasks returns the task list in execution order for a linear graph.
// Requires explicit __start__ and __end__ (validated by validateWorkflowGraph).
func compileGraphToTasks(g *WorkflowGraph) ([]*WorkflowTask, error) {
	nodeByID := make(map[string]*WorkflowNode)
	for _, n := range g.Nodes {
		nodeByID[n.ID] = n
	}
	outEdges := make(map[string][]string)
	for _, e := range g.Edges {
		outEdges[e.From] = append(outEdges[e.From], e.To)
	}
	startNexts := outEdges[WorkflowGraphStartNodeID]
	if len(startNexts) != 1 {
		return nil, fmt.Errorf("workflow graph: must have exactly one edge from %s to the first node", WorkflowGraphStartNodeID)
	}
	start := startNexts[0]

	var order []string
	cur := start
	for cur != "" && cur != WorkflowGraphEndNodeID {
		order = append(order, cur)
		nexts := outEdges[cur]
		if len(nexts) == 0 {
			return nil, fmt.Errorf("workflow graph: must have an edge from the last node to %s", WorkflowGraphEndNodeID)
		}
		cur = nexts[0]
	}
	tasks := make([]*WorkflowTask, 0, len(order))
	for _, id := range order {
		n := nodeByID[id]
		if n == nil {
			continue
		}
		tasks = append(tasks, &WorkflowTask{
			AgentProfile: n.AgentProfile,
			LLMServer:    n.LLMServer,
			Prompt:       n.Prompt,
			UseContext:   n.UseContext,
		})
	}
	return tasks, nil
}
