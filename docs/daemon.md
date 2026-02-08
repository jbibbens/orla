# Daemon and workflows

The Orla daemon runs the Agentic Serving Layer. It orchestrates workflows, applies agent profiles, manages shared context and KV cache policies across LLM backends, and exposes an HTTP API so clients can drive multi-agent workflows. This page explains how to run and configure it.

## Concepts

### Workflow orchestration 
Define workflows as a sequence of tasks. Each task is tied to an agent profile and optionally a specific LLM server. The daemon tracks execution state (current task index) and returns the next task to clients.

### Inference
When a client executes a task, the daemon runs inference. It uses the task’s agent profile to select the right backend and model, applies context (including shared context when configured), and respects KV cache policies.

### Agent profiles
Agent profiles specify which LLM server to use, inference options such as the temperature and max_tokens, and optional tool allow-lists. Different workflow stages can use different profiles e.g. you can use a small model for routing, and a larger model for synthesis.

### LLM server configs
Each LLM server config defines 1) the inference backend such as Ollama, SGLang, vLLM, etc., 2) the model to use, 3) context sharing (`shared: true/false`), and 4) KV cache policy such as preserve the KV Cache on small turns, flush under pressure, etc.

The daemon does *not* run the workflow loop itself. The *client* drives the loop. It start workflows, gets the next task from the daemon, and executes the task via an HTTP call letting the daemon handle inference. See [Driving workflows from a client](#driving-workflows-from-a-client) below.

## Running the daemon

You need an `orla.yaml` that includes an `agentic_serving` section with `mode: daemon`, plus at least one LLM server, one agent profile, and one workflow. Then:

```bash
orla daemon --config orla.yaml
```

By default the daemon listens on `localhost:8081` (configurable via `agentic_serving.daemon.listen_address`). Clients use this URL for the workflow API.

## Configuration

All daemon configuration lives under `agentic_serving` in `orla.yaml`.

### Top-level structure

```yaml
agentic_serving:
  mode: daemon
  daemon:
    listen_address: "localhost:8081"
  llm_servers: [...]
  agent_profiles: [...]
  workflows: [...]
```

### LLM servers

Each entry defines an inference backend, model, and optional context/cache policy:

```yaml
agentic_serving:
  llm_servers:
    - name: "local_ollama_small"
      backend:
        type: "ollama"
        endpoint: "http://localhost:11434"
      model: "ollama:qwen3:0.6b"
      context:
        shared: false
      cache:
        policy: "preserve_on_small_turns"
        small_turn_threshold: 100
        flush_under_pressure: true
        memory_pressure_threshold: 0.85

    - name: "sglang_shared"
      backend:
        type: "ollama"
        endpoint: "http://gpu-server:30001"
      model: "ollama:llama3:8b"
      context:
        shared: true
        sync_interval: 100
      cache:
        policy: "flush_under_pressure"
        memory_pressure_threshold: 0.85
        flush_after_final: true
```

- **Backend**: `type` can be `ollama`, `openai`, etc.; `endpoint` is the server URL. SGLang is typically used via the Ollama-compatible API.
- **Context**: `shared: true` means multiple agents using this server share conversation context (and KV cache). Use for multi-agent workflows (e.g. story-finishing). `shared: false` keeps each agent isolated.
- **Cache**: Policies include `preserve_on_small_turns`, `flush_under_pressure`, `aggressive_flush`, `preserve_within_workflow`. See [RFC 5](rfcs/rfc5.txt) for full semantics.

### Agent profiles

Agent profiles reference an LLM server and set inference options:

```yaml
agentic_serving:
  agent_profiles:
    - name: "router_agent"
      llm_server: "local_ollama_small"
      inference:
        temperature: 0.1
        top_p: 0.9
        max_tokens: 50
      tools:
        allowed: ["fs", "grep"]

    - name: "synthesis_agent"
      llm_server: "sglang_shared"
      inference:
        temperature: 0.7
        top_p: 0.95
        max_tokens: 2000
```

- **llm_server** (Required): Must match an `llm_servers[].name`.
- **inference** (Optional): temperature, top_p, max_tokens.
- **tools** (Optional): `allowed` list; empty or omitted means all tools.

### Workflows

A workflow can be defined either as alinear list of tasks or as a graph. Each task (or node) specifies an agent profile and can override the LLM server or prompt.

#### Linear workflows

```yaml
agentic_serving:
  workflows:
    - name: "story_finishing_game"
      tasks:
        - agent_profile: "story_agent_a"
          llm_server: "sglang_shared"
          use_context: true
        - agent_profile: "story_agent_b"
          llm_server: "sglang_shared"
          use_context: true
        # ... more tasks (e.g. alternating turns)

    - name: "routing_then_synthesis"
      tasks:
        - agent_profile: "router_agent"
        - agent_profile: "synthesis_agent"
```

#### Graph workflows

You can define a workflow as a graph with `nodes` and `edges`. Reserved node ids in edges: `__start__` (entry) and `__end__` (exit). Right now only **linear chains** are supported (one path from start to end), so execution order is the same as a `tasks` list. The graph form gives you **named nodes** (e.g. `writer`, `critic`) instead of indices, an **explicit flow** (edges) that’s easy to read and to visualize, and a **schema that’s ready** for future extensions (e.g. conditional edges or branching). If you don’t care about those, use `tasks`; it’s equivalent for linear workflows.

```yaml
agentic_serving:
  workflows:
    - name: "writer_then_critic"
      graph:
        nodes:
          - id: "writer"
            agent_profile: "writer_agent"
            use_context: true
          - id: "critic"
            agent_profile: "critic_agent"
            prompt: "Review and suggest improvements."
            use_context: true
        edges:
          - from: "__start__"
            to: "writer"
          - from: "writer"
            to: "critic"
          - from: "critic"
            to: "__end__"
```

You must use explicit `__start__` and `__end__` in the edges; implicit entry or exit (e.g. a node with no incoming or outgoing edges) is not allowed.

- **agent_profile** (Required): Must match an `agent_profiles[].name`.
- **llm_server** (Optional): override for this task/node.
- **prompt** (Optional): fixed prompt for this task; if empty, the client supplies the prompt (e.g. accumulated story so far).
- **use_context** (Optional): If true, the task receives accumulated context from previous tasks (and when using shared-context LLM servers, the daemon can sync context across agents).

Workflows enable multi-agent coordination (alternating turns, shared context) and model cascades. The full schema and validation rules are in [RFC 5](rfcs/rfc5.txt).

## Driving workflows from a client

The daemon exposes an HTTP API. The client is responsible for the loop:

1. **Start workflow**: `POST /api/v1/workflow/start` with `{"workflow_name": "..."}`. Returns `execution_id`.
2. **Get next task**: `GET /api/v1/workflow/task/next?execution_id=...`. Returns the next task (and task index) or `complete: true` when the workflow is done.
3. **Execute task**: `POST /api/v1/workflow/task/execute` with execution ID, task index, and prompt. The daemon runs inference and returns the response (and optional metrics like TTFT/TPOT when streaming).
4. **Complete task**: `POST /api/v1/workflow/task/complete` with execution ID, task index, and response. The daemon advances the workflow state.
5. Repeat from step 2 until `complete: true`.

Optional: **Get context** and **Sync context** for servers with `context.shared: true`, so clients can pass shared conversation history into prompts and sync back assistant outputs.

The Go client in [pkg/api](https://github.com/dorcha-inc/orla/tree/main/pkg/api) implements this loop. High-level usage:

```go
executor := api.NewWorkflowExecutor("http://localhost:8081")
responses, err := executor.ExecuteWorkflow(ctx, "story_finishing_game", "Once upon a time", 100)
```

See [pkg/api/README.md](../pkg/api/README.md) for the full API and low-level client usage.

## Cheat Sheet

| You want to… | Do this |
|--------------|--------|
| Run multi-agent workflows with per-step backend selection | Define `llm_servers`, `agent_profiles`, and `workflows` in `agentic_serving`; run `orla daemon`; drive workflows via the HTTP API or `pkg/api`. |
| Share context and KV cache across agents | Set `context.shared: true` on the LLM server and `use_context: true` on tasks; use the context get/sync API as needed. |
| Model cascade | Use different agent profiles (and thus different `llm_server`s) for different workflow tasks. |
