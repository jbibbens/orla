# Agent Guidelines for Orla

This file provides guidance for AI agents working on the Orla codebase.

## Project Overview

Orla is a high-performance agent execution engine written in Go. It runs as a daemon (`orla serve`) or a one-shot CLI (`orla agent`), providing a unified API for building and running agents across multiple LLM backends (SGLang, Ollama, vLLM) via OpenAI-compatible APIs.

## Repository Layout

```
cmd/orla/          CLI entry points (main, serve, agent subcommands)
internal/
  agent/           Agent loop and one-shot executor
  config/          Viper-based YAML configuration
  core/            Shared types (LLMBackend, logger, utils)
  model/           Provider interface, OpenAI provider, mock helpers
  serving/         AgenticLayer, scheduler, backend manager
  serving/api/     HTTP API server and routes
  serving/memory/  Memory manager and cache control
  testing/         Shared test utilities
  tui/             Terminal UI (Charm Bubbletea)
pkg/api/           Public Go client library (includes MCP tool types)
examples/          Example applications and demos
deploy/            Docker Compose configs for backends
scripts/           Install/uninstall scripts
```

## Before Completing Work

Always run these commands before considering a task complete:

```bash
make lint
make test
```

- **`make lint`** runs `go vet` and `golangci-lint` across `./cmd/...`, `./internal/...`, `./pkg/...`, `./examples/...`.
- **`make test`** runs the full unit test suite across those same directories.

Fix any failures before finishing. Do not leave lint errors or failing tests.

Other useful targets:

| Target             | Purpose                                          |
|--------------------|--------------------------------------------------|
| `make format`      | `go fmt` + `go mod tidy`                         |
| `make build`       | Build binary to `bin/orla`                       |
| `make coverage`    | Generate `coverage.html` from internal and pkg   |
| `make test-race`        | Run tests with race detector                |
| `make test-integration` | Integration tests                           |

## Code Style and Conventions

### Go Version and Module

The module is `github.com/dorcha-inc/orla` using Go 1.25+. All code lives under `internal/` (private) or `pkg/` (public API).

### Naming

- Exported identifiers use PascalCase; unexported use camelCase.
- Acronyms stay uppercase: `LLMBackend`, `APIKeyEnvVar`, `TTFTMs`.
- Files: one main type per file; mocks go in `mock_*.go`; shared types go in `types.go`.

### Error Handling

- Wrap errors with context using `fmt.Errorf("description: %w", err)`.
- Include actionable context: `fmt.Errorf("inference failed on server '%s': %w", serverName, err)`.
- Validation errors should list valid options: `fmt.Errorf("log_format must be one of: %s, got '%s'", ...)`.
- Use `core.LogDeferredError(fn)` for deferred cleanup calls that may fail.

### Logging

Uses `go.uber.org/zap` via the global `zap.L()`. Always use structured fields:

```go
zap.L().Info("backend registered", zap.String("name", backendName))
zap.L().Error("inference failed", zap.String("server", name), zap.Error(err))
```

Logs go to stderr to avoid interfering with tool stdout (MCP).

### Concurrency

Shared state is protected with `sync.RWMutex`. Follow the lock/defer-unlock pattern:

```go
s.mu.Lock()
defer s.mu.Unlock()
```

Use `RLock`/`RUnlock` for read-only access.

### Optional Fields

Use pointer types for optional values (`*int`, `*float64`). Helper constructors like `core.IntPtr(n)` exist for building these.

### Interfaces

- Define interfaces in the same package as their primary consumer.
- Keep interfaces small and focused (e.g., `Provider` has three methods: `Name`, `Chat`, `EnsureReady`).
- Use the factory-registration pattern for pluggable implementations: `RegisterProviderFactory("openai", factory)`.

## Testing Conventions

### Structure

Use **table-driven tests** with subtests for multiple cases:

```go
tests := []struct {
    name        string
    // inputs and expected outputs
}{
    {name: "descriptive case name", ...},
}
for _, tt := range tests {
    t.Run(tt.name, func(t *testing.T) {
        // ...
    })
}
```

Use standalone `Test<Function>_<Scenario>` functions for single-case tests.

### Naming

Follow `Test<FunctionOrType>_<Scenario>`:

- `TestNewExecutor` -- constructor happy path
- `TestExecutor_Execute_NoToolCalls` -- method + specific scenario
- `TestLayer_Execute_ServerNotFound` -- nested type + method + scenario

### Assertions

Use `github.com/stretchr/testify`:

- **`require`** for preconditions and setup that must succeed (fails the test immediately).
- **`assert`** for the actual checks under test (continues on failure).

```go
require.NoError(t, err)
assert.Equal(t, expected, actual)
assert.Contains(t, err.Error(), "expected substring")
```

### Mocks

**Hand-written mocks with fluent builders** -- do not use code generators.

`MockProvider` example:

```go
provider := model.NewMockProvider().
    WithContent("response text").
    WithToolCall("tool_name", `{"arg":"value"}`).
    Build()
```

`MockLLMServer` for HTTP-level testing:

```go
server := model.NewMockLLMServer().
    ReturnContent("hello").
    Start()
defer server.Close()
```

Mocks include inspection helpers: `ReceivedMessages()`, `LastInferenceOptions()`, `CallCount()`.

Always verify interface compliance: `var _ Provider = (*MockProvider)(nil)`.

### Test Helpers

- `internal/testing` provides `NewCapturedOutput()` for stdout/stderr capture and `GetTestModelName()`.
- Use `MockLLMServer` for OpenAI-compatible chat API tests; use `httptest.NewServer` for other HTTP tests.
- Use `t.TempDir()` for temporary files and `t.Cleanup()` for teardown.

### Integration Tests

Integration tests use `MockLLMServer` for end-to-end flows without external services. They are named `TestIntegration_*` and run via `make test-integration` (or `make test` which includes them).

## Linting

The `.golangci.yml` enables these linters: `errcheck`, `govet`, `ineffassign`, `staticcheck`, `unused`, `errorlint`, `gosec`, `copyloopvar`, `goconst`, `misspell`.

Key settings:
- `errcheck` checks type assertions and blank identifier assignments.
- `gosec` excludes G204 (subprocess with variable) since Orla intentionally uses variables for tool paths.
- `goconst` flags strings repeated 3+ times that should be constants.

## Configuration

Orla uses Viper for YAML config loading. The config struct (`config.OrlaConfig`) uses both `yaml` and `mapstructure` tags. Valid enum values are stored as `map[T]struct{}` sets with validation in `validateConfig()`.

## Model Identifiers

Models follow the `provider:model-name` format (e.g., `openai:qwen3:0.6b`). The provider prefix selects which `ProviderFactory` creates the `Provider` implementation.
