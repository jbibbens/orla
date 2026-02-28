package main

func main() {
}

// // Run SWE-bench Lite with default Orla client and SGLang backend.
// //
// // Prerequisites:
// //   - Orla daemon running (e.g. docker compose -f deploy/docker-compose.sglang.yaml up -d)
// //   - SGLang serving the model (included in the compose above)
// //
// // Usage:
// //
// //	go run ./examples/swe_bench_lite/cache_strategy/orla_sglang [flags]
// //
// // Flags:
// //   -orla-url: Orla daemon URL (default http://localhost:8081)
// //   -instance: Path to a SWE-bench Lite instance JSON file (one object)
// //   -workdir: Working directory for bash tool (default: current directory)
// //   -max-steps: Max agent steps (default 25)
// //   -output: Output JSONL path for predictions (optional)
// package main

// import (
// 	"context"
// 	"encoding/json"
// 	"flag"
// 	"fmt"
// 	"log"
// 	"os"
// 	"os/exec"
// 	"path/filepath"
// 	"strings"

// 	orla "github.com/dorcha-inc/orla/pkg/api"
// )

// const defaultSystemPrompt = `You are a software engineering agent. You have access to a bash shell to run commands.
// Your task is to fix the issue described in the problem statement. You can run any bash command in the repository.
// Work in the repository root. When you have a fix, you can output a unified diff (e.g. from git diff) or describe the changes.
// Use the run_bash tool to execute commands. Each tool call runs a single bash command.`

// func main() {
// 	orlaURL := flag.String("orla-url", "http://localhost:8081", "Orla daemon URL")
// 	instancePath := flag.String("instance", "", "Path to SWE-bench Lite instance JSON file")
// 	workdir := flag.String("workdir", ".", "Working directory for bash tool")
// 	maxSteps := flag.Int("max-steps", 25, "Max agent steps")
// 	outputPath := flag.String("output", "", "Output JSONL path for predictions (optional)")
// 	flag.Parse()

// 	if *instancePath == "" {
// 		log.Fatal(" -instance is required (path to SWE-bench Lite instance JSON)")
// 	}

// 	ctx := context.Background()

// 	client := orla.NewOrlaClient(*orlaURL)
// 	if err := client.Health(ctx); err != nil {
// 		log.Fatalf("Orla health check failed: %v (is the daemon running?)", err)
// 	}

// 	backend := &orla.LLMBackend{
// 		Name:     "sglang",
// 		Endpoint: "http://sglang:30000",
// 		Type:     "ollama",
// 		ModelID:  "ollama:Qwen/Qwen3-8B",
// 	}
// 	if err := client.RegisterBackend(ctx, backend); err != nil {
// 		log.Fatalf("Register backend: %v", err)
// 	}

// 	agent := orla.NewAgent(client, backend)
// 	agent.SetMaxTokens(4096)

// 	absWorkdir, err := filepath.Abs(*workdir)
// 	if err != nil {
// 		log.Fatalf("workdir: %v", err)
// 	}

// 	bashTool, err := orla.NewTool(
// 		"run_bash",
// 		"Run a single bash command in the repository. Returns stdout, stderr, and exit code.",
// 		orla.ToolSchema{
// 			"type": "object",
// 			"properties": map[string]any{
// 				"command": map[string]any{"type": "string", "description": "The bash command to run (e.g. 'ls -la', 'git status')"},
// 			},
// 			"required": []any{"command"},
// 		},
// 		orla.ToolSchema{
// 			"type": "object",
// 			"properties": map[string]any{
// 				"stdout":   map[string]any{"type": "string"},
// 				"stderr":   map[string]any{"type": "string"},
// 				"exit_code": map[string]any{"type": "integer"},
// 			},
// 		},
// 		orla.ToolRunnerFromSchema(func(ctx context.Context, input orla.ToolSchema) (orla.ToolSchema, error) {
// 			cmdStr, _ := input["command"].(string)
// 			cmdStr = strings.TrimSpace(cmdStr)
// 			if cmdStr == "" {
// 				return orla.ToolSchema{"stdout": "", "stderr": "empty command", "exit_code": 1.0}, nil
// 			}
// 			cmd := exec.CommandContext(ctx, "bash", "-c", cmdStr)
// 			cmd.Dir = absWorkdir
// 			out, err := cmd.CombinedOutput()
// 			stdout := string(out)
// 			stderr := ""
// 			exitCode := 0.0
// 			if err != nil {
// 				if exitErr, ok := err.(*exec.ExitError); ok {
// 					exitCode = float64(exitErr.ExitCode())
// 				} else {
// 					stderr = err.Error()
// 					exitCode = 1
// 				}
// 			}
// 			return orla.ToolSchema{"stdout": stdout, "stderr": stderr, "exit_code": exitCode}, nil
// 		}),
// 	)
// 	if err != nil {
// 		log.Fatal(err)
// 	}
// 	if err := agent.AddTool(bashTool); err != nil {
// 		log.Fatal(err)
// 	}

// 	instance, err := loadInstance(*instancePath)
// 	if err != nil {
// 		log.Fatalf("Load instance: %v", err)
// 	}

// 	userContent := fmt.Sprintf("## Problem statement\n\n%s\n\n## Repository\n- repo: %s\n- base_commit: %s\n\nFix the issue using the run_bash tool. Work in the repository root.",
// 		instance.ProblemStatement, instance.Repo, instance.BaseCommit)

// 	messages := []orla.Message{
// 		{Role: "system", Content: defaultSystemPrompt},
// 		{Role: "user", Content: userContent},
// 	}

// 	var lastContent string
// 	for step := 0; step < *maxSteps; step++ {
// 		resp, err := agent.ExecuteWithMessages(ctx, messages)
// 		if err != nil {
// 			log.Fatalf("Step %d execute: %v", step+1, err)
// 		}
// 		lastContent = resp.Content

// 		if len(resp.ToolCalls) == 0 {
// 			log.Printf("Step %d: model finished (no tool calls)", step+1)
// 			break
// 		}

// 		messages = append(messages, orla.Message{Role: "assistant", Content: resp.Content})
// 		toolMessages, err := agent.RunToolCallsInResponse(ctx, resp)
// 		if err != nil {
// 			log.Fatalf("Step %d run tools: %v", step+1, err)
// 		}
// 		for _, m := range toolMessages {
// 			messages = append(messages, *m)
// 		}
// 	}

// 	if *outputPath != "" {
// 		patch := extractPatch(lastContent)
// 		pred := prediction{
// 			InstanceID:       instance.InstanceID,
// 			ModelNameOrPath:  "orla-sglang",
// 			ModelPatch:       patch,
// 		}
// 		f, err := os.OpenFile(*outputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
// 		if err != nil {
// 			log.Printf("Warning: could not write predictions: %v", err)
// 		} else {
// 			defer f.Close()
// 			enc := json.NewEncoder(f)
// 			enc.SetEscapeHTML(false)
// 			if err := enc.Encode(pred); err != nil {
// 				log.Printf("Warning: encode prediction: %v", err)
// 			}
// 		}
// 	}

// 	fmt.Println("--- Final model output ---")
// 	fmt.Println(lastContent)
// }

// type swebenchInstance struct {
// 	InstanceID       string `json:"instance_id"`
// 	Repo             string `json:"repo"`
// 	BaseCommit       string `json:"base_commit"`
// 	ProblemStatement string `json:"problem_statement"`
// }

// func loadInstance(path string) (*swebenchInstance, error) {
// 	data, err := os.ReadFile(path)
// 	if err != nil {
// 		return nil, err
// 	}
// 	var inst swebenchInstance
// 	if err := json.Unmarshal(data, &inst); err != nil {
// 		return nil, err
// 	}
// 	return &inst, nil
// }

// type prediction struct {
// 	InstanceID      string `json:"instance_id"`
// 	ModelNameOrPath string `json:"model_name_or_path"`
// 	ModelPatch      string `json:"model_patch"`
// }

// func extractPatch(content string) string {
// 	// Naive extraction: look for a diff-like block. Full harness may expect proper unified diff.
// 	if idx := strings.Index(content, "```diff"); idx >= 0 {
// 		start := idx + 7
// 		if end := strings.Index(content[start:], "```"); end >= 0 {
// 			return strings.TrimSpace(content[start : start+end])
// 		}
// 	}
// 	if idx := strings.Index(content, "```"); idx >= 0 {
// 		start := idx + 3
// 		if end := strings.Index(content[start:], "```"); end >= 0 {
// 			return strings.TrimSpace(content[start : start+end])
// 		}
// 	}
// 	return content
// }
