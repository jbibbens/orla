package main

// Run SWE-bench Lite with the Orla client and SGLang backend.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

const defaultSystemPrompt = `You are a software engineering agent. You have access to a bash shell to run commands.
Your task is to fix the issue described in the problem statement. Edit the repository files using the run_bash tool (e.g. run an editor, or use sed/cat to change files). Work in the repository root.
The submitted patch is the git diff of the repository after your edits; Use the run_bash tool to execute commands. Each tool call runs a single bash command.`

const (
	maxSteps    = 256
	orlaURL     = "http://orla:8081"
	sglangURL   = "http://sglang:30000/v1"
	workdirRoot = "/workdir"
	outputPath  = "/output/predictions.jsonl"
)

func main() {
	instancePath := flag.String("instance", "", "Path to SWE-bench Lite instance JSON file")
	flag.Parse()

	if *instancePath == "" {
		log.Fatal("-instance is required (path to SWE-bench Lite instance JSON)")
	}

	instance, err := loadInstance(*instancePath)
	if err != nil {
		log.Fatalf("Load instance: %v", err)
	}

	workdir := filepath.Join(workdirRoot, instance.InstanceID)
	absWorkdir, filePathErr := filepath.Abs(workdir)
	if filePathErr != nil {
		log.Fatalf("workdir: %v", filePathErr)
	}

	ctx := context.Background()

	client := orla.NewOrlaClient(orlaURL)
	if err := client.Health(ctx); err != nil {
		log.Fatalf("Orla health check failed: %v (is the daemon running?)", err)
	}

	backend := orla.NewSGLangBackend("Qwen/Qwen3-8B", sglangURL)
	if err := client.RegisterBackend(ctx, backend); err != nil {
		log.Fatalf("Register backend: %v", err)
	}

	agent := orla.NewAgent(client, backend)
	agent.SetMaxTokens(4096)

	bashTool, err := orla.NewTool(
		"run_bash",
		"Run a single bash command in the repository. Returns stdout, stderr, and exit code.",
		orla.ToolSchema{
			"type": "object",
			"properties": map[string]any{
				"command": map[string]any{"type": "string", "description": "The bash command to run (e.g. 'ls -la', 'git status')"},
			},
			"required": []any{"command"},
		},
		orla.ToolSchema{
			"type": "object",
			"properties": map[string]any{
				"stdout":    map[string]any{"type": "string"},
				"stderr":    map[string]any{"type": "string"},
				"exit_code": map[string]any{"type": "integer"},
			},
		},
		orla.ToolRunnerFromSchema(func(ctx context.Context, input orla.ToolSchema) (orla.ToolSchema, error) {
			cmdStr, ok := input["command"].(string)
			if !ok {
				cmdStr = ""
			}
			cmdStr = strings.TrimSpace(cmdStr)
			if cmdStr == "" {
				return orla.ToolSchema{"stdout": "", "stderr": "empty command", "exit_code": 1.0}, nil
			}
			cmd := exec.CommandContext(ctx, "bash", "-c", cmdStr)
			cmd.Dir = absWorkdir
			out, err := cmd.CombinedOutput()
			stdout := string(out)
			stderr := ""
			exitCode := 0.0
			if err != nil {
				var exitErr *exec.ExitError
				if errors.As(err, &exitErr) {
					exitCode = float64(exitErr.ExitCode())
				} else {
					stderr = err.Error()
					exitCode = 1
				}
			}
			return orla.ToolSchema{"stdout": stdout, "stderr": stderr, "exit_code": exitCode}, nil
		}),
	)

	if err != nil {
		log.Fatal(err)
	}

	if addToolErr := agent.AddTool(bashTool); addToolErr != nil {
		log.Fatal(addToolErr)
	}

	userContent := fmt.Sprintf("## Problem statement\n\n%s\n\n## Repository\n- repo: %s\n- base_commit: %s\n\nFix the issue using the run_bash tool. Work in the repository root.",
		instance.ProblemStatement, instance.Repo, instance.BaseCommit)

	messages := []orla.Message{
		{Role: "system", Content: defaultSystemPrompt},
		{Role: "user", Content: userContent},
	}

	var lastContent string
	for step := range maxSteps {
		resp, err := agent.ExecuteWithMessages(ctx, messages)
		if err != nil {
			log.Fatalf("Step %d execute: %v", step+1, err)
		}
		lastContent = resp.Content

		if len(resp.ToolCalls) == 0 {
			log.Printf("Step %d: model finished (no tool calls)", step+1)
			break
		}

		messages = append(messages, orla.Message{Role: "assistant", Content: resp.Content})
		toolMessages, err := agent.RunToolCallsInResponse(ctx, resp)
		if err != nil {
			log.Fatalf("Step %d run tools: %v", step+1, err)
		}
		for _, m := range toolMessages {
			messages = append(messages, *m)
		}
	}

	if outputPath != "" {
		patch := ""
		if p, ok := patchFromWorkdir(absWorkdir); ok && strings.TrimSpace(p) != "" {
			patch = p
		}
		pred := prediction{
			InstanceID:      instance.InstanceID,
			ModelNameOrPath: "orla-sglang",
			ModelPatch:      patch,
		}
		f, err := os.OpenFile(outputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
		if err != nil {
			log.Printf("Warning: could not write predictions: %v", err)
		} else {
			defer func() {
				if cErr := f.Close(); cErr != nil {
					log.Printf("Warning: close predictions file: %v", cErr)
				}
			}()
			enc := json.NewEncoder(f)
			enc.SetEscapeHTML(false)
			if err := enc.Encode(pred); err != nil {
				log.Printf("Warning: encode prediction: %v", err)
			}
		}
	}

	fmt.Println("--- Final model output ---")
	fmt.Println(lastContent)
}

type swebenchInstance struct {
	InstanceID       string `json:"instance_id"`
	Repo             string `json:"repo"`
	BaseCommit       string `json:"base_commit"`
	ProblemStatement string `json:"problem_statement"`
}

func loadInstance(path string) (*swebenchInstance, error) {
	path = filepath.Clean(path)
	data, err := os.ReadFile(path) // #nosec G304 -- path from -instance flag
	if err != nil {
		return nil, err
	}
	var inst swebenchInstance
	if err := json.Unmarshal(data, &inst); err != nil {
		return nil, err
	}
	return &inst, nil
}

type prediction struct {
	InstanceID      string `json:"instance_id"`
	ModelNameOrPath string `json:"model_name_or_path"`
	ModelPatch      string `json:"model_patch"`
}

// patchFromWorkdir runs "git diff" in workdir and returns the unified diff if the repo has changes.
// Returns ("", false) if git diff fails or is empty (e.g. not a git repo, no edits).
func patchFromWorkdir(workdir string) (string, bool) {
	cmd := exec.CommandContext(context.Background(), "git", "diff")
	cmd.Dir = workdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", false
	}
	return string(out), len(bytes.TrimSpace(out)) > 0
}
