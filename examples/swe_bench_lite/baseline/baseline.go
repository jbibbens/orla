// Package baseline runs the SWE-bench Lite baseline: Orla + SGLang, one run_bash tool,
// ReAct loop over all instances in the dataset.
package baseline

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

const defaultSystemPrompt = `You are a software engineering agent. You have access to a bash shell to run commands.
Your task is to fix the issue described in the problem statement. Edit the repository files using the run_bash tool (e.g. run an editor, or use sed/cat to change files). Work in the repository root.
The submitted patch is the git diff of the repository after your edits; Use the run_bash tool to execute commands. Each tool call runs a single bash command.`

const maxSteps = 256

// Config holds paths and URLs for the baseline run.
type Config struct {
	OrlaURL     string
	SGLangURL   string
	DatasetRoot string
	WorkdirRoot string
	OutputPath  string
}

// DefaultConfig returns config from env or defaults suitable for Docker.
func DefaultConfig() Config {
	return Config{
		OrlaURL:     getEnv("ORLA_URL", "http://orla:8081"),
		SGLangURL:   getEnv("SGLANG_URL", "http://sglang:30000/v1"),
		DatasetRoot: getEnv("DATASET_ROOT", "/dataset"),
		WorkdirRoot: getEnv("WORKDIR_ROOT", "/workdir"),
		OutputPath:  getEnv("OUTPUT_PATH", "/output/predictions.jsonl"),
	}
}

func getEnv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// Run executes the full SWE-bench Lite baseline: discover instances, run agent on each, write predictions.
func Run(ctx context.Context, cfg Config) error {
	paths, err := discoverInstancePaths(cfg.DatasetRoot)
	if err != nil {
		return fmt.Errorf("discover instances: %w", err)
	}
	if len(paths) == 0 {
		return fmt.Errorf("no instance JSON files found under %s", cfg.DatasetRoot)
	}
	log.Printf("Running %d instances from %s", len(paths), cfg.DatasetRoot)

	client := orla.NewOrlaClient(cfg.OrlaURL)
	if err := client.Health(ctx); err != nil {
		return fmt.Errorf("orla health check: %w", err)
	}

	backend := orla.NewSGLangBackend("Qwen/Qwen3-8B", cfg.SGLangURL)
	if err := client.RegisterBackend(ctx, backend); err != nil {
		return fmt.Errorf("register backend: %w", err)
	}

	agent := orla.NewAgent(client, backend)
	agent.SetMaxTokens(4096)

	var currentWorkdir string
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
			cmdStr, _ := input["command"].(string)
			cmdStr = strings.TrimSpace(cmdStr)
			if cmdStr == "" {
				return orla.ToolSchema{"stdout": "", "stderr": "empty command", "exit_code": 1.0}, nil
			}
			cmd := exec.CommandContext(ctx, "bash", "-c", cmdStr)
			cmd.Dir = currentWorkdir
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
		return err
	}
	if err := agent.AddTool(bashTool); err != nil {
		return err
	}

	outFile, err := os.OpenFile(cfg.OutputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open output file: %w", err)
	}
	defer func() {
		if cErr := outFile.Close(); cErr != nil {
			log.Printf("Warning: close predictions file: %v", cErr)
		}
	}()
	enc := json.NewEncoder(outFile)
	enc.SetEscapeHTML(false)

	for i, instancePath := range paths {
		instance, err := loadInstance(instancePath)
		if err != nil {
			log.Printf("Skip %s: load instance: %v", instancePath, err)
			continue
		}

		workdir := filepath.Join(cfg.WorkdirRoot, instance.InstanceID)
		absWorkdir, err := filepath.Abs(workdir)
		if err != nil {
			log.Printf("Skip %s: workdir abs: %v", instance.InstanceID, err)
			continue
		}

		if err := ensureRepo(ctx, absWorkdir, instance.Repo, instance.BaseCommit); err != nil {
			log.Printf("Skip %s: ensure repo: %v", instance.InstanceID, err)
			continue
		}

		currentWorkdir = absWorkdir
		log.Printf("[%d/%d] %s", i+1, len(paths), instance.InstanceID)

		userContent := fmt.Sprintf("## Problem statement\n\n%s\n\n## Repository\n- repo: %s\n- base_commit: %s\n\nFix the issue using the run_bash tool. Work in the repository root.",
			instance.ProblemStatement, instance.Repo, instance.BaseCommit)

		messages := []orla.Message{
			{Role: "system", Content: defaultSystemPrompt},
			{Role: "user", Content: userContent},
		}

		for step := range maxSteps {
			resp, err := agent.ExecuteWithMessages(ctx, messages)
			if err != nil {
				log.Printf("Step %d execute: %v", step+1, err)
				break
			}

			if len(resp.ToolCalls) == 0 {
				log.Printf("  Step %d: model finished", step+1)
				break
			}

			messages = append(messages, orla.Message{Role: "assistant", Content: resp.Content})
			toolMessages, err := agent.RunToolCallsInResponse(ctx, resp)
			if err != nil {
				log.Printf("Step %d run tools: %v", step+1, err)
				break
			}
			for _, m := range toolMessages {
				messages = append(messages, *m)
			}
		}

		patch := ""
		if p, ok := patchFromWorkdir(absWorkdir); ok && strings.TrimSpace(p) != "" {
			patch = p
		}
		pred := prediction{
			InstanceID:      instance.InstanceID,
			ModelNameOrPath: "orla-sglang",
			ModelPatch:      patch,
		}
		if err := enc.Encode(pred); err != nil {
			log.Printf("Warning: encode prediction %s: %v", instance.InstanceID, err)
		}
	}

	log.Printf("Done. Predictions written to %s", cfg.OutputPath)
	return nil
}

func discoverInstancePaths(root string) ([]string, error) {
	var paths []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if info.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".json") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(paths)
	return paths, nil
}

func ensureRepo(ctx context.Context, workdir, repo, baseCommit string) error {
	if err := os.MkdirAll(filepath.Dir(workdir), 0o755); err != nil {
		return err
	}
	if _, err := os.Stat(filepath.Join(workdir, ".git")); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		log.Printf("  Cloning https://github.com/%s.git into %s", repo, workdir)
		clone := exec.CommandContext(ctx, "git", "clone", "https://github.com/"+repo+".git", workdir)
		clone.Stdout = os.Stdout
		clone.Stderr = os.Stderr
		if err := clone.Run(); err != nil {
			return fmt.Errorf("git clone: %w", err)
		}
	}
	fetch := exec.CommandContext(ctx, "git", "fetch", "origin", baseCommit)
	fetch.Dir = workdir
	fetch.Stdout = os.Stdout
	fetch.Stderr = os.Stderr
	_ = fetch.Run()
	checkout := exec.CommandContext(ctx, "git", "checkout", baseCommit)
	checkout.Dir = workdir
	checkout.Stdout = os.Stdout
	checkout.Stderr = os.Stderr
	if err := checkout.Run(); err != nil {
		return fmt.Errorf("git checkout %s: %w", baseCommit, err)
	}
	return nil
}

type swebenchInstance struct {
	InstanceID       string `json:"instance_id"`
	Repo             string `json:"repo"`
	BaseCommit       string `json:"base_commit"`
	ProblemStatement string `json:"problem_statement"`
}

func loadInstance(path string) (*swebenchInstance, error) {
	path = filepath.Clean(path)
	data, err := os.ReadFile(path) // #nosec G304 -- path from dataset dir inside container
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

func patchFromWorkdir(workdir string) (string, bool) {
	cmd := exec.CommandContext(context.Background(), "git", "diff")
	cmd.Dir = workdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", false
	}
	return string(out), len(bytes.TrimSpace(out)) > 0
}
