// Package shared provides common types and helpers for SWE-bench Lite experiments.
// Use LoadDataset, EnsureRepo, NewBashTool, and PatchFromWorkdir from baseline and other experiments.
package shared

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

const (
	// DefaultSystemPrompt is the default system message for SWE-bench agent runs.
	DefaultSystemPrompt = `You are a software engineering agent. You have one tool: run_bash. You are already in the repository root with the base commit checked out.

Follow this workflow:
1. FIND: Use grep -rn or find to locate the relevant file(s). Then use cat or sed -n to read the specific lines you need to change.
2. EDIT: Make the change. Prefer a Python one-liner or heredoc over sed for multi-line or complex edits:
   python3 -c "
   import pathlib; p = pathlib.Path('file.py'); t = p.read_text()
   t = t.replace('old_code', 'new_code'); p.write_text(t)"
   sed -i works for simple single-line substitutions, but note: sed -i returns exit code 0 even if the pattern did not match—always verify.
3. VERIFY: Run git diff to confirm your edit. If git diff is empty, your edit did not apply—re-read the file to check the exact text, then retry with corrected arguments.

Rules:
- Call run_bash with the "command" argument for every action. Do not write commands in code blocks.
- Do not run git clone or git checkout.
- Keep edits minimal and targeted.
- The submitted patch is the git diff after your edits.`
	// MaxSteps is the default cap on ReAct steps per instance.
	MaxSteps        = 256
	MaxOutputTokens = 4096

	MaxIterations = 10

	// Fixed paths/URLs for the Docker compose setup.
	OrlaURL     = "http://orla:8081"
	SGLangURL   = "http://sglang:30000/v1"
	DatasetRoot = "/dataset/test"
	WorkdirRoot = "/workdir"
	OutputPath  = "/output/predictions.jsonl"
	// MetricsPath is the default path for run metrics (end-to-end, per-instance, per-step). Override with METRICS_PATH.
	MetricsPath = "/output/metrics.json"

	// MaxToolOutputBytes caps run_bash stdout/stderr so huge outputs don't blow context.
	// 8KB gives the model enough to read meaningful code while staying within context limits.
	MaxToolOutputBytes = 8192
)

// NoThinking is a map of extra kwargs for the chat template renderer to disable thinking.
var NoThinking = map[string]any{"enable_thinking": false}

// SWEBenchLiteInstance is one instance from the dataset (instance_id, repo, base_commit, problem_statement).
type SWEBenchLiteInstance struct {
	InstanceID       string `json:"instance_id"`
	Repo             string `json:"repo"`
	BaseCommit       string `json:"base_commit"`
	ProblemStatement string `json:"problem_statement"`
}

// SWEBenchLiteDataset is the loaded set of instances (e.g. from LoadDataset).
type SWEBenchLiteDataset struct {
	Instances []SWEBenchLiteInstance
}

func LogDeferredError(fn func() error) {
	if err := fn(); err != nil {
		log.Printf("warning: %v", err)
	}
}

// LoadDataset opens the dataset root (fixed path for Docker) with os.OpenRoot and loads all .json instance files.
func LoadDataset() (*SWEBenchLiteDataset, error) {
	root, err := os.OpenRoot(DatasetRoot)
	if err != nil {
		return nil, fmt.Errorf("open dataset root: %w", err)
	}
	defer func() {
		if cErr := root.Close(); cErr != nil {
			log.Printf("Warning: close dataset root: %v", cErr)
		}
	}()

	var paths []string
	err = fs.WalkDir(root.FS(), ".", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			return nil
		}
		if strings.HasSuffix(path, ".json") {
			paths = append(paths, path)
		}
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("walk dataset root: %w", err)
	}
	sort.Strings(paths)

	dataset := &SWEBenchLiteDataset{}
	for _, path := range paths {
		data, err := root.ReadFile(path)
		if err != nil {
			continue
		}
		var inst SWEBenchLiteInstance
		if json.Unmarshal(data, &inst) != nil {
			continue
		}
		dataset.Instances = append(dataset.Instances, inst)
	}
	return dataset, nil
}

// PrepareWorkdir computes the workdir path for the instance, ensures the repo is cloned and at baseCommit, and returns the absolute workdir path.
func PrepareWorkdir(ctx context.Context, inst SWEBenchLiteInstance) (absWorkdir string, err error) {
	workdir := filepath.Join(WorkdirRoot, inst.InstanceID)
	absWorkdir, err = filepath.Abs(workdir)
	if err != nil {
		return "", fmt.Errorf("workdir abs: %w", err)
	}
	if err := EnsureRepo(ctx, absWorkdir, inst.Repo, inst.BaseCommit); err != nil {
		return "", err
	}
	return absWorkdir, nil
}

// FormatProblemMessage returns the user message content for the standard SWE-bench task (problem statement + repo info).
func FormatProblemMessage(inst SWEBenchLiteInstance) string {
	return fmt.Sprintf("## Problem statement\n\n%s\n\n## Repository\n- repo: %s\n- base_commit: %s\n\nFix the issue using the run_bash tool. Work in the repository root.",
		inst.ProblemStatement, inst.Repo, inst.BaseCommit)
}

func PrepareInitialMessages(inst SWEBenchLiteInstance) []orla.Message {
	return []orla.Message{
		{Role: "system", Content: DefaultSystemPrompt},
		{Role: "user", Content: FormatProblemMessage(inst)},
	}
}

// EnsureRepo clones repo into workdir if needed, then fetches and checks out baseCommit.
func EnsureRepo(ctx context.Context, workdir, repo, baseCommit string) error {
	// #nosec G301 -- workdir is under WorkdirRoot, which is under /workdir in Docker
	if err := os.MkdirAll(filepath.Dir(workdir), 0o755); err != nil {
		return fmt.Errorf("mkdir all: %w", err)
	}

	if _, err := os.Stat(filepath.Join(workdir, ".git")); err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("stat: %w", err)
		}

		log.Printf("  Cloning https://github.com/%s.git into %s", repo, workdir)
		clone := exec.CommandContext(ctx, "git", "clone", "https://github.com/"+repo+".git", workdir)
		clone.Stdout = os.Stdout
		clone.Stderr = os.Stderr
		if err := clone.Run(); err != nil {
			return fmt.Errorf("git clone %s: %w", repo, err)
		}
	}

	fetch := exec.CommandContext(ctx, "git", "fetch", "--quiet", "origin", baseCommit)
	fetch.Dir = workdir
	fetch.Stdout = os.Stdout
	fetch.Stderr = os.Stderr

	if err := fetch.Run(); err != nil {
		return fmt.Errorf("git fetch %s: %w", baseCommit, err)
	}

	checkout := exec.CommandContext(ctx, "git", "checkout", "-q", baseCommit)
	checkout.Dir = workdir
	checkout.Stdout = os.Stdout
	checkout.Stderr = os.Stderr
	if err := checkout.Run(); err != nil {
		return fmt.Errorf("git checkout %s: %w", baseCommit, err)
	}

	// Force clean working tree (avoids submitting leftover changes from a previous run).
	reset := exec.CommandContext(ctx, "git", "reset", "--hard", "-q", baseCommit)
	reset.Dir = workdir
	reset.Stdout = os.Stdout
	reset.Stderr = os.Stderr
	if err := reset.Run(); err != nil {
		return fmt.Errorf("git reset --hard %s: %w", baseCommit, err)
	}
	return nil
}

// PatchFromWorkdir runs "git diff" in workdir and returns the unified diff if there are changes.
func PatchFromWorkdir(workdir string) (string, bool) {
	cmd := exec.CommandContext(context.Background(), "git", "diff")
	cmd.Dir = workdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", false
	}
	return string(out), len(bytes.TrimSpace(out)) > 0
}

// Prediction is one line of predictions.jsonl (instance_id, model_name_or_path, model_patch).
type Prediction struct {
	InstanceID      string `json:"instance_id"`
	ModelNameOrPath string `json:"model_name_or_path"`
	ModelPatch      string `json:"model_patch"`
}

// NewPredictionEncoder returns an encoder that writes Prediction JSON lines to w.
func NewPredictionEncoder(w io.Writer) *PredictionEncoder {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return &PredictionEncoder{enc: enc}
}

// PredictionEncoder encodes Prediction values as JSONL.
type PredictionEncoder struct {
	enc *json.Encoder
}

// Encode writes one prediction line.
func (e *PredictionEncoder) Encode(p Prediction) error {
	return e.enc.Encode(p)
}

// truncateForContext truncates s to at most maxBytes and appends a note so context is not blown by huge tool output.
func truncateForContext(s string, maxBytes int) string {
	if len(s) <= maxBytes {
		return s
	}
	suffix := "\n[... output truncated for context ...]"
	keep := max(maxBytes-len(suffix), 0)
	return s[:keep] + suffix
}

// NewBashTool returns a run_bash tool that runs commands in the directory returned by getWorkdir.
// Pass a function that returns the current instance workdir so the tool runs in the right repo.
func NewBashTool(getWorkdir func() string) (*orla.Tool, error) {
	return orla.NewTool(
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
				return orla.ToolSchema{"stdout": "", "stderr": "command must be a string", "exit_code": 1}, nil
			}

			cmdStr = strings.TrimSpace(cmdStr)
			if cmdStr == "" {
				return orla.ToolSchema{"stdout": "", "stderr": "empty command", "exit_code": 1}, nil
			}
			workdir := getWorkdir()
			cmd := exec.CommandContext(ctx, "bash", "-c", cmdStr)
			cmd.Dir = workdir
			var stdoutBuf, stderrBuf bytes.Buffer
			cmd.Stdout = &stdoutBuf
			cmd.Stderr = &stderrBuf
			err := cmd.Run()
			stdout := truncateForContext(stdoutBuf.String(), MaxToolOutputBytes)
			stderr := truncateForContext(stderrBuf.String(), MaxToolOutputBytes)
			exitCode := 0
			if err != nil {
				var exitErr *exec.ExitError
				if errors.As(err, &exitErr) {
					exitCode = exitErr.ExitCode()
				} else {
					stderr = truncateForContext(err.Error(), MaxToolOutputBytes)
					exitCode = 1
				}
			}
			return orla.ToolSchema{"stdout": stdout, "stderr": stderr, "exit_code": exitCode}, nil
		}),
	)
}

// LogBashCommandsFromResponse logs each run_bash command from the response's tool calls for visibility.
func LogBashCommandsFromResponse(response *orla.InferenceResponse) {
	for _, raw := range response.ToolCalls {
		tc, err := orla.NewToolCallFromRawToolCall(raw)
		if err != nil {
			log.Printf("[tool call] error: new tool call from raw tool call: %v", err)
		}
		if tc.Name != "run_bash" {
			log.Printf("[tool call] error: unknown tool: %s", tc.Name)
		}
		cmd, ok := tc.InputArguments["command"].(string)
		if !ok {
			log.Printf("[tool call] error: command not a string")
		}

		if cmd == "" {
			log.Printf("[tool call] error: empty command")
		}

		log.Printf("[tool call] run_bash: %s", cmd)
	}
}
