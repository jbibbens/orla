// Package shared provides common types and helpers for SWE-bench Lite experiments.
// Use LoadDataset, EnsureRepo, RunAgentLoop, and PatchFromWorkdir from baseline and other experiments.
package shared

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"time"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

const (
	// DefaultSystemPrompt follows the mini-swe-agent pattern: text-based actions with THOUGHT + code block.
	// Uses a unique code fence tag (orla_bash) to avoid collisions with code in model reasoning.
	DefaultSystemPrompt = `You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).
Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format>.

<format>
THOUGHT: Your reasoning and analysis here. Explain why you want to perform the action.

` + "```" + `orla_bash
your_command_here
` + "```" + `
</format>

Failure to follow these rules will cause your response to be rejected.

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Important Rules

- Every response must contain exactly one bash code block
- Directory or environment variable changes are not persistent. Every action is executed in a new subshell. However, you can prefix any action with ` + "`" + `cd /path/to/dir && ...` + "`" + ` or write/load environment variables from files
- Do not run git clone or git checkout
- Keep edits minimal and targeted — modify only source files, not tests or config

## Useful command examples

### View file content with line numbers:
` + "```" + `orla_bash
nl -ba filename.py | sed -n '10,20p'
` + "```" + `

### Edit files with sed:
` + "```" + `orla_bash
sed -i 's/old_string/new_string/g' filename.py
` + "```" + `

### Edit with python3 (for complex changes):
` + "```" + `orla_bash
python3 -c "
import pathlib; p = pathlib.Path('file.py'); t = p.read_text()
t = t.replace('old_code', 'new_code'); p.write_text(t)"
` + "```" + `

### Create a new file:
` + "```" + `orla_bash
cat <<'EOF' > newfile.py
import numpy as np
print("hello")
EOF
` + "```" + ``

	// MaxSteps caps ReAct steps per instance. mini-swe-agent uses 250; we use 100 as a practical limit for local models.
	MaxSteps        = 100
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

	// MaxToolOutputBytes caps command output sent to the model. Outputs longer than this
	// are shown as first-half + last-half with an elision message in between.
	MaxToolOutputBytes = 10000

	// CommandTimeout is the per-command timeout (matches mini-swe-agent's 60s).
	CommandTimeout = 60
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
	return fmt.Sprintf("Please solve this issue:\n\n%s\n\nRepository: %s (base commit: %s)\n\nYou can execute bash commands to explore, edit files, and verify your fix. The submitted patch is the git diff after your edits.",
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

// truncateForLog truncates s to at most maxLen for log messages.
func truncateForLog(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// truncateForContext caps command output for the model. For outputs longer than MaxToolOutputBytes,
// shows the first and last portions with an explicit message asking the model to be more selective.
func truncateForContext(s string) string {
	if len(s) <= MaxToolOutputBytes {
		return s
	}
	half := MaxToolOutputBytes / 2
	elided := len(s) - MaxToolOutputBytes
	return s[:half] +
		fmt.Sprintf("\n\n[%d characters elided — output too long. Use more selective commands (head, tail, sed -n, grep) to view smaller sections.]\n\n", elided) +
		s[len(s)-half:]
}

// bashBlockRe matches ```orla_bash ... ``` code blocks in model output.
// Falls back to ```bash or plain ``` blocks if the model doesn't use the custom tag.
var bashBlockRe = regexp.MustCompile("(?s)```(?:orla_bash|bash)?\\s*\\n(.+?)\\n```")

// ExtractBashCommand parses the first ```bash ... ``` block from the model's text response.
func ExtractBashCommand(content string) string {
	m := bashBlockRe.FindStringSubmatch(content)
	if len(m) < 2 {
		return ""
	}
	return strings.TrimSpace(m[1])
}

// bashEnv contains environment variable overrides to prevent pagers from hanging and progress bars from flooding context.
var bashEnv = []string{
	"PAGER=cat",
	"MANPAGER=cat",
	"LESS=-R",
	"PIP_PROGRESS_BAR=off",
	"TQDM_DISABLE=1",
}

// RunBash executes a bash command in workdir with a timeout and returns a formatted observation string.
func RunBash(ctx context.Context, workdir, cmdStr string) string {
	cmdCtx, cancel := context.WithTimeout(ctx, CommandTimeout*time.Second)
	defer cancel()
	cmd := exec.CommandContext(cmdCtx, "bash", "-c", cmdStr)
	cmd.Dir = workdir
	cmd.Env = append(os.Environ(), bashEnv...)
	var outBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &outBuf
	err := cmd.Run()
	output := truncateForContext(outBuf.String())
	exitCode := 0
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			exitCode = exitErr.ExitCode()
		} else {
			if cmdCtx.Err() == context.DeadlineExceeded {
				output += fmt.Sprintf("\n[command timed out after %ds]", CommandTimeout)
			}
			exitCode = 1
		}
	}
	return fmt.Sprintf("<returncode>%d</returncode>\n<output>\n%s\n</output>", exitCode, strings.TrimRight(output, "\n"))
}

const (
	// maxFormatRetries is how many consecutive format errors we tolerate before stopping.
	maxFormatRetries = 3

	formatErrorMsg = `Format error: no bash code block found in your response.

Every response MUST include exactly ONE bash code block:

<format>
THOUGHT: Your reasoning here.

` + "```" + `orla_bash
your_command_here
` + "```" + `
</format>

Please try again with the correct format.`
)

// RunAgentLoop runs the text-based ReAct loop (no tool calling).
// The model outputs THOUGHT + ` + "```" + `orla_bash ... ` + "```" + `, we parse and execute the command,
// then feed the output back as a user message.
func RunAgentLoop(ctx context.Context, agent *orla.Agent, messages []orla.Message, metrics *RunMetricsRecorder, getWorkdir func() string) error {
	formatRetries := 0
	for step := range MaxSteps {
		log.Printf("step %d: executing", step+1)
		metrics.BeginStep(step + 1)
		resp, err := agent.ExecuteWithMessages(ctx, messages)
		if err != nil {
			return fmt.Errorf("step %d execute: %w", step+1, err)
		}
		content := resp.Content
		if resp.Metrics != nil {
			metrics.RecordStepTokens(resp.Metrics.PromptTokens, resp.Metrics.CompletionTokens)
		}
		log.Printf("step %d: response: %s", step+1, truncateForLog(content, 500))

		cmdStr := ExtractBashCommand(content)
		if cmdStr == "" {
			formatRetries++
			if formatRetries >= maxFormatRetries {
				metrics.EndStep(step + 1)
				log.Printf("step %d: %d consecutive format errors, stopping", step+1, formatRetries)
				return nil
			}
			log.Printf("step %d: no bash block found, sending format error (%d/%d)", step+1, formatRetries, maxFormatRetries)
			messages = append(messages, orla.Message{Role: "assistant", Content: content})
			messages = append(messages, orla.Message{Role: "user", Content: formatErrorMsg})
			metrics.EndStep(step + 1)
			continue
		}
		formatRetries = 0

		log.Printf("step %d: [bash] %s", step+1, cmdStr)
		messages = append(messages, orla.Message{Role: "assistant", Content: content})

		observation := RunBash(ctx, getWorkdir(), cmdStr)
		log.Printf("step %d: observation: %s", step+1, truncateForLog(observation, 300))
		messages = append(messages, orla.Message{Role: "user", Content: observation})
		metrics.EndStep(step + 1)
	}
	return nil
}
