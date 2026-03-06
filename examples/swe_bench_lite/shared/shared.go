// Package shared provides common types and helpers for SWE-bench Lite experiments.
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
)

const (
	MaxOutputTokens = 4096

	MaxIterations = 300

	OrlaURL     = "http://orla:8081"
	SGLangURL   = "http://sglang:30000/v1"
	DatasetRoot = "/dataset/test"
	WorkdirRoot = "/workdir"
	OutputPath  = "/output/predictions.jsonl"
	MetricsPath = "/output/metrics.json"

	MaxToolOutputBytes = 10000
)

var (
	VLLMHeavyURL          = os.Getenv("VLLM_HEAVY_URL")
	VLLMLightURL          = os.Getenv("VLLM_LIGHT_URL")
	BackendProviderIsVLLM = os.Getenv("BACKEND_PROVIDER") == "vllm"
)

// SingleShotSystemPrompt instructs the model to produce a unified diff patch from the
// problem statement and oracle-provided source files.
const SingleShotSystemPrompt = `You are an expert software engineer. You are given a bug report and the relevant source files from the repository. Your task is to produce a minimal unified diff patch that fixes the described issue.

Rules:
- Output ONLY a unified diff (the format produced by "git diff").
- Do not include any explanation, commentary, or markdown fences — just the raw diff.
- Only modify the minimum lines necessary to fix the issue.
- Do not modify tests or configuration files.`

// SWEBenchLiteInstance is one instance from the dataset.
type SWEBenchLiteInstance struct {
	InstanceID       string `json:"instance_id"`
	Repo             string `json:"repo"`
	BaseCommit       string `json:"base_commit"`
	ProblemStatement string `json:"problem_statement"`
	Patch            string `json:"patch,omitempty"`
}

// SWEBenchLiteDataset is the loaded set of instances.
type SWEBenchLiteDataset struct {
	Instances []SWEBenchLiteInstance
}

func LogDeferredError(fn func() error) {
	if err := fn(); err != nil {
		log.Printf("warning: %v", err)
	}
}

// LoadDataset opens the dataset root and loads all .json instance files.
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

// PrepareWorkdir ensures the repo is cloned and at the base commit, returns the absolute workdir path.
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

// EnsureRepo clones repo into workdir if needed, then fetches and checks out baseCommit.
func EnsureRepo(ctx context.Context, workdir, repo, baseCommit string) error {
	// #nosec G301 -- workdir is under WorkdirRoot
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

	reset := exec.CommandContext(ctx, "git", "reset", "--hard", "-q", baseCommit)
	reset.Dir = workdir
	reset.Stdout = os.Stdout
	reset.Stderr = os.Stderr
	if err := reset.Run(); err != nil {
		return fmt.Errorf("git reset --hard %s: %w", baseCommit, err)
	}
	return nil
}

// PatchFromWorkdir runs "git diff" in workdir and returns the unified diff.
func PatchFromWorkdir(workdir string) (string, bool) {
	cmd := exec.CommandContext(context.Background(), "git", "diff")
	cmd.Dir = workdir
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", false
	}
	return string(out), len(bytes.TrimSpace(out)) > 0
}

// diffFileRe matches file paths in unified diff headers: "--- a/path" or "+++ b/path".
var diffFileRe = regexp.MustCompile(`(?m)^(?:---|\+\+\+) [ab]/(.+)$`)

// OracleFilePaths parses a unified diff and returns the deduplicated list of modified file paths.
func OracleFilePaths(patch string) []string {
	matches := diffFileRe.FindAllStringSubmatch(patch, -1)
	seen := map[string]bool{}
	var paths []string
	for _, m := range matches {
		p := m[1]
		if p == "/dev/null" || seen[p] {
			continue
		}
		seen[p] = true
		paths = append(paths, p)
	}
	sort.Strings(paths)
	return paths
}

// GatherOracleContext reads the specified files from workdir and concatenates them
// with file-path headers. Files that cannot be read are skipped with a note.
func GatherOracleContext(workdir string, filePaths []string) string {
	var b strings.Builder
	for _, fp := range filePaths {
		abs := filepath.Join(workdir, fp)
		data, err := os.ReadFile(abs) // #nosec G304 -- paths from gold patch, workdir is controlled
		if err != nil {
			fmt.Fprintf(&b, "### %s\n[file not found at base commit]\n\n", fp)
			continue
		}
		content := string(data)
		if len(content) > MaxToolOutputBytes {
			content = content[:MaxToolOutputBytes] + "\n[truncated]\n"
		}
		fmt.Fprintf(&b, "### %s\n```\n%s\n```\n\n", fp, content)
	}
	return b.String()
}

// BuildSingleShotPrompt constructs the user message for a single-shot patch generation request.
func BuildSingleShotPrompt(inst SWEBenchLiteInstance, oracleContext string) string {
	return fmt.Sprintf("## Bug Report\n\n%s\n\n## Repository\n\n%s (commit %s)\n\n## Relevant Source Files\n\n%s",
		inst.ProblemStatement, inst.Repo, inst.BaseCommit, oracleContext)
}

// Prediction is one line of predictions.jsonl.
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

// PredictionEncoder encodes Prediction values as JSONL and tracks the number written.
type PredictionEncoder struct {
	enc   *json.Encoder
	count int
}

// Encode writes one prediction line and increments the count.
func (e *PredictionEncoder) Encode(p Prediction) error {
	if err := e.enc.Encode(p); err != nil {
		return err
	}
	e.count++
	return nil
}

// Count returns how many predictions have been written so far.
func (e *PredictionEncoder) Count() int {
	return e.count
}

// ModelName returns the model_name_or_path string for predictions based on mode and backend.
func ModelName(mode string) string {
	suffix := ""
	if BackendProviderIsVLLM {
		suffix = "-vllm"
	}
	return "orla-" + mode + suffix
}
