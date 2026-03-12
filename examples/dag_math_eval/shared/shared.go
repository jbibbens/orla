// Package shared provides common types and helpers for DAG-Math memory evaluation experiments.
package shared

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
)

const (
	MaxOutputTokens = 256

	OrlaURL     = "http://orla:8081"
	SGLangURL   = "http://sglang:30000/v1"
	DatasetRoot = "/dataset/test"
	OutputPath  = "/output/results.json"
	MetricsPath = "/output/metrics.json"
)

var MaxInstances = maxInstancesFromEnv()

func maxInstancesFromEnv() int {
	if s := os.Getenv("MAX_INSTANCES"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return 2894
}

// SystemPrompt instructs the model to continue the mathematical reasoning chain.
const SystemPrompt = `You are an expert mathematician solving a problem step by step. You are given a reasoning step from a mathematical proof or derivation. The step includes the logical inference (edge) and the conclusion (node). Your task is to verify and elaborate on this reasoning step, explaining why the conclusion follows from the premises.

Rules:
- Be concise but rigorous.
- Show your work clearly.
- If the step depends on previous conclusions, assume they are correct.`

// DAGMathStep is one step (node) in a DAG-Math solution.
type DAGMathStep struct {
	StepID               int    `json:"step_id"`
	Edge                 string `json:"edge"`
	DirectDependentSteps []int  `json:"direct_dependent_steps"`
	Node                 string `json:"node"`
}

// DAGMathProblem is one problem from the DAG-MATH-Formatted-CoT dataset.
type DAGMathProblem struct {
	ProblemID   int          `json:"problem_id"`
	ProblemText string       `json:"problem_text"`
	FinalAnswer string       `json:"final_answer"`
	Difficulty  float64      `json:"difficulty"`
	Domain      []string     `json:"domain"`
	Steps       []DAGMathStep `json:"steps"`
}

// DAGMathDataset is the loaded set of problems.
type DAGMathDataset struct {
	Problems []DAGMathProblem
}

// LoadDataset opens the dataset root and loads all .json problem files.
func LoadDataset() (*DAGMathDataset, error) {
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

	dataset := &DAGMathDataset{}
	for _, path := range paths {
		data, err := root.ReadFile(path)
		if err != nil {
			continue
		}
		var problem DAGMathProblem
		if json.Unmarshal(data, &problem) != nil {
			continue
		}
		dataset.Problems = append(dataset.Problems, problem)
	}
	sort.Slice(dataset.Problems, func(i, j int) bool {
		return dataset.Problems[i].ProblemID < dataset.Problems[j].ProblemID
	})
	return dataset, nil
}

// BuildStagePrompt constructs the user message for a single DAG-Math step.
// Order is chosen for KV cache reuse: problem first (shared), then previous steps
// (shared among siblings), then current step (unique per stage).
func BuildStagePrompt(problem DAGMathProblem, step DAGMathStep, depResults map[int]string) string {
	var b strings.Builder
	fmt.Fprintf(&b, "## Problem\n\n%s\n\n", problem.ProblemText)

	if len(step.DirectDependentSteps) > 0 && len(depResults) > 0 {
		b.WriteString("## Previous Steps Referenced\n\n")
		for _, depID := range step.DirectDependentSteps {
			if result, ok := depResults[depID]; ok {
				fmt.Fprintf(&b, "**Step %d result:** %s\n\n", depID, result)
			}
		}
	}

	fmt.Fprintf(&b, "## Current Step (step %d)\n\n", step.StepID)
	fmt.Fprintf(&b, "**Inference:** %s\n\n", step.Edge)
	fmt.Fprintf(&b, "**Conclusion:** %s\n\n", step.Node)
	b.WriteString("Verify and elaborate on this reasoning step.")
	return b.String()
}

func LogDeferredError(fn func() error) {
	if err := fn(); err != nil {
		log.Printf("warning: %v", err)
	}
}
