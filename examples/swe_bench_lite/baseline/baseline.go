// Package baseline runs the SWE-bench Lite baseline: Orla + SGLang, one run_bash tool,
// ReAct loop over all instances from the shared dataset.
package baseline

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	orla "github.com/dorcha-inc/orla/pkg/api"

	"github.com/dorcha-inc/orla/examples/swe_bench_lite/shared"
)

// Run loads the dataset from shared, runs the agent on each instance, and appends predictions to shared.OutputPath.
func Run(ctx context.Context, dataset *shared.SWEBenchLiteDataset) error {
	client := orla.NewOrlaClient(shared.OrlaURL)
	if err := client.Health(ctx); err != nil {
		return err
	}

	backend := orla.NewSGLangBackend("Qwen/Qwen3-8B", shared.SGLangURL)
	if err := client.RegisterBackend(ctx, backend); err != nil {
		return err
	}

	agent := orla.NewAgent(client)
	stage := orla.NewAgentStage("baseline", backend)
	stage.SetTemperature(0)
	stage.SetMaxTokens(shared.MaxOutputTokens)
	stage.SetChatTemplateKwargs(shared.NoThinking)

	var currentWorkdir string
	bashTool, err := shared.NewBashTool(func() string { return currentWorkdir })
	if err != nil {
		return fmt.Errorf("new bash tool: %w", err)
	}
	if err := stage.AddTool(bashTool); err != nil {
		return fmt.Errorf("add bash tool: %w", err)
	}
	agent.SetStage(stage)

	outFile, err := os.OpenFile(shared.OutputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open predictions file: %w", err)
	}

	defer shared.LogDeferredError(outFile.Close)

	enc := shared.NewPredictionEncoder(outFile)
	metrics := shared.NewRunMetricsRecorder("baseline")
	metrics.BeginRun()
	defer func() {
		metrics.EndRun()
		if err := metrics.Write(""); err != nil {
			log.Printf("warning: write metrics: %v", err)
		}
	}()

	for i, inst := range dataset.Instances {
		if i >= shared.MaxIterations {
			break
		}

		absWorkdir, err := shared.PrepareWorkdir(ctx, inst)
		if err != nil {
			return fmt.Errorf("prepare workdir: %w", err)
		}

		currentWorkdir = absWorkdir
		log.Printf("running instance %d/%d: %s", i+1, len(dataset.Instances), inst.InstanceID)
		metrics.BeginInstance(inst.InstanceID)

		messages := shared.PrepareInitialMessages(inst)

		for step := range shared.MaxSteps {
			log.Printf("step %d: executing", step+1)
			metrics.BeginStep(step + 1)

			resp, err := agent.ExecuteWithMessages(ctx, messages)

			if err != nil {
				return fmt.Errorf("step %d execute: %w", step+1, err)
			}

			log.Printf("step %d: response: %s", step+1, resp.Content)

			if len(resp.ToolCalls) == 0 {
				metrics.EndStep(step + 1)
				log.Printf("step %d: model finished", step+1)
				break
			}

			messages = append(messages, orla.Message{Role: "assistant", Content: resp.Content})
			shared.LogBashCommandsFromResponse(resp)
			toolMessages, err := agent.RunToolCallsInResponse(ctx, resp)
			if err != nil {
				return fmt.Errorf("step %d run tools: %w", step+1, err)
			}

			metrics.EndStep(step + 1)
			for _, m := range toolMessages {
				log.Printf("step %d: tool message: %s", step+1, m.Content)
				messages = append(messages, *m)
			}
		}

		metrics.EndInstance()

		patch := ""
		if p, ok := shared.PatchFromWorkdir(absWorkdir); ok && strings.TrimSpace(p) != "" {
			patch = p
		}

		patchPreview := patch
		if len(patchPreview) > shared.MaxToolOutputBytes {
			patchPreview = patch[:shared.MaxToolOutputBytes] + "..."
		}
		log.Printf("patch: %s", patchPreview)

		if err := enc.Encode(shared.Prediction{
			InstanceID:      inst.InstanceID,
			ModelNameOrPath: "orla-sglang",
			ModelPatch:      patch,
		}); err != nil {
			log.Printf("Warning: encode prediction %s: %v", inst.InstanceID, err)
		}
	}

	log.Printf("Done. Predictions written to %s, metrics to %s", shared.OutputPath, shared.MetricsPath)
	return nil
}
