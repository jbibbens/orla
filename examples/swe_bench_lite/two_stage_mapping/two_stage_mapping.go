// Package twostagemapping runs the SWE-bench Lite stage-mapping experiment:
// for each instance, a router (on the light model) classifies the task as Light or Heavy,
// then the main ReAct loop runs on the light model for Light tasks and the heavy model for Heavy tasks.
package twostagemapping

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	orla "github.com/dorcha-inc/orla/pkg/api"

	"github.com/dorcha-inc/orla/examples/swe_bench_lite/shared"
)

const (
	// Default heavy (8B) and light (4B) model IDs for stage mapping.
	// Use Qwen3 for both so the "qwen" tool-call parser behaves consistently (Qwen2.5 can emit misparsed tool calls).
	heavyModelID = "Qwen/Qwen3-8B"
	lightModelID = "Qwen/Qwen3-4B-Instruct-2507"
	// Default light backend URL when running two SGLang services (e.g. sglang + sglang-light).
	defaultLightURL = "http://sglang-light:30000/v1"
	// Router prompt: model returns prediction true = light, false = heavy. Balanced criteria; no default bias.
	routerPromptPrefix = `You are classifying a software engineering task for routing. Judge based on the task description only.

Choose true (light) when the fix is likely: a single file or few lines, a clear bug (typo, wrong constant, off-by-one), config/test/doc tweak, or the issue clearly points to an obvious file or location.

Choose false (heavy) when the fix likely needs: multiple files, unclear root cause, design or API changes, or reasoning across several parts of the codebase.

Apply the criteria above; choose the option that best matches the task. Output prediction: `
)

// Run loads the dataset, registers light and heavy backends, and for each instance:
// 1) runs the router (light model) to get Light or Heavy,
// 2) runs the ReAct agent loop on the chosen stage (light or heavy model),
// 3) appends the prediction to shared.OutputPath.
func Run(ctx context.Context, dataset *shared.SWEBenchLiteDataset) error {
	client := orla.NewOrlaClient(shared.OrlaURL)
	if err := client.Health(ctx); err != nil {
		return err
	}

	heavyURL := os.Getenv("SGLANG_HEAVY_URL")
	if heavyURL == "" {
		heavyURL = shared.SGLangURL
	}
	lightURL := os.Getenv("SGLANG_LIGHT_URL")
	if lightURL == "" {
		lightURL = defaultLightURL
	}

	heavyBackend := orla.NewSGLangBackend(heavyModelID, heavyURL)
	if err := client.RegisterBackend(ctx, heavyBackend); err != nil {
		return fmt.Errorf("register heavy backend: %w", err)
	}

	lightBackend := orla.NewSGLangBackend(lightModelID, lightURL)
	if err := client.RegisterBackend(ctx, lightBackend); err != nil {
		return fmt.Errorf("register light backend: %w", err)
	}

	lightStage := orla.NewAgentStage("light", lightBackend)
	heavyStage := orla.NewAgentStage("heavy", heavyBackend)
	lightStage.SetTemperature(0.7)
	heavyStage.SetTemperature(0.7)
	lightStage.SetMaxTokens(shared.MaxOutputTokens)
	heavyStage.SetMaxTokens(shared.MaxOutputTokens)
	lightStage.SetChatTemplateKwargs(shared.NoThinking)
	heavyStage.SetChatTemplateKwargs(shared.NoThinking)

	var currentWorkdir string
	bashTool, bashToolErr := shared.NewBashTool(func() string { return currentWorkdir })
	if bashToolErr != nil {
		return fmt.Errorf("new bash tool: %w", bashToolErr)
	}

	if err := lightStage.AddTool(bashTool); err != nil {
		return fmt.Errorf("add bash tool to light stage: %w", err)
	}
	if err := heavyStage.AddTool(bashTool); err != nil {
		return fmt.Errorf("add bash tool to heavy stage: %w", err)
	}

	// Router runs on the light model (one forward per instance); ReAct loop uses light or heavy per instance.
	mapper := orla.NewOneBitStageMapper(client, lightBackend, lightStage, heavyStage)
	agent := orla.NewAgent(client)

	outFile, err := os.OpenFile(shared.OutputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open predictions file: %w", err)
	}
	defer shared.LogDeferredError(outFile.Close)
	enc := shared.NewPredictionEncoder(outFile)
	metrics := shared.NewRunMetricsRecorder("two_stage_mapping")
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

		routerPrompt := routerPromptPrefix + shared.FormatProblemMessage(inst)
		stage, err := mapper.MapStage(ctx, routerPrompt)
		if err != nil {
			return fmt.Errorf("instance %s map stage: %w", inst.InstanceID, err)
		}

		log.Printf("instance %s: router => %s (prompt_len=%d)", inst.InstanceID, stage.Name, len(routerPrompt))
		metrics.SetMappedStage(stage.Name)
		agent.SetStage(stage)

		messages := shared.PrepareInitialMessages(inst)
		if err := shared.RunAgentLoop(ctx, agent, messages, metrics); err != nil {
			return fmt.Errorf("instance %s: %w", inst.InstanceID, err)
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
			ModelNameOrPath: "orla-two-stage",
			ModelPatch:      patch,
		}); err != nil {
			return fmt.Errorf("failed to encode prediction %s: %w", inst.InstanceID, err)
		}
	}

	log.Printf("Done. Predictions written to %s, metrics to %s", shared.OutputPath, shared.MetricsPath)
	return nil
}
