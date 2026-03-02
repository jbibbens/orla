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
	"sync"

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

// We run one job at a time per backend (light and heavy). The concurrency win is that
// light and heavy backends are used at the same time, not that we parallelize within one backend.

type stageJob struct {
	inst       shared.SWEBenchLiteInstance
	absWorkdir string
	stageName  string
}

// Run loads the dataset, registers light and heavy backends, then:
// 1) Routes all instances (light model) and assigns each to light or heavy.
// 2) Runs light-routed instances in parallel on the light backend and heavy-routed on the heavy backend concurrently.
// 3) Appends predictions to shared.OutputPath (order may differ from dataset).
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

	lightStage := orla.NewStage("light", lightBackend)
	heavyStage := orla.NewStage("heavy", heavyBackend)
	lightStage.SetTemperature(0.7)
	heavyStage.SetTemperature(0.7)
	lightStage.SetMaxTokens(shared.MaxOutputTokens)
	heavyStage.SetMaxTokens(shared.MaxOutputTokens)
	lightStage.SetChatTemplateKwargs(shared.NoThinking)
	heavyStage.SetChatTemplateKwargs(shared.NoThinking)

	lightStage.Client = client
	heavyStage.Client = client
	var mapper orla.StageMapper = orla.NewOneBitStageMapper(client, lightBackend, lightStage, heavyStage)

	outFile, err := os.OpenFile(shared.OutputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open predictions file: %w", err)
	}
	defer shared.LogDeferredError(outFile.Close)
	enc := shared.NewPredictionEncoder(outFile)
	var encMu sync.Mutex
	metrics := shared.NewRunMetricsRecorder("two_stage_mapping")
	metrics.BeginRun()
	defer func() {
		metrics.EndRun()
		if err := metrics.Write(""); err != nil {
			log.Printf("warning: write metrics: %v", err)
		}
	}()

	// Phase 1: route all instances and prepare workdirs
	var lightJobs, heavyJobs []stageJob
	for i, inst := range dataset.Instances {
		if i >= shared.MaxIterations {
			break
		}
		absWorkdir, err := shared.PrepareWorkdir(ctx, inst)
		if err != nil {
			return fmt.Errorf("prepare workdir %s: %w", inst.InstanceID, err)
		}
		routerPrompt := routerPromptPrefix + shared.FormatProblemMessage(inst)
		stage, err := mapper.MapStage(ctx, routerPrompt)
		if err != nil {
			return fmt.Errorf("instance %s map stage: %w", inst.InstanceID, err)
		}
		log.Printf("instance %s: router => %s (prompt_len=%d)", inst.InstanceID, stage.Name, len(routerPrompt))
		job := stageJob{inst: inst, absWorkdir: absWorkdir, stageName: stage.Name}
		if stage.Name == "light" {
			lightJobs = append(lightJobs, job)
		} else {
			heavyJobs = append(heavyJobs, job)
		}
	}

	// Phase 2: one worker per backend; light and heavy run concurrently
	runJob := func(job stageJob, stage *orla.Stage) {
		inst := job.inst
		rec := &shared.InstanceRecorder{}
		rec.BeginInstance(inst.InstanceID, job.stageName)
		messages := shared.PrepareInitialMessages(inst)
		workdir := job.absWorkdir
		if err := shared.RunAgentLoop(ctx, stage, messages, rec, func() string { return workdir }); err != nil {
			log.Printf("instance %s: %v", inst.InstanceID, err)
		}
		metrics.AddInstance(rec.EndInstance())

		patch := ""
		if p, ok := shared.PatchFromWorkdir(workdir); ok && strings.TrimSpace(p) != "" {
			patch = p
		}
		if len(patch) > shared.MaxToolOutputBytes {
			log.Printf("patch: %s...", patch[:shared.MaxToolOutputBytes])
		} else {
			log.Printf("patch: %s", patch)
		}
		encMu.Lock()
		err = enc.Encode(shared.Prediction{
			InstanceID:      inst.InstanceID,
			ModelNameOrPath: "orla-two-stage",
			ModelPatch:      patch,
		})
		if err != nil {
			log.Printf("warning: encode prediction %s: %v", inst.InstanceID, err)
		}
		encMu.Unlock()
	}

	lightChan := make(chan stageJob, len(lightJobs))
	heavyChan := make(chan stageJob, len(heavyJobs))
	for _, j := range lightJobs {
		lightChan <- j
	}
	close(lightChan)
	for _, j := range heavyJobs {
		heavyChan <- j
	}
	close(heavyChan)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for j := range lightChan {
			runJob(j, lightStage)
		}
	}()
	go func() {
		defer wg.Done()
		for j := range heavyChan {
			runJob(j, heavyStage)
		}
	}()
	wg.Wait()

	log.Printf("Done. Predictions written to %s, metrics to %s", shared.OutputPath, shared.MetricsPath)
	return nil
}
