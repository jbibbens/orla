// Package twostagemappingcomplexitysched runs the SWE-bench Lite two-stage mapping
// experiment with complexity-based scheduling: heavy instances are sorted by predicted
// complexity (simplest first) using SJF scheduling.
package twostagemappingcomplexitysched

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
	heavyModelID    = "Qwen/Qwen3-8B"
	lightModelID    = "Qwen/Qwen3-4B-Instruct-2507"
	defaultLightURL = "http://sglang-light:30000/v1"

	routerPromptPrefix = `You are classifying a software engineering task for routing. Judge based on the task description only.

Choose true (light) when the fix is likely: a single file or few lines, a clear bug (typo, wrong constant, off-by-one), config/test/doc tweak, or the issue clearly points to an obvious file or location.

Choose false (heavy) when the fix likely needs: multiple files, unclear root cause, design or API changes, or reasoning across several parts of the codebase.

Apply the criteria above; choose the option that best matches the task. Output prediction: `

	complexityPromptPrefix = `You are estimating the complexity of a software engineering task on a scale of 1-5.

1 = Trivial: single-line fix, obvious typo or constant correction
2 = Simple: one file, small localized change, clear from the error message
3 = Moderate: a few files or functions, some reasoning about call chains
4 = Complex: multiple files, unclear root cause, need to understand module interactions
5 = Very complex: deep architectural issue, cross-cutting changes, significant design reasoning

Judge based on the task description only. Output your estimate as complexity: `
)

type stageJob struct {
	inst          shared.SWEBenchLiteInstance
	absWorkdir    string
	stageName     string
	complexity    int // 1-5 predicted complexity (heavy instances only)
	priority      int // scheduling priority = 6 - complexity (higher = scheduled first)
	queuePosition int // position in the sorted queue (for metrics)
}

// Run loads the dataset, registers light and heavy backends, then:
// 1) Routes all instances (light model) and assigns each to light or heavy.
// 2) Predicts complexity for heavy instances and assigns scheduling priority (SJF: simplest first).
// 3) Sorts heavy queue by priority and runs light/heavy concurrently.
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
	lightStage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
	heavyStage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)
	heavyStage.SetRequestSchedulingPolicy("priority")
	lightStage.SetTemperature(0.7)
	heavyStage.SetTemperature(0.7)
	lightStage.SetMaxTokens(shared.MaxOutputTokens)
	heavyStage.SetMaxTokens(shared.MaxOutputTokens)
	lightStage.SetChatTemplateKwargs(shared.NoThinking)
	heavyStage.SetChatTemplateKwargs(shared.NoThinking)

	lightStage.Client = client
	heavyStage.Client = client
	var mapper orla.StageMapper = orla.NewOneBitStageMapper(client, lightBackend, lightStage, heavyStage)
	scorePredictor := orla.NewScorePredictor(client, lightBackend)

	outFile, err := os.OpenFile(shared.OutputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open predictions file: %w", err)
	}
	defer shared.LogDeferredError(outFile.Close)
	enc := shared.NewPredictionEncoder(outFile)
	var encMu sync.Mutex
	metrics := shared.NewRunMetricsRecorder("two_stage_mapping_complexity_sched")
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

		job := stageJob{inst: inst, absWorkdir: absWorkdir, stageName: stage.Name}
		if stage.Name == "light" {
			log.Printf("instance %s: router => light", inst.InstanceID)
			lightJobs = append(lightJobs, job)
		} else {
			complexityPrompt := complexityPromptPrefix + shared.FormatProblemMessage(inst)
			complexity, cErr := scorePredictor.Predict(ctx, complexityPrompt)
			if cErr != nil {
				log.Printf("instance %s: complexity prediction failed, defaulting to 3: %v", inst.InstanceID, cErr)
				complexity = 3
			}
			job.complexity = complexity
			job.priority = 6 - complexity
			log.Printf("instance %s: router => heavy, complexity=%d, priority=%d", inst.InstanceID, complexity, job.priority)
			heavyJobs = append(heavyJobs, job)
		}
	}

	// Sort heavy jobs by priority descending (simplest first = highest priority first).
	sortByPriority(heavyJobs)
	for i, j := range heavyJobs {
		log.Printf("heavy queue[%d]: %s (complexity=%d, priority=%d)", i, j.inst.InstanceID, j.complexity, j.priority)
	}

	// Phase 2: one worker per backend; light and heavy run concurrently
	runJob := func(job stageJob, stage *orla.Stage) {
		if job.priority > 0 {
			p := job.priority
			stage.SetSchedulingHints(&orla.SchedulingHints{Priority: &p})
		}

		inst := job.inst
		rec := &shared.InstanceRecorder{}
		rec.BeginInstance(inst.InstanceID, job.stageName)
		rec.SetComplexity(job.complexity)
		rec.SetQueuePosition(job.queuePosition)
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
			ModelNameOrPath: "orla-two-stage-complexity-sched",
			ModelPatch:      patch,
		})
		if err != nil {
			log.Printf("warning: encode prediction %s: %v", inst.InstanceID, err)
		}
		encMu.Unlock()
	}

	lightChan := make(chan stageJob, len(lightJobs))
	heavyChan := make(chan stageJob, len(heavyJobs))
	for i, j := range lightJobs {
		j.queuePosition = i
		lightChan <- j
	}
	close(lightChan)
	for i, j := range heavyJobs {
		j.queuePosition = i
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

func sortByPriority(jobs []stageJob) {
	for i := 1; i < len(jobs); i++ {
		key := jobs[i]
		j := i - 1
		for j >= 0 && jobs[j].priority < key.priority {
			jobs[j+1] = jobs[j]
			j--
		}
		jobs[j+1] = key
	}
}
