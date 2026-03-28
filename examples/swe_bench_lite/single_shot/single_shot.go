// Package singleshot runs the SWE-bench Lite single-shot experiment: each instance gets
// one inference call with oracle-provided source files. All instances are submitted
// concurrently to stress Orla's scheduling.
//
// Three modes (set via EXPERIMENT_MODE):
//   - baseline:      all instances → single heavy model, FCFS
//   - stage_mapping: OneBitStageMapper routes to light/heavy, FCFS per backend
//   - sjf:           same routing + priority scheduling on heavy (shorter prompts first)
package singleshot

import (
	"context"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"sync"
	"time"

	orla "github.com/harvard-cns/orla/pkg/api"
	"golang.org/x/sync/errgroup"

	"github.com/harvard-cns/orla/examples/swe_bench_lite/shared"
)

const (
	ModeBaseline     = "baseline"
	ModeStageMapping = "stage_mapping"
	ModeSJF          = "sjf"

	// OrlaMaxConcurrency caps concurrent requests dispatched to the backend.
	// Matches vLLM's default max_num_seqs (128); going higher just queues
	// inside the backend without improving throughput.
	OrlaMaxConcurrency = 128

	heavyModelID    = "Qwen/Qwen3-8B"
	lightModelID    = "Qwen/Qwen3-4B-Instruct-2507"
	defaultLightURL = "http://sglang-light:30000/v1"

	routerPromptPrefix = `You are classifying a software engineering task for routing. Judge based on the task description only.

Choose true (light) when the fix is likely: a single file or few lines, a clear bug (typo, wrong constant, off-by-one), config/test/doc tweak, or the issue clearly points to an obvious file or location.

Choose false (heavy) when the fix likely needs: multiple files, unclear root cause, design or API changes, or reasoning across several parts of the codebase.

Apply the criteria above; choose the option that best matches the task. Output prediction: `
)

func prepParallelismFromEnv() int {
	if s := os.Getenv("PREP_PARALLELISM"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return 16
}

type instanceJob struct {
	inst          shared.SWEBenchLiteInstance
	prompt        string
	promptLen     int
	stage         *orla.Stage
	stageName     string
	priority      int
	queuePosition int
}

// Run executes the single-shot experiment in the given mode.
func Run(ctx context.Context, dataset *shared.SWEBenchLiteDataset, mode string) error {
	client := orla.NewOrlaClient(shared.OrlaURL)
	if err := client.Health(ctx); err != nil {
		return err
	}

	heavyURL, lightURL := resolveURLs()

	var heavyBackend, lightBackend *orla.LLMBackend
	if shared.BackendProviderIsVLLM {
		heavyBackend = orla.NewVLLMBackend(heavyModelID, heavyURL)
	} else {
		heavyBackend = orla.NewSGLangBackend(heavyModelID, heavyURL)
	}
	heavyBackend.SetMaxConcurrency(OrlaMaxConcurrency)
	if err := client.RegisterBackend(ctx, heavyBackend); err != nil {
		return fmt.Errorf("register heavy backend: %w", err)
	}

	needsLight := mode == ModeStageMapping || mode == ModeSJF
	if needsLight {
		if shared.BackendProviderIsVLLM {
			lightBackend = orla.NewVLLMBackend(lightModelID, lightURL)
		} else {
			lightBackend = orla.NewSGLangBackend(lightModelID, lightURL)
		}
		lightBackend.SetMaxConcurrency(OrlaMaxConcurrency)
		if err := client.RegisterBackend(ctx, lightBackend); err != nil {
			return fmt.Errorf("register light backend: %w", err)
		}
	}

	heavyStage := orla.NewStage("heavy", heavyBackend)
	heavyStage.Client = client
	heavyStage.SetMaxTokens(shared.MaxOutputTokens)
	heavyStage.SetTemperature(0)

	var lightStage *orla.Stage
	var mapper orla.StageMapper
	if needsLight {
		lightStage = orla.NewStage("light", lightBackend)
		lightStage.Client = client
		lightStage.SetMaxTokens(shared.MaxOutputTokens)
		lightStage.SetTemperature(0)
		mapper = orla.NewOneBitStageMapper(client, lightBackend, lightStage, heavyStage)
	}

	if mode == ModeSJF {
		heavyStage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)
		heavyStage.SetRequestSchedulingPolicy("priority")
	}

	// Phase 1: build prompts and (optionally) route, in parallel with repo pooling
	prepN := prepParallelismFromEnv()
	log.Printf("Phase 1: building prompts for %d instances (mode=%s, parallelism=%d)", len(dataset.Instances), mode, prepN)
	sem := make(chan struct{}, prepN)
	var jobsMu sync.Mutex
	var jobs []instanceJob
	g, gctx := errgroup.WithContext(ctx)
	for i, inst := range dataset.Instances {
		if i >= shared.MaxIterations {
			break
		}
		g.Go(func() error {
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			case <-gctx.Done():
				return gctx.Err()
			}
			repoDir, err := shared.EnsureRepoClone(gctx, inst)
			if err != nil {
				return fmt.Errorf("ensure repo %s: %w", inst.InstanceID, err)
			}
			filePaths := shared.OracleFilePaths(inst.Patch)
			oracleCtx := shared.GatherOracleContextFromRepo(gctx, repoDir, inst.BaseCommit, filePaths)
			prompt := shared.BuildSingleShotPrompt(inst, oracleCtx)
			log.Printf("  %s: %d oracle files, prompt %d chars", inst.InstanceID, len(filePaths), len(prompt))

			job := instanceJob{
				inst:      inst,
				prompt:    prompt,
				promptLen: len(prompt),
				stage:     heavyStage,
				stageName: "heavy",
			}

			if needsLight {
				routerPrompt := routerPromptPrefix + inst.ProblemStatement
				routed, err := mapper.MapStage(gctx, routerPrompt)
				if err != nil {
					return fmt.Errorf("instance %s map stage: %w", inst.InstanceID, err)
				}
				job.stage = routed
				job.stageName = routed.Name
				log.Printf("  %s: routed => %s", inst.InstanceID, job.stageName)
			}

			jobsMu.Lock()
			jobs = append(jobs, job)
			jobsMu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return err
	}

	// Phase 1b (sjf): assign priority to heavy jobs by prompt length
	if mode == ModeSJF {
		var heavyJobs []*instanceJob
		for i := range jobs {
			if jobs[i].stageName == "heavy" {
				heavyJobs = append(heavyJobs, &jobs[i])
			}
		}
		sort.Slice(heavyJobs, func(i, j int) bool {
			return heavyJobs[i].promptLen < heavyJobs[j].promptLen
		})
		maxPriority := len(heavyJobs)
		for rank, hj := range heavyJobs {
			hj.priority = maxPriority - rank
			hj.queuePosition = rank
			log.Printf("  heavy queue[%d]: %s (prompt=%d chars, priority=%d)",
				rank, hj.inst.InstanceID, hj.promptLen, hj.priority)
		}
	}

	// Phase 2: submit instances with bounded concurrency to avoid overflowing
	// Orla's backend queue. submitConcurrency limits in-flight goroutines so that
	// the queue (capacity 4096) is never saturated even for full SWE-bench (2500+).
	const submitConcurrency = 512
	log.Printf("Phase 2: submitting %d instances (max %d in-flight)", len(jobs), submitConcurrency)

	outFile, err := os.OpenFile(shared.OutputPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open predictions file: %w", err)
	}
	defer shared.LogDeferredError(outFile.Close)
	enc := shared.NewPredictionEncoder(outFile)
	var encMu sync.Mutex

	metrics := shared.NewRunMetricsRecorder("single_shot_" + mode)
	metrics.TotalJobs = len(jobs)
	metrics.BeginRun()
	defer func() {
		metrics.EndRun()
		if err := metrics.Write(""); err != nil {
			log.Printf("warning: write metrics: %v", err)
		}
	}()

	submitSem := make(chan struct{}, submitConcurrency)
	var wg sync.WaitGroup
	for _, job := range jobs {
		wg.Add(1)
		submitSem <- struct{}{}
		go func(j instanceJob) {
			defer wg.Done()
			defer func() { <-submitSem }()
			submitJob(ctx, j, enc, &encMu, outFile, metrics, mode)
		}(job)
	}
	wg.Wait()

	log.Printf("Done. Predictions written to %s, metrics to %s", shared.OutputPath, shared.MetricsPath)
	return nil
}

func submitJob(ctx context.Context, j instanceJob, enc *shared.PredictionEncoder, encMu *sync.Mutex, outFile *os.File, metrics *shared.RunMetricsRecorder, mode string) {
	stage := j.stage
	if j.priority > 0 {
		p := j.priority
		stage.SetSchedulingHints(&orla.SchedulingHints{Priority: &p})
	}

	start := time.Now()
	messages := []orla.Message{
		{Role: "system", Content: shared.SingleShotSystemPrompt},
		{Role: "user", Content: j.prompt},
	}
	resp, err := stage.ExecuteWithMessages(ctx, messages)
	elapsed := time.Since(start)

	inst := shared.InstanceMetrics{
		InstanceID:    j.inst.InstanceID,
		MappedStage:   j.stageName,
		Priority:      j.priority,
		QueuePosition: j.queuePosition,
		PromptLength:  j.promptLen,
		StartTime:     start.UnixMilli(),
		EndTime:       start.Add(elapsed).UnixMilli(),
		DurationMs:    elapsed.Milliseconds(),
	}

	patch := ""
	if err != nil {
		log.Printf("instance %s: execute error: %v", j.inst.InstanceID, err)
	} else {
		patch = resp.Content
		if resp.Metrics != nil {
			inst.PromptTokens = resp.Metrics.PromptTokens
			inst.CompletionTokens = resp.Metrics.CompletionTokens
			inst.QueueWaitMs = resp.Metrics.QueueWaitMs
			inst.SchedulerDecisionMs = resp.Metrics.SchedulerDecisionMs
			inst.DispatchMs = resp.Metrics.DispatchMs
			if resp.Metrics.BackendLatencyMs != nil {
				inst.BackendLatencyMs = *resp.Metrics.BackendLatencyMs
			}
			if resp.Metrics.TTFTMs != nil {
				inst.TTFTMs = *resp.Metrics.TTFTMs
			}
			if resp.Metrics.TPOTMs != nil {
				inst.TPOTMs = *resp.Metrics.TPOTMs
			}
		}
	}

	metrics.AddInstance(inst)

	encMu.Lock()
	predCount := enc.Count()
	if encErr := enc.Encode(shared.Prediction{
		InstanceID:      j.inst.InstanceID,
		ModelNameOrPath: shared.ModelName(mode),
		ModelPatch:      patch,
	}); encErr != nil {
		log.Printf("warning: encode prediction %s: %v", j.inst.InstanceID, encErr)
	} else {
		log.Printf("[prediction %d] %s: patch=%d chars", predCount+1, j.inst.InstanceID, len(patch))
	}
	if syncErr := outFile.Sync(); syncErr != nil {
		log.Printf("warning: sync predictions: %v", syncErr)
	}
	encMu.Unlock()
}

func resolveURLs() (heavyURL, lightURL string) {
	if shared.BackendProviderIsVLLM {
		return shared.VLLMHeavyURL, shared.VLLMLightURL
	}
	heavyURL = os.Getenv("SGLANG_HEAVY_URL")
	if heavyURL == "" {
		heavyURL = shared.SGLangURL
	}
	lightURL = os.Getenv("SGLANG_LIGHT_URL")
	if lightURL == "" {
		lightURL = defaultLightURL
	}
	return heavyURL, lightURL
}
