// Package workflowdemo demonstrates the full Orla abstraction stack:
// Workflow -> Agent -> Stage DAG -> StageMapping -> Scheduling -> Context Passing.
//
// Pipeline: a customer support ticket triage and resolution workflow.
//
//	Workflow
//	  |
//	  +-- Agent "triage" (light model, FCFS)
//	  |     +-- Stage "classify"   (classify ticket: category + entities, structured output)
//	  |     +-- Stage "prioritize" (assign severity & priority score; depends on "classify")
//	  |
//	  +-- Agent "resolver" (heavy model, Priority scheduling; depends on "triage")
//	        +-- Stage "draft_response" (generate personalized reply using triage output)
//	        +-- Stage "qa_check"       (review for policy compliance; depends on "draft_response")
package workflowdemo

import (
	"context"
	"fmt"
	"log"
	"os"

	orla "github.com/dorcha-inc/orla/pkg/api"
)

const (
	defaultOrlaURL    = "http://localhost:8081"
	defaultLightURL   = "http://sglang-light:30000/v1"
	defaultHeavyURL   = "http://sglang:30000/v1"
	defaultLightModel = "Qwen/Qwen3-4B-Instruct-2507"
	defaultHeavyModel = "Qwen/Qwen3-8B"
)

var triageSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"category": map[string]any{
			"type": "string",
			"enum": []any{"billing", "technical", "account", "shipping", "general"},
		},
		"product": map[string]any{
			"type":        "string",
			"description": "Product or service mentioned in the ticket",
		},
		"customer_sentiment": map[string]any{
			"type": "string",
			"enum": []any{"frustrated", "neutral", "positive"},
		},
		"key_issue": map[string]any{
			"type":        "string",
			"description": "One-sentence summary of the core issue",
		},
	},
	"required": []any{"category", "product", "customer_sentiment", "key_issue"},
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// Run executes the customer support workflow demo.
// ticket is the raw customer support message.
func Run(ctx context.Context, ticket string) error {
	orlaURL := envOr("ORLA_URL", defaultOrlaURL)
	client := orla.NewOrlaClient(orlaURL)
	if err := client.Health(ctx); err != nil {
		return fmt.Errorf("orla health check: %w", err)
	}

	// --- Backends ---
	lightBackend := orla.NewSGLangBackend(
		envOr("LIGHT_MODEL", defaultLightModel),
		envOr("SGLANG_LIGHT_URL", defaultLightURL),
	)
	heavyBackend := orla.NewSGLangBackend(
		envOr("HEAVY_MODEL", defaultHeavyModel),
		envOr("SGLANG_HEAVY_URL", defaultHeavyURL),
	)
	if err := client.RegisterBackend(ctx, lightBackend); err != nil {
		return fmt.Errorf("register light backend: %w", err)
	}
	if err := client.RegisterBackend(ctx, heavyBackend); err != nil {
		return fmt.Errorf("register heavy backend: %w", err)
	}

	// --- Agent 1: triage (light model, two-stage DAG) ---
	triage := orla.NewAgent(client)
	triage.Name = "triage"

	classifyStage := orla.NewStage("classify", lightBackend)
	classifyStage.SetMaxTokens(512)
	classifyStage.SetTemperature(0)
	classifyStage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
	classifyStage.SetResponseFormat(orla.NewStructuredOutputRequest("ticket_triage", triageSchema))
	classifyStage.Prompt = fmt.Sprintf(
		"You are a customer support triage system. Classify this support ticket and extract key information.\n\nTicket:\n%s", ticket)

	prioritizeStage := orla.NewStage("prioritize", lightBackend)
	prioritizeStage.SetMaxTokens(256)
	prioritizeStage.SetTemperature(0)
	prioritizeStage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
	prioritizeStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		classification, ok := results[classifyStage.ID]
		if !ok {
			return "", fmt.Errorf("missing classify stage result")
		}
		return fmt.Sprintf(
			"Given this ticket classification, assign a severity (critical / high / medium / low) and explain why in one sentence.\n\nClassification:\n%s",
			classification.Response.Content), nil
	})

	if err := triage.AddStage(classifyStage); err != nil {
		return err
	}
	if err := triage.AddStage(prioritizeStage); err != nil {
		return err
	}
	if err := triage.AddDependency(prioritizeStage.ID, classifyStage.ID); err != nil {
		return err
	}

	// --- Agent 2: resolver (heavy model, two-stage DAG) ---
	resolver := orla.NewAgent(client)
	resolver.Name = "resolver"

	draftStage := orla.NewStage("draft_response", heavyBackend)
	draftStage.SetMaxTokens(1024)
	draftStage.SetTemperature(0.3)
	draftStage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)
	priority := 5
	draftStage.SetSchedulingHints(&orla.SchedulingHints{Priority: &priority})

	qaStage := orla.NewStage("qa_check", heavyBackend)
	qaStage.SetMaxTokens(512)
	qaStage.SetTemperature(0)
	qaStage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)
	qaStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		draft, ok := results[draftStage.ID]
		if !ok {
			return "", fmt.Errorf("missing draft_response stage result")
		}
		return fmt.Sprintf(
			"You are a QA reviewer for customer support responses. Check this draft reply for: (1) policy compliance, (2) professional tone, (3) completeness. If it passes, output APPROVED. If not, explain what to fix.\n\nDraft Reply:\n%s",
			draft.Response.Content), nil
	})

	if err := resolver.AddStage(draftStage); err != nil {
		return err
	}
	if err := resolver.AddStage(qaStage); err != nil {
		return err
	}
	if err := resolver.AddDependency(qaStage.ID, draftStage.ID); err != nil {
		return err
	}

	// --- Stage Mapping (validation) ---
	allStages := []*orla.Stage{classifyStage, prioritizeStage, draftStage, qaStage}
	mapping := &orla.ExplicitStageMapping{}
	output, err := mapping.Map(&orla.StageMappingInput{
		Stages:   allStages,
		Backends: []*orla.LLMBackend{lightBackend, heavyBackend},
	})
	if err != nil {
		return fmt.Errorf("stage mapping: %w", err)
	}
	if err := orla.ApplyStageMappingOutput(allStages, output); err != nil {
		return fmt.Errorf("apply stage mapping: %w", err)
	}
	log.Printf("Stage mapping validated: %d stages assigned to backends", len(output.Assignments))

	// --- Workflow ---
	wf := orla.NewWorkflow()
	if err := wf.AddAgent(triage); err != nil {
		return err
	}
	if err := wf.AddAgent(resolver); err != nil {
		return err
	}
	if err := wf.AddDependency("resolver", "triage"); err != nil {
		return err
	}

	// Context passing: feed triage output into resolver's draft prompt.
	wf.SetContextPassingFn(func(upstreamResults map[string]*orla.AgentResult, downstream *orla.Agent) error {
		if downstream.Name != "resolver" {
			return nil
		}
		triageResult, ok := upstreamResults["triage"]
		if !ok {
			return nil
		}

		var triageOutput string
		for _, sr := range triageResult.StageResults {
			if sr.Response != nil {
				triageOutput += sr.Response.Content + "\n"
			}
		}

		stages := downstream.Stages()
		for _, s := range stages {
			if s.Name == "draft_response" {
				s.Prompt = fmt.Sprintf(
					"You are a customer support agent. Write a helpful, professional reply to this customer based on the triage analysis below. Address their specific issue, apologize if appropriate, and provide clear next steps.\n\nTriage Analysis:\n%s\n\nOriginal Ticket:\n%s",
					triageOutput, ticket)
			}
		}
		return nil
	})

	// --- Execute ---
	log.Println("Executing customer support workflow...")
	results, err := wf.Execute(ctx)
	if err != nil {
		return fmt.Errorf("workflow execution: %w", err)
	}

	// --- Print results ---
	for agentName, agentResult := range results {
		log.Printf("=== Agent: %s ===", agentName)
		for stageID, stageResult := range agentResult.StageResults {
			log.Printf("  Stage %s:", stageID)
			if stageResult.Response != nil {
				content := stageResult.Response.Content
				if len(content) > 500 {
					content = content[:500] + "..."
				}
				log.Printf("    %s", content)
			}
		}
	}

	return nil
}

// SampleTicket is an example customer support ticket for running the demo.
const SampleTicket = `Subject: Charged twice for my subscription - URGENT

Hi,

I just noticed that my credit card was charged $49.99 TWICE for my Pro
subscription this month (Oct 3 and Oct 5). I only have one account and
I definitely did not sign up for a second subscription.

I've been a customer for 2 years and this has never happened before.
I need a refund for the duplicate charge ASAP - I'm on a tight budget
this month and that extra $50 really hurts.

Also, while I have your attention - the dashboard has been loading
really slowly for the past week. Sometimes it takes 30+ seconds.
Is there something going on with the servers?

Thanks,
Alex Johnson
Account: alex.johnson@email.com
Plan: Pro ($49.99/month)`
