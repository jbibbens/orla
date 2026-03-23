// Package workflowdemo demonstrates the full Orla abstraction stack:
// Workflow -> Stage DAG -> StageMapping -> Scheduling.
//
// Pipeline: a customer support ticket triage and resolution workflow.
//
//	Workflow
//	  |
//	  +-- Stage "classify"       (structured output: category, product, key issue, customer request, needs_escalation)
//	  |
//	  +-- Stage "policy_check"   (agent-loop: reads policy YAML via tool, returns accept/deny + reasoning)
//	  |     depends on: classify
//	  |
//	  +-- Stage "reply"          (agent-loop: if escalated, sends acknowledgment; if not, resolves and sends resolution email)
//	  |     depends on: policy_check
//	  |
//	  +-- Stage "route_ticket"   (agent-loop: routes to human team if escalation needed, or notifies team of auto-resolution)
//	        depends on: classify
//
//	classify ──┬──▶ policy_check ──▶ reply
//	           └──▶ route_ticket
//
//	Stage 3 (reply) and Stage 4 (route_ticket) run in parallel once their
//	respective dependencies are satisfied.
package workflowdemo

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	orla "github.com/harvard-cns/orla/pkg/api"
)

const (
	defaultOrlaURL    = "http://localhost:8081"
	defaultLightURL   = "http://sglang-light:30000/v1"
	defaultHeavyURL   = "http://sglang:30000/v1"
	defaultLightModel = "Qwen/Qwen3-4B-Instruct-2507"
	defaultHeavyModel = "Qwen/Qwen3-8B"

	defaultOllamaURL        = "http://ollama:11434"
	defaultOllamaLightModel = "qwen3:0.6b"
	defaultOllamaHeavyModel = "qwen3:1.7b"

	defaultVLLMLightURL = "http://vllm-light:8000/v1"
	defaultVLLMHeavyURL = "http://vllm-heavy:8000/v1"
)

// ---------------------------------------------------------------------------
// Structured-output schemas
// ---------------------------------------------------------------------------

var classifySchema = map[string]any{
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
		"key_issue": map[string]any{
			"type":        "string",
			"description": "One-sentence summary of the core issue",
		},
		"customer_request": map[string]any{
			"type":        "string",
			"description": "What the customer is actually asking for",
		},
		"needs_escalation": map[string]any{
			"type":        "boolean",
			"description": "Whether this ticket requires human team intervention (true) or can be fully resolved automatically (false)",
		},
		"escalation_reason": map[string]any{
			"type":        "string",
			"description": "If needs_escalation is true, explain why human intervention is needed",
		},
	},
	"required": []any{"category", "product", "key_issue", "customer_request", "needs_escalation"},
}

var policyDecisionSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"decision": map[string]any{
			"type": "string",
			"enum": []any{"accept", "deny"},
		},
		"reasoning": map[string]any{
			"type":        "string",
			"description": "Explanation of why the request is accepted or denied per company policy",
		},
		"applicable_policy": map[string]any{
			"type":        "string",
			"description": "The specific policy section that applies",
		},
	},
	"required": []any{"decision", "reasoning", "applicable_policy"},
}

var replyConfirmationSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"email_sent": map[string]any{
			"type":        "boolean",
			"description": "Whether the reply email was sent successfully",
		},
		"summary": map[string]any{
			"type":        "string",
			"description": "Brief summary of the reply that was sent",
		},
	},
	"required": []any{"email_sent", "summary"},
}

// ---------------------------------------------------------------------------
// Mock tool implementations
// ---------------------------------------------------------------------------

// readPolicyYAMLTool simulates reading the company policy document.
func readPolicyYAMLTool() (*orla.Tool, error) {
	return orla.NewTool(
		"read_policy_yaml",
		"Read the company support policy document for a given category. Returns the policy rules as structured text.",
		orla.ToolSchema{
			"type": "object",
			"properties": map[string]any{
				"category": map[string]any{
					"type":        "string",
					"description": "The ticket category to look up policy for (e.g. billing, technical, account, shipping)",
				},
			},
			"required": []any{"category"},
		},
		nil,
		orla.ToolRunnerFromSchema(func(_ context.Context, input orla.ToolSchema) (orla.ToolSchema, error) {
			category, ok := input["category"].(string)
			if !ok {
				return nil, fmt.Errorf("missing or invalid 'category' argument")
			}
			policies := map[string]string{
				"billing": `policy:
  billing:
    duplicate_charges:
      action: refund
      conditions:
        - verified_duplicate: true
        - within_days: 30
      sla: "Refund processed within 3 business days"
    subscription_cancellation:
      action: cancel_and_prorate
      conditions:
        - active_subscription: true
      sla: "Effective end of current billing cycle"`,
				"technical": `policy:
  technical:
    service_degradation:
      action: investigate_and_credit
      conditions:
        - confirmed_outage: true
        - duration_minutes: ">15"
      sla: "Resolution within 4 hours, credit if SLA missed"
    bug_report:
      action: escalate_to_engineering
      conditions:
        - reproducible: true
      sla: "Acknowledgment within 24 hours"`,
				"account": `policy:
  account:
    access_issues:
      action: reset_and_verify
      sla: "Resolved within 1 hour"
    data_request:
      action: export_data
      conditions:
        - identity_verified: true
      sla: "Data export within 48 hours"`,
				"shipping": `policy:
  shipping:
    lost_package:
      action: reship_or_refund
      conditions:
        - tracking_shows_lost: true
      sla: "Replacement shipped within 2 business days"`,
			}
			policy, ok := policies[category]
			if !ok {
				policy = "No specific policy found for category: " + category + ". Apply general support guidelines."
			}
			return orla.ToolSchema{"policy_document": policy}, nil
		}),
	)
}

// sendEmailTool simulates sending an email to the customer.
func sendEmailTool() (*orla.Tool, error) {
	return orla.NewTool(
		"send_email",
		"Send an email to a recipient with the given subject and body.",
		orla.ToolSchema{
			"type": "object",
			"properties": map[string]any{
				"to": map[string]any{
					"type":        "string",
					"description": "Recipient email address",
				},
				"subject": map[string]any{
					"type":        "string",
					"description": "Email subject line",
				},
				"body": map[string]any{
					"type":        "string",
					"description": "Email body text",
				},
			},
			"required": []any{"to", "subject", "body"},
		},
		nil,
		orla.ToolRunnerFromSchema(func(_ context.Context, input orla.ToolSchema) (orla.ToolSchema, error) {
			to, ok := input["to"].(string)
			if !ok {
				return nil, fmt.Errorf("missing or invalid 'to' argument")
			}
			subject, ok := input["subject"].(string)
			if !ok {
				return nil, fmt.Errorf("missing or invalid 'subject' argument")
			}
			log.Printf("[send_email] To: %s | Subject: %s", to, subject)
			return orla.ToolSchema{
				"status":     "sent",
				"message_id": "msg-" + to + "-001",
			}, nil
		}),
	)
}

// readTeamDescriptionsTool simulates reading internal team descriptions.
func readTeamDescriptionsTool() (*orla.Tool, error) {
	return orla.NewTool(
		"read_team_descriptions",
		"Read descriptions of internal support teams to determine the best routing destination.",
		orla.ToolSchema{
			"type":       "object",
			"properties": map[string]any{},
		},
		nil,
		orla.ToolRunnerFromSchema(func(_ context.Context, _ orla.ToolSchema) (orla.ToolSchema, error) {
			return orla.ToolSchema{
				"teams": []any{
					map[string]any{
						"name":        "billing_ops",
						"description": "Handles refunds, subscription changes, payment disputes, and invoice corrections.",
						"email":       "billing-ops@company.com",
					},
					map[string]any{
						"name":        "technical_support",
						"description": "Handles service outages, performance issues, bug reports, and API problems.",
						"email":       "tech-support@company.com",
					},
					map[string]any{
						"name":        "account_management",
						"description": "Handles account access, data requests, plan upgrades, and enterprise onboarding.",
						"email":       "account-mgmt@company.com",
					},
					map[string]any{
						"name":        "escalation_team",
						"description": "Handles critical/VIP issues, multi-department problems, and unresolved complaints.",
						"email":       "escalation@company.com",
					},
				},
			}, nil
		}),
	)
}

// sendTicketTool simulates sending an internal support ticket to a team.
func sendTicketTool() (*orla.Tool, error) {
	return orla.NewTool(
		"send_ticket",
		"Create and send an internal support ticket to the designated team.",
		orla.ToolSchema{
			"type": "object",
			"properties": map[string]any{
				"team": map[string]any{
					"type":        "string",
					"description": "The team to route the ticket to",
				},
				"priority": map[string]any{
					"type":        "string",
					"enum":        []any{"critical", "high", "medium", "low"},
					"description": "Ticket priority level",
				},
				"summary": map[string]any{
					"type":        "string",
					"description": "Brief summary of the issue for the internal ticket",
				},
				"customer_email": map[string]any{
					"type":        "string",
					"description": "Customer email for follow-up",
				},
			},
			"required": []any{"team", "priority", "summary"},
		},
		nil,
		orla.ToolRunnerFromSchema(func(_ context.Context, input orla.ToolSchema) (orla.ToolSchema, error) {
			team, ok := input["team"].(string)
			if !ok {
				return nil, fmt.Errorf("missing or invalid 'team' argument")
			}
			priority, ok := input["priority"].(string)
			if !ok {
				return nil, fmt.Errorf("missing or invalid 'priority' argument")
			}
			log.Printf("[send_ticket] Team: %s | Priority: %s", team, priority)
			return orla.ToolSchema{
				"ticket_id": "TKT-" + team + "-42",
				"status":    "created",
			}, nil
		}),
	)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func setSchedulingPolicy(stage *orla.Stage, optsPolicy, defaultPolicy string) {
	switch optsPolicy {
	case orla.SchedulingPolicyFCFS:
		stage.SetSchedulingPolicy(orla.SchedulingPolicyFCFS)
	case orla.SchedulingPolicyPriority:
		stage.SetSchedulingPolicy(orla.SchedulingPolicyPriority)
	default:
		stage.SetSchedulingPolicy(defaultPolicy)
	}
}

// Options configures RunWithOptions for demo variants (e.g. Part 1/2/3).
type Options struct {
	// SchedulingPolicy sets all stages to "fcfs" or "priority".
	SchedulingPolicy string
	// MemoryPolicy sets the workflow memory policy. If nil, uses default (preserve + flush at boundary).
	MemoryPolicy orla.MemoryPolicy
	// RunSecondWorkflow runs a small second workflow after the main one to demonstrate flush at boundary.
	RunSecondWorkflow bool
}

// Run executes the customer support workflow demo with default options.
// ticket is the raw customer support message.
func Run(ctx context.Context, ticket string) error {
	return RunWithOptions(ctx, ticket, Options{})
}

// RunWithOptions executes the customer support workflow with the given options.
func RunWithOptions(ctx context.Context, ticket string, opts Options) error {
	orlaURL := envOr("ORLA_URL", defaultOrlaURL)
	client := orla.NewOrlaClient(orlaURL)
	if err := client.Health(ctx); err != nil {
		return fmt.Errorf("orla health check: %w", err)
	}

	// --- Backends ---
	//
	// BACKEND env selects the inference backend: "ollama", "vllm", or "sglang" (default).
	// Each has sensible Docker-internal defaults so no extra env vars are needed
	// when running the matching compose file.
	var lightBackend, heavyBackend *orla.LLMBackend
	switch os.Getenv("BACKEND") {
	case "ollama":
		ollamaURL := envOr("OLLAMA_URL", defaultOllamaURL)
		lightBackend = orla.NewOllamaBackend(envOr("LIGHT_MODEL", defaultOllamaLightModel), ollamaURL)
		heavyBackend = orla.NewOllamaBackend(envOr("HEAVY_MODEL", defaultOllamaHeavyModel), ollamaURL)
	case "vllm":
		lightBackend = orla.NewVLLMBackend(
			envOr("LIGHT_MODEL", defaultLightModel),
			envOr("VLLM_LIGHT_URL", defaultVLLMLightURL),
		)
		heavyBackend = orla.NewVLLMBackend(
			envOr("HEAVY_MODEL", defaultHeavyModel),
			envOr("VLLM_HEAVY_URL", defaultVLLMHeavyURL),
		)
	default: // sglang
		lightBackend = orla.NewSGLangBackend(
			envOr("LIGHT_MODEL", defaultLightModel),
			envOr("SGLANG_LIGHT_URL", defaultLightURL),
		)
		heavyBackend = orla.NewSGLangBackend(
			envOr("HEAVY_MODEL", defaultHeavyModel),
			envOr("SGLANG_HEAVY_URL", defaultHeavyURL),
		)
	}
	if err := client.RegisterBackend(ctx, lightBackend); err != nil {
		return fmt.Errorf("register light backend: %w", err)
	}
	if err := client.RegisterBackend(ctx, heavyBackend); err != nil {
		return fmt.Errorf("register heavy backend: %w", err)
	}

	noThinking := map[string]any{"enable_thinking": false}
	wf := orla.NewWorkflow(client)
	if opts.MemoryPolicy != nil {
		wf.SetMemoryPolicy(opts.MemoryPolicy)
	}

	// -----------------------------------------------------------------------
	// Stage 1: classify (single-shot, structured output, light model)
	//
	//   Input:  Email + System Prompt
	//   Output: Structured JSON {category, product, key_issue, customer_request}
	// -----------------------------------------------------------------------

	classifyStage := orla.NewStage("classify", lightBackend)
	classifyStage.SetMaxTokens(512)
	classifyStage.SetTemperature(0)
	setSchedulingPolicy(classifyStage, opts.SchedulingPolicy, orla.SchedulingPolicyFCFS)
	classifyStage.SetResponseFormat(orla.NewStructuredOutputRequest("ticket_classify", classifySchema))
	classifyStage.Prompt = fmt.Sprintf(
		"You are a customer support triage system. Classify this support ticket.\n"+
			"Extract the category, product, a one-sentence summary of the core issue, "+
			"and what the customer is actually asking for.\n\n"+
			"Also decide whether this ticket needs human team escalation. Set needs_escalation "+
			"to true if the issue is ambiguous, involves multiple departments, requires manual "+
			"verification, or cannot be resolved by automated policy lookup alone. Set it to "+
			"false if standard policy can fully resolve it.\n\nTicket:\n%s", ticket)

	// -----------------------------------------------------------------------
	// Stage 2: policy_check (agent-loop, tool: read_policy_yaml, heavy model)
	//
	//   Input:  Email + Output(classify) + System Prompt
	//   Flow:   Gen → Tool(read_policy_yaml) → Gen → Structured Output
	//   Output: {decision: accept|deny, reasoning, applicable_policy}
	//   Depends on: classify
	// -----------------------------------------------------------------------

	policyTool, err := readPolicyYAMLTool()
	if err != nil {
		return fmt.Errorf("create read_policy_yaml tool: %w", err)
	}

	policyStage := orla.NewStage("policy_check", heavyBackend)
	policyStage.SetExecutionMode(orla.ExecutionModeAgentLoop)
	policyStage.SetMaxTurns(5)
	policyStage.SetMaxTokens(1024)
	policyStage.SetTemperature(0)
	setSchedulingPolicy(policyStage, opts.SchedulingPolicy, orla.SchedulingPolicyPriority)
	policyStage.SetChatTemplateKwargs(noThinking)
	policyStage.SetResponseFormat(orla.NewStructuredOutputRequest("policy_decision", policyDecisionSchema))
	if err := policyStage.AddTool(policyTool); err != nil {
		return fmt.Errorf("add read_policy_yaml tool: %w", err)
	}
	policyStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		classification, ok := results[classifyStage.ID]
		if !ok {
			return "", fmt.Errorf("missing classify stage result")
		}
		return fmt.Sprintf(
			"You are a support policy specialist. You have access to a tool called "+
				"read_policy_yaml that lets you look up the company's support policy for a "+
				"given category.\n\n"+
				"Step 1: Use the read_policy_yaml tool to retrieve the policy for the ticket's category.\n"+
				"Step 2: Based on the policy, decide whether to ACCEPT or DENY the customer's request.\n\n"+
				"Ticket Classification:\n%s\n\nOriginal Ticket:\n%s",
			classification.Response.Content, ticket), nil
	})

	// -----------------------------------------------------------------------
	// Stage 3: reply (agent-loop, tool: send_email, heavy model)
	//
	//   Input:  Output(policy_check) + Output(classify) + Email + System Prompt
	//   If needs_escalation: send a brief acknowledgment that the request is being escalated
	//   If !needs_escalation: resolve the ticket — confirm action or explain denial
	//   Flow:   Gen → Tool(send_email) → Gen → Structured Output
	//   Output: {email_sent, summary}
	//   Depends on: policy_check
	// -----------------------------------------------------------------------

	emailTool, err := sendEmailTool()
	if err != nil {
		return fmt.Errorf("create send_email tool: %w", err)
	}

	replyStage := orla.NewStage("reply", heavyBackend)
	replyStage.SetExecutionMode(orla.ExecutionModeAgentLoop)
	replyStage.SetMaxTurns(5)
	replyStage.SetMaxTokens(1024)
	replyStage.SetTemperature(0.3)
	setSchedulingPolicy(replyStage, opts.SchedulingPolicy, orla.SchedulingPolicyPriority)
	replyStage.SetChatTemplateKwargs(noThinking)
	replyStage.SetResponseFormat(orla.NewStructuredOutputRequest("reply_confirmation", replyConfirmationSchema))
	if err := replyStage.AddTool(emailTool); err != nil {
		return fmt.Errorf("add send_email tool to reply: %w", err)
	}
	replyStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		classification, ok := results[classifyStage.ID]
		if !ok {
			return "", fmt.Errorf("missing classify stage result")
		}
		policyResult, ok := results[policyStage.ID]
		if !ok {
			return "", fmt.Errorf("missing policy_check stage result")
		}

		var classifyData struct {
			Category        string `json:"category"`
			NeedsEscalation bool   `json:"needs_escalation"`
		}
		if err := json.Unmarshal([]byte(classification.Response.Content), &classifyData); err != nil {
			log.Printf("warning: failed to parse classify output: %v", err)
		}
		priority := 5
		if classifyData.Category == "billing" || classifyData.Category == "technical" {
			priority = 8
		}
		replyStage.SetSchedulingHints(&orla.SchedulingHints{Priority: &priority})

		if classifyData.NeedsEscalation {
			return fmt.Sprintf(
				"You are a customer support agent. The triage system determined this ticket "+
					"NEEDS HUMAN ESCALATION and it is being routed to the appropriate team.\n\n"+
					"Compose a brief, professional email to the customer letting them know their "+
					"request has been received and is being escalated to a specialist team for "+
					"further review. Do NOT resolve the issue or make promises about the outcome. "+
					"Just acknowledge receipt and set expectations for follow-up.\n\n"+
					"Send the email using the send_email tool. "+
					"Extract the customer's email from the original ticket for the 'to' field.\n\n"+
					"Policy Decision:\n%s\n\nTicket Classification:\n%s\n\nOriginal Ticket:\n%s",
				policyResult.Response.Content, classification.Response.Content, ticket), nil
		}

		return fmt.Sprintf(
			"You are a customer support agent. Based on the policy decision and ticket "+
				"classification below, compose a professional reply to the customer and send "+
				"it using the send_email tool.\n\n"+
				"If the request is ACCEPTED, confirm the action being taken and provide an ETA.\n"+
				"If DENIED, explain why politely and offer alternatives.\n\n"+
				"Extract the customer's email from the original ticket for the 'to' field.\n\n"+
				"Policy Decision:\n%s\n\nTicket Classification:\n%s\n\nOriginal Ticket:\n%s",
			policyResult.Response.Content, classification.Response.Content, ticket), nil
	})

	// -----------------------------------------------------------------------
	// Stage 4: route_ticket (agent-loop, tools: send_email, read_team_descriptions,
	//          send_ticket; heavy model)
	//
	//   Input:  Output(classify) + Email + System Prompt
	//   If needs_escalation: read teams → route ticket to human team
	//   If !needs_escalation: notify team that ticket is being resolved automatically
	//   Output: Free-text summary
	//   Depends on: classify
	// -----------------------------------------------------------------------

	emailToolRoute, err := sendEmailTool()
	if err != nil {
		return fmt.Errorf("create send_email tool for route: %w", err)
	}
	teamsTool, err := readTeamDescriptionsTool()
	if err != nil {
		return fmt.Errorf("create read_team_descriptions tool: %w", err)
	}
	ticketTool, err := sendTicketTool()
	if err != nil {
		return fmt.Errorf("create send_ticket tool: %w", err)
	}

	routeStage := orla.NewStage("route_ticket", heavyBackend)
	routeStage.SetExecutionMode(orla.ExecutionModeAgentLoop)
	routeStage.SetMaxTurns(10)
	routeStage.SetMaxTokens(1024)
	routeStage.SetTemperature(0)
	setSchedulingPolicy(routeStage, opts.SchedulingPolicy, orla.SchedulingPolicyPriority)
	routeStage.SetChatTemplateKwargs(noThinking)
	if err := routeStage.AddTool(emailToolRoute); err != nil {
		return fmt.Errorf("add send_email tool to route: %w", err)
	}
	if err := routeStage.AddTool(teamsTool); err != nil {
		return fmt.Errorf("add read_team_descriptions tool: %w", err)
	}
	if err := routeStage.AddTool(ticketTool); err != nil {
		return fmt.Errorf("add send_ticket tool: %w", err)
	}
	routeStage.SetPromptBuilder(func(results map[string]*orla.StageResult) (string, error) {
		classification, ok := results[classifyStage.ID]
		if !ok {
			return "", fmt.Errorf("missing classify stage result")
		}

		var classifyOut struct {
			NeedsEscalation  bool   `json:"needs_escalation"`
			EscalationReason string `json:"escalation_reason"`
		}
		if err := json.Unmarshal([]byte(classification.Response.Content), &classifyOut); err != nil {
			log.Printf("warning: failed to parse classify output for escalation decision: %v", err)
		}

		if classifyOut.NeedsEscalation {
			return fmt.Sprintf(
				"You are an internal support ticket router. The triage system determined "+
					"this ticket NEEDS HUMAN ESCALATION.\n\n"+
					"Escalation reason: %s\n\n"+
					"Your job is to:\n"+
					"1. Read the available team descriptions (use the read_team_descriptions tool) "+
					"to determine which team should handle this ticket.\n"+
					"2. Create an internal support ticket routed to the appropriate team "+
					"(use the send_ticket tool). Include the escalation reason.\n"+
					"3. Send an email to the assigned team (use the send_email tool) alerting them "+
					"to the escalated ticket.\n\n"+
					"After completing all steps, provide a brief summary of what was done.\n\n"+
					"Ticket Classification:\n%s\n\nOriginal Ticket:\n%s",
				classifyOut.EscalationReason,
				classification.Response.Content, ticket), nil
		}

		return fmt.Sprintf(
			"You are an internal support ticket router. The triage system determined "+
				"this ticket does NOT need human escalation -- it is being resolved "+
				"automatically by the policy check and reply stages.\n\n"+
				"Your job is to:\n"+
				"1. Read the available team descriptions (use the read_team_descriptions tool) "+
				"to identify the team responsible for this category.\n"+
				"2. Send an informational email to that team (use the send_email tool) letting "+
				"them know the ticket is being handled automatically by the system.\n\n"+
				"After completing all steps, provide a brief summary of what was done.\n\n"+
				"Ticket Classification:\n%s\n\nOriginal Ticket:\n%s",
			classification.Response.Content, ticket), nil
	})

	// --- Add all stages ---
	allStages := []*orla.Stage{classifyStage, policyStage, replyStage, routeStage}
	for _, s := range allStages {
		if err := wf.AddStage(s); err != nil {
			return err
		}
	}

	// --- Stage Mapping (validation) ---
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

	// --- Dependencies ---
	//   classify → policy_check → reply → route_ticket
	if err := wf.AddDependency(policyStage.ID, classifyStage.ID); err != nil {
		return err
	}
	if err := wf.AddDependency(replyStage.ID, policyStage.ID); err != nil {
		return err
	}
	if err := wf.AddDependency(routeStage.ID, classifyStage.ID); err != nil {
		return err
	}

	// --- Execute ---
	log.Println("Executing customer support workflow...")
	log.Println("  Stage DAG:")
	log.Println("    classify ──┬──▶ policy_check ──▶ reply")
	log.Println("               └──▶ route_ticket")
	results, err := wf.Execute(ctx)
	if err != nil {
		return fmt.Errorf("workflow execution: %w", err)
	}

	// --- Print results ---
	stageOrder := []struct {
		id   string
		name string
	}{
		{classifyStage.ID, "classify"},
		{policyStage.ID, "policy_check"},
		{replyStage.ID, "reply"},
		{routeStage.ID, "route_ticket"},
	}
	for _, s := range stageOrder {
		result, ok := results[s.id]
		if !ok {
			continue
		}
		log.Printf("  %s:", s.name)
		if result.Response != nil {
			content := result.Response.Content
			if len(content) > 500 {
				content = content[:500] + "..."
			}
			log.Printf("    %s", content)
		}
		if len(result.Messages) > 0 {
			toolCalls := 0
			for _, msg := range result.Messages {
				if msg.Role == "tool" {
					toolCalls++
				}
			}
			if toolCalls > 0 {
				log.Printf("    (tool calls executed: %d)", toolCalls)
			}
		}
	}

	// --- Second workflow (demonstrates flush at boundary) ---
	if opts.RunSecondWorkflow {
		log.Println("")
		log.Println("Running second workflow (Orla flushes cache at workflow boundary)...")
		wf2 := orla.NewWorkflow(client)
		if opts.MemoryPolicy != nil {
			wf2.SetMemoryPolicy(opts.MemoryPolicy)
		}
		summarizeStage := orla.NewStage("summarize", heavyBackend)
		summarizeStage.SetMaxTokens(128)
		summarizeStage.SetTemperature(0)
		setSchedulingPolicy(summarizeStage, opts.SchedulingPolicy, orla.SchedulingPolicyPriority)
		summarizeStage.Prompt = "In one sentence, summarize what a customer support ticket triage workflow does."
		if err := wf2.AddStage(summarizeStage); err != nil {
			return err
		}
		_, err := wf2.Execute(ctx)
		if err != nil {
			return fmt.Errorf("second workflow: %w", err)
		}
		log.Println("Second workflow complete.")
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
Leonhard Euler
Account: leonhard.euler@email.com
Plan: Pro ($49.99/month)`
