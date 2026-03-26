"""Customer support triage demo using pyorla Workflow API."""

from __future__ import annotations

import json
import logging
import os
import sys

from pyorla import (
    EXECUTION_MODE_AGENT_LOOP,
    SCHEDULING_POLICY_FCFS,
    SCHEDULING_POLICY_PRIORITY,
    ExplicitStageMapping,
    OrlaClient,
    SchedulingHints,
    StageMappingInput,
    Stage,
    StructuredOutputRequest,
    Workflow,
    apply_stage_mapping_output,
    new_ollama_backend,
    new_sglang_backend,
    new_vllm_backend,
)
from pyorla.stage import StageResultData

from examples.workflow_demo.mock_tools import (
    read_policy_yaml,
    read_team_descriptions,
    send_email,
    send_ticket,
)
from examples.workflow_demo.schemas import (
    CLASSIFY_SCHEMA,
    POLICY_DECISION_SCHEMA,
    REPLY_CONFIRMATION_SCHEMA,
    SAMPLE_TICKET,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def run(ticket: str) -> None:
    orla_url = _env("ORLA_URL", "http://localhost:8081")
    client = OrlaClient(orla_url)
    client.health()

    backend_type = os.environ.get("BACKEND", "sglang")
    if backend_type == "ollama":
        ollama_url = _env("OLLAMA_URL", "http://ollama:11434")
        light = new_ollama_backend(_env("LIGHT_MODEL", "qwen3:0.6b"), ollama_url)
        heavy = new_ollama_backend(_env("HEAVY_MODEL", "qwen3:1.7b"), ollama_url)
    elif backend_type == "vllm":
        light = new_vllm_backend(
            _env("LIGHT_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
            _env("VLLM_LIGHT_URL", "http://vllm-light:8000/v1"),
        )
        heavy = new_vllm_backend(
            _env("HEAVY_MODEL", "Qwen/Qwen3-8B"),
            _env("VLLM_HEAVY_URL", "http://vllm-heavy:8000/v1"),
        )
    else:
        light = new_sglang_backend(
            _env("LIGHT_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
            _env("SGLANG_LIGHT_URL", "http://sglang-light:30000/v1"),
        )
        heavy = new_sglang_backend(
            _env("HEAVY_MODEL", "Qwen/Qwen3-8B"),
            _env("SGLANG_HEAVY_URL", "http://sglang:30000/v1"),
        )

    client.register_backend(light)
    client.register_backend(heavy)

    no_thinking = {"enable_thinking": False}
    wf = Workflow(client)

    # --- Stage 1: classify ---
    classify = Stage("classify", light)
    classify.set_max_tokens(512)
    classify.set_temperature(0)
    classify.set_scheduling_policy(SCHEDULING_POLICY_FCFS)
    classify.set_response_format(StructuredOutputRequest(name="ticket_classify", schema=CLASSIFY_SCHEMA))
    classify.prompt = (
        "You are a customer support triage system. Classify this support ticket.\n"
        "Extract the category, product, a one-sentence summary of the core issue, "
        "and what the customer is actually asking for.\n\n"
        "Also decide whether this ticket needs human team escalation. Set needs_escalation "
        "to true if the issue is ambiguous, involves multiple departments, requires manual "
        "verification, or cannot be resolved by automated policy lookup alone. Set it to "
        "false if standard policy can fully resolve it.\n\n"
        f"Ticket:\n{ticket}"
    )

    # --- Stage 2: policy_check ---
    policy_stage = Stage("policy_check", heavy)
    policy_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
    policy_stage.set_max_turns(5)
    policy_stage.set_max_tokens(1024)
    policy_stage.set_temperature(0)
    policy_stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
    policy_stage.set_chat_template_kwargs(no_thinking)
    policy_stage.set_response_format(StructuredOutputRequest(name="policy_decision", schema=POLICY_DECISION_SCHEMA))
    policy_stage.add_tool(read_policy_yaml)

    def policy_prompt_builder(results: dict[str, StageResultData]) -> str:
        classification = results.get(classify.id)
        if classification is None or classification.response is None:
            raise ValueError("missing classify stage result")
        return (
            "You are a support policy specialist. You have access to a tool called "
            "read_policy_yaml that lets you look up the company's support policy for a "
            "given category.\n\n"
            "Step 1: Use the read_policy_yaml tool to retrieve the policy for the ticket's category.\n"
            "Step 2: Based on the policy, decide whether to ACCEPT or DENY the customer's request.\n\n"
            f"Ticket Classification:\n{classification.response.content}\n\nOriginal Ticket:\n{ticket}"
        )

    policy_stage.set_prompt_builder(policy_prompt_builder)

    # --- Stage 3: reply ---
    reply_stage = Stage("reply", heavy)
    reply_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
    reply_stage.set_max_turns(5)
    reply_stage.set_max_tokens(1024)
    reply_stage.set_temperature(0.3)
    reply_stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
    reply_stage.set_chat_template_kwargs(no_thinking)
    reply_stage.set_response_format(StructuredOutputRequest(name="reply_confirmation", schema=REPLY_CONFIRMATION_SCHEMA))
    reply_stage.add_tool(send_email)

    def reply_prompt_builder(results: dict[str, StageResultData]) -> str:
        classification = results.get(classify.id)
        policy_result = results.get(policy_stage.id)
        if classification is None or classification.response is None:
            raise ValueError("missing classify stage result")
        if policy_result is None or policy_result.response is None:
            raise ValueError("missing policy_check stage result")

        try:
            classify_data = json.loads(classification.response.content)
        except json.JSONDecodeError:
            classify_data = {}

        priority = 5
        if classify_data.get("category") in ("billing", "technical"):
            priority = 8
        reply_stage.set_scheduling_hints(SchedulingHints(priority=priority))

        if classify_data.get("needs_escalation"):
            return (
                "You are a customer support agent. The triage system determined this ticket "
                "NEEDS HUMAN ESCALATION and it is being routed to the appropriate team.\n\n"
                "Compose a brief, professional email to the customer letting them know their "
                "request has been received and is being escalated to a specialist team for "
                "further review. Do NOT resolve the issue or make promises about the outcome. "
                "Just acknowledge receipt and set expectations for follow-up.\n\n"
                "Send the email using the send_email tool. "
                "Extract the customer's email from the original ticket for the 'to' field.\n\n"
                f"Policy Decision:\n{policy_result.response.content}\n\n"
                f"Ticket Classification:\n{classification.response.content}\n\n"
                f"Original Ticket:\n{ticket}"
            )

        return (
            "You are a customer support agent. Based on the policy decision and ticket "
            "classification below, compose a professional reply to the customer and send "
            "it using the send_email tool.\n\n"
            "If the request is ACCEPTED, confirm the action being taken and provide an ETA.\n"
            "If DENIED, explain why politely and offer alternatives.\n\n"
            "Extract the customer's email from the original ticket for the 'to' field.\n\n"
            f"Policy Decision:\n{policy_result.response.content}\n\n"
            f"Ticket Classification:\n{classification.response.content}\n\n"
            f"Original Ticket:\n{ticket}"
        )

    reply_stage.set_prompt_builder(reply_prompt_builder)

    # --- Stage 4: route_ticket ---
    route_stage = Stage("route_ticket", heavy)
    route_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
    route_stage.set_max_turns(10)
    route_stage.set_max_tokens(1024)
    route_stage.set_temperature(0)
    route_stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
    route_stage.set_chat_template_kwargs(no_thinking)
    route_stage.add_tool(send_email)
    route_stage.add_tool(read_team_descriptions)
    route_stage.add_tool(send_ticket)

    def route_prompt_builder(results: dict[str, StageResultData]) -> str:
        classification = results.get(classify.id)
        if classification is None or classification.response is None:
            raise ValueError("missing classify stage result")

        try:
            classify_out = json.loads(classification.response.content)
        except json.JSONDecodeError:
            classify_out = {}

        if classify_out.get("needs_escalation"):
            reason = classify_out.get("escalation_reason", "")
            return (
                "You are an internal support ticket router. The triage system determined "
                "this ticket NEEDS HUMAN ESCALATION.\n\n"
                f"Escalation reason: {reason}\n\n"
                "Your job is to:\n"
                "1. Read the available team descriptions (use the read_team_descriptions tool).\n"
                "2. Create an internal support ticket routed to the appropriate team "
                "(use the send_ticket tool). Include the escalation reason.\n"
                "3. Send an email to the assigned team alerting them to the escalated ticket.\n\n"
                "After completing all steps, provide a brief summary of what was done.\n\n"
                f"Ticket Classification:\n{classification.response.content}\n\n"
                f"Original Ticket:\n{ticket}"
            )

        return (
            "You are an internal support ticket router. The triage system determined "
            "this ticket does NOT need human escalation — it is being resolved "
            "automatically by the policy check and reply stages.\n\n"
            "Your job is to:\n"
            "1. Read the available team descriptions (use the read_team_descriptions tool).\n"
            "2. Send an informational email to the responsible team letting them know "
            "the ticket is being handled automatically.\n\n"
            "After completing all steps, provide a brief summary.\n\n"
            f"Ticket Classification:\n{classification.response.content}\n\n"
            f"Original Ticket:\n{ticket}"
        )

    route_stage.set_prompt_builder(route_prompt_builder)

    # --- Add stages to workflow ---
    all_stages = [classify, policy_stage, reply_stage, route_stage]
    for s in all_stages:
        wf.add_stage(s)

    # --- Stage mapping validation ---
    mapping = ExplicitStageMapping()
    output = mapping.map(StageMappingInput(stages=all_stages, backends=[light, heavy]))
    apply_stage_mapping_output(all_stages, output)
    log.info("Stage mapping validated: %d stages assigned to backends", len(output.assignments))

    # --- Dependencies ---
    wf.add_dependency(policy_stage.id, classify.id)
    wf.add_dependency(reply_stage.id, policy_stage.id)
    wf.add_dependency(route_stage.id, classify.id)

    # --- Execute ---
    log.info("Executing customer support workflow...")
    log.info("  classify ──┬──▶ policy_check ──▶ reply")
    log.info("             └──▶ route_ticket")
    results = wf.execute()

    for sid, name in [
        (classify.id, "classify"),
        (policy_stage.id, "policy_check"),
        (reply_stage.id, "reply"),
        (route_stage.id, "route_ticket"),
    ]:
        result = results.get(sid)
        if result is None:
            continue
        log.info("  %s:", name)
        if result.response:
            content = result.response.content
            if len(content) > 500:
                content = content[:500] + "..."
            log.info("    %s", content)


if __name__ == "__main__":
    ticket_text = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_TICKET
    run(ticket_text)
