"""Customer support triage demo using LangGraph + pyorla.

Same pipeline as run_workflow.py but orchestrated via LangGraph StateGraph:

    classify ──┬──▶ policy_check ──▶ reply
               └──▶ route_ticket

This demonstrates how Orla's scheduling, stage mapping, and KV cache
management integrate with LangGraph's graph execution model.
"""

from __future__ import annotations

import json
import logging
import operator
import os
import sys
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from pyorla import (
    ChatOrla,
    ExplicitStageMapping,
    OrlaClient,
    SchedulingHints,
    StageMappingInput,
    Stage,
    StructuredOutputRequest,
    apply_stage_mapping_output,
    new_ollama_backend,
    new_sglang_backend,
    new_vllm_backend,
)
from pyorla.types import SCHEDULING_POLICY_FCFS, SCHEDULING_POLICY_PRIORITY, EXECUTION_MODE_AGENT_LOOP

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


# ======================================================================
# LangGraph state
# ======================================================================


class WorkflowState(TypedDict, total=False):
    ticket: str
    classify_result: str
    policy_result: str
    reply_result: str
    route_result: str
    completed_stages: Annotated[list[str], operator.add]


# ======================================================================
# Nodes
# ======================================================================

def classify_node(state: WorkflowState, *, classify_llm: ChatOrla) -> dict[str, Any]:
    """Stage 1: classify the ticket (single-shot, structured output)."""
    ticket = state["ticket"]
    prompt = (
        "You are a customer support triage system. Classify this support ticket.\n"
        "Extract the category, product, a one-sentence summary of the core issue, "
        "and what the customer is actually asking for.\n\n"
        "Also decide whether this ticket needs human team escalation. Set needs_escalation "
        "to true if the issue is ambiguous, involves multiple departments, requires manual "
        "verification, or cannot be resolved by automated policy lookup alone.\n\n"
        f"Ticket:\n{ticket}"
    )
    from langchain_core.messages import HumanMessage
    resp = classify_llm.invoke([HumanMessage(content=prompt)])
    return {"classify_result": resp.content, "completed_stages": ["classify"]}


def policy_check_node(state: WorkflowState, *, policy_stage: Stage) -> dict[str, Any]:
    """Stage 2: policy check (agent-loop with read_policy_yaml tool)."""
    ticket = state["ticket"]
    classification = state["classify_result"]
    from pyorla.types import Message

    prompt = (
        "You are a support policy specialist. You have access to a tool called "
        "read_policy_yaml that lets you look up the company's support policy.\n\n"
        "Step 1: Use the read_policy_yaml tool to retrieve the policy for the ticket's category.\n"
        "Step 2: Based on the policy, decide whether to ACCEPT or DENY the customer's request.\n\n"
        f"Ticket Classification:\n{classification}\n\nOriginal Ticket:\n{ticket}"
    )

    messages = [Message(role="user", content=prompt)]
    max_turns = 5
    last_content = ""

    for _ in range(max_turns):
        resp = policy_stage.execute_with_messages(messages)
        last_content = resp.content
        messages.append(Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls))
        if not resp.tool_calls:
            break
        tool_results = policy_stage.run_tool_calls_in_response(resp)
        for tr in tool_results:
            msg_dict = tr.to_message_dict()
            messages.append(Message(
                role="tool",
                content=msg_dict["content"],
                tool_call_id=msg_dict.get("tool_call_id", ""),
                tool_name=msg_dict.get("tool_name", ""),
            ))

    return {"policy_result": last_content, "completed_stages": ["policy_check"]}


def reply_node(state: WorkflowState, *, reply_stage: Stage) -> dict[str, Any]:
    """Stage 3: compose and send reply email (agent-loop with send_email)."""
    ticket = state["ticket"]
    classification = state["classify_result"]
    policy_result = state["policy_result"]

    try:
        classify_data = json.loads(classification)
    except json.JSONDecodeError:
        classify_data = {}

    priority = 8 if classify_data.get("category") in ("billing", "technical") else 5
    reply_stage.set_scheduling_hints(SchedulingHints(priority=priority))

    if classify_data.get("needs_escalation"):
        prompt = (
            "You are a customer support agent. The triage system determined this ticket "
            "NEEDS HUMAN ESCALATION.\n\n"
            "Compose a brief, professional email acknowledging receipt and escalation. "
            "Send it via the send_email tool.\n\n"
            f"Policy Decision:\n{policy_result}\n\n"
            f"Ticket Classification:\n{classification}\n\n"
            f"Original Ticket:\n{ticket}"
        )
    else:
        prompt = (
            "You are a customer support agent. Based on the policy decision, compose a "
            "professional reply and send it using the send_email tool.\n\n"
            f"Policy Decision:\n{policy_result}\n\n"
            f"Ticket Classification:\n{classification}\n\n"
            f"Original Ticket:\n{ticket}"
        )

    from pyorla.types import Message

    messages = [Message(role="user", content=prompt)]
    last_content = ""

    for _ in range(5):
        resp = reply_stage.execute_with_messages(messages)
        last_content = resp.content
        messages.append(Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls))
        if not resp.tool_calls:
            break
        tool_results = reply_stage.run_tool_calls_in_response(resp)
        for tr in tool_results:
            msg_dict = tr.to_message_dict()
            messages.append(Message(
                role="tool",
                content=msg_dict["content"],
                tool_call_id=msg_dict.get("tool_call_id", ""),
                tool_name=msg_dict.get("tool_name", ""),
            ))

    return {"reply_result": last_content, "completed_stages": ["reply"]}


def route_ticket_node(state: WorkflowState, *, route_stage: Stage) -> dict[str, Any]:
    """Stage 4: route ticket to internal team (agent-loop with tools)."""
    ticket = state["ticket"]
    classification = state["classify_result"]

    try:
        classify_out = json.loads(classification)
    except json.JSONDecodeError:
        classify_out = {}

    if classify_out.get("needs_escalation"):
        reason = classify_out.get("escalation_reason", "")
        prompt = (
            "You are an internal support ticket router. This ticket NEEDS HUMAN ESCALATION.\n\n"
            f"Escalation reason: {reason}\n\n"
            "1. read_team_descriptions to find the right team.\n"
            "2. send_ticket to route the ticket.\n"
            "3. send_email to alert the team.\n\n"
            f"Ticket Classification:\n{classification}\n\nOriginal Ticket:\n{ticket}"
        )
    else:
        prompt = (
            "You are an internal support ticket router. This ticket is being resolved "
            "automatically.\n\n"
            "1. read_team_descriptions to find the responsible team.\n"
            "2. send_email to inform them.\n\n"
            f"Ticket Classification:\n{classification}\n\nOriginal Ticket:\n{ticket}"
        )

    from pyorla.types import Message

    messages = [Message(role="user", content=prompt)]
    last_content = ""

    for _ in range(10):
        resp = route_stage.execute_with_messages(messages)
        last_content = resp.content
        messages.append(Message(role="assistant", content=resp.content, tool_calls=resp.tool_calls))
        if not resp.tool_calls:
            break
        tool_results = route_stage.run_tool_calls_in_response(resp)
        for tr in tool_results:
            msg_dict = tr.to_message_dict()
            messages.append(Message(
                role="tool",
                content=msg_dict["content"],
                tool_call_id=msg_dict.get("tool_call_id", ""),
                tool_name=msg_dict.get("tool_name", ""),
            ))

    return {"route_result": last_content, "completed_stages": ["route_ticket"]}


# ======================================================================
# Graph construction
# ======================================================================


def build_graph(
    classify_llm: ChatOrla,
    policy_stage: Stage,
    reply_stage: Stage,
    route_stage: Stage,
) -> StateGraph:
    """Build the LangGraph workflow.

    ::

        classify ──┬──▶ policy_check ──▶ reply ──▶ END
                   └──▶ route_ticket ────────────▶ END
    """
    graph = StateGraph(WorkflowState)

    graph.add_node("classify", lambda s: classify_node(s, classify_llm=classify_llm))
    graph.add_node("policy_check", lambda s: policy_check_node(s, policy_stage=policy_stage))
    graph.add_node("reply", lambda s: reply_node(s, reply_stage=reply_stage))
    graph.add_node("route_ticket", lambda s: route_ticket_node(s, route_stage=route_stage))

    graph.set_entry_point("classify")

    graph.add_edge("classify", "policy_check")
    graph.add_edge("classify", "route_ticket")
    graph.add_edge("policy_check", "reply")
    graph.add_edge("reply", END)
    graph.add_edge("route_ticket", END)

    return graph


# ======================================================================
# Main
# ======================================================================


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

    # --- Stages ---
    classify_stage = Stage("classify", light)
    classify_stage.client = client
    classify_stage.set_max_tokens(512)
    classify_stage.set_temperature(0)
    classify_stage.set_scheduling_policy(SCHEDULING_POLICY_FCFS)
    classify_stage.set_response_format(
        StructuredOutputRequest(name="ticket_classify", schema=CLASSIFY_SCHEMA)
    )
    classify_llm = classify_stage.as_chat_model()

    policy_stage = Stage("policy_check", heavy)
    policy_stage.client = client
    policy_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
    policy_stage.set_max_turns(5)
    policy_stage.set_max_tokens(1024)
    policy_stage.set_temperature(0)
    policy_stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
    policy_stage.set_chat_template_kwargs(no_thinking)
    policy_stage.set_response_format(
        StructuredOutputRequest(name="policy_decision", schema=POLICY_DECISION_SCHEMA)
    )
    policy_stage.add_tool(read_policy_yaml)

    reply_stage = Stage("reply", heavy)
    reply_stage.client = client
    reply_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
    reply_stage.set_max_turns(5)
    reply_stage.set_max_tokens(1024)
    reply_stage.set_temperature(0.3)
    reply_stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
    reply_stage.set_chat_template_kwargs(no_thinking)
    reply_stage.set_response_format(
        StructuredOutputRequest(name="reply_confirmation", schema=REPLY_CONFIRMATION_SCHEMA)
    )
    reply_stage.add_tool(send_email)

    route_stage = Stage("route_ticket", heavy)
    route_stage.client = client
    route_stage.set_execution_mode(EXECUTION_MODE_AGENT_LOOP)
    route_stage.set_max_turns(10)
    route_stage.set_max_tokens(1024)
    route_stage.set_temperature(0)
    route_stage.set_scheduling_policy(SCHEDULING_POLICY_PRIORITY)
    route_stage.set_chat_template_kwargs(no_thinking)
    route_stage.add_tool(send_email)
    route_stage.add_tool(read_team_descriptions)
    route_stage.add_tool(send_ticket)

    # --- Stage mapping ---
    all_stages = [classify_stage, policy_stage, reply_stage, route_stage]
    mapping = ExplicitStageMapping()
    output = mapping.map(StageMappingInput(stages=all_stages, backends=[light, heavy]))
    apply_stage_mapping_output(all_stages, output)
    log.info("Stage mapping validated: %d stages", len(output.assignments))

    # --- Build & run the LangGraph ---
    graph = build_graph(classify_llm, policy_stage, reply_stage, route_stage)
    app = graph.compile()

    log.info("Running LangGraph workflow...")
    log.info("  classify ──┬──▶ policy_check ──▶ reply")
    log.info("             └──▶ route_ticket")

    final_state = app.invoke({"ticket": ticket, "completed_stages": []})

    for key in ["classify_result", "policy_result", "reply_result", "route_result"]:
        value = final_state.get(key, "")
        if value:
            name = key.replace("_result", "")
            content = value[:500] + "..." if len(value) > 500 else value
            log.info("  %s: %s", name, content)


if __name__ == "__main__":
    ticket_text = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_TICKET
    run(ticket_text)
