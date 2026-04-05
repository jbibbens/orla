"""Document analysis pipeline with access control using LangGraph + Orla.

Demonstrates Orla's access control policies in a two-stage document analysis
pipeline where different teams have different permissions for backends, tools,
and data.

Pipeline
--------
1. **Extract** — read a document and extract key facts (may contain PII).
2. **Summarize** — produce a concise summary from the extracted facts.

Backends (all open-weight on Bedrock Mantle, overridable via env vars):

- **cheap** — Ministral 3B, low cost (quality 0.3)
- **mid** — Qwen3 32B, moderate cost (quality 0.6)
- **strong** — Qwen3 235B MoE, high cost, external API (quality 0.9)

Teams and their permissions:

- **interns** — cheap model only, no ``query_hr_database`` tool
- **engineering** — cheap and mid models, all tools
- **research** — all models, all tools

Data label propagation:

When the extract stage processes a document containing PII, the summarize
stage inherits the PII label through the registered DAG edge. If the
summarize stage targets an external backend, the daemon blocks it — even
though no one explicitly labeled the summarize stage.

Running
-------
::

    cd pyorla
    uv sync --group examples
    uv run python examples/access_control_demo/run.py

Override the endpoint::

    OPENAI_BASE_URL=http://localhost:8000/v1 uv run python examples/access_control_demo/run.py
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from pyorla import (
    AccessPolicy,
    LLMBackend,
    OrlaClient,
    OrlaError,
    Stage,
    orla_runtime,
)
from pyorla.types import (
    ACCESS_ACTION_ALLOW,
    ACCESS_ACTION_DENY,
    CostModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MANTLE_BASE_URL = "https://bedrock-mantle.us-east-2.api.aws/v1"
DEFAULT_CHEAP_MODEL = "openai:mistral.ministral-3-3b-instruct"
DEFAULT_MID_MODEL = "openai:qwen.qwen3-32b-v1:0"
DEFAULT_STRONG_MODEL = "openai:qwen.qwen3-235b-a22b-2507-v1:0"

SAMPLE_DOCUMENT = """\
EMPLOYEE RECORD — CONFIDENTIAL
Name: Jane Doe
SSN: 123-45-6789
Department: Engineering
Salary: $185,000
Performance: Exceeds expectations. Promoted to Staff Engineer in Q3.
Notes: Requested transfer to the London office starting January 2027.
"""


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def query_hr_database(employee_name: str) -> dict[str, str]:
    """Query the HR database for an employee's record. Restricted to HR team."""
    return {
        "name": employee_name,
        "department": "Engineering",
        "status": "active",
        "manager": "Alice Smith",
    }


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def register_backends(client: OrlaClient) -> tuple[LLMBackend, LLMBackend, LLMBackend]:
    api_key_env = "OPENAI_API_KEY"
    base_url = _env("OPENAI_BASE_URL", MANTLE_BASE_URL)

    cheap = LLMBackend(
        name="cheap",
        endpoint=base_url,
        type="openai",
        model_id=_env("CHEAP_MODEL", DEFAULT_CHEAP_MODEL),
        api_key_env_var=api_key_env,
        quality=0.30,
        cost_model=CostModel(input_cost_per_mtoken=0.10, output_cost_per_mtoken=0.10),
    )
    mid = LLMBackend(
        name="mid",
        endpoint=base_url,
        type="openai",
        model_id=_env("MID_MODEL", DEFAULT_MID_MODEL),
        api_key_env_var=api_key_env,
        quality=0.60,
        cost_model=CostModel(input_cost_per_mtoken=0.15, output_cost_per_mtoken=0.60),
    )
    strong = LLMBackend(
        name="strong",
        endpoint=base_url,
        type="openai",
        model_id=_env("STRONG_MODEL", DEFAULT_STRONG_MODEL),
        api_key_env_var=api_key_env,
        quality=0.90,
        cost_model=CostModel(input_cost_per_mtoken=0.53, output_cost_per_mtoken=2.66),
    )
    for b in (cheap, mid, strong):
        client.register_backend(b)
    return cheap, mid, strong


def install_policies(client: OrlaClient) -> None:
    """Install access control policies."""

    # --- Model access ---
    # Interns: allow only cheap. No deny needed — once a subject has any policy,
    # resources without an explicit allow are denied automatically.
    client.add_policy(AccessPolicy(
        name="intern-allow-cheap",
        subjects=["tenant:interns"],
        resources=["backend:cheap"],
        action=ACCESS_ACTION_ALLOW,
    ))
    # Engineering: allow all backends, then deny strong.
    client.add_policy(AccessPolicy(
        name="eng-allow-all",
        subjects=["tenant:engineering"],
        resources=["backend:*"],
        action=ACCESS_ACTION_ALLOW,
    ))
    client.add_policy(AccessPolicy(
        name="eng-deny-strong",
        subjects=["tenant:engineering"],
        resources=["backend:strong"],
        action=ACCESS_ACTION_DENY,
    ))

    # --- Tool access ---
    # Interns can use all tools except the HR database.
    client.add_policy(AccessPolicy(
        name="intern-allow-all-tools",
        subjects=["tenant:interns"],
        resources=["tool:*"],
        action=ACCESS_ACTION_ALLOW,
    ))
    client.add_policy(AccessPolicy(
        name="intern-no-hr-db",
        subjects=["tenant:interns"],
        resources=["tool:query_hr_database"],
        action=ACCESS_ACTION_DENY,
    ))

    # --- Data labels ---
    # PII cannot flow to the strong (external) backend.
    client.add_policy(AccessPolicy(
        name="pii-no-external",
        subjects=["backend:strong"],
        resources=["data:pii"],
        action=ACCESS_ACTION_DENY,
    ))


# ---------------------------------------------------------------------------
# LangGraph pipeline
# ---------------------------------------------------------------------------


@dataclass
class DocState:
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)


def build_pipeline(
    client: OrlaClient,
    extract_backend: LLMBackend,
    summarize_backend: LLMBackend,
    *,
    tenant: str,
    data_labels: list[str] | None = None,
    tools: list[Any] | None = None,
) -> Any:
    """Build an extract → summarize LangGraph pipeline.

    When tools are provided, the extract node can call them. A ToolNode
    executes the calls and loops back to extract until the model responds
    with text, then the pipeline continues to summarize.
    """

    extract_stage = Stage("extract", extract_backend)
    extract_stage.client = client
    extract_stage.set_tags({"tenant": tenant})
    extract_stage.set_temperature(0.0)
    extract_stage.set_max_tokens(512)
    if data_labels:
        extract_stage.set_data_labels(data_labels)

    summarize_stage = Stage("summarize", summarize_backend)
    summarize_stage.client = client
    summarize_stage.set_tags({"tenant": tenant})
    summarize_stage.set_temperature(0.3)
    summarize_stage.set_max_tokens(256)

    extract_llm = extract_stage.as_chat_model()
    if tools:
        extract_llm = extract_llm.bind_tools(tools)
    summarize_llm = summarize_stage.as_chat_model()

    def extract_node(state: DocState) -> dict[str, Any]:
        reply = extract_llm.invoke([
            SystemMessage(
                content="Extract the key facts from this document as bullet points. "
                "Use any available tools to look up additional information if needed."
            ),
            HumanMessage(content=str(state.messages[-1].content)),
        ])
        return {"messages": [reply]}

    def should_use_tools(state: DocState) -> str:
        last = state.messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "summarize"

    def summarize_node(state: DocState) -> dict[str, Any]:
        # Find the last AI message with text content (skip tool call messages).
        facts = ""
        for m in reversed(state.messages):
            if isinstance(m, AIMessage) and m.content and not m.tool_calls:
                facts = str(m.content)
                break
        reply = summarize_llm.invoke([
            SystemMessage(content="Write a one-paragraph summary from these facts."),
            HumanMessage(content=facts),
        ])
        return {"messages": [reply]}

    g = StateGraph(DocState)
    g.add_node("extract", extract_node)
    g.add_node("summarize", summarize_node)

    if tools:
        tool_node = ToolNode(tools)
        g.add_node("tools", tool_node)
        g.add_edge(START, "extract")
        g.add_conditional_edges("extract", should_use_tools, {"tools": "tools", "summarize": "summarize"})
        g.add_edge("tools", "extract")
        g.add_edge("summarize", END)
    else:
        g.add_edge(START, "extract")
        g.add_edge("extract", "summarize")
        g.add_edge("summarize", END)

    compiled = g.compile()

    # Register DAG with Orla for data label propagation.
    stages = {"extract": extract_stage, "summarize": summarize_stage}
    client.register_workflow_from_langgraph(
        f"wf-{extract_stage.id}",
        compiled,
        stages,
    )

    return compiled


def build_sanitized_pipeline(
    client: OrlaClient,
    extract_backend: LLMBackend,
    sanitize_backend: LLMBackend,
    summarize_backend: LLMBackend,
    *,
    tenant: str,
) -> Any:
    """Build an extract → sanitize → summarize pipeline with declassification.

    The sanitize stage declassifies PII, so the summarize stage downstream
    does NOT inherit the PII label and can use an external backend.
    """

    extract_stage = Stage("extract", extract_backend)
    extract_stage.client = client
    extract_stage.set_tags({"tenant": tenant})
    extract_stage.set_temperature(0.0)
    extract_stage.set_max_tokens(512)
    extract_stage.set_data_labels(["pii"])

    sanitize_stage = Stage("sanitize", sanitize_backend)
    sanitize_stage.client = client
    sanitize_stage.set_tags({"tenant": tenant})
    sanitize_stage.set_temperature(0.0)
    sanitize_stage.set_max_tokens(512)
    sanitize_stage.set_declassifies(["pii"])

    summarize_stage = Stage("summarize", summarize_backend)
    summarize_stage.client = client
    summarize_stage.set_tags({"tenant": tenant})
    summarize_stage.set_temperature(0.3)
    summarize_stage.set_max_tokens(256)

    extract_llm = extract_stage.as_chat_model()
    sanitize_llm = sanitize_stage.as_chat_model()
    summarize_llm = summarize_stage.as_chat_model()

    def extract_node(state: DocState) -> dict[str, Any]:
        reply = extract_llm.invoke([
            SystemMessage(content="Extract the key facts from this document as bullet points."),
            HumanMessage(content=str(state.messages[-1].content)),
        ])
        return {"messages": [reply]}

    def sanitize_node(state: DocState) -> dict[str, Any]:
        facts = str(state.messages[-1].content)
        reply = sanitize_llm.invoke([
            SystemMessage(
                content="Remove all personally identifiable information (names, SSNs, emails) "
                "from the following text. Replace them with [REDACTED]."
            ),
            HumanMessage(content=facts),
        ])
        return {"messages": [reply]}

    def summarize_node(state: DocState) -> dict[str, Any]:
        facts = str(state.messages[-1].content)
        reply = summarize_llm.invoke([
            SystemMessage(content="Write a one-paragraph summary from these facts."),
            HumanMessage(content=facts),
        ])
        return {"messages": [reply]}

    g = StateGraph(DocState)
    g.add_node("extract", extract_node)
    g.add_node("sanitize", sanitize_node)
    g.add_node("summarize", summarize_node)
    g.add_edge(START, "extract")
    g.add_edge("extract", "sanitize")
    g.add_edge("sanitize", "summarize")
    g.add_edge("summarize", END)
    compiled = g.compile()

    client.register_workflow_from_langgraph(
        f"wf-{extract_stage.id}",
        compiled,
        {"extract": extract_stage, "sanitize": sanitize_stage, "summarize": summarize_stage},
    )

    return compiled


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_sanitized_scenario(
    label: str,
    client: OrlaClient,
    extract_be: LLMBackend,
    sanitize_be: LLMBackend,
    summarize_be: LLMBackend,
    *,
    tenant: str,
    document: str,
) -> None:
    log.info("  [%s] tenant=%s, extract=%s, sanitize=%s, summarize=%s",
             label, tenant, extract_be.name, sanitize_be.name, summarize_be.name)
    try:
        pipeline = build_sanitized_pipeline(
            client, extract_be, sanitize_be, summarize_be, tenant=tenant,
        )
        result = pipeline.invoke({"messages": [HumanMessage(content=document)]})
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        answer = str(ai_msgs[-1].content)[:120].replace("\n", " ") if ai_msgs else "(empty)"
        log.info("    ALLOWED — %s...", answer)
    except (OrlaError, Exception) as e:
        msg = str(e)
        if any(k in msg for k in ("access denied", "tool access denied", "data access denied", "403")):
            reason = msg.split(": ", 1)[-1][:150] if ": " in msg else msg[:150]
            log.info("    DENIED  — %s", reason)
        else:
            log.info("    ERROR   — %s", msg[:150])


def run_scenario(
    label: str,
    client: OrlaClient,
    extract_be: LLMBackend,
    summarize_be: LLMBackend,
    *,
    tenant: str,
    document: str,
    data_labels: list[str] | None = None,
    tools: list[Any] | None = None,
) -> None:
    log.info("  [%s] tenant=%s, extract=%s, summarize=%s",
             label, tenant, extract_be.name, summarize_be.name)
    try:
        pipeline = build_pipeline(
            client, extract_be, summarize_be,
            tenant=tenant, data_labels=data_labels, tools=tools,
        )
        result = pipeline.invoke({"messages": [HumanMessage(content=document)]})
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        answer = str(ai_msgs[-1].content)[:120].replace("\n", " ") if ai_msgs else "(empty)"
        log.info("    ALLOWED — %s...", answer)
    except (OrlaError, Exception) as e:
        msg = str(e)
        if any(k in msg for k in ("access denied", "tool access denied", "data access denied", "403")):
            reason = msg.split(": ", 1)[-1][:150] if ": " in msg else msg[:150]
            log.info("    DENIED  — %s", reason)
        else:
            log.info("    ERROR   — %s", msg[:150])


def run() -> None:
    with orla_runtime(quiet=True) as client:
        cheap, mid, strong = register_backends(client)
        install_policies(client)

        log.info("Installed policies:")
        for p in client.list_policies():
            log.info("  %-25s %-5s subjects=%-22s resources=%s",
                     p.name, p.action, p.subjects, p.resources)

        # --- 1. Model access control ---
        log.info("")
        log.info("=== Model access ===")
        run_scenario("intern→cheap",    client, cheap, cheap,  tenant="interns",     document=SAMPLE_DOCUMENT)
        run_scenario("intern→strong",   client, cheap, strong, tenant="interns",     document=SAMPLE_DOCUMENT)
        run_scenario("eng→mid",         client, cheap, mid,    tenant="engineering", document=SAMPLE_DOCUMENT)
        run_scenario("eng→strong",      client, cheap, strong, tenant="engineering", document=SAMPLE_DOCUMENT)
        run_scenario("research→strong", client, cheap, strong, tenant="research",    document=SAMPLE_DOCUMENT)

        # --- 2. Tool access control ---
        log.info("")
        log.info("=== Tool access ===")
        run_scenario("intern+hr_db",  client, cheap, cheap, tenant="interns",     document=SAMPLE_DOCUMENT, tools=[query_hr_database])
        run_scenario("eng+hr_db",     client, cheap, mid,   tenant="engineering", document=SAMPLE_DOCUMENT, tools=[query_hr_database])

        # --- 3. Data label propagation ---
        # Extract processes PII. Summarize inherits the label via the DAG.
        log.info("")
        log.info("=== Data label propagation ===")
        run_scenario("pii→mid",    client, cheap, mid,    tenant="engineering", document=SAMPLE_DOCUMENT, data_labels=["pii"])
        run_scenario("pii→strong", client, cheap, strong, tenant="research",    document=SAMPLE_DOCUMENT, data_labels=["pii"])

        # --- 4. Declassification ---
        # A sanitize stage strips PII before summarization.
        # extract(pii) → sanitize(declassifies pii) → summarize on strong (allowed!)
        log.info("")
        log.info("=== Declassification ===")
        run_sanitized_scenario(
            "pii→sanitize→strong", client, cheap, mid, strong,
            tenant="research", document=SAMPLE_DOCUMENT,
        )

        log.info("")
        log.info("Done.")


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        sys.exit(130)
