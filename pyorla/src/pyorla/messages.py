"""LangChain <-> Orla message conversion."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from pyorla.types import InferenceResponse, Message


# ======================================================================
# LangChain -> Orla
# ======================================================================


def _tool_call_args_from_lc(tc: dict[str, Any]) -> Any:
    """Normalize LangChain / OpenAI tool call payloads to argument object for Orla."""
    if "args" in tc:
        return tc["args"]
    if "arguments" in tc:
        a = tc["arguments"]
        if isinstance(a, str):
            try:
                return json.loads(a) if a.strip() else {}
            except json.JSONDecodeError:
                return {}
        return a if a is not None else {}
    fn = tc.get("function")
    if isinstance(fn, dict):
        raw = fn.get("arguments", "{}")
        if isinstance(raw, str):
            try:
                return json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError:
                return {}
        return raw if raw is not None else {}
    return {}


def _tool_call_name_from_lc(tc: dict[str, Any]) -> str:
    if tc.get("name"):
        return str(tc["name"])
    fn = tc.get("function")
    if isinstance(fn, dict) and fn.get("name"):
        return str(fn["name"])
    return ""


def _lc_tool_call_to_orla_wire(tc: dict[str, Any]) -> dict[str, Any]:
    """Build tool_calls[] entry matching Go's ``ToolCallWithID`` JSON (``McpCallToolParams``)."""
    name = _tool_call_name_from_lc(tc)
    args = _tool_call_args_from_lc(tc)
    return {
        "id": str(tc.get("id", "")),
        "McpCallToolParams": {
            "name": name,
            "arguments": args if args is not None else {},
        },
    }


def langchain_to_orla(msgs: Sequence[BaseMessage]) -> list[Message]:
    """Convert a list of LangChain messages to Orla Messages."""
    return [_lc_msg_to_orla(m) for m in msgs]


def _lc_msg_to_orla(msg: BaseMessage) -> Message:
    if isinstance(msg, SystemMessage):
        return Message(role="system", content=str(msg.content))

    if isinstance(msg, HumanMessage):
        return Message(role="user", content=str(msg.content))

    if isinstance(msg, AIMessage):
        orla_msg = Message(role="assistant", content=str(msg.content))
        if msg.tool_calls:
            orla_msg.tool_calls = [
                _lc_tool_call_to_orla_wire(tc if isinstance(tc, dict) else dict(tc))
                for tc in msg.tool_calls
            ]
        return orla_msg

    if isinstance(msg, ToolMessage):
        return Message(
            role="tool",
            content=str(msg.content),
            tool_call_id=msg.tool_call_id or "",
            tool_name=msg.name or "",
        )

    return Message(role=msg.type, content=str(msg.content))


# ======================================================================
# Orla -> LangChain
# ======================================================================


def orla_response_to_ai_message(resp: InferenceResponse) -> AIMessage:
    """Convert an Orla InferenceResponse to a LangChain AIMessage."""
    tool_calls: list[dict[str, Any]] = []
    for raw_tc in resp.tool_calls or []:
        tc = _parse_raw_tool_call(raw_tc)
        if tc is not None:
            tool_calls.append(tc)

    kwargs: dict[str, Any] = {}
    if resp.metrics:
        meta: dict[str, Any] = {
            "prompt_tokens": resp.metrics.prompt_tokens,
            "completion_tokens": resp.metrics.completion_tokens,
            "queue_wait_ms": resp.metrics.queue_wait_ms,
        }
        if resp.metrics.ttft_ms is not None:
            meta["ttft_ms"] = resp.metrics.ttft_ms
        if resp.metrics.tpot_ms is not None:
            meta["tpot_ms"] = resp.metrics.tpot_ms
        if resp.metrics.backend_latency_ms is not None:
            meta["backend_latency_ms"] = resp.metrics.backend_latency_ms
        kwargs["response_metadata"] = meta

    return AIMessage(
        content=resp.content,
        tool_calls=tool_calls,
        **kwargs,
    )


def _parse_raw_tool_call(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a raw tool call into LangChain tool_call format."""
    call_id = raw.get("id", "")
    params = raw.get("McpCallToolParams", raw.get("params", {}))
    name = params.get("name", raw.get("name", ""))
    arguments = params.get("arguments", raw.get("arguments", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not name:
        return None
    return {"id": call_id, "name": name, "args": arguments or {}}


def orla_to_langchain(msgs: list[Message]) -> list[BaseMessage]:
    """Convert a list of Orla Messages to LangChain messages."""
    return [_orla_msg_to_lc(m) for m in msgs]


def _orla_msg_to_lc(msg: Message) -> BaseMessage:
    if msg.role == "system":
        return SystemMessage(content=msg.content)
    if msg.role == "user":
        return HumanMessage(content=msg.content)
    if msg.role == "assistant":
        tool_calls: list[dict[str, Any]] = []
        for raw_tc in msg.tool_calls:
            tc = _parse_raw_tool_call(raw_tc)
            if tc is not None:
                tool_calls.append(tc)
        return AIMessage(content=msg.content, tool_calls=tool_calls)
    if msg.role == "tool":
        return ToolMessage(
            content=msg.content,
            tool_call_id=msg.tool_call_id,
            name=msg.tool_name,
        )
    return HumanMessage(content=msg.content)
