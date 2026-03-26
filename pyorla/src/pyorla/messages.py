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
                {
                    "id": tc["id"],
                    "method": "tools/call",
                    "params": {
                        "name": tc["name"],
                        "arguments": tc["args"],
                    },
                }
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
        kwargs["response_metadata"] = {
            "ttft_ms": resp.metrics.ttft_ms,
            "tpot_ms": resp.metrics.tpot_ms,
            "prompt_tokens": resp.metrics.prompt_tokens,
            "completion_tokens": resp.metrics.completion_tokens,
            "queue_wait_ms": resp.metrics.queue_wait_ms,
            "backend_latency_ms": resp.metrics.backend_latency_ms,
        }

    return AIMessage(
        content=resp.content,
        tool_calls=tool_calls,
        **kwargs,
    )


def _parse_raw_tool_call(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Parse a raw tool call into LangChain tool_call format."""
    call_id = raw.get("id", "")
    params = raw.get("params", raw.get("McpCallToolParams", {}))
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
