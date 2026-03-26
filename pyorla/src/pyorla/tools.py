"""Tool definitions and runners for client-side tool execution."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


ToolSchema = dict[str, Any]
ToolRunner = Callable[[ToolSchema], "ToolResult"]


@dataclass
class Tool:
    """A callable tool with JSON Schema input and optional runner."""

    name: str
    description: str
    input_schema: ToolSchema
    output_schema: ToolSchema | None = None
    run: ToolRunner | None = None

    def to_mcp(self) -> dict[str, Any]:
        """Return the MCP wire-format dict for the execute request."""
        d: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.output_schema:
            d["outputSchema"] = self.output_schema
        return d


@dataclass
class ToolCall:
    """A parsed tool call from an InferenceResponse."""

    id: str
    name: str
    input_arguments: ToolSchema = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of executing a tool call."""

    id: str = ""
    name: str = ""
    output_values: ToolSchema = field(default_factory=dict)
    error: str = ""
    is_error: bool = False

    def to_message_dict(self) -> dict[str, Any]:
        """Convert to an Orla Message dict (role=tool)."""
        msg: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": self.id,
            "tool_name": self.name,
        }
        if self.is_error:
            prefix = "tool execution error"
            if self.error:
                prefix += ": " + self.error
            msg["content"] = prefix
        else:
            msg["content"] = json.dumps(self.output_values)
        return msg


def new_tool(
    name: str,
    description: str,
    input_schema: ToolSchema,
    run: ToolRunner,
    output_schema: ToolSchema | None = None,
) -> Tool:
    """Create a new :class:`Tool`."""
    return Tool(
        name=name,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        run=run,
    )


def tool_call_from_raw(raw: dict[str, Any]) -> ToolCall:
    """Parse a raw tool call dict from InferenceResponse.tool_calls.

    Expected format is the MCP ``toolCallWithID`` envelope::

        {"id": "...", "method": "tools/call", "params": {"name": "...", "arguments": {...}}}
    """
    call_id = raw.get("id", "")
    params = raw.get("params", raw.get("McpCallToolParams", {}))
    name = params.get("name", raw.get("name", ""))
    arguments = params.get("arguments", raw.get("arguments", {}))
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    return ToolCall(id=call_id, name=name, input_arguments=arguments or {})


def tool_runner_from_schema(
    fn: Callable[[ToolSchema], ToolSchema],
) -> ToolRunner:
    """Wrap a simple dict-to-dict function as a :class:`ToolRunner`."""

    def _runner(input_args: ToolSchema) -> ToolResult:
        try:
            out = fn(input_args)
            return ToolResult(output_values=out or {})
        except Exception as exc:
            return ToolResult(error=str(exc), is_error=True)

    return _runner
