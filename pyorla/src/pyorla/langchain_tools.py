"""Wrap LangChain ``BaseTool`` instances as executable ``pyorla.tools.Tool``."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.tools import BaseTool

from pyorla.tool_output import tool_output_to_values
from pyorla.tools import Tool, ToolResult, ToolSchema


def langchain_input_schema(lc_tool: BaseTool) -> dict[str, Any]:
    """JSON Schema for tool arguments (same shape as LangChain / OpenAI tools)."""
    args = getattr(lc_tool, "args_schema", None)
    if args is not None and not isinstance(args, dict):
        return cast(Any, args).model_json_schema()
    if hasattr(lc_tool, "get_input_schema"):
        return lc_tool.get_input_schema().model_json_schema()
    return {"type": "object", "properties": {}}


def tool_from_langchain(lc_tool: BaseTool) -> Tool:
    """Convert a LangChain tool into a ``pyorla.Tool`` with ``run`` calling ``invoke``."""
    name = lc_tool.name
    description = getattr(lc_tool, "description", None) or ""
    schema = langchain_input_schema(lc_tool)

    def _run(input_args: ToolSchema) -> ToolResult:
        try:
            out = lc_tool.invoke(input_args)
            return ToolResult(output_values=tool_output_to_values(out))
        except Exception as exc:
            return ToolResult(error=str(exc), is_error=True)

    return Tool(
        name=name,
        description=description,
        input_schema=schema,
        run=_run,
    )
