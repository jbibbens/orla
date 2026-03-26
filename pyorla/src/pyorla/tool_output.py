"""Map Python return values to JSON-friendly ``ToolResult.output_values``."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from pyorla.tools import ToolSchema


def tool_output_to_values(value: Any) -> ToolSchema:
    """Normalize a tool function return value into a dict for ``ToolResult.output_values``.

    Used by ``@orla_tool`` and ``tool_from_langchain`` so ``ToolResult.to_message_dict``
    can ``json.dumps`` the payload.
    """
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        dumped = value.model_dump(mode="json")
        return dumped if isinstance(dumped, dict) else {"result": dumped}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (str, int, float, bool)):
        return {"result": value}
    try:
        json.dumps(value)
        return {"result": value}
    except (TypeError, ValueError):
        return {"result": str(value)}
