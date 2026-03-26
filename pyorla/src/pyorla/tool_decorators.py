"""Build ``pyorla.tools.Tool`` from Python functions (``@orla_tool``)."""

from __future__ import annotations

import asyncio
import inspect
import sys
from collections.abc import Callable
from typing import Any, TypeVar, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, create_model


class _OrlaToolArgsBase(BaseModel):
    model_config = ConfigDict(extra="ignore")

from pyorla.tool_output import tool_output_to_values
from pyorla.tools import Tool, ToolResult, ToolSchema

F = TypeVar("F", bound=Callable[..., Any])


def _callable_label(fn: Callable[..., Any]) -> str:
    q = getattr(fn, "__qualname__", None)
    if isinstance(q, str):
        return q
    n = getattr(fn, "__name__", None)
    if isinstance(n, str):
        return n
    return repr(fn)


def _first_line(doc: str | None) -> str:
    if not doc:
        return ""
    return doc.strip().split("\n", 1)[0].strip()


def _signature_to_model(fn: Callable[..., Any]) -> type[BaseModel]:
    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}
    mod = sys.modules.get(fn.__module__)
    globalns = mod.__dict__ if mod is not None else {}
    try:
        hints = get_type_hints(fn, globalns=globalns, include_extras=True)
    except Exception:
        hints = {}
    for pname, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise TypeError(
                f"orla_tool: *args and **kwargs are not supported "
                f"({_callable_label(fn)!r} parameter {pname!r})"
            )
        ann: Any
        if pname in hints:
            ann = hints[pname]
        elif param.annotation is not inspect.Parameter.empty:
            ann = param.annotation
        else:
            ann = Any
        if param.default is inspect.Parameter.empty:
            fields[pname] = (ann, ...)
        else:
            fields[pname] = (ann, param.default)
    model_name = f"_{getattr(fn, '__name__', 'tool')}_OrlaToolArgs"
    if not fields:
        return create_model(model_name, __base__=_OrlaToolArgsBase)
    return create_model(model_name, __base__=_OrlaToolArgsBase, **fields)


def _build_tool_from_function(
    fn: Callable[..., Any],
    *,
    name: str | None,
    description: str | None,
    args_schema: type[BaseModel] | None,
) -> Tool:
    if asyncio.iscoroutinefunction(fn):
        raise TypeError(
            f"orla_tool: async functions are not supported ({_callable_label(fn)!r})"
        )
    t_name = name if name is not None else getattr(fn, "__name__", "tool")
    t_desc = description if description is not None else _first_line(inspect.getdoc(fn))
    input_model = args_schema if args_schema is not None else _signature_to_model(fn)
    schema = input_model.model_json_schema()

    def _run(input_args: ToolSchema) -> ToolResult:
        try:
            validated = input_model.model_validate(input_args)
            kwargs = validated.model_dump()
            out = fn(**kwargs)
            return ToolResult(output_values=tool_output_to_values(out))
        except Exception as exc:
            return ToolResult(error=str(exc), is_error=True)

    return Tool(
        name=t_name,
        description=t_desc,
        input_schema=schema,
        run=_run,
    )


@overload
def orla_tool(fn: F) -> Tool: ...


@overload
def orla_tool(
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
) -> Callable[[F], Tool]: ...


def orla_tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    args_schema: type[BaseModel] | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorate a **sync** function to produce a ``Tool`` with schema and ``run`` set.

    The LLM-facing JSON schema is derived from type annotations (via Pydantic) unless
    ``args_schema`` is provided. The first line of the docstring becomes the default
    description.

    Example::

        @orla_tool
        def search(query: str, limit: int = 10) -> dict[str, Any]:
            \"\"\"Search the knowledge base.\"\"\"
            return {\"hits\": []}
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        return _build_tool_from_function(
            f, name=name, description=description, args_schema=args_schema
        )

    if fn is not None:
        return decorator(fn)
    return decorator
