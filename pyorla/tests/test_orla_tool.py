"""Tests for pyorla.tool_decorators.orla_tool."""

from typing import Any, Literal

import pytest
from pydantic import BaseModel

from pyorla.tool_decorators import orla_tool


def test_orla_tool_happy_path():
    @orla_tool
    def add(a: int, b: int) -> dict[str, int]:
        """Add two integers."""
        return {"sum": a + b}

    assert add.name == "add"
    assert "Add two integers" in add.description
    assert add.run is not None
    r = add.run({"a": 2, "b": 3})
    assert not r.is_error
    assert r.output_values == {"sum": 5}


def test_orla_tool_validation_error():
    @orla_tool
    def only_int(x: int) -> dict[str, Any]:
        """Takes an int."""
        return {"x": x}

    assert only_int.run is not None
    r = only_int.run({"x": "not-int"})
    assert r.is_error


def test_orla_tool_scalar_return():
    @orla_tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}"

    assert greet.run is not None
    r = greet.run({"name": "Ada"})
    assert r.output_values == {"result": "Hello, Ada"}


def test_orla_tool_no_args():
    @orla_tool
    def ping() -> dict[str, str]:
        """Health check."""
        return {"status": "ok"}

    assert ping.run is not None
    r = ping.run({})
    assert r.output_values == {"status": "ok"}


def test_orla_tool_explicit_name_description():
    @orla_tool(name="custom", description="Custom desc")
    def f(x: str) -> dict[str, str]:
        """Ignored first line."""
        return {"x": x}

    assert f.name == "custom"
    assert f.description == "Custom desc"


def test_orla_tool_args_schema_override():
    class Args(BaseModel):
        q: str

    @orla_tool(args_schema=Args)
    def search(q: str) -> dict[str, Any]:
        return {"q": q}

    assert search.run is not None
    r = search.run({"q": "hi"})
    assert r.output_values == {"q": "hi"}


def test_orla_tool_runtime_exception():
    @orla_tool
    def boom(x: str) -> dict[str, str]:
        """Boom."""
        raise ValueError("nope")

    assert boom.run is not None
    r = boom.run({"x": "a"})
    assert r.is_error
    assert "nope" in r.error


def test_orla_tool_rejects_async():
    async def af() -> dict[str, str]:
        return {}

    with pytest.raises(TypeError, match="async"):
        orla_tool(af)


def test_orla_tool_rejects_var_kwargs():
    with pytest.raises(TypeError, match="\\*args"):

        @orla_tool
        def bad(**kwargs: Any) -> dict[str, Any]:
            return kwargs


def test_orla_tool_literal_enum_in_schema():
    @orla_tool
    def prio(level: Literal["low", "high"]) -> dict[str, str]:
        """Prio."""
        return {"level": level}

    assert prio.run is not None
    r = prio.run({"level": "low"})
    assert r.output_values == {"level": "low"}
    r2 = prio.run({"level": "medium"})
    assert r2.is_error
