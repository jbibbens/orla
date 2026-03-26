"""Tests for pyorla.langchain_tools."""

from langchain_core.tools import tool

from pyorla.langchain_tools import tool_from_langchain


@tool
def lc_add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def test_tool_from_langchain_invoke():
    t = tool_from_langchain(lc_add)
    assert t.name == "lc_add"
    assert t.run is not None
    r = t.run({"a": 1, "b": 2})
    assert not r.is_error
    assert r.output_values == {"result": 3}


def test_tool_from_langchain_error():
    t = tool_from_langchain(lc_add)
    assert t.run is not None
    r = t.run({"a": "x", "b": 1})
    assert r.is_error
