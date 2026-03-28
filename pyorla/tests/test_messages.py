"""Tests for pyorla.messages."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from pyorla.messages import (
    _lc_tool_call_to_orla_wire,
    langchain_to_orla,
    orla_response_to_ai_message,
    orla_to_langchain,
)
from pyorla.types import InferenceResponse, InferenceResponseMetrics, Message


def test_langchain_to_orla_basic():
    msgs = [
        SystemMessage(content="You are helpful"),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there"),
    ]
    orla_msgs = langchain_to_orla(msgs)
    assert len(orla_msgs) == 3
    assert orla_msgs[0].role == "system"
    assert orla_msgs[1].role == "user"
    assert orla_msgs[2].role == "assistant"


def test_langchain_to_orla_tool_calls():
    ai = AIMessage(
        content="",
        tool_calls=[{"id": "tc-1", "name": "search", "args": {"q": "hello"}}],
    )
    orla_msgs = langchain_to_orla([ai])
    assert len(orla_msgs) == 1
    assert orla_msgs[0].role == "assistant"
    assert len(orla_msgs[0].tool_calls) == 1
    tc = orla_msgs[0].tool_calls[0]
    assert tc["McpCallToolParams"]["name"] == "search"
    assert tc["McpCallToolParams"]["arguments"] == {"q": "hello"}


def test_lc_tool_call_to_orla_wire_openai_function_shape():
    """API payloads sometimes use function.name / function.arguments (JSON string)."""
    tc = {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "add", "arguments": '{"a":3,"b":4}'},
    }
    out = _lc_tool_call_to_orla_wire(tc)
    assert out["McpCallToolParams"]["name"] == "add"
    assert out["McpCallToolParams"]["arguments"] == {"a": 3, "b": 4}


def test_langchain_to_orla_tool_message():
    tm = ToolMessage(content='{"result": 42}', tool_call_id="tc-1", name="search")
    orla_msgs = langchain_to_orla([tm])
    assert orla_msgs[0].role == "tool"
    assert orla_msgs[0].tool_call_id == "tc-1"
    assert orla_msgs[0].tool_name == "search"


def test_orla_response_to_ai_message():
    resp = InferenceResponse(
        content="Hello world",
        metrics=InferenceResponseMetrics(ttft_ms=50, prompt_tokens=10, completion_tokens=5),
    )
    ai = orla_response_to_ai_message(resp)
    assert ai.content == "Hello world"
    assert ai.response_metadata["ttft_ms"] == 50
    assert ai.response_metadata["prompt_tokens"] == 10


def test_orla_response_to_ai_message_with_tool_calls():
    resp = InferenceResponse(
        content="",
        tool_calls=[
            {"id": "tc-1", "params": {"name": "greet", "arguments": {"x": 1}}},
        ],
    )
    ai = orla_response_to_ai_message(resp)
    assert len(ai.tool_calls) == 1
    assert ai.tool_calls[0]["name"] == "greet"
    assert ai.tool_calls[0]["args"] == {"x": 1}


def test_orla_to_langchain_roundtrip():
    orla_msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="tool", content="result", tool_call_id="tc-1", tool_name="fn"),
    ]
    lc_msgs = orla_to_langchain(orla_msgs)
    assert isinstance(lc_msgs[0], SystemMessage)
    assert isinstance(lc_msgs[1], HumanMessage)
    assert isinstance(lc_msgs[2], AIMessage)
    assert isinstance(lc_msgs[3], ToolMessage)
    assert lc_msgs[3].tool_call_id == "tc-1"
