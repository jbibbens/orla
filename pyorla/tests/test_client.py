"""Tests for pyorla.client."""

import pytest
import httpx

from pyorla.client import OrlaClient, OrlaError, _parse_execute_response, _raise_http
from pyorla.types import InferenceResponseMetrics


def test_orla_client_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORLA_URL", "http://example:9090")
    c = OrlaClient.from_env()
    assert c.base_url == "http://example:9090"
    c.close()


def test_orla_client_from_env_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORLA_URL", raising=False)
    c = OrlaClient.from_env("http://localhost:9999")
    assert c.base_url == "http://localhost:9999"
    c.close()


def test_raise_http_maps_status() -> None:
    req = httpx.Request("GET", "http://test/api/v1/health")
    resp = httpx.Response(502, request=req, text="bad gateway")
    with pytest.raises(OrlaError) as exc_info:
        _raise_http(resp)
    err = exc_info.value
    assert err.status_code == 502
    assert "bad gateway" in (err.body or "")


def test_raise_http_includes_json_error_field() -> None:
    req = httpx.Request("POST", "http://test/api/v1/execute")
    payload = '{"success":false,"error":"inference failed: model not found"}'
    resp = httpx.Response(500, request=req, text=payload)
    with pytest.raises(OrlaError) as exc_info:
        _raise_http(resp)
    err = exc_info.value
    assert err.status_code == 500
    assert "model not found" in str(err)


def test_orla_error_message_str() -> None:
    e = OrlaError("x", status_code=418)
    assert str(e) == "x"
    assert e.status_code == 418


def test_parse_execute_response_metrics_null() -> None:
    """Orla may send ``metrics: null``; client must not assume a dict."""
    data = {
        "success": True,
        "response": {"content": "ok", "thinking": "", "metrics": None},
    }
    r = _parse_execute_response(data)
    assert r.content == "ok"
    assert r.metrics is None


def test_parse_execute_response_metrics_dict() -> None:
    data = {
        "success": True,
        "response": {
            "content": "",
            "metrics": {"ttft_ms": 12, "prompt_tokens": 3, "completion_tokens": 1},
        },
    }
    r = _parse_execute_response(data)
    assert r.metrics == InferenceResponseMetrics(
        ttft_ms=12,
        tpot_ms=0,
        prompt_tokens=3,
        completion_tokens=1,
        queue_wait_ms=0,
        scheduler_decision_ms=0,
        dispatch_ms=0,
        backend_latency_ms=0,
    )
