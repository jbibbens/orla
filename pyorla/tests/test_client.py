"""Tests for pyorla.client."""

import pytest
import httpx

from pyorla.client import OrlaClient, OrlaError, _raise_http


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


def test_orla_error_message_str() -> None:
    e = OrlaError("x", status_code=418)
    assert str(e) == "x"
    assert e.status_code == 418
