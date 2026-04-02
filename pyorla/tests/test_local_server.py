"""Tests for pyorla.local_server."""

from unittest.mock import MagicMock, patch

import pytest

from pyorla.local_server import (
    OrlaBinaryNotFoundError,
    orla_runtime,
    pick_free_port,
    resolve_orla_binary,
)


def test_resolve_orla_binary_explicit():
    assert resolve_orla_binary("/usr/bin/orla") == "/usr/bin/orla"


def test_resolve_orla_binary_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ORLA_BIN", raising=False)
    monkeypatch.setattr("pyorla.local_server.shutil.which", lambda _: None)
    with pytest.raises(OrlaBinaryNotFoundError):
        resolve_orla_binary()


def test_resolve_orla_binary_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORLA_BIN", "/opt/orla/bin/orla")
    assert resolve_orla_binary() == "/opt/orla/bin/orla"


def test_pick_free_port_localhost():
    p = pick_free_port("127.0.0.1")
    assert 1024 <= p <= 65535


def test_orla_runtime_starts_and_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_proc = MagicMock()
    mock_proc.wait = MagicMock(return_value=0)
    mock_proc.terminate = MagicMock()
    mock_popen = MagicMock(return_value=mock_proc)

    with (
        patch("pyorla.local_server.subprocess.Popen", mock_popen),
        patch("pyorla.local_server.resolve_orla_binary", return_value="/fake/orla"),
        patch("pyorla.local_server.pick_free_port", return_value=54321),
        patch("pyorla.local_server._wait_for_health") as mock_wait,
    ):
        mock_wait.side_effect = lambda base_url, **kw: None

        with orla_runtime(orla_bin="/fake/orla", quiet=True) as client:
            assert client.base_url == "http://127.0.0.1:54321"

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    assert args[0] == ["/fake/orla", "serve", "--listen-address", "127.0.0.1:54321"]
    mock_proc.terminate.assert_called_once()
    mock_proc.wait.assert_called()


def test_orla_runtime_passes_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    mock_proc = MagicMock()
    mock_proc.wait = MagicMock(return_value=0)
    mock_proc.terminate = MagicMock()
    mock_popen = MagicMock(return_value=mock_proc)

    with (
        patch("pyorla.local_server.subprocess.Popen", mock_popen),
        patch("pyorla.local_server.resolve_orla_binary", return_value="/fake/orla"),
        patch("pyorla.local_server.pick_free_port", return_value=54323),
        patch("pyorla.local_server._wait_for_health", lambda *a, **k: None),
    ):
        with orla_runtime(quiet=True, timeout=1800) as client:
            assert client._sync.timeout == httpx.Timeout(1800)
            assert client._async.timeout == httpx.Timeout(1800)


def test_orla_runtime_terminates_on_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_proc = MagicMock()
    mock_proc.wait = MagicMock(return_value=0)
    mock_proc.terminate = MagicMock()
    mock_popen = MagicMock(return_value=mock_proc)

    with (
        patch("pyorla.local_server.subprocess.Popen", mock_popen),
        patch("pyorla.local_server.resolve_orla_binary", return_value="/fake/orla"),
        patch("pyorla.local_server.pick_free_port", return_value=54322),
        patch("pyorla.local_server._wait_for_health", lambda *a, **k: None),
    ):
        with pytest.raises(RuntimeError):
            with orla_runtime(quiet=True):
                raise RuntimeError("boom")

    mock_proc.terminate.assert_called_once()


@pytest.mark.skipif(
    not __import__("shutil").which("orla"),
    reason="orla binary not on PATH",
)
def test_orla_runtime_integration() -> None:
    """Opt-in: requires ``orla`` installed; skipped in most CI."""
    with orla_runtime(health_timeout_s=60.0, quiet=True) as client:
        client.health()


def test_wait_for_health_timeout():
    from pyorla.local_server import _wait_for_health

    def always_fail(url: str, timeout: float = 0) -> MagicMock:
        r = MagicMock()
        r.status_code = 503
        return r

    with pytest.raises(TimeoutError, match="did not become healthy"):
        _wait_for_health("http://127.0.0.1:59999", timeout_s=0.25, interval_s=0.05, get=always_fail)
