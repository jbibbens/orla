"""Start a local ``orla serve`` subprocess (loopback) for development and tests."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from collections.abc import Generator
from contextlib import contextmanager
from typing import IO, Any

import httpx

from pyorla.client import OrlaClient

_ORLA_BIN_ENV = "ORLA_BIN"
_DEFAULT_HOST = "127.0.0.1"


class OrlaBinaryNotFoundError(FileNotFoundError):
    """Could not find the ``orla`` executable."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or (
                "orla executable not found. Install the Orla CLI from "
                "https://github.com/harvard-cns/orla (e.g. build from source or use "
                "your package manager), ensure `orla` is on PATH, or set ORLA_BIN to "
                "the binary path."
            )
        )


def resolve_orla_binary(explicit: str | None = None) -> str:
    """Return path to ``orla``: ``explicit``, then ``$ORLA_BIN``, then ``shutil.which``."""
    if explicit:
        return explicit
    env = os.environ.get(_ORLA_BIN_ENV)
    if env:
        return env
    found = shutil.which("orla")
    if not found:
        raise OrlaBinaryNotFoundError()
    return found


def pick_free_port(host: str = _DEFAULT_HOST) -> int:
    """Bind an ephemeral TCP port on *host* (small race before ``orla serve`` binds)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_health(
    base_url: str,
    *,
    timeout_s: float = 30.0,
    interval_s: float = 0.1,
    get: Any = None,
) -> None:
    """Poll ``GET /api/v1/health`` until 200 or *timeout_s*."""
    getter = get if get is not None else httpx.get
    deadline = time.monotonic() + timeout_s
    url = f"{base_url.rstrip('/')}/api/v1/health"
    last_exc: BaseException | None = None
    while time.monotonic() < deadline:
        try:
            r = getter(url, timeout=2.0)
            code = getattr(r, "status_code", None)
            if code == 200:
                return
        except BaseException as exc:
            last_exc = exc
        time.sleep(interval_s)
    msg = f"Orla server did not become healthy at {base_url!r} within {timeout_s}s"
    if last_exc is not None:
        msg += f": {last_exc!r}"
    raise TimeoutError(msg)


@contextmanager
def orla_runtime(
    *,
    orla_bin: str | None = None,
    host: str = _DEFAULT_HOST,
    health_timeout_s: float = 30.0,
    quiet: bool = True,
) -> Generator[OrlaClient, None, None]:
    """Run a local Orla server in a subprocess and yield an ``OrlaClient``.

    Picks a free TCP port, runs ``orla serve --listen-address <host>:<port>``, waits for
    ``/api/v1/health``, then yields an ``OrlaClient`` at ``http://<host>:<port>``.
    On context exit the client is closed and the process is terminated.

    This is not an in-process embed: it is **subprocess + loopback HTTP**, same as
    running ``orla serve`` yourself.

    Binary resolution: *orla_bin* argument, ``$ORLA_BIN``, then ``PATH`` (see
    :func:`resolve_orla_binary`).
    """
    bin_path = resolve_orla_binary(orla_bin)
    port = pick_free_port(host)
    addr = f"{host}:{port}"
    base_url = f"http://{addr}"
    cmd = [bin_path, "serve", "--listen-address", addr]
    out_err: int | IO[Any] | None = subprocess.DEVNULL if quiet else None
    proc = subprocess.Popen(cmd, stdout=out_err, stderr=out_err)
    client = OrlaClient(base_url)
    try:
        _wait_for_health(base_url, timeout_s=health_timeout_s)
        yield client
    finally:
        client.close()
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
