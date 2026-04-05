# pyorla

Python SDK for [Orla](https://github.com/harvard-cns/orla).

## Install

From [PyPI](https://pypi.org/project/pyorla/):

```bash
pip install pyorla
```

With [uv](https://docs.astral.sh/uv/), either install into the active environment:

```bash
uv pip install pyorla
```

Or, if you are already in a uv project i.e a directory with a `pyproject.toml`, add it as a dependency:

```bash
uv add pyorla
```

pyorla talks to a running `orla serve` daemon over HTTP. Install the Orla binary separately (see the [Quickstart](https://orlaserver.github.io/docs/#/README) for Homebrew, pyorla, and related setup).

### Developing pyorla

From a clone of the Orla repo, in the `pyorla` directory:

```bash
uv sync
```

To run checks, use the following make targets from the repository root:

```bash
make pyorla-lint
```

and 

```bash
make pyorla-test
```

### Releasing to PyPI

1. Bump `version` in `pyproject.toml`.
2. Commit and push a tag: `pyorla-vX.Y.Z` (must match the version in `pyproject.toml`).
3. The [pyorla-publish](https://github.com/harvard-cns/orla/blob/main/.github/workflows/pyorla-publish.yml) workflow builds and uploads to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/).

## Remote server

Point `OrlaClient` at a running daemon:

```python
from pyorla import OrlaClient

client = OrlaClient("https://orla.example.com")
client.health()
```

Register backends and run workflows / `Stage` / `ChatOrla` as in the package examples.

## Local server from Python

For development or notebooks, you can spawn `orla serve` on loopback and get a client back (requires the `orla` CLI on `PATH` or `ORLA_BIN`):

```python
from pyorla import orla_runtime

with orla_runtime() as client:
    client.health()
    # register backends, run execute, etc.
```

This starts a subprocess (`orla serve --listen-address 127.0.0.1:<ephemeral-port>`), waits for `/api/v1/health`, then terminates the process when the block exits.

## Colab and remote notebooks

Colab cannot see `localhost` on your laptop. Use either:

- An Orla daemon on a **public URL** (VM, Kubernetes, tunnel), and pass that URL to `OrlaClient`, or
- A **tunnel** (ngrok, Cloudflare Tunnel, etc.) from the machine where `orla serve` runs to a URL you paste into the notebook.
