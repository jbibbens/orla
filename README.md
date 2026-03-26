<p align="center">
  <img src="share/orla_banner_no_caption.png" alt="Orla" width="500">
</p>

<p align="center">
  <a href="https://golang.org/"><img src="https://img.shields.io/badge/Go-1.25+-00ADD8?style=flat&logo=go" alt="Go Version"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
  <a href="https://goreportcard.com/report/github.com/harvard-cns/orla"><img src="https://img.shields.io/badge/go%20report-A+-brightgreen.svg?style=flat" alt="Go Report Card"></a>
  <a href="https://www.bestpractices.dev/projects/6573"><img src="https://www.bestpractices.dev/projects/6573/badge" alt="OpenSSF Best Practices"></a>
  <a href="https://github.com/harvard-cns/orla/actions/workflows/build.yml"><img src="https://github.com/harvard-cns/orla/actions/workflows/build.yml/badge.svg" alt="Build"></a>
  <a href="https://github.com/harvard-cns/orla/actions/workflows/pyorla-ci.yml"><img src="https://github.com/harvard-cns/orla/actions/workflows/pyorla-ci.yml/badge.svg" alt="pyorla CI"></a>
  <a href="https://pypi.org/project/pyorla/"><img src="https://img.shields.io/pypi/v/pyorla" alt="pyorla on PyPI"></a>
  <a href="https://pypi.org/project/pyorla/"><img src="https://img.shields.io/pypi/pyversions/pyorla" alt="pyorla Python versions"></a>
  <a href="https://discord.gg/bzKYCFewPT"><img src="https://img.shields.io/badge/Discord-5865F2?style=flat&logo=discord&logoColor=white" alt="Discord"></a>
</p>

Orla is a library for building and running LLM-based agentic systems. Modern agentic applications are workflows that combine multiple LLM calls, tool invocations, and heterogeneous infrastructure. Today, developers often stitch these pieces together manually using orchestration code, LLM serving engines, and tool execution logic.

Orla simplifies this process by separating workflow-level decisions from request execution. Developers define workflows as stages, while Orla handles how those stages are mapped to models and backends, scheduled and executed, and coordinated through shared inference state.

The system exposes three core components: a Stage Mapper for heterogeneous model routing, a Workflow Orchestrator for executing and scheduling stages, and a Memory Manager that manages KV cache across workflow stages.

<p align="center">
  <img src="share/main.gif" alt="Orla CLI and API demo" width="700">
</p>

Installing the orla daemon:

```bash
brew install --cask harvard-cns/orla/orla
```

Installing the orla client SDK:

```bash
pip install pyorla
```

## Documentation

For the complete documentation, go to our website, [https://orlaserver.github.io](https://orlaserver.github.io).