# Deploying Orla Server with Docker Compose

This directory contains Docker Compose setups to run the Orla agentic server with a single LLM backend. You only need to edit the provided `orla-*.yaml` (or your own) to configure workflows, agent profiles, and models.

## Prerequisites

- Docker and Docker Compose
- For vLLM and SGLang: NVIDIA GPU with drivers and Docker GPU support (e.g. [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux, or Docker Desktop with GPU on supported hosts)
- For Ollama: optional GPU for faster inference; runs on CPU otherwise

### Installing on Linux

1. **Docker Engine and Compose**  
   Install Docker Engine (Compose V2 is included as a plugin). See [Install Docker Engine](https://docs.docker.com/engine/install/) for your distro. Example on Ubuntu/Debian:
   ```bash
   sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin
   sudo usermod -aG docker $USER   # then log out and back in
   ```

2. **NVIDIA GPU support (for vLLM / SGLang)**  
   Install [NVIDIA drivers](https://www.nvidia.com/drivers) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then configure Docker to use the NVIDIA runtime:
   ```bash
   # Ubuntu/Debian (after adding NVIDIA's package repo per the install guide)
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```
   Verify: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

## Quick start

### Ollama (CPU or GPU)

```bash
# From the repo root
docker compose -f deploy/docker-compose.ollama.yaml up -d

# Pull a model (then update deploy/orla-ollama.yaml model if you use a different one)
docker compose -f deploy/docker-compose.ollama.yaml exec ollama ollama pull llama3.2:3b
```

Daemon API: **http://localhost:8081**

### vLLM (GPU)

Defaults to **Qwen3-4B-Instruct-2507** (open weights, no Hugging Face login required). Override with `VLLM_MODEL` if needed.

```bash
# Optional: for gated models
export HF_TOKEN=your_token
# Optional: different model (default: Qwen/Qwen3-4B-Instruct-2507)
export VLLM_MODEL=Qwen/Qwen3-8B

docker compose -f deploy/docker-compose.vllm.yaml up -d
```

Daemon API: **http://localhost:8081**. Update `deploy/orla-vllm.yaml` so `llm_servers[0].model` matches the model vLLM serves (e.g. `openai:Qwen/Qwen3-4B-Instruct-2507`).

### SGLang (GPU)

Defaults to **Qwen3-8B** (open weights, no Hugging Face login required). Override with `SGLANG_MODEL` if needed.

```bash
export HF_TOKEN=your_token   # only for gated models
# Optional: different model (default: Qwen/Qwen3-8B)
export SGLANG_MODEL=Qwen/Qwen3-8B

docker compose -f deploy/docker-compose.sglang.yaml up -d
```

Daemon API: **http://localhost:8081**. Update `deploy/orla-sglang.yaml` so `llm_servers[0].model` matches the model SGLang serves.

## Config files

| Backend | Compose file                  | Orla config (mount)   |
|---------|--------------------------------|------------------------|
| Ollama  | `docker-compose.ollama.yaml`   | `orla-ollama.yaml`     |
| vLLM    | `docker-compose.vllm.yaml`     | `orla-vllm.yaml`       |
| SGLang  | `docker-compose.sglang.yaml`   | `orla-sglang.yaml`     |

Edit the corresponding `orla-*.yaml` to change:

- **LLM server**: `backend.endpoint` (compose service hostname), `model`, `context`, `cache`
- **Agent profiles**: `agent_profiles[].llm_server`, inference options
- **Workflows**: `workflows[].tasks` or `workflows[].graph`

Then restart the orla service so it reloads config:

```bash
docker compose -f deploy/docker-compose.ollama.yaml restart orla
```

## Building the Orla image

The compose files build the Orla daemon image from the repo root Dockerfile. First build:

```bash
docker compose -f deploy/docker-compose.ollama.yaml build orla
```

Or build the image once and reuse:

```bash
docker build -t orla:latest .
```

Then in each compose file you can replace `build: ...` with `image: orla:latest` for the `orla` service.

## Stopping

```bash
docker compose -f deploy/docker-compose.ollama.yaml down
# or .vllm.yaml / .sglang.yaml
```

Add `-v` to remove named volumes (e.g. Ollama models, Hugging Face cache).
