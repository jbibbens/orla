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

Daemon API: **http://localhost:8081**. When registering backends via the API, use `model_id: "openai:Qwen/Qwen3-4B-Instruct-2507"` to match the model vLLM serves.

### SGLang (GPU)

Defaults to **Qwen3-8B** (open weights, no Hugging Face login required). Override with `SGLANG_MODEL` if needed.

```bash
export HF_TOKEN=your_token   # only for gated models
# Optional: different model (default: Qwen/Qwen3-8B)
export SGLANG_MODEL=Qwen/Qwen3-8B

docker compose -f deploy/docker-compose.sglang.yaml up -d
```

Daemon API: **http://localhost:8081**. When registering backends via the API, use `model_id` to match the model SGLang serves.

## Multi-backend stacks (workflow demo and SWE-bench)

For the **workflow demo** (customer support triage + resolution) and **SWE-bench Lite** experiments you can use either SGLang or vLLM for the heavy and light models:

| Stack        | SGLang compose                               | vLLM compose                                       |
|-------------|------------------------------------------------|----------------------------------------------------|
| Workflow demo | `docker-compose.workflow-demo.yaml`          | `docker-compose.workflow-demo.vllm.yaml`          |
| SWE-bench Lite | `docker-compose.swebench-lite.yaml`        | `docker-compose.swebench-lite.vllm.yaml`          |

- **Workflow demo**: Start the stack, then run the Go client (see [Multi-Agent Workflow](https://orlaserver.github.io/docs/#/research/orla_workflow_customer_support) in the docs). For vLLM, set `VLLM_LIGHT_URL` and `VLLM_HEAVY_URL` to the service URLs the Orla *container* can resolve (e.g. `VLLM_LIGHT_URL=http://vllm-light:8000/v1 VLLM_HEAVY_URL=http://vllm-heavy:8000/v1 go run ./examples/workflow_demo/cmd/workflow_demo`).
- **SWE-bench**: The vLLM compose sets `BACKEND_PROVIDER=vllm` and `VLLM_HEAVY_URL` / `VLLM_LIGHT_URL` for the run container. Three modes are available: `single_shot_baseline`, `single_shot_stage_mapping`, and `single_shot_sjf`. Set `RUN_TARGET` to select the mode. Default is SWE-bench Lite (~300 instances). For full SWE-bench (~2,294 instances), set `FULL_SWE_BENCH=1` when building and running: `FULL_SWE_BENCH=1 docker compose -f deploy/docker-compose.swebench-lite.vllm.yaml build` and `FULL_SWE_BENCH=1 RUN_TARGET=single_shot_baseline docker compose ... up run`. Use `MAX_INSTANCES=50` for a quick subset test. The dataset zips are committed; run `FULL_SWE_BENCH=1 python examples/swe_bench_lite/scripts/prepare_dataset.py` only to regenerate them.

Both vLLM stacks run two vLLM services (heavy on port 8000, light on 8001). Two GPUs are recommended; with one GPU you may need to run the two vLLM containers on different devices or one at a time.

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
