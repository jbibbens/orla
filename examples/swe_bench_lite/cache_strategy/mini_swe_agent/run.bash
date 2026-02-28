#!/usr/bin/env bash
set -e

# Use SGLang's OpenAI-compatible API (same as Orla). LiteLLM uses OPENAI_BASE_URL and openai/<model>.
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:30000/v1}"
# Dummy key for local SGLang (no auth required)
OPENAI_API_KEY="${OPENAI_API_KEY:-sk-no-key-required}"

# Model: openai/ prefix routes to OpenAI-compatible endpoint; name must match what SGLang serves.
# Same model as Orla (Qwen/Qwen3-8B) for comparable runs; use same docker-compose.sglang.yaml.
MODEL="${MINI_SWE_MODEL:-openai/Qwen/Qwen3-8B}"
OUTPUT_DIR="mini_swe_agent_preds"

export OPENAI_BASE_URL OPENAI_API_KEY

exec mini-extra swebench \
  --model "$MODEL" \
  --subset lite \
  --split dev \
  -o "$OUTPUT_DIR" \
  "$@"
