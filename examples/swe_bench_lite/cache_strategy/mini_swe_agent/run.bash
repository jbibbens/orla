#!/usr/bin/env bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:30000}"

# Model name must match what SGLang serves via Ollama API
MODEL="${MINI_SWE_MODEL:-ollama/qwen3:8b}"
OUTPUT_DIR="mini_swe_agent_preds"

export OLLAMA_HOST

exec mini-extra swebench \
  --model "$MODEL" \
  --subset lite \
  --split dev \
  -o "$OUTPUT_DIR" \
  "$@"
