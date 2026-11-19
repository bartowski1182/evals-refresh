#!/usr/bin/env bash
# MRCR on RedHatAI/GLM-5.3-NVFP4 against an already-running vLLM server.
# Sampling: checkpoint generation_config.json sets temperature 1.0; the base zai-org/GLM-5.3
# generation_config.json also sets top_p 0.95 (RedHatAI's file omits it). No top_k in either.
# Thinking always on; highest effort is max (--reasoning-effort max).
# -M client_timeout raises Inspect's HTTP timeout (default 600s).
set -euo pipefail

MODEL="${MODEL:-RedHatAI/GLM-5.3-NVFP4}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

# KV pool ~950k tokens/DP replica at max_model_len=262144 (~3.62x); 6 keeps
# ~3 concurrent full-window histories per replica after DP load-balance.
inspect eval inspect_evals/mrcr \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" \
  --temperature 1.0 --top-p 0.95 \
  --reasoning-effort max \
  -T max_model_len=262144 \
  -T 'gen_kwargs={"temperature": 1.0, "top_p": 0.95}' \
  --max-connections "${MAX_CONNECTIONS:-6}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
