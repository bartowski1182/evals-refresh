#!/usr/bin/env bash
# MRCR on Qwen/Qwen3.5-27B against an already-running vLLM server.
# Sampling params come from the checkpoint's generation_config.json (temp 0.6, top_p 0.95, top_k 20).
# The model card recommends temp 1.0 for thinking-mode general tasks; rule 2 follows generation_config.json.
# Inspect's vllm provider drops --top-k from the request; MRCR's gen_kwargs forwards it via extra_body.
# Thinking is enabled server-side; Qwen3.5 has no reasoning_effort levels (unlike Qwen3.8).
# MRCR's solver sizes each request's max_tokens from the remaining context window (power-of-2).
# -M client_timeout raises Inspect's HTTP timeout (default 600s).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

# max_model_len nests under text_config; pass the served window explicitly (matches serve.sh).
inspect eval inspect_evals/mrcr \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" \
  --temperature 0.6 --top-p 0.95 --top-k 20 \
  -T max_model_len=262144 \
  -T 'gen_kwargs={"temperature": 0.6, "top_p": 0.95, "top_k": 20}' \
  --max-connections "${MAX_CONNECTIONS:-16}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
