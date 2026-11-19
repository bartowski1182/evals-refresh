#!/usr/bin/env bash
# MRCR on Qwen/Qwen3-30B-A3B-Instruct-2507 against an already-running vLLM server.
# Sampling params come from the checkpoint's generation_config.json (temp 0.7, top_p 0.8, top_k 20).
# Inspect's vllm provider drops --top-k from the request; MRCR's gen_kwargs forwards it (and the
# other sampling fields) via extra_body so the request carries the full generation_config.
# This checkpoint is Instruct-2507 (non-thinking): no --reasoning-effort.
# MRCR's solver sizes each request's max_tokens from the remaining context window (power-of-2), so
# no --max-tokens is set here.
# -M client_timeout raises Inspect's HTTP timeout (default 600s): long prompts take longer than
# 10 min and would otherwise be cut off and retried from scratch forever.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

# max_model_len is top-level in this checkpoint's config.json (262144); pass it explicitly so the
# filter matches the served window even if HF auto-detect races the server.
inspect eval inspect_evals/mrcr \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" \
  --temperature 0.7 --top-p 0.8 --top-k 20 \
  -T max_model_len=262144 \
  -T 'gen_kwargs={"temperature": 0.7, "top_p": 0.8, "top_k": 20}' \
  --max-connections "${MAX_CONNECTIONS:-16}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
