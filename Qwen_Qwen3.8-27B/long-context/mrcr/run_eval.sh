#!/usr/bin/env bash
# MRCR on Qwen/Qwen3.8-27B against an already-running vLLM server.
# Sampling params come from the checkpoint's generation_config.json (temp 1.0, top_p 0.95, top_k 20).
# Inspect's vllm provider drops --top-k from the request; MRCR's gen_kwargs forwards it (and the
# other sampling fields) via extra_body so the request carries the full generation_config.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (see the recipe).
# MRCR's solver sizes each request's max_tokens from the remaining context window (power-of-2), so
# no --max-tokens is set here.
# -M client_timeout raises Inspect's HTTP timeout (default 600s): long prompts and reasoning traces
# take longer than 10 min and would otherwise be cut off and retried from scratch forever.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

# max_model_len: this checkpoint nests max_position_embeddings under text_config, so
# MRCR's HF-config auto-detect misses it; pass the served window explicitly (matches serve.sh).
inspect eval inspect_evals/mrcr \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  -T max_model_len=262144 \
  -T 'gen_kwargs={"temperature": 1.0, "top_p": 0.95, "top_k": 20}' \
  --max-connections "${MAX_CONNECTIONS:-16}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
