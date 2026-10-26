#!/usr/bin/env bash
# AIME 2026 on RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 against an already-running vLLM server.
# Sampling params come from the checkpoint's generation_config.json (temp 1.0, top_p 1.0, top_k 64). The model card
# recommends the same temperature and top_p and says "top_k = disabled"; the generation_config.json (identical to the
# base model's) sets 64, and the checkpoint is followed. Inspect's vllm provider does not forward --top-k; vLLM applies
# top_k=64 from generation_config.json as the server default (logged at startup), so the effective value is the
# checkpoint's.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM passes it to the chat
# template as `reasoning_effort` (xhigh, strength 512, is the highest of low/medium/high/xhigh and the template's
# default; the card recommends high or xhigh for coding and agentic tasks).
# -M client_timeout raises Inspect's HTTP timeout (default 600s): long reasoning traces take longer
# than 10 min to generate and would otherwise be cut off and retried from scratch forever.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_evals/aime2026 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 1.0 --top-k 64 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-131072}" \
  --max-connections "${MAX_CONNECTIONS:-128}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
