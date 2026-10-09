#!/usr/bin/env bash
# AIME 2026 on amd/Qwen3.8-27B-Quark-AWQ-MXFP4 against an already-running vLLM server.
# Sampling params come from the model's generation_config.json (temp 1.0, top_p 0.95, top_k 20), identical to
# the base model Qwen/Qwen3.8-27B's; everything below matches its run_eval.sh.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (see the recipe).
# -M client_timeout raises Inspect's HTTP timeout (default 600s): long reasoning traces take longer
# than 10 min to generate and would otherwise be cut off and retried from scratch forever.
set -euo pipefail

MODEL="${MODEL:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-3}"   # avg@3 (the base model ran avg@8)
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_evals/aime2026 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-131072}" \
  --max-connections "${MAX_CONNECTIONS:-128}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
