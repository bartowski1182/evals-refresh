#!/usr/bin/env bash
# AIME 2026 on Qwen/Qwen3.8-Flash-Next against an already-running vLLM server.
# Sampling params come from the model's generation_config.json (temp 1.0, top_p 0.95, top_k 20), the thinking-mode
# values of the Qwen/Qwen3.8-Flash-Next model card. The card's min_p=0.0,
# presence_penalty=0.0 and repetition_penalty=1.0 are vLLM's defaults and no generation_config.json overrides them.
# Inspect's vllm provider does not forward --top-k; vLLM applies top_k=20 from the checkpoint's generation_config.json
# as the server default (logged at startup), so the effective value is the card's.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (xhigh is the template's highest level).
# -M client_timeout raises Inspect's HTTP timeout (default 600s): long reasoning traces take longer
# than 10 min to generate and would otherwise be cut off and retried from scratch forever.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.8-Flash-Next}"
PORT="${PORT:-8003}"
EPOCHS="${EPOCHS:-1}"
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
