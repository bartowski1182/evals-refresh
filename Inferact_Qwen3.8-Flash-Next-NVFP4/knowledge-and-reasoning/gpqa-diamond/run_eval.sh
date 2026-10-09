#!/usr/bin/env bash
# GPQA-Diamond on Inferact/Qwen3.8-Flash-Next-NVFP4 against an already-running vLLM server.
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

MODEL="${MODEL:-Inferact/Qwen3.8-Flash-Next-NVFP4}"
PORT="${PORT:-8001}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_evals/gpqa_diamond \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-131072}" \
  --max-connections "${MAX_CONNECTIONS:-256}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
