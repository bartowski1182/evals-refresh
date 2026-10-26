#!/usr/bin/env bash
# Serve the user-simulator model, Qwen/Qwen3.8-27B, with vLLM on a second port (used by local_orchestrator.sh).
# Flags follow the official vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
set -euo pipefail

USER_MODEL="${USER_MODEL:-Qwen/Qwen3.8-27B}"
USER_PORT="${USER_PORT:-8001}"
USER_TP="${USER_TP:-4}"

exec vllm serve "$USER_MODEL" \
  --tensor-parallel-size "$USER_TP" \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --served-model-name "$USER_MODEL" \
  --host 0.0.0.0 --port "$USER_PORT"
