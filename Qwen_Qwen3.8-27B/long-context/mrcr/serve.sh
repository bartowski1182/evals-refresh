#!/usr/bin/env bash
# Serve Qwen/Qwen3.8-27B with vLLM.
# Flags follow the official vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
# Speculative decoding (MTP / DSpark) is an opt-in recipe feature and is left off.
# --language-model-only: MRCR is text-only; the checkpoint is multimodal (vision tower).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8000}"
TP="${TP:-1}"   # BF16 fits one B200
DP="${DP:-4}"   # 4 replicas behind one endpoint

exec vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --language-model-only \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
