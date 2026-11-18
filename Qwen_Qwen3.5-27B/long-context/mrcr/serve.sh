#!/usr/bin/env bash
# Serve Qwen/Qwen3.5-27B with vLLM.
# Flags follow the official vLLM recipe: https://recipes.vllm.ai/Qwen/Qwen3.5-27B
# Speculative decoding (MTP) and YaRN long-context are opt-in recipe features and are left off.
# --language-model-only: MRCR is text-only; the checkpoint is multimodal (vision tower).
# Thinking is on by default for Qwen3.5; set enable_thinking explicitly on both sides.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8000}"
TP="${TP:-1}"   # BF16 ~65 GB fits one B200
DP="${DP:-4}"   # 4 replicas behind one endpoint

exec vllm serve "$MODEL_PATH" \
  --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --language-model-only \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
