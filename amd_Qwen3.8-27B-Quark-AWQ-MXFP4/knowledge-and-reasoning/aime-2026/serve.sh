#!/usr/bin/env bash
# Serve amd/Qwen3.8-27B-Quark-AWQ-MXFP4 with vLLM.
# Flags follow the official vLLM recipe of the base model: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
# (the recipe has no MXFP4 variant; vLLM reads the quantization scheme from the checkpoint's config.json).
# --attention-backend is the one flag not from the recipe (which has no AMD path): vLLM 0.31.0's default on
# MI355X, ROCM_ATTN, decodes about 14x slower at 100k context (5 vs 77 tok/s per stream). AITER's faster unified
# attention is not used: it produces garbage for RedHatAI/Qwen3.8-27B-MXFP4 with --kv-cache-dtype bfloat16.
set -euo pipefail

MODEL="${MODEL:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}"
PORT="${PORT:-8000}"
TP="${TP:-1}"   # 20 GB of MXFP4 weights fit one MI355X (288 GB)
DP="${DP:-8}"   # 8 replicas behind one endpoint -> all 8 GPUs

exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --attention-backend TRITON_ATTN \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
