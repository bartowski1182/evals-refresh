#!/usr/bin/env bash
# Serve nvidia/Qwen3.8-Flash-Next-NVFP4 with vLLM.
# Flags follow the official vLLM recipe for the model family, Blackwell hardware, features tool_calling, reasoning and
# text_only: https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next
# The recipe requires a vLLM nightly (PyPI releases are not supported for this model).
set -euo pipefail

MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8002}"
TP="${TP:-1}"   # one B200 per replica
DP="${DP:-4}"   # 4 replicas behind one endpoint -> 4 GPUs; without expert parallelism the MoE experts are sharded across the DP group

exec vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len 262144 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --kv-cache-dtype fp8 \
  --attention-config.indexer_kv_dtype fp8 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --language-model-only \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
