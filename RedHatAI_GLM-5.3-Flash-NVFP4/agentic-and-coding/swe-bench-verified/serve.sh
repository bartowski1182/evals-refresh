#!/usr/bin/env bash
# Serve RedHatAI/GLM-5.3-Flash-NVFP4 with vLLM.
# Flags follow the official vLLM recipe for the model family (Blackwell, features reasoning + tool_calling):
# https://recipes.vllm.ai/zai-org/GLM-5.3-Flash
# The RedHatAI NVFP4 checkpoint (base zai-org/GLM-5.3-Flash, arch glm5_next) requires NVIDIA Blackwell.
# The recipe needs vLLM 0.29.0+ with the glm5_next integration (a recent nightly) and FlashInfer 0.6.17+
# for the NoPE sparse-MLA kernels; MTP speculative decoding is the recipe's primary serving mode.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/GLM-5.3-Flash-NVFP4}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8000}"
TP="${TP:-8}"   # one replica across all 8 B200s: a coherent prefix cache for multi-turn agent histories
DP="${DP:-1}"

# flashinfer 0.7.1 provides the sparse-MLA kernels; point it at the CUDA toolkit so its JIT path is available.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# MTP keeps its weights in FP8 and needs headroom; the recipe lowers gpu-mem-util and disables custom all-reduce.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len 262144 \
  --max-num-seqs 256 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --disable-flashinfer-autotune \
  --disable-custom-all-reduce \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --reasoning-parser glm47 \
  --default-chat-template-kwargs '{"reasoning_effort": "max"}' \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --language-model-only \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
