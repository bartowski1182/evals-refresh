#!/usr/bin/env bash
# Serve RedHatAI/Qwen3.8-27B-MXFP4xFP8 with vLLM.
# Flags follow the official vLLM recipe of Qwen/Qwen3.8-27B: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
# The recipe has no MXFP4xFP8 variant (its MXFP4 variant serves the W4A4 sibling with the base flags); vLLM reads the
# quantization scheme from the checkpoint's config.json (compressed-tensors: MXFP4 weights, dynamic FP8 activations).
# Needs vLLM >= 0.32.0: the dense MXFP4-weight / FP8-activation scheme (CompressedTensorsW4A8Mxfp4) is not in 0.31.
# Check the start-up log: it must say "Using HummingMxFp4LinearKernel for MXFP4 GEMM" (in 0.32.0 the only MXFP4 linear
# kernel that takes FP8 activations; humming-kernels is a dependency of vLLM's CUDA wheels). A warning "No MXFP4 linear
# kernel supports ... falling back to MXFP4 activations" means the server runs this checkpoint as W4A4, i.e. as
# RedHatAI/Qwen3.8-27B-MXFP4, and the comparison is void.
# Identical in the three dirs of the Qwen3.8-27B comparison set (Qwen/Qwen3.8-27B, RedHatAI/Qwen3.8-27B-MXFP4,
# RedHatAI/Qwen3.8-27B-MXFP4xFP8) apart from MODEL: same vLLM release, topology, context, parsers, thinking default and
# KV-cache dtype, so the three runs differ only in the linear layers' quantization.
# --kv-cache-dtype bfloat16: the default for the BF16 base model, spelled out because the MXFP4 checkpoint declares an
# FP8 KV-cache scheme that vLLM would apply under "auto" (the MXFP4xFP8 checkpoint may too).
# TP=2 x DP=4: four replicas on 8 GPUs, the smallest topology that holds the BF16 model with its full 262k context on
# 80 GB GPUs; kept for the quantized checkpoints so that the three are served alike. The multi-turn tasks export
# TP=8 DP=1 (one replica, one prefix cache) in local_orchestrator.sh.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Qwen3.8-27B-MXFP4xFP8}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8000}"
TP="${TP:-2}"
DP="${DP:-4}"

exec vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len 262144 \
  --kv-cache-dtype bfloat16 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
