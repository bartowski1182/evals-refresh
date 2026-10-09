#!/usr/bin/env bash
# Serve amd/Qwen3.8-27B-Quark-AWQ-MXFP4 with vLLM.
# Flags follow the official vLLM recipe of the base model: https://recipes.vllm.ai/Qwen/Qwen3.8-27B
# (the recipe has no MXFP4 variant; the model card's serving command only adds --trust-remote-code, which is not
# needed: config.json has no auto_map). The W4A4 MXFP4 scheme is read from the checkpoint's quantization_config
# (Quark); the KV cache stays in the model dtype (bf16).
# Not from the recipe: --safetensors-load-strategy eager reads the checkpoint into RAM before copying it to the GPU.
# On this ROCm host every copy from the default memory-mapped file costs ~1 s regardless of size (~330 s per load);
# it only changes how weights are read, not what is computed.
# Not from the recipe: --attention-backend TRITON_ATTN. vLLM's ROCm default (ROCM_ATTN) decodes a single sequence at
# 64k context at ~8 tok/s here, TRITON_ATTN at ~90 tok/s (and +50% batched); same math, different kernel.
set -euo pipefail

MODEL="${MODEL:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}"
PORT="${PORT:-8000}"
TP="${TP:-1}"   # 27B MXFP4 (20 GB on disk) fits one GPU with room for the full 262k context
DP="${DP:-8}"   # 8 replicas behind one endpoint -> all 8 GPUs

exec vllm serve "$MODEL" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len 262144 \
  --safetensors-load-strategy eager \
  --attention-backend TRITON_ATTN \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
