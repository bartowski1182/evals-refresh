#!/usr/bin/env bash
# Serve RedHatAI/GLM-5.3-NVFP4 with vLLM.
# No page at recipes.vllm.ai for this HF id; flags follow the model card:
#   https://huggingface.co/RedHatAI/GLM-5.3-NVFP4
# Closest base recipe: https://recipes.vllm.ai/zai-org/GLM-5.3
# Speculative decoding (MTP / DSpark) is opt-in on the card and is left off for quality measurement.
# Thinking is always on for GLM-5.3; pin the highest effort (max) server-side.
# Cap max_model_len at 262144: the checkpoint's native 1M window OOMs the DSA sparse-indexer
# JIT warm-up (AGENTS.md §9).
set -euo pipefail

MODEL="${MODEL:-RedHatAI/GLM-5.3-NVFP4}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"
PORT="${PORT:-8000}"
TP="${TP:-2}"   # user request: TP=2 × DP=2 on four B200s
DP="${DP:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

# SymmMem / FlashInfer-MNNVL hang on non-contiguous CUDA_VISIBLE_DEVICES (TP×DP):
# disable custom all-reduce, ShardQ DirectCP, and fused allreduce+RMS (which
# lazily builds an MNNVL FlashInfer workspace even when
# VLLM_ALLREDUCE_USE_FLASHINFER=0). Companion envs in local_orchestrator.sh.
exec vllm serve "$MODEL_PATH" \
  --trust-remote-code \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-dtype fp8_e4m3 \
  --disable-custom-all-reduce \
  --attention-config.sparse_mla_shard_q=false \
  --compilation-config.pass_config.fuse_allreduce_rms=false \
  --reasoning-parser glm45 \
  --chat-template-content-format string \
  --default-chat-template-kwargs '{"reasoning_effort": "max"}' \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
