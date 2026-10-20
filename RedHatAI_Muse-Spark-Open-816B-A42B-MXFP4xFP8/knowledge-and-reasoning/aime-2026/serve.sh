#!/usr/bin/env bash
# Serve RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 with vLLM.
# Flags follow the official vLLM recipe for the model family, NVIDIA, strategy "Tensor + Expert Parallel" (TP8 +
# --enable-expert-parallel on one 8-GPU node), features reasoning and tool_calling:
# https://recipes.vllm.ai/meta-models/Muse-Spark-Open-816B-A42B
# The recipe's checkpoints are Meta's BF16 and NVFP4; this dir serves RedHatAI's compressed-tensors checkpoint with
# the same flags (vLLM reads the quantization scheme from config.json: MXFP4 expert weights with dynamic FP8
# activations, FP8-block attention and expert in/out projections, BF16 router, embeddings and lm_head).
# The recipe's DSpark speculative decoding is an opt-in feature (on by default only for the NVFP4 variant) and is not
# used. Muse Spark support is not in a vLLM release: the recipe requires the nightly/PR build (see README, Setup).
# --language-model-only and --limit-mm-per-prompt 0: the tasks here are text-only, so the perception encoder and audio
# tower are not loaded (the recipe's text-only commands use the same flag) and no multimodal budget is reserved.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8000}"
TP="${TP:-8}"   # one replica on 8 GPUs: 453 GB of weights plus KV cache (recipe: TP8 + expert parallel per node)
DP="${DP:-1}"

# The recipe's commands set VLLM_USE_RUST_FRONTEND=1 and offer the Python frontend (0, vLLM's own default) as the
# alternative. 0 here: the Rust frontend needs the separately built vllm-rs binary, which a source-built image may lack.
export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-0}"

exec vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --enable-expert-parallel \
  --max-model-len "${MAX_MODEL_LEN:-1048576}" \
  --reasoning-parser muse \
  --enable-auto-tool-choice --tool-call-parser muse \
  --default-chat-template-kwargs '{"reasoning_effort": "xhigh"}' \
  --language-model-only \
  --limit-mm-per-prompt '{"image": 0, "video": 0, "audio": 0}' \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
