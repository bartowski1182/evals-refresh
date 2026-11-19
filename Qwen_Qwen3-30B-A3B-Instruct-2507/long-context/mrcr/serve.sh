#!/usr/bin/env bash
# Serve Qwen/Qwen3-30B-A3B-Instruct-2507 with vLLM.
# No page at recipes.vllm.ai for this id (404). Serving flags follow the model card:
#   https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
#   vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 262144
# Instruct-2507 is non-thinking only (no <think> blocks; no reasoning parser).
# Closest sibling recipe (tools only, also non-thinking): https://recipes.vllm.ai/Qwen/Qwen3-235B-A22B-Instruct-2507
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8000}"
TP="${TP:-1}"   # ~74 GB BF16 MoE fits one B200
DP="${DP:-4}"   # 4 replicas behind one endpoint

exec vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len 262144 \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
