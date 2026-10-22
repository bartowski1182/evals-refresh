#!/usr/bin/env bash
# Serve RedHatAI/GLM-5.3-MXFP4xFP8 with vLLM.
# Flags follow the model card's vLLM serving command, which is identical to the base model recipe's
# single-node Hopper section (the recipes site has no MXFP4xFP8 entry): https://recipes.vllm.ai/zai-org/GLM-5.3
# The card requires vLLM v0.32+ on Hopper for this mixed MXFP4/FP8 checkpoint.
# Thinking is always on for GLM-5.3; the server-side default pins the highest reasoning effort
# (max), mirrored per request in run_eval.sh.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/GLM-5.3-MXFP4xFP8}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8000}"
TP="${TP:-8}"   # one TP=8 replica across all 8 GPUs (the recipe's single-node H200 topology)
DP="${DP:-1}"   # keep DP=1: vLLM's prefix cache is per replica, and multi-turn agent runs
                 # need every turn of a sample on one replica
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"   # recipe default is the checkpoint's 1M window; the DSA sparse
                 # indexer's JIT warm-up then allocates 24 GiB per GPU (96 seqs x 64 heads x 1M x fp32) and
                 # OOMs on H200 with the default gpu-memory-utilization. 262144 matches the repo's other
                 # multi-turn agent dirs and keeps per-turn 65536 and Inspect's 90% compaction threshold sane.

exec vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 5}' \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort": "max"}' \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
