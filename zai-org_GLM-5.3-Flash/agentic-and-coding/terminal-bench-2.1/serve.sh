#!/usr/bin/env bash
# Serve zai-org/GLM-5.3-Flash (the native FP8 checkpoint, the recipe's default variant) with vLLM.
# Flags follow the official vLLM recipe: https://recipes.vllm.ai/zai-org/GLM-5.3-Flash
# Taken from it: the Tensor Parallel strategy (one replica across the node's GPUs), MTP speculative decoding (the
# recipe's default spec-decoding mode, 5 draft tokens from the checkpoint's MTP layer), the glm47 reasoning and
# tool-call parsers with automatic tool choice, and the Hopper override `--no-enable-flashinfer-autotune`. On Hopper
# the recipe runs the KV cache in BF16 (its `--kv-cache-dtype fp8` is a Blackwell-only override: "FP8 KV cache only
# works on Blackwell for this model; Hopper runs BF16 KV"); it is pinned explicitly so that a checkpoint-declared KV
# scheme cannot switch it, and so that the FP8 base model and this checkpoint are served alike.
# Departures from the recipe: --max-model-len 262144 instead of the checkpoint's 1M window (the sparse-MLA indexer's
# warm-up allocates per-sequence buffers proportional to max_model_len and OOMs on H200 at 1M, see AGENTS.md,
# section 9; 262144 is what the repo's other multi-turn agent dirs use and leaves per-turn 65536 and Inspect's
# compaction threshold sane), and --language-model-only (the tasks are text-only, so the vision encoder is not loaded).
# Thinking is always on for GLM-5.3-Flash; the server-side default pins the highest reasoning effort (max, of
# low/high/max), mirrored per request in run_eval.sh.
# Checkpoint: the native FP8 weights (e4m3, dynamic activation scaling, `quant_method: fp8` in config.json; the
# attention projections, embeddings, lm_head and the mHC/KDA parameters stay unquantized). Check the start-up log:
# the quantization method must be fp8 and the KV cache dtype bfloat16. vLLM 0.32.0 is the first release whose model
# registry carries glm5_next.
set -euo pipefail

MODEL="${MODEL:-zai-org/GLM-5.3-Flash}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
PORT="${PORT:-8000}"
TP="${TP:-8}"   # one TP=8 replica across all 8 GPUs (the recipe's Tensor Parallel strategy on one node)
DP="${DP:-1}"   # keep DP=1: vLLM's prefix cache is per replica, and multi-turn agent runs
                # need every turn of a sample on one replica
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"

exec vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-cache-dtype bfloat16 \
  --no-enable-flashinfer-autotune \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --reasoning-parser glm47 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"reasoning_effort": "max"}' \
  --language-model-only \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 --port "$PORT"
