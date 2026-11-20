#!/usr/bin/env bash
# Serve zai-org/GLM-5.3 with vLLM, speculative decoding against the DFlash2 draft model
# RedHatAI/GLM-5.3-speculator.dflash2-preview, and per-request spec-decode acceptance metrics.
#
# Target flags follow the official vLLM recipe for the model family (Blackwell, reasoning + tool calling):
# https://recipes.vllm.ai/zai-org/GLM-5.3 . The speculator's own config declares zai-org/GLM-5.3 as its verifier,
# a 5-layer qwen3-style DFlash2DraftModel (arch resolved by vLLM to qwen3_dflash2), aux target layers
# [18, 30, 42, 54, 66, 74], and greedy proposals. method="dflash" is the value vLLM 0.32.0 knows for DFlash2
# drafts (the DFlash2DraftModel architecture selects the dflash2 implementation).
#
# Draft weights revision: these results were produced with RedHatAI/GLM-5.3-speculator.dflash2-preview at
# revision 3325df9c8ddad6b11afc9623ca6c3833d70fa4ef. The repo's main has since moved to 83c1f7eb (2026-11-13,
# "Replace draft weights with validated 4-epoch DFlash2 checkpoint"): same architecture, same serving-relevant
# config, new weights. Serve either by pinning the download (hf download <repo> --revision <sha>) and passing
# the snapshot path as SPECULATOR; no flag here changes between the two revisions.
#
# num_speculative_tokens=8 matches the speculator card's acceptance tables ("eight draft tokens with adaptive
# verification disabled"); the card's speedup example uses 7 + adaptive verification, which is a different
# acceptance regime and would make per-position MRATs harder to interpret. Adaptive verification is left at
# its default (off).
#
# vLLM build: main @ commit ace415c06 (nightly wheel 0.34.0rc1.dev253+gace415c06). This matters: on 0.32.0 and
# 0.33.0 a DFlash/DFlash2 draft served with the default FULL_AND_PIECEWISE CUDA graphs can silently lose all
# draft acceptance (MAL ~ 1.00, or a trailing run of zero-accepted steps) while output text stays correct,
# cutting throughput ~3x. That is vLLM issue #63587, fixed by PR #64059 (commit d3a323093, merged 2026-11-04),
# which adds DraftModelSpeculator._map_kernel_block_tables() and calls it on the non-FULL path in
# DFlashSpeculator: before the fix the persistent kernel block tables a FULL graph reads were only refreshed
# on the FULL path, so a non-FULL replay read stale tables and rejected every draft token. ace415c06 is a
# descendant of d3a323093, so the default graphs are used here unchanged.
# CUDAGRAPH_MODE is the escape hatch: unset (default) = vLLM's own choice (FULL_AND_PIECEWISE);
# PIECEWISE / NONE reproduce the pre-fix workaround if some future build regresses. monitor_spec_decode.py
# canary + audit are what decide whether acceptance actually collapsed.
#
# --per-request-spec-decode-metrics is experimental in vLLM 0.32.0: it puts `metrics.speculative_decoding`
# in every chat-completion response (single-sequence requests only).
set -euo pipefail

MODEL="${MODEL:-zai-org/GLM-5.3}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
SPECULATOR="${SPECULATOR:-RedHatAI/GLM-5.3-speculator.dflash2-preview}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"   # card's acceptance tables: 8 draft tokens, adaptive verification off
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-}"   # "" = vLLM default graphs; PIECEWISE/NONE = #63587 escape hatch
PORT="${PORT:-8000}"
TP="${TP:-8}"   # one TP=8 replica across all 8 B200s: ~756 GB of FP8 weights, ~95 GB per GPU
DP="${DP:-1}"   # one replica = one prefix cache for the agent's multi-turn histories

# flashinfer 0.7.x provides the sparse-MLA (DSA) kernels for this arch (GlmMoeDsaForCausalLM); point it at
# the CUDA toolkit so its JIT path is available (there is no flashinfer-cubin wheel for 0.7.0.post1 on PyPI).
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Loading the ~756 GB FP8 checkpoint across 8 GPUs plus first-start kernel compilation takes longer than
# vLLM's default 600 s engine start-up budget.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# NOTE on --tokenizer-mode: vLLM main has a "glm5_next" tokenizer mode for this family (0.32.0 did not).
# This checkpoint's tokenizer_config lists tokenizer_class TokenizersBackend with a chat_template.jinja
# (GLM-5 / glm47 family); both "glm5_next" and "auto" resolve to the same tokenizer here (verified offline:
# CachedTokenizersBackend, 154,856 tokens), so the explicit family mode costs nothing and documents intent.
exec vllm serve "$MODEL_PATH" \
  --tokenizer-mode glm5_next \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-128}" \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  ${CUDAGRAPH_MODE:+--compilation-config "{\"cudagraph_mode\":\"${CUDAGRAPH_MODE}\"}"} \
  --reasoning-parser glm47 \
  --default-chat-template-kwargs '{"reasoning_effort": "max"}' \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --speculative-config "{\"method\":\"dflash\",\"model\":\"${SPECULATOR}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
  --per-request-spec-decode-metrics detailed \
  --served-model-name "$SPECULATOR" \
  --host 0.0.0.0 --port "$PORT"
