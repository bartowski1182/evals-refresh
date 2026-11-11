#!/usr/bin/env bash
# Serve zai-org/GLM-5.3 with vLLM, speculative decoding against the DSpark draft model
# RedHatAI/GLM-5.3-speculator.dspark, and per-request spec-decode acceptance metrics in every response.
#
# Target flags follow the official vLLM recipe for the model family (Blackwell, reasoning + tool calling):
# https://recipes.vllm.ai/zai-org/GLM-5.3 . The speculator's own config declares zai-org/GLM-5.3 as its
# verifier and a DSpark block_size / speculative_tokens of 8, which is what --speculative-config below uses.
# --per-request-spec-decode-metrics is experimental in vLLM 0.32.0: it puts
# `metrics.speculative_decoding` in every chat-completion response (single-sequence requests only).
set -euo pipefail

MODEL="${MODEL:-zai-org/GLM-5.3}"
MODEL_PATH="${MODEL_PATH:-$MODEL}"   # HF id, or a local copy of the checkpoint
SPECULATOR="${SPECULATOR:-RedHatAI/GLM-5.3-speculator.dspark}"
NUM_SPEC_TOKENS="${NUM_SPEC_TOKENS:-8}"   # speculator config: block_size 8, speculative_tokens 8
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

# NOTE on --tokenizer-mode: this checkpoint's tokenizer_config lists tokenizer_class TokenizersBackend with
# a chat_template.jinja (GLM-5 / glm47 family), so the default "auto" resolves the HF fast tokenizer.
# vLLM 0.32.0 has no "glm5_next" tokenizer mode (choices: auto, cohere, deepseek_v32, deepseek_v4, hf,
# inkling, kimi_k3, mistral); "auto" is correct here.
exec vllm serve "$MODEL_PATH" \
  --tokenizer-mode auto \
  --tensor-parallel-size "$TP" \
  --data-parallel-size "$DP" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --max-num-seqs "${MAX_NUM_SEQS:-128}" \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --reasoning-parser glm47 \
  --default-chat-template-kwargs '{"reasoning_effort": "max"}' \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --speculative-config "{\"method\":\"dspark\",\"model\":\"${SPECULATOR}\",\"num_speculative_tokens\":${NUM_SPEC_TOKENS}}" \
  --per-request-spec-decode-metrics detailed \
  --served-model-name "$SPECULATOR" \
  --host 0.0.0.0 --port "$PORT"
