#!/usr/bin/env bash
# AIME 2026 on zai-org/GLM-5.3-Flash against an already-running vLLM server.
# Identical in the two dirs of the GLM-5.3-Flash pair (zai-org/GLM-5.3-Flash, RedHatAI/GLM-5.3-Flash-MXFP4xFP8) apart
# from the MODEL default.
# Sampling params: temperature 1.0, top_p 0.95, no top_k, the generation_config.json of zai-org/GLM-5.3-Flash
# (byte-identical in the MXFP4xFP8 checkpoint).
# --reasoning-effort max: the model's highest level (low/high/max) and the card's setting for benchmark reproduction;
# Inspect forwards it as the OpenAI `reasoning_effort` field, and serve.sh sets the same server-side default.
# --max-tokens 163840: the maximum generation length the card states for its single-turn evaluations (HLE), strictly
# below serve.sh's --max-model-len 262144.
# -M client_timeout raises Inspect's HTTP timeout (default 600s): long reasoning traces take longer
# than 10 min to generate and would otherwise be cut off and retried from scratch forever.
set -euo pipefail

MODEL="${MODEL:-zai-org/GLM-5.3-Flash}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-8}"   # avg@8, at the user's request (30 problems: one epoch is far too noisy for the pair's comparison)
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_evals/aime2026 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 \
  --reasoning-effort max \
  --max-tokens "${MAX_TOKENS:-163840}" \
  --max-connections "${MAX_CONNECTIONS:-128}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
