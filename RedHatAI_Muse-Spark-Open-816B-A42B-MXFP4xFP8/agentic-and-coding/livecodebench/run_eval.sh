#!/usr/bin/env bash
# LiveCodeBench via the Harbor registry on RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 against an already-running
# vLLM server.
#
# What this measures: a fixed 100-task sample of LiveCodeBench release_v6, solved AGENTICALLY.
# Each task runs in its own Docker container; the model gets bash/python tools, must write
# /app/solution.py and may iterate against public tests; hidden tests decide pass/fail.
# This is NOT the single-shot pass@1 protocol of LiveCodeBench leaderboards; the Muse Spark Open card does not
# report LiveCodeBench.
#
# Sampling params come from the checkpoint's generation_config.json (temp 1.0, top_p 1.0, top_k 64). The model card
# recommends the same temperature and top_p and says "top_k = disabled"; the generation_config.json (identical to the
# base model's) sets 64, and the checkpoint is followed. Inspect's vllm provider does not forward --top-k; vLLM applies
# top_k=64 from generation_config.json as the server default (logged at startup), so the effective value is the
# checkpoint's.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM passes it to the chat
# template as `reasoning_effort` (xhigh, strength 512, is the highest of low/medium/high/xhigh and the template's
# default; the card recommends high or xhigh for coding and agentic tasks).
# --max-tokens is per turn; 65536 leaves room for many turns inside the 1M context.
# --time-limit caps wall time per sample; 3600 s because xhigh reasoning turns are long (Harbor's own
# agent timeout for these tasks is 360 s).
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
# assets/harbor_registry.json is the Harbor registry entry for livecodebench@6.0 pinned to a commit that
# carries our upstream fixes (see README, Patches); swap back to `inspect_harbor/livecodebench` once merged.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_harbor/harbor \
  -T registry_path="$(dirname "$0")/assets/harbor_registry.json" -T dataset_name_version=livecodebench@6.0 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 1.0 --top-k 64 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-100}" \
  --max-sandboxes "${MAX_SANDBOXES:-100}" \
  --time-limit "${TIME_LIMIT:-3600}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
