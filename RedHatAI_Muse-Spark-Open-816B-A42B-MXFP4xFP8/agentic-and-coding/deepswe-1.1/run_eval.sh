#!/usr/bin/env bash
# DeepSWE 1.1 via the Harbor hub on RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 against an already-running vLLM
# server.
#
# What this measures: the 113 tasks of DeepSWE 1.1 (Harbor hub package datacurve/deep-swe-1-1, pinned to the digest below):
# original long-horizon engineering tasks with prebuilt images (public.ecr.aws) and hidden test verifiers.
# Each task runs in its own container; the model gets bash/python tools through Inspect's ReAct agent
# (no message limit; automatic context compaction). The Muse Spark Open card reports 60 with its "preferred
# harness", so the numbers are not directly comparable (see README).
#
# --max-tokens is per turn; 65536 leaves room for many turns inside the 1M context.
# Harbor's per-task agent budget is 5400 s; a uniform --time-limit 14400 is used (see SWE-bench Verified: about
# 1 to 3 turns/min per sample on the shared server, so 5400 s would cut long samples short).
# -T override_cpus=0 drops the per-container CPU cap (0 means no limit), as in the other DeepSWE dirs of this repo.
# Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 48 in flight, as in
# the Qwen3.8-Flash-Next dirs (histories of about 120k tokens; raise it if the server's KV cache holds more without
# evicting them, see the prefix-cache hit rate in serve.log).
# Retune a running eval with `inspect ctl config --key vllm/<model> N --max-samples N --key docker N`.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
#
# Sampling params come from the checkpoint's generation_config.json (temp 1.0, top_p 1.0, top_k 64). The model card
# recommends the same temperature and top_p and says "top_k = disabled"; the generation_config.json (identical to the
# base model's) sets 64, and the checkpoint is followed. Inspect's vllm provider does not forward --top-k; vLLM applies
# top_k=64 from generation_config.json as the server default (logged at startup), so the effective value is the
# checkpoint's.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM passes it to the chat
# template as `reasoning_effort` (xhigh, strength 512, is the highest of low/medium/high/xhigh and the template's
# default; the card recommends high or xhigh for coding and agentic tasks).
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

# The tasks are loaded through assets/harbor_registry.json, which pins them to the fork commit that carries the
# verifier fix of datacurve-ai/deep-swe#100 (see README, Patches). Requires inspect-harbor>=1.0 (verifier.collect
# support since 0.7.6, no login shell since 1.0). Swap back to `inspect_harbor/datacurve_deep_swe_1_1 -T ref=<digest>`
# once merged and republished.
inspect eval inspect_harbor/harbor \
  -T registry_path="$(dirname "$0")/assets/harbor_registry.json" -T dataset_name_version=deep-swe-1-1@1.1 \
  -T override_cpus=0 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 1.0 --top-k 64 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-48}" \
  --max-sandboxes "${MAX_SANDBOXES:-48}" \
  --time-limit "${TIME_LIMIT:-14400}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
