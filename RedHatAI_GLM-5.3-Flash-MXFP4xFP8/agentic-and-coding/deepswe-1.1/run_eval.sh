#!/usr/bin/env bash
# DeepSWE 1.1 via the Harbor hub on RedHatAI/GLM-5.3-Flash-MXFP4xFP8 against an already-running vLLM server.
#
# What this measures: the 113 tasks of DeepSWE 1.1 (Harbor hub package datacurve/deep-swe-1-1, pinned to the digest below):
# original long-horizon engineering tasks with prebuilt images (public.ecr.aws) and hidden test verifiers.
# Each task runs in its own container; the model gets bash/python tools through Inspect's ReAct agent
# (no message limit; automatic context compaction). The GLM-5.3-Flash card reports DeepSWE 63.4 with the
# mini-swe-agent harness (temp 0.95, top_p 1.0, 400K context, 6h), so this Inspect ReAct number is not directly
# comparable (see README).
#
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# Harbor's per-task agent budget is 5400 s; a uniform --time-limit 14400 is used (see SWE-bench Verified: about
# 1 to 3 turns/min per sample on the shared server, so 5400 s would cut long samples short).
# -T override_cpus=0 drops the per-container CPU cap (0 means no limit), as in the other DeepSWE dirs of this repo.
# Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 48 in flight, as in the
# GLM-5.3-Flash-NVFP4 dir: GLM-5.3-Flash is a hybrid linear-attention / sparse-MLA model with a small per-token KV
# footprint, so one TP=8 replica holds that many ~120k-token histories. Read the "GPU KV cache size" line in
# logs/serve.log before the full run (BF16 KV on Hopper, so half the pool of the NVFP4 run's FP8 KV) and lower the
# concurrency if 48 x 120k tokens does not fit; oversubscribing evicts prefix-cached histories and re-prefills them
# every turn. Retune a running eval with `inspect ctl config --key vllm/<model> N --max-samples N --key docker N`.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
#
# Sampling params come from zai-org/GLM-5.3-Flash's generation_config.json (temperature 1.0, top_p 0.95; no top_k,
# min_p or penalties, so vLLM's defaults apply), which the quantized checkpoint is expected to carry unchanged
# (verify before the run, see README). GLM-5.3-Flash has thinking always on; --reasoning-effort is forwarded by
# Inspect as the OpenAI `reasoning_effort` field and vLLM turns it into chat_template_kwargs {"reasoning_effort":
# "max"} (max is the highest of low/high/max, the default, and what the card prescribes for benchmark reproduction).
set -euo pipefail

MODEL="${MODEL:-RedHatAI/GLM-5.3-Flash-MXFP4xFP8}"
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
  --temperature 1.0 --top-p 0.95 \
  --reasoning-effort max \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-48}" \
  --max-sandboxes "${MAX_SANDBOXES:-48}" \
  --time-limit "${TIME_LIMIT:-14400}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
