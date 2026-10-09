#!/usr/bin/env bash
# DeepSWE 1.1 via the Harbor hub on Qwen/Qwen3.8-Flash-Next against an already-running vLLM server.
#
# What this measures: the 113 tasks of DeepSWE 1.1 (Harbor hub package datacurve/deep-swe-1-1, pinned to the digest below):
# original long-horizon engineering tasks with prebuilt images (public.ecr.aws) and hidden test verifiers.
# Each task runs in its own container; the model gets bash/python tools through Inspect's ReAct agent
# (no message limit; automatic context compaction). The Qwen model card uses the Claude Code harness, so the
# numbers are not directly comparable (see README).
#
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# Harbor's per-task agent budget is 5400 s; a uniform --time-limit 14400 is used (see SWE-bench Verified: about
# 1 to 3 turns/min per sample on the shared server, so 5400 s would cut long samples short).
# -T override_cpus=0 drops the per-container CPU cap (0 means no limit), as in the other DeepSWE dirs of this repo.
# Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 48 in flight (the
# Qwen3.8-27B DeepSWE dir used 24, what an H100 TP=8 server's 4M-token KV cache held); this TP=4 server's 20M-token KV
# cache holds 48 histories of about 120k tokens.
# Retune a running eval with `inspect ctl config --key vllm/<model> N --max-samples N --key docker N`.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
#
# Sampling params come from the model's generation_config.json (temp 1.0, top_p 0.95, top_k 20), the thinking-mode
# values of the Qwen/Qwen3.8-Flash-Next model card. The card's min_p=0.0,
# presence_penalty=0.0 and repetition_penalty=1.0 are vLLM's defaults and no generation_config.json overrides them.
# Inspect's vllm provider does not forward --top-k; vLLM applies top_k=20 from the checkpoint's generation_config.json
# as the server default (logged at startup), so the effective value is the card's.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (xhigh is the template's highest level).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.8-Flash-Next}"
PORT="${PORT:-8003}"
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
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-48}" \
  --max-sandboxes "${MAX_SANDBOXES:-48}" \
  --time-limit "${TIME_LIMIT:-14400}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
