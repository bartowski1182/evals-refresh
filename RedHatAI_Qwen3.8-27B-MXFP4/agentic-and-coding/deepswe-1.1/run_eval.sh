#!/usr/bin/env bash
# DeepSWE 1.1 via the Harbor hub on RedHatAI/Qwen3.8-27B-MXFP4 against an already-running vLLM server.
#
# What this measures: the 113 tasks of DeepSWE 1.1 (Harbor hub package datacurve/deep-swe-1-1, pinned to the digest below):
# original long-horizon engineering tasks with prebuilt images (public.ecr.aws) and hidden test verifiers.
# Each task runs in its own container; the model gets bash/python tools through Inspect's ReAct agent
# (no message limit; automatic context compaction). The Qwen model card uses the Claude Code harness, so the
# numbers are not directly comparable (see README).
#
# Copied from the Qwen3.8-Flash-Next DeepSWE dir, which carries the solution-leak fixes (AGENTS.md, section 11): the
# agent runs offline through the inspect_harbor fork pinned in the README.
# Identical in the three dirs of the Qwen3.8-27B comparison set (Qwen/Qwen3.8-27B, RedHatAI/Qwen3.8-27B-MXFP4,
# RedHatAI/Qwen3.8-27B-MXFP4xFP8) apart from the MODEL default; one epoch (pass@1) in each.
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# Harbor's per-task agent budget is 5400 s; a uniform --time-limit 14400 is used (see SWE-bench Verified: about
# 1 to 3 turns/min per sample on the shared server, so 5400 s would cut long samples short).
# -T override_cpus=0 drops the per-container CPU cap (0 means no limit), as in the other DeepSWE dirs of this repo.
# Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 48 in flight, as in the
# Flash-Next dirs; the histories average about 120k tokens, and the 27B hybrid model (16 full-attention layers) keeps
# about 90 KB per token of BF16 KV cache, so one TP=8 replica on 80 GB GPUs holds far more. Retune a running eval with
# `inspect ctl config --key vllm/<model> N --max-samples N --key docker N`.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
#
# Sampling params: temperature 1.0, top_p 0.95, top_k 20, the generation_config.json values of Qwen/Qwen3.8-27B and
# RedHatAI/Qwen3.8-27B-MXFP4 (the MXFP4xFP8 repository was not public when these dirs were written: check its
# generation_config.json before running it). Inspect drops --top-k; vLLM applies the checkpoint's value as the server
# default (logged at start-up).
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (see the recipe).
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Qwen3.8-27B-MXFP4}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

# The tasks are loaded through assets/harbor_registry.json, which pins them to the fork commit that carries the
# verifier fix of datacurve-ai/deep-swe#100 (see README, Patches). Requires inspect-harbor>=1.0 (verifier.collect
# support since 0.7.6, no login shell since 1.0) from the fork that enforces the tasks' network policy. Swap back to
# `inspect_harbor/datacurve_deep_swe_1_1 -T ref=<digest>` once merged and republished.
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
