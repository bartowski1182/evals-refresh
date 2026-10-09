#!/usr/bin/env bash
# SWE-bench Pro via the Harbor hub on amd/Qwen3.8-27B-Quark-AWQ-MXFP4 against an already-running vLLM server.
#
# What this measures: the 731 public tasks of SWE-bench Pro (Harbor hub package scale-ai/swe-bench-pro, digest
# sha256:88411d32ff27e53a4c1a7e29f0c2aeba180c8e5d60f221cab5ed56325f33549d, loaded from a patched local copy):
# long-horizon enterprise software-engineering tasks in Python/JS/TS/Go with hidden test verifiers.
# Each task runs in its own container; the model gets bash/python tools through Inspect's ReAct agent
# (no message limit; automatic context compaction). The Qwen model card uses the Claude Code harness, so the
# numbers are not directly comparable (see README).
#
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# Harbor's per-task agent budget is 3000 s; a uniform --time-limit 14400 is used (see SWE-bench Verified: about
# 1 to 3 turns/min per sample on the shared server, so 3000 s would cut long samples short).
# -T override_cpus=0 drops the per-container CPU cap: rootless Docker without CPU-controller delegation rejects
# `cpus`; 0 means no limit. Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 56 in flight is what
# the server's KV cache holds (4.0M tokens at TP=8) once agent histories average 45k tokens; above that vLLM
# preempts requests and evicts prefix-cache entries. Retune a running eval with `inspect ctl config --key vllm/<model> N --max-samples N`.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
#
# Sampling params come from the model's generation_config.json (temp 1.0, top_p 0.95, top_k 20), identical to
# the base model Qwen/Qwen3.8-27B's; everything below matches its run_eval.sh.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (see the recipe).
set -euo pipefail

MODEL="${MODEL:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

# The tasks are loaded through assets/harbor_registry.json, which pins them to the fork commit that carries the
# verifier fixes of harbor-framework/harbor-datasets#258 (see README, Patches); swap back to
# `inspect_harbor/scale_ai_swe_bench_pro -T ref=<digest>` once merged and republished.
inspect eval inspect_harbor/harbor \
  -T registry_path="$(dirname "$0")/assets/harbor_registry.json" -T dataset_name_version=swebenchpro@1.0 \
  -T override_cpus=0 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-56}" \
  --max-sandboxes "${MAX_SANDBOXES:-56}" \
  --time-limit "${TIME_LIMIT:-14400}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
