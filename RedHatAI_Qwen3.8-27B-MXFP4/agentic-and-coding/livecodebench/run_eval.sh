#!/usr/bin/env bash
# LiveCodeBench via the Harbor registry on RedHatAI/Qwen3.8-27B-MXFP4
# against an already-running vLLM server.
#
# What this measures: a fixed 100-task sample of LiveCodeBench release_v6, solved AGENTICALLY.
# Each task runs in its own Docker container; the model gets bash/python tools, must write
# /app/solution.py and may iterate against public tests; hidden tests decide pass/fail.
# This is NOT the single-shot pass@1 protocol behind the model card's "LiveCodeBench v6" number.
#
# Sampling params come from the model's generation_config.json (temp 1.0, top_p 0.95, top_k 20).
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (see the recipe).
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# --time-limit caps wall time per sample; 3600 s because xhigh reasoning turns are long (Harbor's own
# agent timeout for these tasks is 360 s).
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
# -T override_cpus=0 drops the tasks' 1-CPU cap: rootless Docker on this host has no CPU-controller delegation and
# rejects `cpus` (the base Qwen run used rootful Docker, with the cap). Memory caps are kept.
# assets/harbor_registry.json is the Harbor registry entry for livecodebench@6.0 pinned to a commit that
# carries our upstream fixes (see README, Patches); swap back to `inspect_harbor/livecodebench` once merged.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Qwen3.8-27B-MXFP4}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-3}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_harbor/harbor \
  -T registry_path="$(dirname "$0")/assets/harbor_registry.json" -T dataset_name_version=livecodebench@6.0 \
  -T override_cpus=0 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-100}" \
  --max-sandboxes "${MAX_SANDBOXES:-100}" \
  --time-limit "${TIME_LIMIT:-3600}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
