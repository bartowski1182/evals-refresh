#!/usr/bin/env bash
# Terminal-Bench 2.1 via the Harbor hub on RedHatAI/GLM-5.3-MXFP4xFP8 against an already-running vLLM server.
#
# What this measures: the 89 tasks of Terminal-Bench 2.1 (Harbor hub package terminal-bench/terminal-bench-2-1,
# pinned to the digest below), solved agentically. Each task runs in its own prebuilt Docker container; the model
# gets bash/python tools through Inspect's ReAct agent and a hidden verifier decides pass/fail.
# The leaderboard and the GLM-5.3 model card use the Claude Code harness; this scaffold is different, so the
# numbers are not directly comparable (see README).
#
# Sampling params come from the checkpoint's generation_config.json on the HF hub (temperature 1.0, top_p 0.95;
# no top_k, unlike Qwen). --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field;
# vLLM turns it into chat_template_kwargs {"reasoning_effort": "max"} (max is GLM-5.3's highest and default level,
# and what the model card prescribes for benchmark reproduction).
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context window.
# --time-limit caps wall time per sample. Harbor's per-task agent timeouts range from 600 s to 12000 s (most 900 s);
# Inspect only supports one uniform limit, so 7200 s is used for every task, as in the other Terminal-Bench dirs.
# -T override_cpus=0 drops the per-container CPU cap (Harbor: 1 CPU for 83 of 89 tasks), as in the other
# Terminal-Bench dirs of this repo; 0 means no limit. Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 32 in flight, not all
# 89 at once: Terminal-Bench histories reach ~30-60k tokens and this model's ~1.19M-token KV pool (see serve.log)
# would evict prefix-cached histories late in the run; 32 keeps the pool covered while tasks churn through.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/GLM-5.3-MXFP4xFP8}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_harbor/terminal_bench_2_1 \
  -T ref=sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a \
  -T override_cpus=0 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 \
  --reasoning-effort max \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-32}" \
  --max-sandboxes "${MAX_SANDBOXES:-32}" \
  --time-limit "${TIME_LIMIT:-7200}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
