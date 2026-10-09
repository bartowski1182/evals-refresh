#!/usr/bin/env bash
# Terminal-Bench 2.1 via the Harbor hub on amd/Qwen3.8-27B-Quark-AWQ-MXFP4 against an already-running vLLM server.
#
# What this measures: the 89 tasks of Terminal-Bench 2.1 (Harbor hub package terminal-bench/terminal-bench-2-1,
# pinned to the digest below), solved agentically. Each task runs in its own prebuilt Docker container; the model
# gets bash/python tools through Inspect's ReAct agent and a hidden verifier decides pass/fail.
# The leaderboard and the Qwen model card use the Terminus 2 agent; this scaffold is different, so the numbers
# are not directly comparable (see README).
#
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# --time-limit caps wall time per sample. Harbor's per-task agent timeouts range from 600 s to 12000 s (most 900 s);
# Inspect only supports one uniform limit, so 7200 s is used for every task (3600 s cut 57% of Qwen samples short).
# -T override_cpus=0 drops the per-container CPU cap (Harbor: 1 CPU for 83 of 89 tasks). Rootless Docker without
# CPU-controller delegation rejects `cpus`; 0 means no limit. Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 267 = all 89 tasks x 3
# epochs at once; wall time is dominated by tool execution, so this keeps the GPUs busy.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
#
# Sampling params come from the model's generation_config.json (temp 1.0, top_p 0.95, top_k 20).
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (see the recipe).
set -euo pipefail

MODEL="${MODEL:-amd/Qwen3.8-27B-Quark-AWQ-MXFP4}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-3}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_harbor/terminal_bench_2_1 \
  -T ref=sha256:7d7bdc1cbedad549fc1140404bd4dc45e5fd0ea7c4186773687d177ad3a0699a \
  -T override_cpus=0 \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-267}" \
  --max-sandboxes "${MAX_SANDBOXES:-267}" \
  --time-limit "${TIME_LIMIT:-7200}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
