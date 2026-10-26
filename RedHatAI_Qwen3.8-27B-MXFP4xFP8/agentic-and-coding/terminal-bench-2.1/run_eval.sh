#!/usr/bin/env bash
# Terminal-Bench 2.1 via the Harbor hub on RedHatAI/Qwen3.8-27B-MXFP4xFP8 against an already-running vLLM server.
#
# What this measures: the 89 tasks of Terminal-Bench 2.1 (Harbor hub package terminal-bench/terminal-bench-2-1,
# pinned to the digest below), solved agentically. Each task runs in its own prebuilt Docker container; the model
# gets bash/python tools through Inspect's ReAct agent and a hidden verifier decides pass/fail.
# The leaderboard and the Qwen model card use the Terminus 2 agent; this scaffold is different, so the numbers
# are not directly comparable (see README).
#
# Identical in the three dirs of the Qwen3.8-27B comparison set (Qwen/Qwen3.8-27B, RedHatAI/Qwen3.8-27B-MXFP4,
# RedHatAI/Qwen3.8-27B-MXFP4xFP8) apart from the MODEL default; one epoch (pass@1) in each.
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# --time-limit caps wall time per sample. Harbor's per-task agent timeouts range from 600 s to 12000 s (most 900 s);
# Inspect only supports one uniform limit, so 7200 s is used for every task (3600 s cut 57% of Qwen samples short).
# -T override_cpus=0 drops the per-container CPU cap (Harbor: 1 CPU for 83 of 89 tasks). Rootless Docker without
# CPU-controller delegation rejects `cpus`; 0 means no limit. Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 89 = all tasks at once
# for one epoch (the superseded 3-epoch runs used 267); wall time is dominated by tool execution, so this keeps the GPUs busy.
# The limit is wall-clock per sample while all samples share one server, so fewer samples in flight give each one
# more generation budget; the three dirs use the same concurrency, so they are comparable with each other.
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

MODEL="${MODEL:-RedHatAI/Qwen3.8-27B-MXFP4xFP8}"
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
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-89}" \
  --max-sandboxes "${MAX_SANDBOXES:-89}" \
  --time-limit "${TIME_LIMIT:-7200}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
