#!/usr/bin/env bash
# Terminal-Bench 2.1 via the Harbor hub on RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 against an already-running
# vLLM server.
#
# What this measures: the 89 tasks of Terminal-Bench 2.1 (Harbor hub package terminal-bench/terminal-bench-2-1,
# pinned to the digest below), solved agentically. Each task runs in its own prebuilt Docker container; the model
# gets bash/python tools through Inspect's ReAct agent and a hidden verifier decides pass/fail.
# The leaderboard and the model card (76.4) use the Terminus 2 agent; this scaffold is different, so the numbers are
# not directly comparable (see README).
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
# --time-limit caps wall time per sample. Harbor's per-task agent timeouts range from 600 s to 12000 s (most 900 s);
# Inspect only supports one uniform limit, so 7200 s is used for every task, as in the other Terminal-Bench dirs.
# -T override_cpus=0 drops the per-container CPU cap (Harbor: 1 CPU for 83 of 89 tasks), as in the other
# Terminal-Bench dirs of this repo; 0 means no limit. Memory caps are kept (inspect_harbor floors them at 6 GB).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 45 as in the
# Qwen3.8-Flash-Next BF16 dir (one replica for all samples; raise it if the server's KV cache holds more histories
# without evicting them, see the prefix-cache hit rate in serve.log).
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8}"
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
  --temperature 1.0 --top-p 1.0 --top-k 64 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-45}" \
  --max-sandboxes "${MAX_SANDBOXES:-45}" \
  --time-limit "${TIME_LIMIT:-7200}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
