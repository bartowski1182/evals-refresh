#!/usr/bin/env bash
# SWE-bench Verified on amd/Qwen3.8-27B-Quark-AWQ-MXFP4 against an already-running vLLM server.
#
# What this measures: the 500 instances of SWE-bench Verified (HF princeton-nlp/SWE-bench_Verified, revision pinned by
# inspect_evals 0.20.0), solved agentically with inspect_evals's default ReAct agent (python, bash_session, text_editor;
# 210 s tool timeout) inside Epoch AI's prebuilt per-instance Docker images, no internet. The official swebench harness
# grades FAIL_TO_PASS and PASS_TO_PASS tests. Neither model card reports SWE-bench Verified.
#
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# --message-limit 200 (about 100 agent turns; the package default of 30 cut smoke samples off after 13 turns);
# --time-limit 14400 caps wall time per sample as a runaway guard (about 3 turns/min per sample with 56 in flight and a warm prefix cache).
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 56 in flight is what
# the server's KV cache holds (4.0M tokens at TP=8; mean request carries ~41k tokens of history after an hour); above that vLLM
# preempts requests and evicts prefix-cache entries, and per-sample speed collapses (0.4 turns/min at 234, 0.55 at 128, 0.5 at 96). Retune a running eval with `inspect ctl config --key vllm/<model> N --max-samples N`.
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

inspect eval inspect_evals/swe_bench \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --message-limit "${MESSAGE_LIMIT:-200}" \
  --max-connections "${MAX_CONNECTIONS:-56}" \
  --max-sandboxes "${MAX_SANDBOXES:-56}" \
  --time-limit "${TIME_LIMIT:-14400}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
