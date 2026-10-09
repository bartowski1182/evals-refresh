#!/usr/bin/env bash
# SWE-bench Verified on Qwen/Qwen3.8-Flash-Next against an already-running vLLM server.
#
# What this measures: the 500 instances of SWE-bench Verified (HF princeton-nlp/SWE-bench_Verified, revision pinned by
# inspect_evals 0.20.0), solved agentically with inspect_evals's default ReAct agent (python, bash_session, text_editor;
# 210 s tool timeout), no internet. The official swebench harness grades FAIL_TO_PASS and PASS_TO_PASS tests. The
# Qwen/Qwen3.8-Flash-Next card does not report SWE-bench Verified.
#
# -T image_name_template selects the official SWE-bench images (Docker Hub, built with the current swebench harness,
# which removes the remote, deletes tags newer than the base commit and prunes unreachable objects). inspect_evals'
# default Epoch AI images (ghcr.io) keep every release tag of the project, so the reference fix of an instance can be
# read offline with `git show <later tag>:<file>` (see README).
#
# Sampling params come from the model's generation_config.json (temp 1.0, top_p 0.95, top_k 20), the thinking-mode
# values of the Qwen/Qwen3.8-Flash-Next model card. The card's min_p=0.0,
# presence_penalty=0.0 and repetition_penalty=1.0 are vLLM's defaults and no generation_config.json overrides them.
# Inspect's vllm provider does not forward --top-k; vLLM applies top_k=20 from the checkpoint's generation_config.json
# as the server default (logged at startup), so the effective value is the card's.
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (xhigh is the template's highest level).
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# --message-limit 200 (about 100 agent turns; the package default of 30 cuts samples off after about 13 turns), as in
# the other SWE-bench Verified dirs; --time-limit 14400 caps wall time per sample as a runaway guard.
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.8-Flash-Next}"
PORT="${PORT:-8003}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_evals/swe_bench \
  -T image_name_template='swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest' \
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
