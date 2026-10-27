#!/usr/bin/env bash
# SWE-bench Verified on RedHatAI/GLM-5.3-Flash-MXFP4xFP8 against an already-running vLLM server.
#
# What this measures: the 500 instances of SWE-bench Verified (HF princeton-nlp/SWE-bench_Verified, revision pinned by
# inspect_evals 0.20.0), solved agentically with inspect_evals's default ReAct agent (python, bash_session, text_editor;
# 210 s tool timeout), no internet. The official swebench harness grades FAIL_TO_PASS and PASS_TO_PASS tests. The
# GLM-5.3-Flash card does not report SWE-bench Verified.
#
# -T image_name_template selects the official SWE-bench images (Docker Hub, built with the current swebench harness,
# which removes the remote, deletes tags newer than the base commit and prunes unreachable objects). inspect_evals'
# default Epoch AI images (ghcr.io) keep every release tag of the project, so the reference fix of an instance can be
# read offline with `git show <later tag>:<file>` (see README).
#
# Sampling params come from zai-org/GLM-5.3-Flash's generation_config.json (temperature 1.0, top_p 0.95; no top_k,
# min_p or penalties, so vLLM's defaults apply), which the quantized checkpoint is expected to carry unchanged
# (verify before the run, see README). GLM-5.3-Flash has thinking always on; --reasoning-effort is forwarded by
# Inspect as the OpenAI `reasoning_effort` field and vLLM turns it into chat_template_kwargs {"reasoning_effort":
# "max"} (max is the highest of low/high/max, the default, and what the card prescribes for benchmark reproduction).
# --max-tokens is per turn; 65536 leaves room for several turns inside the 262k context.
# --message-limit 200 (about 100 agent turns; the package default of 30 cuts samples off after about 13 turns), as in
# the other SWE-bench Verified dirs; --time-limit 14400 caps wall time per sample as a runaway guard.
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample. 48 in flight, what the
# GLM-5.3-Flash-NVFP4 run used; read the "GPU KV cache size" line in logs/serve.log before the full run (BF16 KV on
# Hopper) and lower the concurrency if the histories do not fit, see the DeepSWE dir.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/GLM-5.3-Flash-MXFP4xFP8}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_evals/swe_bench \
  -T image_name_template='swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest' \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 \
  --reasoning-effort max \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --message-limit "${MESSAGE_LIMIT:-200}" \
  --max-connections "${MAX_CONNECTIONS:-48}" \
  --max-sandboxes "${MAX_SANDBOXES:-48}" \
  --time-limit "${TIME_LIMIT:-14400}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
