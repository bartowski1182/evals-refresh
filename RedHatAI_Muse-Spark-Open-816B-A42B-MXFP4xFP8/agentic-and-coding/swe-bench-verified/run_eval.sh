#!/usr/bin/env bash
# SWE-bench Verified on RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8 against an already-running vLLM server.
#
# What this measures: the 500 instances of SWE-bench Verified (HF princeton-nlp/SWE-bench_Verified, revision pinned by
# inspect_evals 0.20.0), solved agentically with inspect_evals's default ReAct agent (python, bash_session, text_editor;
# 210 s tool timeout), no internet. The official swebench harness grades FAIL_TO_PASS and PASS_TO_PASS tests. The
# Muse Spark Open card does not report SWE-bench Verified.
#
# -T image_name_template selects the official SWE-bench images (Docker Hub, built with the current swebench harness,
# which removes the remote, deletes tags newer than the base commit and prunes unreachable objects). inspect_evals'
# default Epoch AI images (ghcr.io) keep every release tag of the project, so the reference fix of an instance can be
# read offline with `git show <later tag>:<file>` (see README).
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
# --message-limit 200 (about 100 agent turns; the package default of 30 cuts samples off after about 13 turns), as in
# the other SWE-bench Verified dirs; --time-limit 14400 caps wall time per sample as a runaway guard.
# --max-sandboxes equals --max-connections: Inspect holds one container per in-flight sample.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Muse-Spark-Open-816B-A42B-MXFP4xFP8}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_evals/swe_bench \
  -T image_name_template='swebench/sweb.eval.{arch}.{org}_1776_{repo}-{issue}:latest' \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 1.0 --top-k 64 \
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
