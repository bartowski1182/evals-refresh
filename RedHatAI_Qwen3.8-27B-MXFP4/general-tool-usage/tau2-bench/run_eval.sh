#!/usr/bin/env bash
# tau2-bench (inspect_evals native implementation) on RedHatAI/Qwen3.8-27B-MXFP4 against an already-running vLLM
# server, with Qwen/Qwen3.8-27B as the user simulator on a second vLLM server (USER_PORT).
#
# What this measures: the 375 "base" tasks of tau2-bench >= 1.0 (airline 50, retail 114, telecom 114, banking_knowledge 97;
# the same task set as the Harbor tau3-bench package, minus its voice modality), run with inspect_evals's native
# implementation: the domain tools are Python tools in-process (no sandbox), a user-simulator model plays the
# customer (model role `user`, temperature 0 as in tau2-bench), and each conversation is scored on the final
# database state plus the information the agent had to communicate (pass@1 per task; the airline natural-language
# assertions are not graded by this implementation). No model card reports tau2/tau3-bench.
#
# The user simulator is Qwen/Qwen3.8-27B for every model of this repo (a fixed simulator, as on the tau-bench
# leaderboards). It runs on its own server (see local_orchestrator.sh; the Gemma pattern) and is bound to the `user`
# role with its URL, in all three dirs of the comparison set, so it is served identically for the three models and a
# quantized model under test never plays its own customer.
# Identical in the three dirs of the Qwen3.8-27B comparison set (Qwen/Qwen3.8-27B, RedHatAI/Qwen3.8-27B-MXFP4,
# RedHatAI/Qwen3.8-27B-MXFP4xFP8) apart from the MODEL default; one epoch (pass@1) in each.
# --max-tokens is per turn. --time-limit caps wall time per sample as a runaway guard; the task's own loop limit is
# 100 user/agent exchanges (task parameter message_limit). No sandbox: --max-connections bounds in-flight samples.
# -M client_timeout raises Inspect's HTTP timeout (default 600s) for long reasoning turns.
# --fail-on-error 0.1 aborts only if >10% of samples hit infrastructure errors.
#
# Sampling params: temperature 1.0, top_p 0.95, top_k 20, the generation_config.json values of Qwen/Qwen3.8-27B and
# RedHatAI/Qwen3.8-27B-MXFP4 (the MXFP4xFP8 repository was not public when these dirs were written: check its
# generation_config.json before running it). Inspect drops --top-k; vLLM applies the checkpoint's value as the server
# default (logged at start-up). The task's own temperature=0.0 only applies to the user simulator
# (Inspect's get_model() returns the eval's model with its CLI config).
# --reasoning-effort is forwarded by Inspect as the OpenAI `reasoning_effort` field; vLLM turns it
# into chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"} (see the recipe).
set -euo pipefail

MODEL="${MODEL:-RedHatAI/Qwen3.8-27B-MXFP4}"
PORT="${PORT:-8000}"
USER_MODEL="${USER_MODEL:-Qwen/Qwen3.8-27B}"
USER_PORT="${USER_PORT:-8001}"
USER_BASE_URL="${USER_BASE_URL:-http://localhost:${USER_PORT}/v1}"   # the user simulator's server
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

inspect eval inspect_evals/tau2_airline inspect_evals/tau2_retail inspect_evals/tau2_telecom inspect_evals/tau2_banking \
  --model "vllm/${MODEL}" \
  --model-role "user={model: vllm/${USER_MODEL}, model_args: {base_url: ${USER_BASE_URL}, client_timeout: 7200}}" \
  --epochs "$EPOCHS" --epochs-reducer mean \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --reasoning-effort xhigh \
  --max-tokens "${MAX_TOKENS:-65536}" \
  --max-connections "${MAX_CONNECTIONS:-56}" \
  --time-limit "${TIME_LIMIT:-3600}" \
  --fail-on-error 0.1 \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
