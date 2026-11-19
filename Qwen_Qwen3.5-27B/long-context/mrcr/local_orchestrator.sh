#!/usr/bin/env bash
# Local glue, specific to one server: venvs, GPUs, serve in background, eval, tear down.
# User request for this host: DP=4 on CUDA_VISIBLE_DEVICES=2,3,6,7.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,6,7}"
export PORT="${PORT:-8000}"
# Server uses the default HF hub cache (weights under ~/.cache/huggingface/hub).
# Do not export HF_HUB_OFFLINE to the Inspect client.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
mkdir -p logs

setsid bash -c 'source ~/.venvs/vllm/bin/activate && export HF_HUB_OFFLINE=1 && exec ./serve.sh' > logs/serve.log 2>&1 &
SERVE_PID=$!
SERVE_PGID=$(ps -o pgid= -p "$SERVE_PID" | tr -d ' ')

cleanup() {
  if [ -n "${SERVE_PGID:-}" ]; then
    kill -9 -- "-$SERVE_PGID" 2>/dev/null || true
  fi
  local gpu
  for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
    while true; do
      used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo 0)
      [ "${used:-0}" = "0" ] && break
      sleep 2
    done
  done
}
trap cleanup EXIT

until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

unset HF_HUB_OFFLINE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
source ~/.venvs/inspect/bin/activate
./run_eval.sh "$@"
