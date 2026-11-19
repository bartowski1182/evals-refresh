#!/usr/bin/env bash
# Local glue: venvs, GPUs, serve, eval, tear down.
# User request: DP=2 TP=2 on CUDA_VISIBLE_DEVICES=2,3,6,7.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,6,7}"
export PORT="${PORT:-8000}"
export TP="${TP:-2}"
export DP="${DP:-2}"
# Weights live under /data/imarkov/hub (readable). Do not export HF_HUB_OFFLINE to the client.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
# TP×DP on non-contiguous CUDA_VISIBLE_DEVICES: SymmMem / FlashInfer-MNNVL
# hang in multicast rendezvous (IMEX/FABRIC not permitted). Force NCCL for
# standalone AR; force trtllm for the eager fused allreduce+RMS path (GLM
# always calls it; auto picks mnnvl and never returns).
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}"
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC="${VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:-0}"
export VLLM_FLASHINFER_ALLREDUCE_BACKEND="${VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}"
mkdir -p logs

setsid bash -c 'source ~/.venvs/vllm/bin/activate && export HF_HUB_CACHE="${HF_HUB_CACHE:-/data/imarkov/hub}" HF_HOME="${HF_HOME:-$HF_HUB_CACHE}" HF_HUB_OFFLINE=1 VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}" VLLM_ALLREDUCE_USE_FLASHINFER="${VLLM_ALLREDUCE_USE_FLASHINFER:-0}" VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC="${VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC:-0}" VLLM_FLASHINFER_ALLREDUCE_BACKEND="${VLLM_FLASHINFER_ALLREDUCE_BACKEND:-trtllm}" PYTHONUNBUFFERED=1 && exec ./serve.sh' > logs/serve.log 2>&1 &
SERVE_PID=$!
SERVE_PGID=$(ps -o pgid= -p "$SERVE_PID" | tr -d ' ')

cleanup() {
  if [ -n "${SERVE_PGID:-}" ]; then
    kill -9 -- "-$SERVE_PGID" 2>/dev/null || true
  fi
  local gpu
  for gpu in ${CUDA_VISIBLE_DEVICES//,/ }; do
    local tries=0
    while true; do
      used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' || echo 0)
      # allow a few MiB of driver residual so teardown cannot hang forever
      [ "${used:-0}" -le 32 ] 2>/dev/null && break
      tries=$((tries + 1))
      [ "$tries" -ge 60 ] && break
      sleep 2
    done
  done
}
trap cleanup EXIT

until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

# Client keeps the default HF cache so openai/mrcr resolves; model weights are
# symlinked into ~/.cache/huggingface/hub from /data/imarkov/hub.
unset HF_HUB_CACHE HF_HOME HF_HUB_OFFLINE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
source ~/.venvs/inspect/bin/activate
./run_eval.sh "$@"
