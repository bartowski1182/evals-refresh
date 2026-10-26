#!/usr/bin/env bash
# Local glue, specific to one server: server and client environments, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt GPUs/environments to your own host.
# This host is an 8x H200 node. Muse Spark support is not in a vLLM release (see README, Setup), so the server runs from
# a podman image built from the vLLM branch (VLLM_IMAGE, or DEFAULT_VLLM_IMAGE below) or from a venv with the branch
# installed (e.g. VLLM_VENV=~/.venvs/vllm-muse ./local_orchestrator.sh). Set at most one.
# tools/vllm_server.sh at the repo root starts serve.sh unchanged in either; its header says what the podman mode mounts
# and why. INSPECT_VENV selects the client venv.
# The checkpoint (453 GB of weights) takes all 8 GPUs as a single TP=8 replica with expert parallelism, the recipe's
# single-node strategy.
# HF_HUB_OFFLINE=1: the checkpoint is pre-downloaded to MODEL_PATH (453 GB; hf download, on local disk).
# VLLM_ENGINE_READY_TIMEOUT_S=3600: loading the checkpoint and the first compile of a new config take long.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-8000}"
export TP="${TP:-8}" DP="${DP:-1}"
export MODEL_PATH="${MODEL_PATH:-$HOME/Muse-Spark-Open-816B-A42B-MXFP4xFP8}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
DEFAULT_VLLM_IMAGE="localhost/vllm-dev:latest"
# The image's vLLM is an editable install of this source checkout (the vLLM branch with Muse Spark support), mounted
# where the image expects it.
VLLM_SRC="${VLLM_SRC:-$HOME/vllm}"
VLLM_PODMAN_ARGS=(-v "$VLLM_SRC:/workspace/vllm")
INSPECT_VENV="${INSPECT_VENV:-$HOME/.venvs/inspect}"
source "$(dirname "${BASH_SOURCE[0]}")/../../../tools/vllm_server.sh"
mkdir -p logs

vllm_start "$PORT" "$CUDA_VISIBLE_DEVICES" logs/serve.log ./serve.sh \
  MODEL MODEL_PATH PORT TP DP MAX_MODEL_LEN VLLM_USE_RUST_FRONTEND
vllm_wait

source "$INSPECT_VENV/bin/activate"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
./run_eval.sh "$@"
