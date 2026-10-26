#!/usr/bin/env bash
# Local glue, specific to one server: server and client environments, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt GPUs/environments to your own host.
# The vLLM server runs from a podman image (VLLM_IMAGE, or DEFAULT_VLLM_IMAGE below: the vLLM 0.32.0 image pinned by
# digest; see README, Setup) or from a venv (e.g. VLLM_VENV=~/.venvs/vllm ./local_orchestrator.sh). Set at most one.
# tools/vllm_server.sh at the repo root starts serve.sh unchanged in either; its header says what the podman mode mounts
# and why. INSPECT_VENV selects the client venv. The checkpoint is served from the Hub id (MODEL, through the HF cache)
# or from a local copy (MODEL_PATH).
# Identical in the three dirs of the Qwen3.8-27B comparison set apart from the MODEL_PATH hint.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-8000}"
# export MODEL_PATH=/path/to/Qwen3.8-27B-MXFP4xFP8   # serve a local copy of the checkpoint instead of the HF id
DEFAULT_VLLM_IMAGE="docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"
INSPECT_VENV="${INSPECT_VENV:-$HOME/.venvs/inspect-harbor}"
# Sandboxes on a rootless Docker daemon (see local_env_notes.md: data-root on a large disk, containerd snapshotter,
# runc with NoNewKeyring, enough default-address-pools); drop the export with a rootful daemon. A podman image only
# serves the model.
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
source "$(dirname "${BASH_SOURCE[0]}")/../../../tools/vllm_server.sh"
mkdir -p logs

vllm_start "$PORT" "$CUDA_VISIBLE_DEVICES" logs/serve.log ./serve.sh MODEL MODEL_PATH PORT TP DP
vllm_wait

source "$INSPECT_VENV/bin/activate"
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
./run_eval.sh "$@"
