#!/usr/bin/env bash
# Local glue, specific to one server: container, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs/image to your own environment.
# This host is an 8-GPU NVIDIA node. serve.sh runs inside the vLLM 0.32.0 podman image (IMAGE, pinned by digest; see
# README, Setup) and the Inspect client runs from a venv on the host. The task directory and the host's HF cache are
# mounted where serve.sh and vLLM expect them, so ./serve.sh runs unchanged inside the container and serves the
# checkpoint from the Hub id (MODEL, through the mounted cache) or from a local copy (MODEL_PATH, mounted read-only).
# Identical in the three dirs of the Qwen3.8-27B comparison set apart from the MODEL_PATH hint.
# One TP=8 replica instead of serve.sh's TP=2 x DP=4: vLLM's prefix cache is per replica and DP routes each turn
# of a sample to a random replica, so multi-turn agent runs would re-prefill their history on every turn.
# --device nvidia.com/gpu=all: GPU access through the NVIDIA Container Toolkit's CDI spec (`nvidia-ctk cdi generate
# --output=/etc/cdi/nvidia.yaml` once per host); CUDA_VISIBLE_DEVICES then selects GPUs inside the container.
# --ipc host: NCCL and vLLM's TP workers share memory. --pids-limit -1: podman's default of 2048 tasks is exhausted by
# vLLM's processes, each starting one thread per CPU. --init forwards `podman stop`'s SIGTERM to vLLM, which ignores
# it as PID 1. label=disable: with SELinux enforcing, the container cannot list the bind-mounted directories otherwise.
# Only PORT is published, on 127.0.0.1: `localhost` resolves to ::1 first and the server listens on IPv4.
# ~/.cache/vllm-podman keeps vLLM's torch.compile and FlashInfer JIT caches across runs: the first start of a new
# config compiles for several minutes, past vLLM's default 600 s engine start-up budget (VLLM_ENGINE_READY_TIMEOUT_S).
# --ulimit memlock/stack/nproc: unlimited pinned memory and processes, 64 MB thread stacks, for NCCL and the TP workers.
# HF_HOME is mounted on the container's HF cache: download the checkpoint once on the host (`hf download`) and export
# HF_HUB_OFFLINE=1, so that the engines don't call the Hub at start-up (anonymous calls are rate-limited per IP);
# HF_TOKEN and HF_HUB_OFFLINE are passed through when set.
# The container runs as root (vLLM's Dockerfile), so /root/.cache is its cache directory.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-8000}"
export TP="${TP:-8}" DP="${DP:-1}"
# export MODEL_PATH=/path/to/Qwen3.8-27B-MXFP4   # serve a local copy of the checkpoint instead of the HF id (mounted below)
IMAGE="${IMAGE:-docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Sandboxes on a rootless Docker daemon (see local_env_notes.md: data-root on a large disk, containerd snapshotter,
# runc with NoNewKeyring, enough default-address-pools); drop the export with a rootful daemon. podman only serves the model.
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
mkdir -p logs ~/.cache/vllm-podman "$HF_HOME"

podman run --rm --name "vllm-$PORT" --init --pids-limit -1 --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nproc=-1:-1 \
  --device nvidia.com/gpu=all --security-opt label=disable \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -e CUDA_VISIBLE_DEVICES -e PORT -e TP -e DP -e MODEL -e MODEL_PATH \
  -e HF_TOKEN -e HF_HUB_OFFLINE -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
  -v ~/.cache/vllm-podman:/root/.cache -v "$HF_HOME:/root/.cache/huggingface" \
  ${MODEL_PATH:+-v "$MODEL_PATH:$MODEL_PATH:ro"} \
  -v "$PWD:$PWD" -w "$PWD" \
  --entrypoint ./serve.sh "$IMAGE" > logs/serve.log 2>&1 &
SERVE_PID=$!
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do
  # give up if the server died during startup (e.g. a GPU was busy), instead of polling forever
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

source ~/.venvs/inspect/bin/activate
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
./run_eval.sh "$@"

podman stop -t 60 "vllm-$PORT" >/dev/null
