#!/usr/bin/env bash
# Local glue, specific to one server: container, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs/image to your own environment.
# This host is an 8x H200 node. Muse Spark support is not in a vLLM release (see README, Setup), so serve.sh runs
# inside a podman image built from the vLLM branch (`vllm-dev` here); the Inspect client runs from a venv on the host.
# The task directory and the checkpoint are mounted at the same paths, so ./serve.sh runs unchanged inside the container.
# The checkpoint (453 GB of weights) takes all 8 GPUs as a single TP=8 replica with expert parallelism, the recipe's
# single-node strategy.
# --device nvidia.com/gpu=all: GPU access through the NVIDIA Container Toolkit's CDI spec (`nvidia-ctk cdi generate
# --output=/etc/cdi/nvidia.yaml` once per host); CUDA_VISIBLE_DEVICES then selects GPUs inside the container.
# --ipc host: NCCL and vLLM's TP workers share memory. --pids-limit -1: podman's default of 2048 tasks is exhausted by
# vLLM's processes, each starting one thread per CPU. --init forwards `podman stop`'s SIGTERM to vLLM, which ignores
# it as PID 1. label=disable: with SELinux enforcing, the container cannot list the bind-mounted directories otherwise.
# Only PORT is published, on 127.0.0.1: `localhost` resolves to ::1 first and the server listens on IPv4.
# ~/.cache/vllm-dev keeps vLLM's torch.compile and FlashInfer JIT caches across runs: the first start of a new config
# compiles for several minutes, past vLLM's default 600 s engine start-up budget (VLLM_ENGINE_READY_TIMEOUT_S).
# --ulimit memlock/stack/nproc: unlimited pinned memory and processes, 64 MB thread stacks, for NCCL and the TP workers.
# HF_HUB_OFFLINE=1: the checkpoint is pre-downloaded to MODEL_PATH (453 GB; hf download, on local disk).
# The container runs as root (vLLM's Dockerfile), so /root/.cache is its cache directory.
export PATH="$HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PORT="${PORT:-8000}"
export TP="${TP:-8}" DP="${DP:-1}"
export MODEL_PATH="${MODEL_PATH:-$HOME/Muse-Spark-Open-816B-A42B-MXFP4xFP8}"
IMAGE="${IMAGE:-localhost/vllm-dev:latest}"
# The image's vLLM is an editable install of this source checkout (the vLLM branch with Muse Spark support), mounted
# where the image expects it.
VLLM_SRC="${VLLM_SRC:-$HOME/vllm}"
mkdir -p logs ~/.cache/vllm-dev

podman run --rm --name "vllm-$PORT" --init --pids-limit -1 --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nproc=-1:-1 \
  --device nvidia.com/gpu=all --security-opt label=disable \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -e CUDA_VISIBLE_DEVICES -e PORT -e TP -e DP -e MODEL_PATH -e MAX_MODEL_LEN -e VLLM_USE_RUST_FRONTEND \
  -e HF_HUB_OFFLINE=1 -e HF_TOKEN -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}" \
  -v "$MODEL_PATH:$MODEL_PATH:ro" -v "$VLLM_SRC:/workspace/vllm" -v ~/.cache/vllm-dev:/root/.cache \
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
