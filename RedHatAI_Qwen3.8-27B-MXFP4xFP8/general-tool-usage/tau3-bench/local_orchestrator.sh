#!/usr/bin/env bash
# Local glue, specific to one server: container, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs/image to your own environment.
# This host is an 8-GPU NVIDIA node. serve.sh runs inside the vLLM 0.32.0 podman image (IMAGE, pinned by digest; see
# README, Setup) and the Inspect client runs from a venv on the host. The task directory and the host's HF cache are
# mounted where serve.sh and vLLM expect them, so ./serve.sh runs unchanged inside the container and serves the
# checkpoint from the Hub id (MODEL, through the mounted cache) or from a local copy (MODEL_PATH, mounted read-only).
# Identical in the three dirs of the Qwen3.8-27B comparison set apart from the MODEL_PATH hint.
# Two vLLM servers on this host, each in its own container of the image: the model under test on GPUs 0-3 (port PORT)
# and the Qwen/Qwen3.8-27B user simulator on GPUs 4-7 (port USER_PORT), one TP=4 replica each (a single replica per
# model, so every turn of a conversation hits one prefix cache). The simulator is bound to Inspect's `user` model role
# in run_eval.sh (OPENAI_BASE_URL, with the NL-assertion judge).
# --device nvidia.com/gpu=all: GPU access through the NVIDIA Container Toolkit's CDI spec (`nvidia-ctk cdi generate
# --output=/etc/cdi/nvidia.yaml` once per host); CUDA_VISIBLE_DEVICES then selects GPUs inside the container.
# --ipc host: NCCL and vLLM's TP workers share memory. --pids-limit -1: podman's default of 2048 tasks is exhausted by
# vLLM's processes, each starting one thread per CPU. --init forwards `podman stop`'s SIGTERM to vLLM, which ignores
# it as PID 1. label=disable: with SELinux enforcing, the container cannot list the bind-mounted directories otherwise.
# PORT is published on 127.0.0.1 (`localhost` resolves to ::1 first); USER_PORT on all interfaces, because the
# task containers (rootless Docker) call the user simulator at the host's primary IP, CONTAINER_HOST_IP.
# ~/.cache/vllm-podman keeps vLLM's torch.compile and FlashInfer JIT caches across runs: the first start of a new
# config compiles for several minutes, past vLLM's default 600 s engine start-up budget (VLLM_ENGINE_READY_TIMEOUT_S).
# --ulimit memlock/stack/nproc: unlimited pinned memory and processes, 64 MB thread stacks, for NCCL and the TP workers.
# HF_HOME is mounted on the container's HF cache: download the checkpoint once on the host (`hf download`) and export
# HF_HUB_OFFLINE=1, so that the engines don't call the Hub at start-up (anonymous calls are rate-limited per IP);
# HF_TOKEN and HF_HUB_OFFLINE are passed through when set.
# The container runs as root (vLLM's Dockerfile), so /root/.cache is its cache directory.
export TP="${TP:-4}" DP="${DP:-1}"
export PORT="${PORT:-8000}"
export USER_PORT="${USER_PORT:-8001}" USER_TP="${USER_TP:-4}"
# export MODEL_PATH=/path/to/Qwen3.8-27B-MXFP4xFP8   # serve a local copy of the checkpoint instead of the HF id (mounted below)
IMAGE="${IMAGE:-docker.io/vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Sandboxes on a rootless Docker daemon (see local_env_notes.md); drop the export with a rootful daemon. Every sample is
# a two-service compose project with its own network: the daemon's default-address-pools must hold at least
# --max-sandboxes networks (Docker's default runs out near 40). podman only serves the two models.
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
# The task containers call the user-simulator / judge server on the host: rootless containers reach the host at its
# primary IP (not localhost); run_eval.sh builds OPENAI_BASE_URL from it.
export CONTAINER_HOST_IP="${CONTAINER_HOST_IP:-$(hostname -I | awk '{print $1}')}"
mkdir -p logs ~/.cache/vllm-podman "$HF_HOME"

podman run --rm --name "vllm-$PORT" --init --pids-limit -1 --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nproc=-1:-1 \
  --device nvidia.com/gpu=all --security-opt label=disable \
  -p "127.0.0.1:${PORT}:${PORT}" \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e PORT -e TP -e DP -e MODEL -e MODEL_PATH \
  -e HF_TOKEN -e HF_HUB_OFFLINE -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
  -v ~/.cache/vllm-podman:/root/.cache -v "$HF_HOME:/root/.cache/huggingface" \
  ${MODEL_PATH:+-v "$MODEL_PATH:$MODEL_PATH:ro"} \
  -v "$PWD:$PWD" -w "$PWD" \
  --entrypoint ./serve.sh "$IMAGE" > logs/serve.log 2>&1 &
SERVE_PID=$!
podman run --rm --name "vllm-$USER_PORT" --init --pids-limit -1 --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nproc=-1:-1 \
  --device nvidia.com/gpu=all --security-opt label=disable \
  -p "${USER_PORT}:${USER_PORT}" \
  -e CUDA_VISIBLE_DEVICES=4,5,6,7 -e USER_MODEL -e USER_PORT -e USER_TP \
  -e HF_TOKEN -e HF_HUB_OFFLINE -e VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}" \
  -v ~/.cache/vllm-podman:/root/.cache -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$PWD:$PWD" -w "$PWD" \
  --entrypoint ./assets/serve_user_model.sh "$IMAGE" > logs/serve_user.log 2>&1 &
USER_SERVE_PID=$!
for p in "$PORT" "$USER_PORT"; do
  until curl -sf "http://127.0.0.1:${p}/health" >/dev/null; do
    # give up if a server died during startup (e.g. a GPU was busy), instead of polling forever
    if ! kill -0 $SERVE_PID 2>/dev/null || ! kill -0 $USER_SERVE_PID 2>/dev/null; then
      echo "a vLLM server exited before becoming healthy, see logs/serve.log and logs/serve_user.log" >&2
      podman stop -t 60 "vllm-$PORT" "vllm-$USER_PORT" >/dev/null 2>&1; exit 1
    fi
    sleep 10
  done
done

source ~/.venvs/inspect-harbor/bin/activate
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
./run_eval.sh "$@"

podman stop -t 60 "vllm-$PORT" "vllm-$USER_PORT" >/dev/null
