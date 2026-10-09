#!/usr/bin/env bash
# Local glue, specific to one server: venvs, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs to your own environment.
# This host (8x MI355X, RHEL 9, glibc 2.34) cannot install vLLM's ROCm wheels (they need glibc >= 2.39), so serve.sh
# runs inside the official vLLM ROCm image under rootless podman (image store under ~/.local/share/containers).
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8000}"
IMAGE="${IMAGE:-docker.io/vllm/vllm-openai-rocm:v0.31.0@sha256:749f6f3f944f12af49966ac523c1f8573e4b229594b541954bd5f879b4496b1f}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/hf_hub_cache}"
# Sandboxes run on a rootless Docker daemon (image store under ~/.local/share/docker; see local_env_notes.md).
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
mkdir -p logs ~/.cache/vllm-rocm

# ~/.cache/vllm-rocm keeps vLLM's compile caches across runs. --pids-limit -1: podman's default of 2048 tasks per
# container is exhausted by 16 vLLM processes (8 API servers, 8 engines) each starting a thread per CPU.
# Only PORT is published (no --network host), so two servers on disjoint GPUs cannot clash on vLLM's internal ports;
# it is published on 127.0.0.1 only because `localhost` resolves to ::1 first and the server listens on IPv4.
# HF_HUB_OFFLINE=1: the weights are pre-downloaded (see README); Hub API calls from 8 engines get rate-limited (429).
podman run --rm --device /dev/kfd --device /dev/dri --group-add keep-groups --security-opt seccomp=unconfined \
  -p "127.0.0.1:${PORT}:${PORT}" --ipc host --pids-limit -1 \
  -e HIP_VISIBLE_DEVICES -e PORT -e TP -e DP -e HF_HUB_OFFLINE=1 -e HF_HUB_CACHE=/hf_hub_cache \
  -v "$HF_HUB_CACHE":/hf_hub_cache -v ~/.cache/vllm-rocm:/root/.cache -v "$PWD":/task -w /task \
  --entrypoint bash "$IMAGE" ./serve.sh > logs/serve.log 2>&1 &
SERVE_PID=$!
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
  # give up if the server died during startup (e.g. a GPU was busy), instead of polling forever
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

source ~/.venvs/inspect-harbor/bin/activate
./run_eval.sh "$@"

kill $SERVE_PID
