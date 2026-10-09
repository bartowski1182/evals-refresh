#!/usr/bin/env bash
# Local glue, specific to one server: container, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs to your own environment.
# This host is 8x AMD MI355X on RHEL 9 (glibc 2.34); vLLM's ROCm wheels need glibc >= 2.39, so the server runs
# from the official vLLM ROCm image with rootless podman (image store under ~/.local/share/containers).
# The task directory is mounted at the same path, so ./serve.sh runs unchanged inside the container.
# vLLM 0.31.0 rather than the base model's 0.29.0: ~15-20% faster decode on this host, same eval config.
# TP stays 1: RCCL's peer-to-peer transport fails in rootless containers here (hipIpcGetMemHandle), so TP>1 does not start.
# --pids-limit -1: podman's default of 2048 is exhausted by vLLM's processes, each starting one BLAS thread per CPU
# (128 here); --init forwards `podman stop`'s SIGTERM to vLLM, which ignores it as PID 1; label=disable because SELinux
# (enforcing here) denies the container directory listings in the bind-mounted HF cache. VLLM_ENGINE_READY_TIMEOUT_S:
# the first start on a new vLLM config compiles for ~10 min here, past vLLM's default of 600 s.
# The published port answers on IPv4 only (rootless podman, pasta) while `localhost` resolves to ::1 first: use 127.0.0.1.
# numactl --interleave=all: NUMA auto-balancing is on here (kernel.numa_balancing=1, no root to turn it off; AMD's ROCm
# tuning guides say to disable it) and stalls vLLM start-up for many minutes; the kernel does not balance processes that
# have an explicit memory policy, and the policy is inherited by the container's processes.
# Sandboxes: rootless Docker (image store under ~/.local/share/docker on /home; ~/.config/docker/daemon.json sets a
# runc runtime with NoNewKeyring, as this host allows 200 kernel keys per user, and default-address-pools for one
# network per sample). See local_env_notes.md.
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
DP="${DP:-8}"
VLLM_IMAGE="${VLLM_IMAGE:-docker.io/vllm/vllm-openai-rocm:v0.31.0}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
export LOG_DIR="${LOG_DIR:-./logs}"   # run_eval.sh writes the .eval here too
mkdir -p "$LOG_DIR" "$HF_CACHE" "$HOME/.cache/vllm"

numactl --interleave=all podman run --rm --name "vllm-$PORT" --init --pids-limit -1 \
  --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined --security-opt label=disable --ipc=host -p "$PORT:$PORT" \
  -e HIP_VISIBLE_DEVICES="$GPUS" -e PORT="$PORT" -e TP="$TP" -e DP="$DP" -e HF_TOKEN -e VLLM_ENGINE_READY_TIMEOUT_S=1800 \
  -v "$HF_CACHE:/root/.cache/huggingface" -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$PWD:$PWD" -w "$PWD" --entrypoint ./serve.sh "$VLLM_IMAGE" > "$LOG_DIR/serve.log" 2>&1 &
SERVE_PID=$!
until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; do
  # give up if the server died during startup (e.g. a GPU was busy), instead of polling forever
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see $LOG_DIR/serve.log" >&2; exit 1; }
  sleep 10
done

source ~/.venvs/inspect-harbor-1.0/bin/activate
export VLLM_BASE_URL="http://127.0.0.1:${PORT}/v1"
./run_eval.sh "$@"

podman stop -t 60 "vllm-$PORT" >/dev/null
