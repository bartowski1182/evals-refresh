# Start, wait for and stop vLLM servers from a venv or a podman image. Sourced by local_orchestrator.sh; not run.
#
# The server environment comes from one of two variables; set at most one, or sourcing this file fails:
#   VLLM_VENV=/path/to/venv         a venv directory (must contain bin/activate)
#   VLLM_IMAGE=<image reference>    a podman image, e.g. docker.io/vllm/vllm-openai:v0.32.0@sha256:...
# With neither set, the podman image DEFAULT_VLLM_IMAGE is used, which the orchestrator sets to the environment that
# produced its dir's results. A leftover export of the other variable would otherwise switch the vLLM build silently.
# serve.sh runs unchanged in both: from the activated venv, or as the container's entrypoint with the task dir mounted
# at its host path. The first line of each server log records the environment used.
#
#   vllm_start [--public] <port> <gpus> <log> <script> [VAR ...]   start <script> in the background on <gpus>
#   vllm_wait                                                      poll /health of every started server
#   vllm_stop                                                      stop every started server (also run on exit)
#   vllm_freeze                                                    `pip freeze` of the selected environment
#
# VAR ...: the variables <script> reads (MODEL, PORT, TP, ...). A venv server inherits the whole environment, so they
# only matter for podman, which forwards each one that is set (`-e VAR`); unset ones fall back to the script's defaults.
# --public: publish the podman port on all interfaces, for servers that task sandboxes call at the host's IP (a venv
# server binds whatever serve.sh says). Otherwise the port is published on 127.0.0.1 (`localhost` resolves to ::1 first).
#
# Podman mode is for NVIDIA hosts (see local_env_notes.md, "Serving from a podman image on NVIDIA"):
# --device nvidia.com/gpu=all: GPU access through the NVIDIA Container Toolkit's CDI spec (`nvidia-ctk cdi generate
# --output=/etc/cdi/nvidia.yaml` once per host); CUDA_VISIBLE_DEVICES then selects GPUs inside the container.
# --ipc host: NCCL and vLLM's TP workers share memory. --pids-limit -1: podman's default of 2048 tasks is exhausted by
# vLLM's processes, each starting one thread per CPU. --init forwards `podman stop`'s SIGTERM to vLLM, which ignores
# it as PID 1. label=disable: with SELinux enforcing, the container cannot list the bind-mounted directories otherwise.
# --ulimit memlock/stack/nproc: unlimited pinned memory and processes, 64 MB thread stacks, for NCCL and the TP workers.
# ~/.cache/vllm-podman keeps vLLM's torch.compile and FlashInfer JIT caches across runs (the container runs as root,
# so /root/.cache is its cache directory). HF_HOME is mounted on the container's HF cache: download the checkpoint once
# on the host (`hf download`) and export HF_HUB_OFFLINE=1, so that the engines don't call the Hub at start-up
# (anonymous calls are rate-limited per IP); HF_TOKEN and HF_HUB_OFFLINE are passed through when set. MODEL_PATH, when
# set, is mounted read-only at its host path. VLLM_PODMAN_ARGS, an array the orchestrator may set before calling
# vllm_start, adds `podman run` arguments (e.g. the mount of a vLLM source checkout an image installs in editable mode).
# Both modes: the first start of a new config compiles for several minutes, past vLLM's default 600 s engine start-up
# budget, hence VLLM_ENGINE_READY_TIMEOUT_S=1800 unless set.

if [ -n "${VLLM_VENV:-}" ] && [ -n "${VLLM_IMAGE:-}" ]; then
  echo "set VLLM_VENV or VLLM_IMAGE, not both (VLLM_VENV=$VLLM_VENV, VLLM_IMAGE=$VLLM_IMAGE)" >&2; exit 1
elif [ -n "${VLLM_VENV:-}" ]; then
  [ -f "$VLLM_VENV/bin/activate" ] || { echo "VLLM_VENV=$VLLM_VENV has no bin/activate" >&2; exit 1; }
  VLLM_MODE=venv
else
  VLLM_IMAGE="${VLLM_IMAGE:-${DEFAULT_VLLM_IMAGE:?set VLLM_VENV, VLLM_IMAGE or DEFAULT_VLLM_IMAGE before sourcing vllm_server.sh}}"
  VLLM_MODE=podman
fi
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-1800}"
_VLLM_PIDS=() _VLLM_PORTS=() _VLLM_LOGS=()

vllm_start() {
  local publish=127.0.0.1:
  [ "$1" = --public ] && { publish=; shift; }
  local port=$1 gpus=$2 log=$3 script=$4; shift 4
  echo "# vLLM env ($VLLM_MODE): ${VLLM_VENV:-$VLLM_IMAGE}" > "$log"
  if [ "$VLLM_MODE" = venv ]; then
    ( source "$VLLM_VENV/bin/activate" && CUDA_VISIBLE_DEVICES="$gpus" exec "$script" ) >> "$log" 2>&1 &
  else
    local hf_home="${HF_HOME:-$HOME/.cache/huggingface}" fwd=() v
    for v in "$@" HF_TOKEN HF_HUB_OFFLINE VLLM_ENGINE_READY_TIMEOUT_S; do fwd+=(-e "$v"); done
    mkdir -p ~/.cache/vllm-podman "$hf_home"
    podman run --rm --name "vllm-$port" --init --pids-limit -1 --ipc host \
      --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nproc=-1:-1 \
      --device nvidia.com/gpu=all --security-opt label=disable \
      -p "${publish}${port}:${port}" \
      -e CUDA_VISIBLE_DEVICES="$gpus" "${fwd[@]}" \
      -v ~/.cache/vllm-podman:/root/.cache -v "$hf_home:/root/.cache/huggingface" \
      ${MODEL_PATH:+-v "$MODEL_PATH:$MODEL_PATH:ro"} "${VLLM_PODMAN_ARGS[@]}" \
      -v "$PWD:$PWD" -w "$PWD" \
      --entrypoint "$script" "$VLLM_IMAGE" >> "$log" 2>&1 &
  fi
  _VLLM_PIDS+=($!) _VLLM_PORTS+=("$port") _VLLM_LOGS+=("$log")
  trap vllm_stop EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
}

vllm_wait() {
  local i
  for i in "${!_VLLM_PORTS[@]}"; do
    until curl -sf "http://127.0.0.1:${_VLLM_PORTS[i]}/health" >/dev/null; do
      # give up if a server died during startup (e.g. a GPU was busy), instead of polling forever; the EXIT trap
      # stops the others
      kill -0 "${_VLLM_PIDS[i]}" 2>/dev/null || {
        echo "vLLM server on port ${_VLLM_PORTS[i]} exited before becoming healthy, see ${_VLLM_LOGS[i]}" >&2; exit 1; }
      sleep 10
    done
  done
}

vllm_stop() {
  local i pid
  for i in "${!_VLLM_PIDS[@]}"; do
    pid=${_VLLM_PIDS[i]}
    if [ "$VLLM_MODE" = podman ]; then
      # also when the `podman run` client is gone: the container can outlive it
      podman stop -t 60 "vllm-${_VLLM_PORTS[i]}" >/dev/null 2>&1
    elif kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -9 "$pid" 2>/dev/null
    fi
  done
  _VLLM_PIDS=() _VLLM_PORTS=() _VLLM_LOGS=()
}

vllm_freeze() {
  if [ "$VLLM_MODE" = venv ]; then
    "$VLLM_VENV/bin/python" -m pip freeze
  else
    podman run --rm --entrypoint python3 "$VLLM_IMAGE" -m pip freeze
  fi
}
