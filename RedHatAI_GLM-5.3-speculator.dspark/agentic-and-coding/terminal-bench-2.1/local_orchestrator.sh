#!/usr/bin/env bash
# Local glue, specific to one server: venvs, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs to your own environment.
#
# This host: 8x B200 (183 GB each). The target is ~756 GB of FP8 weights, so it takes all 8 GPUs
# (TP=8, DP=1 -- the recipe's single-node topology).
#
# This host is shared: never touch the GPUs without a canhazgpu reservation. This script therefore
# refuses to run without CUDA_VISIBLE_DEVICES and tells you how to reserve. Launch it as:
#   canhazgpu run --gpus 8 --note "TB-2.1 GLM-5.3 DSpark" -- ./local_orchestrator.sh --limit 2
# (or hold a manual reservation with `canhazgpu reserve` and export exactly the reserved GPU ids).
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "CUDA_VISIBLE_DEVICES is not set. Reserve GPUs first, e.g.:" >&2
  echo "  canhazgpu run --gpus 8 -- ./local_orchestrator.sh \"\$@\"" >&2
  exit 1
fi
# serve.sh defaults to TP=8 x DP=1; refuse to start a server that needs more GPUs than were reserved.
TP="${TP:-8}"
DP="${DP:-1}"
NGPU=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
if (( NGPU < TP * DP )); then
  echo "TP=$TP x DP=$DP needs $((TP * DP)) GPUs, but CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES has $NGPU." >&2
  echo "Reserve more (canhazgpu run --gpus $((TP * DP)) -- ...) or lower TP." >&2
  exit 1
fi

# Docker on this host is rootful and the daemon is shared (other tenants' containers are running on it);
# our user is in the docker group, so DOCKER_HOST stays unset. The image store lives under
# /var/lib/docker. Docker Compose v2 is installed user-locally in ~/.docker/cli-plugins.
PORT="${PORT:-8000}"
# Checkpoint and HF caches live on the big local /data volume; weights are pre-downloaded there.
export HF_HOME="${HF_HOME:-/data/reed-meyerson/hf}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Loading the ~756 GB FP8 checkpoint across 8 GPUs plus first-start kernel compilation takes longer
# than vLLM's default 600 s engine start-up budget.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
mkdir -p logs

source ~/.venvs/vllm/bin/activate
# setsid: serve.sh and its vLLM workers get their own process group, so teardown can kill the whole
# tree. A bare `kill $SERVE_PID` only signals the launcher; EngineCore/Worker grandchildren survive
# and keep their GPU memory while canhazgpu has already released the reservation.
HF_HUB_OFFLINE=1 setsid ./serve.sh > logs/serve.log 2>&1 &
SERVE_PID=$!

wait_healthy() {
  # VLLM_ENGINE_READY_TIMEOUT_S bounds the engine start-up inside vLLM; this bounds the whole wait.
  local deadline=$(( SECONDS + 5400 ))
  until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
    if ! kill -0 "$SERVE_PID" 2>/dev/null; then
      echo "vLLM server died during startup; see logs/serve.log" >&2
      tail -40 logs/serve.log >&2
      exit 1
    fi
    if (( SECONDS > deadline )); then
      echo "vLLM server did not become healthy within 90 minutes; see logs/serve.log" >&2
      tail -40 logs/serve.log >&2
      exit 1
    fi
    sleep 10
  done
}
wait_healthy
echo "vLLM server healthy on port ${PORT}"

source ~/.venvs/inspect-harbor/bin/activate
# Do not let a failing eval (or an Inspect abort) skip the teardown below.
set +e
./run_eval.sh "$@"
STATUS=$?
set -e

# Teardown: SIGKILL the whole process group (vLLM ignores SIGTERM and hangs), then wait until *our*
# GPUs really show 0 MiB before exiting, so canhazgpu's auto-release does not hand out GPUs that our
# own workers are still occupying. Only CUDA_VISIBLE_DEVICES is checked: this host is shared, so
# summing all 8 GPUs would never reach zero while other tenants hold the rest.
echo "teardown: killing vLLM process group $SERVE_PID"
kill -9 -- "-$SERVE_PID" 2>/dev/null || kill -9 "$SERVE_PID" 2>/dev/null || true
if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -ra OUR_GPUS <<< "$CUDA_VISIBLE_DEVICES"
  for _ in $(seq 1 60); do
    busy=0
    for id in "${OUR_GPUS[@]}"; do
      m=$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
      [[ -n "$m" ]] && (( m > 0 )) && busy=1
    done
    (( busy == 0 )) && break
    sleep 2
  done
  echo "teardown: GPUs $CUDA_VISIBLE_DEVICES now hold: $(for id in "${OUR_GPUS[@]}"; do nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader | tr -d ' '; done | paste -sd' ')"
fi
exit "$STATUS"
