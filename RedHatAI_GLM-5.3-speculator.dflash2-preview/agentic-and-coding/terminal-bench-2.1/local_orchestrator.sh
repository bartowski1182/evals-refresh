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

# Pre-flight: a GPU can be "available" in canhazgpu while the previous tenant's processes are still winding
# down (its memory shows up as used for a few seconds to minutes). vLLM refuses to start if free memory on any
# GPU is below gpu-memory-utilization x total (0.85 x ~178 GiB), so wait until every reserved GPU is really
# idle instead of failing after a 30-minute engine start. Wheels: 90.7 GiB of FP8 weights per B200.
echo "preflight: waiting for reserved GPUs $CUDA_VISIBLE_DEVICES to be idle"
IFS=',' read -ra PRE_GPUS <<< "$CUDA_VISIBLE_DEVICES"
for _ in $(seq 1 120); do   # up to 20 minutes, in 10 s slices
  busy=""
  for id in "${PRE_GPUS[@]}"; do
    m=$(nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    [[ -n "$m" ]] && (( m > 2048 )) && busy="$busy $id:${m}MiB"
  done
  [[ -z "$busy" ]] && break
  echo "preflight: still in use:$busy"
  sleep 10
done
if [[ -n "$busy" ]]; then
  echo "preflight: GPUs never went idle:$busy — aborting instead of failing engine startup" >&2
  exit 1
fi
echo "preflight: GPUs idle"

# vLLM main @ ace415c06 (nightly wheel), NOT 0.32.0: 0.32.0/0.33.0 contain the DFlash acceptance-collapse bug
# (vLLM #63587). ~/.venvs/vllm-nightly holds 0.34.0rc1.dev253+gace415c06, which includes the fix.
source ~/.venvs/vllm-nightly/bin/activate
echo "serve: vllm $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
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

# Serve-side audit for vLLM #63587 (DFlash/DFlash2 silent acceptance collapse). serve.sh already serves with
# PIECEWISE cudagraphs to avoid the bug; these are the two fingerprints the bug report names. Neither is
# fatal by itself (the second is informational on some configs), but both are recorded in the run log.
if grep -q "no KV cache group could be identified as the draft model's" logs/serve.log; then
  echo "WARNING: serve.log has the draft-KV-cache-group warning associated with vLLM #63587;" \
       "monitor_spec_decode.py canary will decide whether acceptance actually collapses." >&2
fi
echo "serve: cudagraph workaround -> $(grep -oE 'cudagraph_mode[^,}]*' logs/serve.log | head -1 || true)"

# Side-channel probe for acceptance collapse, one capped-length request every 2 minutes. It is independent of
# Inspect: the definitive per-call evidence is the eval log, which monitor_spec_decode.py audit post-processes.
CANARY_MODEL="${SPECULATOR:-RedHatAI/GLM-5.3-speculator.dflash2-preview}"
rm -f logs/canary.stop logs/collapse_detected.jsonl
nohup python monitor_spec_decode.py canary \
  --base-url "http://localhost:${PORT}/v1" --model "$CANARY_MODEL" \
  --interval 120 --max-tokens 1024 \
  --log logs/canary.jsonl --flag logs/collapse_detected.jsonl --stop-file logs/canary.stop \
  > logs/canary.log 2>&1 &
CANARY_PID=$!

source ~/.venvs/inspect-harbor/bin/activate
# Do not let a failing eval (or an Inspect abort) skip the teardown below.
set +e
./run_eval.sh "$@"
STATUS=$?
set -e

# Stop the canary (it sleeps in 5 s slices, so it exits promptly), then report where its evidence lives.
touch logs/canary.stop
wait "$CANARY_PID" 2>/dev/null || true
if [[ -s logs/collapse_detected.jsonl ]]; then
  echo "WARNING: canary detected acceptance collapse $(grep -c . logs/collapse_detected.jsonl) time(s); see logs/collapse_detected.jsonl" >&2
else
  echo "canary: no acceptance collapse detected in $(wc -l < logs/canary.jsonl 2>/dev/null || echo 0) probe(s)"
fi

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
      [[ -n "$m" ]] && (( m > 512 )) && busy=1
    done
    (( busy == 0 )) && break
    sleep 2
  done
  echo "teardown: GPUs $CUDA_VISIBLE_DEVICES now hold: $(for id in "${OUR_GPUS[@]}"; do nvidia-smi --id="$id" --query-gpu=memory.used --format=csv,noheader | tr -d ' '; done | paste -sd' ')"
fi
exit "$STATUS"
