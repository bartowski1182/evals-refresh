#!/usr/bin/env bash
# Local glue, specific to one server: venvs, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs to your own environment.
# This host: 8x H200; the model is the only tenant and takes all 8 GPUs (TP=8, the recipe's topology).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT="${PORT:-8000}"
# Checkpoint and HF caches live on the big local /data volume; weights are pre-downloaded there.
export HF_HOME="${HF_HOME:-/data/playground/dsikka/hf}"
# Loading the 415 GB MXFP4/FP8 checkpoint across 8 GPUs plus first-start compilation takes longer
# than vLLM's default 600 s engine start-up budget.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
# Rootless Docker (image store on /data, see local_env_notes.md) with its own static CLI and the compose plugin.
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
export PATH="$HOME/.local/docker-rootless/bin:$PATH"
mkdir -p logs

source /data/playground/dsikka/venvs/vllm-glm53/bin/activate
HF_HUB_OFFLINE=1 ./serve.sh > logs/serve.log 2>&1 &
SERVE_PID=$!
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
  # give up if the server died during startup (e.g. a GPU was busy), instead of polling forever
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

source /data/playground/dsikka/venvs/inspect-harbor-fix/bin/activate
./run_eval.sh "$@"

kill $SERVE_PID
