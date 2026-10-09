#!/usr/bin/env bash
# Local glue, specific to one server: venvs, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs to your own environment.
# The BF16 model takes all 8 GPUs (port 8003) for this task, unlike the other BF16 dirs (GPUs 4-7).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT="${PORT:-8003}"
export MODEL_PATH="${MODEL_PATH:-/data/eldarkurtic/hf_models/Qwen/Qwen3.8-Flash-Next}"
# One TP=8 replica. One replica because vLLM's prefix cache is per replica and DP routes each turn of a sample to any
# replica, so multi-turn agent runs would prefill their history again on every turn. TP=8 rather than TP=4 because a
# BF16 TP=4 replica has room for 9.5M KV-cache tokens (the NVFP4 TP=4 replicas: 20.4M); with 56 long agent sessions
# it evicts their histories and re-prefills them every turn. TP=8 has room for 17.1M.
export TP="${TP:-8}" DP="${DP:-1}"
# Copying the 51B n-gram embedding table from disk into pinned host memory takes several minutes, more than
# vLLM's default 600 s engine start-up budget.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
# Rootless Docker (image store on /data, see local_env_notes.md) with its own static CLI and the compose plugin.
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
export PATH="$HOME/.local/docker-rootless/bin:$PATH"
mkdir -p logs

source ~/.venvs/vllm-nightly/bin/activate
./serve.sh > logs/serve.log 2>&1 &
SERVE_PID=$!
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
  # give up if the server died during startup (e.g. a GPU was busy), instead of polling forever
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

# inspect_harbor from the eldarkurtic fork: applies [agent] network_mode = no-network (see README, Patches).
source ~/.venvs/inspect-harbor-fix/bin/activate
./run_eval.sh "$@"

kill $SERVE_PID
