#!/usr/bin/env bash
# Local glue, specific to one server: venvs, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs to your own environment.
# The model takes half of the 8 GPUs (0-3, port 8000); the Inferact NVFP4 dirs use GPUs 4-7 and port 8001,
# so the two checkpoints can be evaluated side by side.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PORT="${PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-/data/eldarkurtic/hf_models/RedHatAI/Qwen3.8-Flash-Next-NVFP4}"
# One TP=4 replica instead of TP=1 x DP=4: vLLM's prefix cache is per replica and DP routes each turn of a sample
# to any replica, so multi-turn agent runs would prefill their history again on every turn.
export TP="${TP:-4}" DP="${DP:-1}"
# Copying the 51B n-gram embedding table from disk into pinned host memory takes several minutes, more than
# vLLM's default 600 s engine start-up budget.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
# Rootless Docker (image store on /data, see local_env_notes.md) with its own static CLI and the compose plugin.
export DOCKER_HOST="${DOCKER_HOST:-unix:///run/user/$(id -u)/docker.sock}"
export PATH="$HOME/.local/docker-rootless/bin:$PATH"
mkdir -p logs

source ~/.venvs/vllm-nightly-ple-fix/bin/activate
./serve.sh > logs/serve.log 2>&1 &
SERVE_PID=$!
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
  # give up if the server died during startup (e.g. a GPU was busy), instead of polling forever
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

source ~/.venvs/inspect-harbor-fix/bin/activate
./run_eval.sh "$@"

kill $SERVE_PID
