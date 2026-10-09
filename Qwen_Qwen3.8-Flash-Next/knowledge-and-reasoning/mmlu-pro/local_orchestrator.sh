#!/usr/bin/env bash
# Local glue, specific to one server: venvs, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs to your own environment.
# The BF16 model takes half of the 8 GPUs (4-7, port 8003); the nvidia NVFP4 dirs use GPUs 0-3 and port 8002,
# so the two checkpoints can be evaluated side by side.
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PORT="${PORT:-8003}"
export MODEL_PATH="${MODEL_PATH:-/data/eldarkurtic/hf_models/Qwen/Qwen3.8-Flash-Next}"
# Copying the 51B n-gram embedding table from disk into pinned host memory takes several minutes, more than
# vLLM's default 600 s engine start-up budget.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
mkdir -p logs

source ~/.venvs/vllm-nightly/bin/activate
./serve.sh > logs/serve.log 2>&1 &
SERVE_PID=$!
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
  # give up if the server died during startup (e.g. a GPU was busy), instead of polling forever
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

source ~/.venvs/inspect/bin/activate
./run_eval.sh "$@"

kill $SERVE_PID
