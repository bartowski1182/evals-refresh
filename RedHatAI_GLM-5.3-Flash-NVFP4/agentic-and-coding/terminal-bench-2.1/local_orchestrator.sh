#!/usr/bin/env bash
# Local glue, specific to one server: venvs, GPUs, serve in background, eval, tear down.
# Treat this as a template and adapt paths/GPUs to your own environment.
# This host has 8 idle B200s and only this model under test, so one TP=8 replica uses all of them: a single
# coherent prefix cache, which multi-turn agent histories need (DP would route each turn to a random replica).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PORT="${PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-/data/kylesayrs/hub/RedHatAI/GLM-5.3-Flash-NVFP4}"
export TP="${TP:-8}" DP="${DP:-1}"
# GLM-5.3-Flash is ~169B (NVFP4 ~150 GB) plus an MTP module and a torch.compile warmup; the first start can
# exceed vLLM's default 600 s engine start-up budget.
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
# This host runs the system (rootful) Docker daemon on the default socket (/var/run/docker.sock); the invoking
# user is in the docker group, so nothing to override here.
mkdir -p logs

source /home/kylesayrs/vllm/.venv/bin/activate
./serve.sh > logs/serve.log 2>&1 &
SERVE_PID=$!
until curl -sf "http://localhost:${PORT}/health" >/dev/null; do
  # give up if the server died during startup (e.g. a GPU was busy), instead of polling forever
  kill -0 $SERVE_PID 2>/dev/null || { echo "vLLM server exited before becoming healthy, see logs/serve.log" >&2; exit 1; }
  sleep 10
done

source /home/kylesayrs/.venvs/glm-inspect-harbor/bin/activate
./run_eval.sh "$@"

kill $SERVE_PID
