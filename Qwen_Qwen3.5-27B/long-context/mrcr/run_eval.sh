#!/usr/bin/env bash
# MRCR on Qwen/Qwen3.5-27B against an already-running vLLM server.
# Sampling follows the model card's thinking-mode general-task recommendation (user request),
# not generation_config.json (temp 0.6). Forwarded via gen_kwargs / extra_body so top_k,
# presence_penalty, and repetition_penalty reach vLLM (Inspect's GenerateConfig drops some).
# do_sample is accepted but ignored by vLLM's OpenAI-compatible server (HF generate() kwarg).
# Thinking is enabled server-side; Qwen3.5 has no reasoning_effort levels.
# -M client_timeout raises Inspect's HTTP timeout (default 600s).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
PORT="${PORT:-8000}"
EPOCHS="${EPOCHS:-1}"
LOG_DIR="${LOG_DIR:-./logs}"
# Card thinking-mode general tasks; override with GEN_KWARGS='k=v,...' if needed.
export GEN_KWARGS="${GEN_KWARGS:-temperature=1.0,top_p=0.95,top_k=20,presence_penalty=1.5,repetition_penalty=1.0,do_sample=true}"

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-local}"

# Convert GEN_KWARGS=k=v,... into a JSON object for -T gen_kwargs=...
GEN_KWARGS_JSON=$(
  python3 -c '
import json, os, sys
raw = os.environ["GEN_KWARGS"]
out = {}
for part in raw.split(","):
    part = part.strip()
    if not part:
        continue
    k, _, v = part.partition("=")
    k, v = k.strip(), v.strip()
    if v.lower() in ("true", "false"):
        out[k] = v.lower() == "true"
    else:
        try:
            out[k] = int(v) if v.isdigit() or (v.startswith("-") and v[1:].isdigit()) else float(v)
        except ValueError:
            out[k] = v
print(json.dumps(out))
' 
)

inspect eval inspect_evals/mrcr \
  --model "vllm/${MODEL}" \
  --epochs "$EPOCHS" \
  --temperature 1.0 --top-p 0.95 --top-k 20 --presence-penalty 1.5 \
  -T max_model_len=262144 \
  -T "gen_kwargs=${GEN_KWARGS_JSON}" \
  --max-connections "${MAX_CONNECTIONS:-16}" \
  -M client_timeout=7200 \
  --log-dir "$LOG_DIR" \
  "$@"
