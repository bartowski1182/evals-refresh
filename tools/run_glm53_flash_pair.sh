#!/usr/bin/env bash
# Run the GLM-5.3-Flash pair's agentic benchmarks one after the other, interactively (meant for a tmux session).
# For each (model, task) in order: smoke test (--limit 2, logs in logs/smoke/), a summary of its samples and a pause so
# you can read them, the full run (logs in logs/), the section 11 leak audit (tools/leak_audit.py), the two
# requirements files, and a pause. Every run goes through the task dir's own local_orchestrator.sh, so what is
# measured stays in serve.sh / run_eval.sh; this script only sequences the dirs and waits for the GPUs between them.
#
#   tools/run_glm53_flash_pair.sh                      everything: 2 models x 3 tasks, shortest task first
#   MODELS="zai-org_GLM-5.3-Flash" TASKS="terminal-bench-2.1" tools/run_glm53_flash_pair.sh      a subset
#   SKIP_SMOKE=1        no smoke tests               ONLY_SMOKE=1   stop after each smoke test
#   AUTO=1              never pause (continue at every prompt; for an unattended run)
#   RUN_ARGS="..."      extra args for every full run (e.g. --sandbox-prebuilt)
#   INSPECT_VENV_DEEPSWE / INSPECT_VENV_SWEBENCH / INSPECT_VENV_TB    client venvs (defaults = the orchestrators')
#   MODEL_PATH_FP8 / MODEL_PATH_MXFP4XFP8    local checkpoint copies (else the Hub id through HF_HOME; export
#                                            HF_HUB_OFFLINE=1 once downloaded)
#   VLLM_VENV or VLLM_IMAGE, HF_HOME, HF_TOKEN, DOCKER_HOST, PORT, MAX_CONNECTIONS, ...   pass through as usual
# At a prompt: c = continue, s = skip the rest of this task, q = quit (the server is already stopped by then).
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS="${MODELS:-zai-org_GLM-5.3-Flash RedHatAI_GLM-5.3-Flash-MXFP4xFP8}"
TASKS="${TASKS:-terminal-bench-2.1 swe-bench-verified deepswe-1.1}"

venv_for() {
  case "$1" in
    deepswe-1.1) echo "${INSPECT_VENV_DEEPSWE:-$HOME/.venvs/inspect-harbor-fix}" ;;
    swe-bench-verified) echo "${INSPECT_VENV_SWEBENCH:-$HOME/.venvs/inspect-swebench}" ;;
    terminal-bench-2.1) echo "${INSPECT_VENV_TB:-$HOME/.venvs/inspect-harbor}" ;;
    *) echo "unknown task $1" >&2; exit 1 ;;
  esac
}
model_path_for() {
  case "$1" in
    zai-org_GLM-5.3-Flash) echo "${MODEL_PATH_FP8:-}" ;;
    RedHatAI_GLM-5.3-Flash-MXFP4xFP8) echo "${MODEL_PATH_MXFP4XFP8:-}" ;;
    *) echo "" ;;
  esac
}

ask() {  # sets REPLY to c, s or q
  if [ "${AUTO:-0}" = 1 ]; then REPLY=c; return; fi
  while true; do
    read -r -p "$1 [c]ontinue / [s]kip rest of this task / [q]uit: " REPLY
    case "$REPLY" in c|s|q) return ;; esac
  done
}

wait_gpus_idle() {  # a server from the previous run must be gone before the next one starts (AGENTS.md, section 9)
  local used i
  for i in $(seq 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    [ "${used:-0}" -lt 1024 ] && return 0
    echo "waiting for the GPUs to free up (max ${used} MiB in use)"; sleep 10
  done
  echo "GPUs still busy after 15 min; check nvidia-smi" >&2; return 1
}

smoke_summary() {  # $1 = log dir
  "$INSPECT_VENV/bin/python" - "$1" <<'PY'
import glob, sys
from inspect_ai.log import read_eval_log
for f in sorted(glob.glob(sys.argv[1] + "/*.eval")):
    log = read_eval_log(f)
    print(f"{f}: status={log.status}")
    print("  model_generate_config:", log.eval.model_generate_config)
    for s in log.samples or []:
        reasoning = any(getattr(c, "type", None) == "reasoning"
                        for m in s.messages if m.role == "assistant" and not isinstance(m.content, str)
                        for c in m.content)
        print(f"  {s.id}: scores={ {k: v.value for k, v in (s.scores or {}).items()} }"
              f" stop_reason={s.output.stop_reason if s.output else None} reasoning_part={reasoning}"
              f" error={bool(s.error)} limit={s.limit.type if s.limit else None}"
              f" tokens={s.output.usage if s.output and s.output.usage else None}")
PY
}

write_requirements() {  # headers as in the other dirs: date, GPUs, OS, driver, CUDA, Python
  local date gpus driver cuda os env_line env_kind env_ref spy cpy
  date=$(date +%F)
  gpus="$(nvidia-smi -L | wc -l)x $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 | sed 's/, / /')"
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
  cuda=$(nvidia-smi | grep -o 'CUDA Version: [0-9.]*' | head -1)
  os=$(. /etc/os-release && echo "$PRETTY_NAME")
  env_line=$(head -1 logs/serve.log)                       # "# vLLM env (podman|venv): <ref>", written by tools/vllm_server.sh
  env_kind=$(sed -n 's/^# vLLM env (\([a-z]*\)): .*/\1/p' <<<"$env_line")
  env_ref=${env_line#*: }
  if [ "$env_kind" = venv ]; then
    spy=$("$env_ref/bin/python" --version 2>&1)
    { echo "# vLLM serving env (serve.sh). Generated $date with pip freeze from the venv that ran the eval."
      echo "# Host: $gpus, $os, NVIDIA driver $driver, $cuda, $spy."
      "$env_ref/bin/python" -m pip freeze; } > server_requirements.txt
  else
    spy=$(podman run --rm --entrypoint python3 "$env_ref" --version 2>&1)
    { echo "# vLLM serving env (serve.sh). Generated $date with pip freeze inside the podman image that ran the eval ($env_ref)."
      echo "# Host: $gpus, $os, NVIDIA driver $driver, $cuda, $spy."
      podman run --rm --entrypoint python3 "$env_ref" -m pip freeze; } > server_requirements.txt
  fi
  cpy=$("$INSPECT_VENV/bin/python" --version 2>&1)
  { echo "# Inspect eval client env (run_eval.sh). Generated $date with pip freeze from the venv that ran the eval."
    echo "# Host: $os, $cpy."
    "$INSPECT_VENV/bin/python" -m pip freeze; } > client_requirements.txt
  echo "wrote server_requirements.txt and client_requirements.txt"
}

run_one() {
  local model=$1 task=$2 dir status mp
  dir="$ROOT/$model/agentic-and-coding/$task"
  cd "$dir" || exit 1
  export INSPECT_VENV; INSPECT_VENV=$(venv_for "$task")
  [ -x "$INSPECT_VENV/bin/python" ] || { echo "client venv $INSPECT_VENV missing (see $dir/README.md, Setup)" >&2; exit 1; }
  mp=$(model_path_for "$model"); if [ -n "$mp" ]; then export MODEL_PATH="$mp"; else unset MODEL_PATH; fi
  echo; echo "################ $model / $task  ($(date))"; echo

  if [ "${SKIP_SMOKE:-0}" != 1 ]; then
    wait_gpus_idle || exit 1
    mkdir -p logs/smoke
    echo "== smoke test (--limit 2), logs in $dir/logs/smoke"
    LOG_DIR=./logs/smoke EPOCHS=1 ./local_orchestrator.sh --limit 2 2>&1 | tee logs/smoke/full.log
    status=${PIPESTATUS[0]}
    echo "== smoke test exited with $status"
    smoke_summary logs/smoke || true
    echo "Read the samples (another shell: inspect view --log-dir $dir/logs/smoke). Check: reasoning parsed into a"
    echo "separate part, stop_reason=stop, the intended config recorded, and in logs/serve.log the quantization"
    echo "method, KV cache dtype (bfloat16), MTP, and the GPU KV cache size against the task's concurrency."
    ask "Start the full run of $model / $task?"
    case "$REPLY" in s) return ;; q) exit 0 ;; esac
    [ "${ONLY_SMOKE:-0}" = 1 ] && return
  fi

  wait_gpus_idle || exit 1
  echo "== full run, logs in $dir/logs"
  # shellcheck disable=SC2086
  ./local_orchestrator.sh ${RUN_ARGS:-} 2>&1 | tee logs/full.log
  status=${PIPESTATUS[0]}
  echo "== full run exited with $status ($(date))"

  echo "== leak audit (AGENTS.md, section 11)"
  local offline=()
  [ "$task" != terminal-bench-2.1 ] && offline=(--offline)
  "$INSPECT_VENV/bin/python" "$ROOT/tools/leak_audit.py" "${offline[@]}" --out logs/leak_hits.jsonl logs/*.eval \
    2>&1 | tee logs/leak_audit.txt | tail -5
  echo "full audit output in $dir/logs/leak_audit.txt; review the hits by hand (section 11 lists what counts as a leak)"

  write_requirements
  ask "Done with $model / $task."
  case "$REPLY" in q) exit 0 ;; esac
}

echo "Models: $MODELS"; echo "Tasks: $TASKS"; echo "Repo: $ROOT"
for model in $MODELS; do
  for task in $TASKS; do
    [ -d "$ROOT/$model/agentic-and-coding/$task" ] || { echo "no dir $model/agentic-and-coding/$task" >&2; exit 1; }
    [ -x "$(venv_for "$task")/bin/python" ] || echo "WARNING: client venv for $task not found at $(venv_for "$task")"
  done
done
for model in $MODELS; do
  for task in $TASKS; do
    run_one "$model" "$task"
  done
done
echo "all done ($(date))"
