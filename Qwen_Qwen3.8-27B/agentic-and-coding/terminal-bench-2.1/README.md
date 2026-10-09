# Qwen/Qwen3.8-27B — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task, bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-27B |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

**Protocol note.** The model card reports 73.0 for "Terminal Bench 2.1 (Terminus)", i.e. with the Terminus 2 agent used by the official leaderboard. `inspect_harbor` solves every task with Inspect's generic ReAct agent
(`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context compaction), not
Terminus 2. Harbor's per-task agent timeouts (600 s to 12000 s, 48 of 89 tasks at 900 s) are not
applied by `inspect_harbor`; a single `--time-limit 7200` is used for every task instead. Containers run
without Harbor's per-task CPU cap (`override_cpus=0`, see `run_eval.sh`) and with `inspect_harbor`'s 6 GB
memory floor. All 267 samples share one vLLM server, so the wall-clock limit binds harder than it would
against an API endpoint: per-sample generation speed drops with the number of samples in flight. The two
numbers are therefore not directly comparable.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here
  (parallelism, context length, reasoning/tool parsers), taken from the official recipe.
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and package digest, epochs, sampling params, per-turn max tokens, concurrency, sandbox count, time limit).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below. The client venv is the `inspect-harbor` one (see Run).
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the
  venvs, sets GPUs, points Docker at the rootless daemon, starts `serve.sh` in the background,
  runs `run_eval.sh`, tears down. **This is the only file you should modify** to match the
  environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and `client_requirements.txt`
are the exact `pip freeze` of the environments that produced the results below (Python 3.12, host details in their
headers): installing from them reproduces those environments; the commented alternative gives an equivalent fresh
install with the same pinned versions.

```bash
# server: vLLM 0.29.0 (CUDA 13 wheels; add the matching https://download.pytorch.org/whl/<cuXXX> index if pip cannot resolve torch)
python3.12 -m venv ~/.venvs/vllm && source ~/.venvs/vllm/bin/activate
pip install -r server_requirements.txt          # or: pip install vllm==0.29.0
# client: Inspect
python3.12 -m venv ~/.venvs/inspect-harbor && source ~/.venvs/inspect-harbor/bin/activate
pip install -r client_requirements.txt          # or:
# pip install "inspect-harbor==0.7.5" httpx2 "openai==3.14.0"
```

`inspect-harbor` pulls in `litellm`, which pins `openai<3`, while Inspect's OpenAI-compatible providers need
`openai>=3`; installing `openai==3.14.0` last overrides the pin (pip prints a conflict warning that is harmless
here because Inspect's agent path never imports `litellm`). `httpx2` is imported by `inspect-ai` but not declared.

Model weights and datasets are fetched from the Hugging Face Hub on first use into the usual cache (`HF_HOME`
if set). Export `HF_TOKEN` in your shell before running (never in the repo); nothing here is gated, but anonymous Hub
downloads are rate-limited.

Docker (with the compose plugin) must be reachable by your user. `local_orchestrator.sh` exports `DOCKER_HOST` for
a rootless daemon (`unix:///run/user/<uid>/docker.sock`, image store under `~/.local/share/docker`); with a
rootful daemon drop that export. Check the image-store disk space quoted below before the first run. Rootless
Docker needs a runc runtime with `NoNewKeyring` in `daemon.json` (one kernel key per container otherwise
exhausts a user's quota of 200) and cannot apply CPU limits, hence `-T override_cpus=0` in `run_eval.sh`.

### Run the eval

```bash
./local_orchestrator.sh
inspect view --log-dir ./logs
```

Always smoke-test first: `EPOCHS=1 ./local_orchestrator.sh --limit 2`

Requires Docker with roughly 70 GB free in its image store. **What to expect on the first run:**
Harbor downloads the 89 task definitions into `~/.cache/harbor/tasks/packages/terminal-bench/`
(59 MB), then Inspect pulls one prebuilt image per task from Docker Hub as containers start
(21 GB compressed, about 70 GB on disk, nothing is built). Docker Hub's anonymous limit is 100 pulls per
hour per IP, so the first run may stall on pulls; pre-pull with `docker pull` if needed. Later runs reuse
the images. Once samples start, all 267 samples (89 tasks x 3 epochs) run at once; memory caps total
1.6 TB but real usage is far lower, and CPU is uncapped.

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=267, max_sandboxes=267, time_limit=7200s, client_timeout=7200, override_cpus=0, epochs=3 (mean)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (avg@3) | 0.493 |
| stderr | 0.0449 |
| per-epoch pass rate | 0.562 / 0.449 / 0.466 |
| samples | 267 (89 tasks x 3 epochs) |
| tasks passing 3/3, 2/3 or 1/3, 0/3 | 29 / 29 / 31 |
| time-limit hits (7200 s) | 138 (19 of them still passed the verifier) |
| samples ending without `submit()` | 266 |
| turns truncated at max_tokens | 9 |
| context compactions | 0 |
| sample errors | 1 (feal-differential-cryptanalysis epoch 3: verifier exceeded Harbor's 1800 s test timeout; excluded from the score) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 726,177 | 34,280 | 26,942 |
| median | 345,120 | 23,565 | 18,938 |
| max | 5,942,977 | 100,274 | 92,169 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 19, mean 23, max 88. Wall time per sample: median 7,207 s,
mean 4,612 s, max 7,764 s. The verifier scores the container state whether or not the agent
called `submit()`, so ending without it is not a failure by itself. Total run time about 2 h 30 min on
8x H100 (267 concurrent samples).

Always-failing tasks (0/3): build-pov-ray, chess-best-move, cobol-modernization, compile-compcert, dna-assembly, extract-moves-from-video, filter-js-from-html, fix-ocaml-gc, gcode-to-text, gpt2-codegolf, install-windows-3.11, largest-eigenval, mailman, make-doom-for-mips, make-mips-interpreter, mteb-leaderboard, mteb-retrieve, path-tracing, path-tracing-reverse, qemu-alpine-ssh, qemu-startup, query-optimize, raman-fitting, regex-chess, sanitize-git-repo, schemelike-metacircular-eval, torch-pipeline-parallelism, tune-mjcf, video-processing, winning-avg-corewars, write-compressor.

## Notes for future runs

- A first full run with `--time-limit 3600` (same config otherwise) scored 0.449 avg@3 with 153 of 267
  samples hitting the limit; raising it to 7200 s gave 0.493 with 138 hits, a gain within one stderr.
- The 119 samples that timed out and failed had already generated a median 36,600 output tokens over
  23 turns (passing samples: 15,700 tokens, 14 turns) and spent a median 3,400 s of their 7,280 s
  generating; the rest was tool execution (builds, training). More time is unlikely to flip them.
- 25 tasks timed out in all three epochs; Harbor's own budgets for 24 of them are 900 to 3600 s, so
  7200 s is already 2 to 8x the official budget. Only `build-pov-ray` (12000 s) has a larger one.
- The limit is wall-clock per sample while all in-flight samples share one vLLM server, so per-sample
  generation speed scales with 1/concurrency (about 9 tokens/s at 267). Running 89 at a time would
  give each sample about 3x the generation budget in the same 7200 s, at about 3x the run time, and
  would not change tool-execution time. Per-task Harbor timeouts would be the faithful fix but
  `inspect_harbor` does not support them.

## Patches for upstream

None.
