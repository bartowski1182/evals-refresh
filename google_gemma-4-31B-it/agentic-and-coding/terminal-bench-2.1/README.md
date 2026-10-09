# google/gemma-4-31B-it — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task, bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/Google/gemma-4-31B-it |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

**Protocol note.** The model card does not report Terminal-Bench; the official leaderboard uses the Terminus 2 agent. `inspect_harbor` solves every task with Inspect's generic ReAct agent
(`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context compaction), not
Terminus 2. Harbor's per-task agent timeouts (600 s to 12000 s, 48 of 89 tasks at 900 s) are not
applied by `inspect_harbor`; a single `--time-limit 7200` is used for every task instead. Containers run
without Harbor's per-task CPU cap (`override_cpus=0`, see `run_eval.sh`) and with `inspect_harbor`'s 6 GB
memory floor. All 267 samples share one vLLM server, so the wall-clock limit binds harder than it would
against an API endpoint: per-sample generation speed drops with the number of samples in flight. The two
numbers are therefore not directly comparable.

## Files

- `assets/generate_config.json` — Inspect generate config that cannot be given as a CLI flag; here it
  carries the per-request `chat_template_kwargs` that enable thinking.
- `assets/tool_chat_template_gemma4.jinja` — vLLM's Gemma 4 chat template (from the recipe), used by `serve.sh`.
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

`temperature=1.0, top_p=0.95, top_k=64, chat_template_kwargs={"enable_thinking": true}, max_tokens=65536 (per turn), max_connections=267, max_sandboxes=267, time_limit=7200s, client_timeout=7200, override_cpus=0, epochs=3 (mean)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (avg@3) | 0.397 |
| stderr | 0.0452 |
| per-epoch pass rate | 0.382 / 0.404 / 0.404 |
| samples | 267 (89 tasks x 3 epochs) |
| tasks passing 3/3, 2/3 or 1/3, 0/3 | 23 / 24 / 42 |
| time-limit hits (7200 s) | 7 (0 of them still passed the verifier) |
| samples ending without `submit()` | 267 |
| turns truncated at max_tokens | 6 |
| context compactions | 1 |
| sample errors | 0 |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 741,363 | 15,572 | 8,195 |
| median | 273,063 | 10,560 | 6,594 |
| max | 18,594,301 | 150,805 | 82,651 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per
sample: median 20, mean 29, max 342. Wall time per sample: median 2,120 s,
mean 2,406 s, max 7,271 s. The verifier scores the container state whether or not the agent
called `submit()`, so ending without it is not a failure by itself. Total run time about 2 h 01 min on
8x H100 (267 concurrent samples).

Always-failing tasks (0/3): break-filter-js-from-html, build-cython-ext, caffe-cifar-10, chess-best-move, circuit-fibsqrt, code-from-image, db-wal-recovery, dna-assembly, dna-insert, extract-elf, extract-moves-from-video, filter-js-from-html, financial-document-processor, fix-ocaml-gc, gcode-to-text, gpt2-codegolf, install-windows-3.11, log-summary-date-ranges, make-doom-for-mips, make-mips-interpreter, merge-diff-arc-agi-task, model-extraction-relu-logits, mteb-leaderboard, mteb-retrieve, openssl-selfsigned-cert, path-tracing, path-tracing-reverse, protein-assembly, pytorch-model-cli, qemu-alpine-ssh, qemu-startup, raman-fitting, regex-chess, sam-cell-seg, schemelike-metacircular-eval, sqlite-db-truncate, torch-pipeline-parallelism, train-fasttext, tune-mjcf, video-processing, winning-avg-corewars, write-compressor.

## Notes for future runs

- Only 7 of 267 samples hit the 7200 s limit, so the limit does not bind for this model. The limit
  is wall-clock per sample while all in-flight samples share one vLLM server, so per-sample
  generation speed scales with 1/concurrency; lower concurrency would give each sample more
  generation budget at proportionally longer run time. Per-task Harbor timeouts would be the
  faithful fix but `inspect_harbor` does not support them.
- 24 of the 42 always-failing tasks also always fail for Qwen3.8-27B in this scaffold (see that README).

## Patches for upstream

None.
