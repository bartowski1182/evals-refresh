# Qwen/Qwen3.8-Flash-Next — Terminal-Bench 2.1

| | |
|---|---|
| Task | `inspect_harbor/terminal_bench_2_1` (89 tasks, Harbor hub package `terminal-bench/terminal-bench-2-1` pinned to digest `sha256:7d7bdc1c…`) |
| Category | agentic-and-coding (subclass: swe-agent) |
| Exec tier | `agent` — one prebuilt Docker container per task (Docker Hub), bash/python tools, hidden verifier |
| vLLM recipe | https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next |
| Task source | https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1 |

**Protocol note.** The official leaderboard uses the Terminus 2 agent; `inspect_harbor` solves every task with
Inspect's generic ReAct agent (`bash`, `python`, `update_plan`, `submit`; 300 s tool timeout; automatic context
compaction). Harbor's per-task agent timeouts (600 s to 12000 s, most 900 s) are not applied; a single
`--time-limit 7200` is used for every task. Containers run without Harbor's per-task CPU cap (`override_cpus=0`)
and with `inspect_harbor`'s 6 GB memory floor; the tasks have public network by design. One epoch (pass@1) instead
of the avg@3 of the other Terminal-Bench dirs of this repo, by request. The Qwen/Qwen3.8-Flash-Next card does not
report Terminal-Bench 2.1, and the numbers are not comparable to Terminus 2 leaderboard entries. Checkpoint: the
original BF16 weights, revision `de4b8e4`.

## Files

- `serve.sh` — deploys the model with a vLLM server. vLLM-serve arguments live here (parallelism, context length,
  KV-cache dtypes, reasoning/tool parsers, text-only mode), taken from the official recipe (Blackwell, features
  tool_calling, reasoning, text_only).
- `run_eval.sh` — runs the Inspect task against that server. Inspect-side params live here
  (task and dataset pin, epochs, sampling params, reasoning effort, per-turn max tokens, time limit, concurrency).
- `server_requirements.txt` / `client_requirements.txt` — `pip freeze` of the vLLM and Inspect
  venvs that produced the results below, with the host (GPUs, driver, CUDA, Python) in the header.
- `local_orchestrator.sh` — runs everything in your local dev environment: activates the venvs, sets GPUs
  (4-7, port 8003), one TP=4 replica (so each sample's prefix cache stays on one replica), the local
  checkpoint path and the Docker socket, starts `serve.sh` in the background, runs `run_eval.sh`, tears down.
  **This is the only file you should modify** to match the environment you run in.

## Run

### Setup (once per host)

Create a server and a client environment as below. The commands use the paths that `local_orchestrator.sh` activates;
if you keep the environments elsewhere, change the paths there. `server_requirements.txt` and
`client_requirements.txt` are the exact `pip freeze` of the environments that produced the results below (Python 3.12,
host details in their headers).

```bash
# server: vLLM nightly 0.30.1rc1.dev418+g50319916a (the recipe requires a nightly build; CUDA 13 wheels)
uv venv ~/.venvs/vllm-nightly --python 3.12 && source ~/.venvs/vllm-nightly/bin/activate
uv pip install -r server_requirements.txt --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/50319916a3c72ec1358f9e29c7c282d316d86f78 \
  --extra-index-url https://download.pytorch.org/whl/cu130
# client: Inspect + inspect_harbor
uv venv ~/.venvs/inspect-harbor --python 3.12 && source ~/.venvs/inspect-harbor/bin/activate
uv pip install -r client_requirements.txt   # or: uv pip install "inspect-harbor==1.0.0" httpx2 && uv pip install "openai==3.14.0"
```

Docker with the compose plugin (v2.x or later) and buildx >= 0.17 is required. The results below were produced with
a rootless Docker 29 daemon (containerd image store on local NVMe, 512 network pools, runc with `NoNewKeyring`);
`local_orchestrator.sh` points `DOCKER_HOST` at its socket. Inspect builds a task's image before its first sample
if it is missing (serially, before any sample starts). If you pre-built the `hb__<hash>` images, pass
`--sandbox-prebuilt` to skip that step.

`local_orchestrator.sh` serves the checkpoint from a local copy (`MODEL_PATH`); unset it to let vLLM fetch
`Qwen/Qwen3.8-Flash-Next` from the Hugging Face Hub. Export `HF_TOKEN` in your shell before running (never in the
repo); the Harbor task files are fetched from GitHub at the pinned commit.

### Run the eval

```bash
./local_orchestrator.sh                  # add --sandbox-prebuilt if the task images are pre-built
inspect view --log-dir ./logs
```

Always smoke-test first: `./local_orchestrator.sh --limit 2`

## Results

With model generation config:

`temperature=1.0, top_p=0.95, top_k=20, reasoning_effort=xhigh, max_tokens=65536 (per turn), max_connections=45, max_sandboxes=45, time_limit=7200s, client_timeout=7200, override_cpus=0, TP=4 DP=1, epochs=1; errored samples retried once with `inspect eval-retry` (completed samples kept)`

the following scores and token stats are obtained:

| metric | value |
|---|---|
| accuracy (pass@1, 1 epoch, sample errors excluded) | 0.855 (71 of 83 scored tasks) |
| accuracy counting sample errors as failures | 0.798 (71 of 89) |
| samples that downloaded their own task files (tests, reference solution) from a public benchmark repo | 0 |
| stderr | 0.039 |
| samples | 89 (89 tasks x 1 epoch) |
| time-limit hits (7200 s) | 0 |
| turns truncated at max_tokens | 1 |
| context compactions | 21 |
| sample errors | 6 (adaptive-rejection-sampler, extract-moves-from-video, gpt2-codegolf, make-doom-for-mips, make-mips-interpreter, winning-avg-corewars) |

| tokens / sample | input | output | reasoning |
|---|---|---|---|
| mean | 3,405,925 | 65,178 | 44,925 |
| median | 1,221,085 | 40,196 | 24,903 |
| max | 25,356,641 | 335,929 | 280,576 |

Input tokens are cumulative over agent turns (history is resent each turn). Agent turns per sample: median 29, mean
62, max 479. Wall time per sample: median 1,159 s, mean 1,486 s, max 4,811 s. Total run time about 2.7 h including
the first pass and the retry on 4x B200 (one TP=4 replica, 45 concurrent samples), with the other model's runs on
the other four GPUs of the host at the same time.

Failed tasks: bn-fit-modify, chess-best-move, configure-git-webserver, distribution-search, filter-js-from-html,
llm-inference-batching-scheduler, mteb-retrieve, prove-plus-comm, pytorch-model-cli, qemu-alpine-ssh, qemu-startup,
video-processing.

## Patches for upstream

No dataset patch is needed. Known issues in the tooling, documented here for a future run: (1) Inspect 0.3.273's
`vllm` provider learns the served context window (262,144) from the server's `/v1/models` on a sample's first model
call, but the ReAct agent's compaction resolves its threshold (90% of the window) at sample start, so samples that
start before the first call fall back to Inspect's 128,000 default. Both thresholds (115,200 and 235,929 tokens)
occur in these runs. Pinning the window needs `set_model_info()` in Python (no CLI option), or an Inspect fix that
resolves the threshold after registration. (2) Trajectories whose own messages exceed the threshold after
compaction end with `RuntimeError: Compaction insufficient` and are not graded; they are counted both ways above.
(3) The tasks have public network by design (several verifiers `apt-get install` at test time), and nothing stops
an agent from fetching the task's own files, including its reference solution, from the public
`harbor-framework/terminal-bench-2-1` repository or mirrors; such samples are flagged above.
